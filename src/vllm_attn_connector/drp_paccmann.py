# SPDX-License-Identifier: Apache-2.0
"""Capture Paccmann MCA attention at inference. The torch half.

Paccmann MCA is a bimodal drug-sensitivity regressor: a SMILES string and a
gene-expression vector in, one IC50 out. It attends over both, and the weights
are already computed on the inference path.

Why hooks work here and not in vLLM
-----------------------------------
``probe.py`` overrides the *attention backend impl* rather than registering an
``nn.Module`` forward hook, because ``torch.compile`` traces through
``Attention.forward`` and CUDA graphs would stop a module hook from firing.

Paccmann has neither. It runs eager, uncompiled, one pass. So the plain
``register_forward_hook`` that would not work in vLLM is exactly the right tool
here -- and it means **no Paccmann source is patched**, matching this project's
stance on vLLM.

What gets captured
------------------
``MCA.forward`` builds ``smiles_alphas`` as one tensor per (layer, head) --
with ``multiheads=[4,4,4,4]`` that is 16 vectors of length
``smiles_padding_length`` (``paccmann.py:244-253``). It then averages them over
the head axis into ``prediction_dict['smiles_attention']``
(``paccmann.py:273-279``), **destroying the per-head structure** that the vLLM
payload goes to some trouble to preserve.

Hooking ``ContextAttentionLayer`` catches the 16 vectors *before* that average,
so ``attn_peak`` and ``attn_argmax_head`` are real rather than reconstructed
from a mean. The gene axis is single-headed (``gene_attention_layer``,
``paccmann.py:231``) and is captured the same way.

Both are also thrown away by the caller: ``test_paccmann.py:161`` binds
``pred_dict`` and never reads it. This module is the difference between that
data existing and that data being recorded.

Usage
-----
::

    from vllm_attn_connector.drp_paccmann import PaccmannCapture

    with Flowcept("vllm", workflow_id=wf, workflow_name="paccmann"):
        with PaccmannCapture(model, workflow_id=wf) as cap:
            cap.send_workflow(params)
            for smiles, gep, y in loader:
                y_hat, _ = model(torch.squeeze(smiles), gep)
                cap.emit(sample_ids=[...])      # one call per batch
"""

from __future__ import annotations

from typing import Any

from .drp_capture import AttentionAxis, AttentionCapture

__all__ = ["GENE_AXIS", "SMILES_AXIS", "PaccmannCapture"]

SMILES_AXIS = "smiles"
GENE_AXIS = "gene"

#: Workflow keys worth recording: what a later reader needs to interpret a bare
#: attention vector -- the vocabulary it was tokenised with, the widths, and
#: which checkpoint produced it.
_WORKFLOW_KEYS = (
    "smiles_language_filepath",
    "smiles_vocabulary_size",
    "smiles_padding_length",
    "number_of_genes",
    "multiheads",
    "modelpath",
)


class PaccmannCapture(AttentionCapture):
    """Collect ``ContextAttentionLayer`` outputs from an MCA model.

    Attaches one forward hook per attention layer. Each hook stores the alphas
    for the most recent forward pass; :meth:`emit` drains them.

    Parameters
    ----------
    model:
        A Paccmann MCA model, in ``eval()`` mode.
    workflow_id:
        Flowcept workflow to attach records to.
    interceptor:
        Defaults to Flowcept's ``VLLMInterceptor``. Injectable for tests.
    """

    MODEL_NAME = "Paccmann_MCA"
    FRAMEWORK = "torch"
    # Additive (Bahdanau): alphas come from
    # `alpha_projection(tanh(reference_attention + context_attention))`
    # (layers.py:232-234), not from a scaled dot product.
    METRIC_REFERENCE = "paccmann_mca_context_attention_additive"

    def __init__(self, model: Any, workflow_id: str, **kwargs: Any) -> None:
        super().__init__(model, workflow_id, **kwargs)
        # Ordered: SMILES heads are registered in the order MCA builds them, so
        # index i is `multiheads[0] * layer + head` -- the same `ind` the model
        # computes at paccmann.py:248. That makes `attn_argmax_head` directly
        # interpretable as a (layer, head) pair.
        self._smiles: list[Any] = []
        self._gene: list[Any] = []
        self._handles: list[Any] = []
        self._attach()

    def _attach(self) -> None:
        # Imported here, not at module scope: this package must stay importable
        # without Paccmann on the path (the unit tests rely on that).
        from paccmann_predictor.utils.layers import ContextAttentionLayer

        for module in self.model.modules():
            if isinstance(module, ContextAttentionLayer):
                self._handles.append(module.register_forward_hook(self._smiles_hook))

        gene_layer = getattr(self.model, "gene_attention_layer", None)
        if gene_layer is not None:
            self._handles.append(gene_layer.register_forward_hook(self._gene_hook))

        if not self._handles:
            raise RuntimeError(
                "no attention layers found on this model; expected at least one "
                "ContextAttentionLayer. Is this an MCA model?"
            )

    def _smiles_hook(self, module: Any, inputs: Any, output: Any) -> None:
        # ContextAttentionLayer returns (output, alphas); alphas is [bs, T].
        self._smiles.append(output[1].detach())

    def _gene_hook(self, module: Any, inputs: Any, output: Any) -> None:
        # gene_attention_layer returns the alphas directly, [bs, n_genes].
        self._gene.append(output.detach() if hasattr(output, "detach") else output)

    @property
    def n_smiles_heads(self) -> int:
        """Attention heads seen in the last forward pass."""
        return len(self._smiles)

    def clear(self) -> None:
        """Drop everything captured so far without emitting it."""
        self._smiles.clear()
        self._gene.clear()

    def collect(self, n_samples: int | None = None, **_: Any) -> list[AttentionAxis]:
        """Drain the hooks into axes. Always clears, including on failure.

        A retained batch would otherwise be silently prepended to the next one,
        which would misattribute every subsequent record.
        """
        try:
            if not self._smiles and not self._gene:
                return []

            n = n_samples if n_samples is not None else self._infer_batch_size()
            axes: list[AttentionAxis] = []

            if self._smiles:
                # Captured as [n_heads][bs, T]; regroup to [bs][n_heads][T] so
                # each sample carries its own per-head stack.
                axes.append(
                    AttentionAxis(
                        name=SMILES_AXIS,
                        per_sample_heads=[[h[i] for h in self._smiles] for i in range(n)],
                    )
                )

            if self._gene:
                gene = self._gene[0]
                axes.append(
                    AttentionAxis(
                        name=GENE_AXIS,
                        per_sample_heads=[[gene[i]] for i in range(n)],
                    )
                )

            return axes
        finally:
            self.clear()

    def _infer_batch_size(self) -> int:
        source = self._smiles[0] if self._smiles else self._gene[0]
        return len(source)

    def emit(self, sample_ids, metadata=None, **kwargs: Any) -> list[str]:
        """Emit the most recent forward pass. See :meth:`AttentionCapture.emit`."""
        return super().emit(
            sample_ids, metadata=metadata, n_samples=len(sample_ids), **kwargs
        )

    def workflow_conf(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return {k: params[k] for k in _WORKFLOW_KEYS if params and k in params}

    def close(self) -> None:
        """Detach every hook. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    # Kept as an alias: `remove()` reads naturally for hooks, and the original
    # API used it.
    remove = close
