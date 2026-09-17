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
``smiles_padding_length`` (paccmann.py:244-253). It then averages them over the
head axis into ``prediction_dict['smiles_attention']``
(paccmann.py:273-279), **destroying the per-head structure** that the vLLM
payload goes to some trouble to preserve.

Hooking ``ContextAttentionLayer`` catches the 16 vectors *before* that average,
so ``attn_peak`` and ``attn_argmax_head`` are real rather than reconstructed
from a mean. The gene axis is single-headed (``gene_attention_layer``,
paccmann.py:231) and is captured the same way.

Both are also thrown away by the caller: ``test_paccmann.py:161`` binds
``pred_dict`` and never reads it. This module is the difference between that
data existing and that data being recorded.

Usage
-----
::

    from vllm_attn_connector.drp_paccmann import PaccmannCapture

    with Flowcept("vllm", workflow_id=wf, workflow_name="paccmann") as fc:
        cap = PaccmannCapture(model, workflow_id=wf)
        cap.send_workflow(params)
        for smiles, gep, y in loader:
            y_hat, _ = model(torch.squeeze(smiles), gep)
            cap.emit(sample_ids=[...])      # one call per batch
        cap.remove()

``remove()`` detaches the hooks. ``PaccmannCapture`` is also a context manager,
which is the safer spelling when an exception can skip the teardown.
"""

from __future__ import annotations

from typing import Any, Sequence

from .drp_emit import ATTENTION_ACTIVITY, emit_batch

__all__ = ["PaccmannCapture"]


# Paccmann's attention is additive (Bahdanau): the alphas come from
# `alpha_projection(tanh(reference_attention + context_attention))`
# (layers.py:232-234), not from a scaled dot product. Recorded so these numbers
# are never silently compared against the vLLM connector's, which measure
# `softmax(q.K^T/sqrt d)`.
METRIC_REFERENCE = "paccmann_mca_context_attention_additive"


class PaccmannCapture:
    """Collect ``ContextAttentionLayer`` outputs from an MCA model.

    Attaches one forward hook per attention layer. Each hook stores the alphas
    for the most recent forward pass; ``emit`` drains them.
    """

    def __init__(
        self,
        model: Any,
        workflow_id: str,
        interceptor: Any = None,
        activity: str = ATTENTION_ACTIVITY,
    ) -> None:
        self.model = model
        self.workflow_id = workflow_id
        self.activity = activity

        if interceptor is None:
            from flowcept.flowceptor.adapters.vllm.vllm_interceptor import (
                VLLMInterceptor,
            )

            interceptor = VLLMInterceptor.get_instance()
        self.interceptor = interceptor

        # Ordered: the SMILES heads are registered in the order MCA builds them,
        # so index i is `multiheads[0] * layer + head` -- the same `ind` the
        # model computes at paccmann.py:248. That makes `attn_argmax_head`
        # directly interpretable as a (layer, head) pair.
        self._smiles: list[Any] = []
        self._gene: list[Any] = []
        self._handles: list[Any] = []
        self._attach()

    def _attach(self) -> None:
        # Imported here, not at module scope: this package must stay importable
        # without Paccmann on the path (the tests rely on that).
        from paccmann_predictor.utils.layers import ContextAttentionLayer

        model = self.model
        gene_layer = getattr(model, "gene_attention_layer", None)

        for module in model.modules():
            if isinstance(module, ContextAttentionLayer):
                self._handles.append(
                    module.register_forward_hook(self._smiles_hook)
                )

        if gene_layer is not None:
            self._handles.append(
                gene_layer.register_forward_hook(self._gene_hook)
            )

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

    def emit(
        self,
        sample_ids: Sequence[str],
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Emit the most recent forward pass and clear the buffers.

        ``sample_ids`` must have one entry per row of the batch. Returns the
        task ids written.
        """
        if not self._smiles and not self._gene:
            raise RuntimeError(
                "nothing captured; call emit() after a forward pass, and check "
                "the model is in eval() mode"
            )

        n = len(sample_ids)
        meta = {
            "model": "Paccmann_MCA",
            "metric": "attention_weights",
            "metric_reference": METRIC_REFERENCE,
            "framework": "torch",
        }
        if metadata:
            meta.update(metadata)

        axes: list[dict[str, Any]] = []

        if self._smiles:
            # Captured as [n_heads][bs, T]; regroup to [bs][n_heads][T] so each
            # sample carries its own per-head stack.
            per_sample = [[h[i] for h in self._smiles] for i in range(n)]
            axes.append({"name": "smiles", "per_head": per_sample})

        if self._gene:
            # Single-headed, but wrapped in a one-element list so the gene axis
            # goes through exactly the same reduction as the SMILES axis.
            g = self._gene[0]
            axes.append(
                {"name": "gene", "per_head": [[g[i]] for i in range(n)]}
            )

        try:
            return emit_batch(
                self.interceptor,
                workflow_id=self.workflow_id,
                sample_ids=sample_ids,
                axes=axes,
                metadata=meta,
                activity=self.activity,
            )
        finally:
            # Always clear, including on a failed emit: a retained batch would
            # otherwise be silently prepended to the next one.
            self.clear()

    def send_workflow(self, params: dict[str, Any]) -> None:
        """Record model identity on the workflow, once per run.

        The vLLM connector sends tokenizer identity so token ids can be decoded
        later; the analogue here is the SMILES language and the checkpoint.
        """
        keep = (
            "smiles_language_filepath",
            "smiles_vocabulary_size",
            "smiles_padding_length",
            "number_of_genes",
            "multiheads",
            "modelpath",
        )
        conf = {k: params[k] for k in keep if k in params}
        conf["model"] = "Paccmann_MCA"
        conf["metric_reference"] = METRIC_REFERENCE
        self.interceptor.send_model_workflow(self.workflow_id, conf)

    def remove(self) -> None:
        """Detach every hook. Safe to call more than once."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> "PaccmannCapture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()
