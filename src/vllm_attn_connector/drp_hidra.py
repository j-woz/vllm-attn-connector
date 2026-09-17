# SPDX-License-Identifier: Apache-2.0
"""Capture HiDRA attention at inference. The Keras half.

HiDRA predicts drug response from gene expression organised by KEGG pathway. It
is hierarchically attentive, and both levels are interesting:

*Gene level*
    One softmax per pathway, over the genes in that pathway
    (``<PATHWAY>_Attention_Softmax``, hidra_utils.py:91-92). With the shipped
    ``geneset.gmt`` that is 186 pathways, so 186 separate distributions.
*Pathway level*
    One softmax over the pathways themselves (``Sample_Attention_Softmax``,
    hidra_utils.py:111-112) -- which pathway mattered for this prediction.

The pathway level is the more useful of the two for attribution: it is a single
186-wide distribution per sample, directly comparable across samples, and it
answers "which biology drove this prediction". The gene level is captured on
request (``include_gene_level=True``) but is 186 separate vectors per sample,
which is a lot of provenance for one number.

Why this needs no HiDRA source change
-------------------------------------
Every attention layer is explicitly ``name=``d when the graph is built. Keras
lets you construct a second model over the same weights that outputs any named
intermediate tensor, so the attention comes out with **no edit to HiDRA and no
retraining** -- the extraction model shares the trained layers.

That is a different mechanism from the torch side (forward hooks) and from vLLM
(backend override), which is why the three capture paths are separate modules
over one shared emitter rather than one clever abstraction.

Attention here is a learned gate
--------------------------------
``Dense(tanh) -> softmax`` over concatenated gene and drug features, not
``softmax(q.K^T/sqrt d)`` and not Bahdanau either. ``metric_reference`` says so,
so these numbers are never silently pooled with the vLLM or Paccmann records.

Usage
-----
::

    from vllm_attn_connector.drp_hidra import HidraCapture

    with Flowcept("vllm", workflow_id=wf, workflow_name="hidra") as fc:
        cap = HidraCapture(model, workflow_id=wf)
        cap.send_workflow({"modelpath": ...})
        cap.emit(inputs=X_batch, sample_ids=[...])
"""

from __future__ import annotations

from typing import Any, Sequence

from .drp_emit import ATTENTION_ACTIVITY, emit_batch

__all__ = ["HidraCapture", "PATHWAY_LAYER", "GENE_LAYER_SUFFIX"]


PATHWAY_LAYER = "Sample_Attention_Softmax"
GENE_LAYER_SUFFIX = "_Attention_Softmax"

METRIC_REFERENCE = "hidra_hierarchical_attention_softmax_gate"


class HidraCapture:
    """Extract attention from a trained HiDRA Keras model.

    Builds one auxiliary model whose outputs are the attention tensors. The
    trained layers are shared, not copied, so this adds no parameters and
    changes no weights.
    """

    def __init__(
        self,
        model: Any,
        workflow_id: str,
        interceptor: Any = None,
        include_gene_level: bool = False,
        activity: str = ATTENTION_ACTIVITY,
    ) -> None:
        self.model = model
        self.workflow_id = workflow_id
        self.activity = activity
        self.include_gene_level = include_gene_level

        if interceptor is None:
            from flowcept.flowceptor.adapters.vllm.vllm_interceptor import (
                VLLMInterceptor,
            )

            interceptor = VLLMInterceptor.get_instance()
        self.interceptor = interceptor

        self.pathway_names = self._pathway_names()
        self._extractor = self._build_extractor()

    def _pathway_names(self) -> list[str]:
        """Gene-level attention layer names, in graph order.

        Graph order is also the order the pathway-level softmax indexes, so
        position i in the pathway distribution is ``pathway_names[i]``. That
        mapping is what makes the emitted vector interpretable.
        """
        return [
            layer.name[: -len(GENE_LAYER_SUFFIX)]
            for layer in self.model.layers
            if layer.name.endswith(GENE_LAYER_SUFFIX)
            and layer.name != PATHWAY_LAYER
        ]

    def _build_extractor(self) -> Any:
        from tensorflow.keras.models import Model

        try:
            outputs = [self.model.get_layer(PATHWAY_LAYER).output]
        except ValueError as exc:  # pragma: no cover - wrong model shape
            raise RuntimeError(
                f"layer {PATHWAY_LAYER!r} not found; this does not look like a "
                f"HiDRA model"
            ) from exc

        if self.include_gene_level:
            outputs += [
                self.model.get_layer(p + GENE_LAYER_SUFFIX).output
                for p in self.pathway_names
            ]

        return Model(inputs=self.model.inputs, outputs=outputs)

    def emit(
        self,
        inputs: Any,
        sample_ids: Sequence[str],
        metadata: dict[str, Any] | None = None,
        verbose: int = 0,
    ) -> list[str]:
        """Run the extractor over one batch and emit the attention.

        ``inputs`` is whatever the trained model takes -- for HiDRA, the list of
        per-pathway expression arrays plus the drug array, exactly as
        ``MultiGenerator`` yields it.
        """
        outs = self._extractor.predict(inputs, verbose=verbose)
        if not isinstance(outs, list):
            outs = [outs]

        pathway_attn = outs[0]
        n = len(sample_ids)
        if len(pathway_attn) != n:
            raise ValueError(
                f"got {len(pathway_attn)} rows of attention for {n} sample_ids"
            )

        meta = {
            "model": "HiDRA",
            "metric": "attention_weights",
            "metric_reference": METRIC_REFERENCE,
            "framework": "tensorflow",
            "n_pathways": len(self.pathway_names),
        }
        if metadata:
            meta.update(metadata)

        # Single-headed, so each sample's vector is wrapped in a one-element
        # list: the same reduction path as Paccmann's 16 heads, where sum and
        # peak simply coincide.
        axes: list[dict[str, Any]] = [
            {
                "name": "pathway",
                "per_head": [[pathway_attn[i]] for i in range(n)],
            }
        ]

        if self.include_gene_level:
            # One axis per pathway. Deliberately opt-in: this multiplies the
            # record count by the number of pathways.
            for p_idx, p_name in enumerate(self.pathway_names):
                arr = outs[1 + p_idx]
                axes.append(
                    {
                        "name": f"gene::{p_name}",
                        "per_head": [[arr[i]] for i in range(n)],
                    }
                )

        return emit_batch(
            self.interceptor,
            workflow_id=self.workflow_id,
            sample_ids=sample_ids,
            axes=axes,
            metadata=meta,
            activity=self.activity,
        )

    def send_workflow(self, params: dict[str, Any] | None = None) -> None:
        """Record model identity, and the pathway order, on the workflow.

        The pathway list is the decoder for the emitted vectors: without it,
        position 41 of a 186-wide distribution means nothing. It is the
        counterpart of the tokenizer identity the vLLM connector sends.
        """
        conf = {
            "model": "HiDRA",
            "metric_reference": METRIC_REFERENCE,
            "n_pathways": len(self.pathway_names),
            "pathway_order": self.pathway_names,
        }
        if params:
            conf.update(params)
        self.interceptor.send_model_workflow(self.workflow_id, conf)
