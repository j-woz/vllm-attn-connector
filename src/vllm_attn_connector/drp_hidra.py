# SPDX-License-Identifier: Apache-2.0
"""Capture HiDRA attention at inference. The Keras half.

HiDRA predicts drug response from gene expression organised by KEGG pathway. It
is hierarchically attentive, and both levels are interesting:

*Gene level*
    One softmax per pathway, over the genes in that pathway
    (``<PATHWAY>_Attention_Softmax``, ``hidra_utils.py:91-92``). With the
    shipped ``geneset.gmt`` that is 186 pathways, so 186 separate
    distributions.
*Pathway level*
    One softmax over the pathways themselves (``Sample_Attention_Softmax``,
    ``hidra_utils.py:111-112``) -- which pathway mattered for this prediction.

The pathway level is the more useful of the two for attribution: a single
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
over one shared base rather than one strained abstraction.

Usage
-----
::

    from vllm_attn_connector.drp_hidra import HidraCapture

    with Flowcept("vllm", workflow_id=wf, workflow_name="hidra"):
        cap = HidraCapture(model, workflow_id=wf)
        cap.send_workflow({"modelpath": ...})
        cap.emit(inputs=X_batch, sample_ids=[...])
"""

from __future__ import annotations

from typing import Any

from .drp_capture import AttentionAxis, AttentionCapture

__all__ = [
    "GENE_LAYER_SUFFIX",
    "PATHWAY_AXIS",
    "PATHWAY_LAYER",
    "HidraCapture",
]

PATHWAY_LAYER = "Sample_Attention_Softmax"
GENE_LAYER_SUFFIX = "_Attention_Softmax"
PATHWAY_AXIS = "pathway"


class HidraCapture(AttentionCapture):
    """Extract attention from a trained HiDRA Keras model.

    Builds one auxiliary model whose outputs are the attention tensors. The
    trained layers are shared, not copied, so this adds no parameters and
    changes no weights.

    Parameters
    ----------
    model:
        A trained HiDRA model, loaded with ``compile=False``.
    workflow_id:
        Flowcept workflow to attach records to.
    include_gene_level:
        Also emit the per-pathway gene distributions. Opt-in: this multiplies
        the record count by the number of pathways.
    """

    MODEL_NAME = "HiDRA"
    FRAMEWORK = "tensorflow"
    # A learned gate -- Dense(tanh) -> softmax over concatenated gene and drug
    # features -- not softmax(q.K^T/sqrt d), and not Bahdanau either.
    METRIC_REFERENCE = "hidra_hierarchical_attention_softmax_gate"

    def __init__(
        self,
        model: Any,
        workflow_id: str,
        include_gene_level: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, workflow_id, **kwargs)
        self.include_gene_level = include_gene_level
        self.pathway_names = self._pathway_names()
        self._extractor = self._build_extractor()

    def _pathway_names(self) -> list[str]:
        """Gene-level attention layer names, in graph order.

        Graph order is also the order the pathway-level softmax indexes, so
        position i in the pathway distribution is ``pathway_names[i]``. That
        mapping is what makes the emitted vector interpretable, which is why
        it goes on the workflow record.
        """
        return [
            layer.name[: -len(GENE_LAYER_SUFFIX)]
            for layer in self.model.layers
            if layer.name.endswith(GENE_LAYER_SUFFIX) and layer.name != PATHWAY_LAYER
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

    def collect(
        self,
        inputs: Any = None,
        n_samples: int | None = None,
        verbose: int = 0,
        **_: Any,
    ) -> list[AttentionAxis]:
        """Run the extractor over one batch.

        ``inputs`` is whatever the trained model takes -- for HiDRA, the list of
        per-pathway expression arrays plus the drug array, exactly as
        ``MultiGenerator`` yields it.
        """
        if inputs is None:
            raise ValueError("HidraCapture.emit() requires inputs=")

        outs = self._extractor.predict(inputs, verbose=verbose)
        if not isinstance(outs, list):
            outs = [outs]

        pathway_attn = outs[0]
        n = n_samples if n_samples is not None else len(pathway_attn)
        if len(pathway_attn) != n:
            raise ValueError(
                f"got {len(pathway_attn)} rows of attention for {n} sample_ids"
            )

        # Single-headed, so each sample's vector is wrapped in a one-element
        # list: the same reduction path as Paccmann's 16 heads, where sum and
        # peak simply coincide.
        axes = [
            AttentionAxis(
                name=PATHWAY_AXIS,
                per_sample_heads=[[pathway_attn[i]] for i in range(n)],
                feature_names=self.pathway_names,
            )
        ]

        if self.include_gene_level:
            for offset, pathway in enumerate(self.pathway_names):
                arr = outs[1 + offset]
                axes.append(
                    AttentionAxis(
                        name=f"gene::{pathway}",
                        per_sample_heads=[[arr[i]] for i in range(n)],
                    )
                )

        return axes

    def emit(self, sample_ids, metadata=None, **kwargs: Any) -> list[str]:
        """Emit one batch. See :meth:`AttentionCapture.emit`."""
        meta = {"n_pathways": len(self.pathway_names)}
        if metadata:
            meta.update(metadata)
        return super().emit(
            sample_ids, metadata=meta, n_samples=len(sample_ids), **kwargs
        )

    def workflow_conf(self, params: dict[str, Any] | None) -> dict[str, Any]:
        # The pathway list is the decoder for the emitted vectors: without it,
        # position 41 of a 186-wide distribution means nothing. It is the
        # counterpart of the tokenizer identity the vLLM connector sends.
        conf: dict[str, Any] = {
            "n_pathways": len(self.pathway_names),
            "pathway_order": self.pathway_names,
        }
        if params:
            conf.update(params)
        return conf
