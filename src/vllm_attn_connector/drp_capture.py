# SPDX-License-Identifier: Apache-2.0
"""Shared base for drug-response attention capture.

Each model family needs its own extraction mechanism -- torch forward hooks for
Paccmann, a second Keras model for HiDRA -- but everything downstream of
"attention is now in hand" is identical: resolve an interceptor, assemble
metadata, reduce over heads, emit, record model identity on the workflow.

``AttentionCapture`` owns that shared half. A subclass supplies three things:

``MODEL_NAME`` / ``FRAMEWORK`` / ``METRIC_REFERENCE``
    Identity, recorded on every task and on the workflow. ``METRIC_REFERENCE``
    matters most: Paccmann's attention is additive/Bahdanau, HiDRA's is a
    learned softmax gate, and the vLLM connector's is ``softmax(q.K^T/sqrt d)``.
    They are not the same measurement, and records from different producers
    must never be silently pooled.

``collect(...)``
    Return the axes captured for one batch, as ``AttentionAxis`` objects.

``workflow_conf(params)`` (optional)
    Extra identity for the workflow record -- the tokenizer, the checkpoint,
    the feature ordering. Whatever a later reader needs to interpret a bare
    vector.

Subclasses that attach resources (hooks, handles) override ``close``;
``AttentionCapture`` is a context manager either way, which is the safer
spelling when an exception can skip teardown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from typing_extensions import Self

from .drp_emit import ATTENTION_ACTIVITY, emit_batch

__all__ = ["AttentionAxis", "AttentionCapture"]


@dataclass
class AttentionAxis:
    """One attention axis for one batch.

    Attributes
    ----------
    name:
        Axis label, recorded in metadata and used to select axes when reading
        back (``"gene"``, ``"smiles"``, ``"pathway"``).
    per_sample_heads:
        One entry per sample; each entry is that sample's list of per-head
        attention vectors. A single-headed axis passes a one-element list per
        sample, so it goes through exactly the same reduction as a 16-headed
        one -- sum and peak simply coincide, which is correct rather than
        degenerate.
    input_ids:
        Optional token ids per sample, when the axis is tokenised.
    feature_names:
        Optional labels for the axis positions. The store keeps the vector; a
        bare index is not interpretable without these, so when they are known
        they belong on the workflow record.
    """

    name: str
    per_sample_heads: list[Any]
    input_ids: list[Any] | None = None
    feature_names: list[str] | None = field(default=None, repr=False)

    def as_emit_axis(self) -> dict[str, Any]:
        """Shape this axis the way :func:`drp_emit.emit_batch` expects."""
        axis: dict[str, Any] = {
            "name": self.name,
            "per_head": self.per_sample_heads,
        }
        if self.input_ids is not None:
            axis["input_ids"] = self.input_ids
        return axis


class AttentionCapture:
    """Base class for model-specific attention capture.

    Subclasses implement :meth:`collect`; everything else is shared.
    """

    #: Model identity, recorded on every task. Subclasses must set these.
    MODEL_NAME: str = ""
    FRAMEWORK: str = ""
    #: What quantity the numbers are. See the module docstring.
    METRIC_REFERENCE: str = ""

    def __init__(
        self,
        model: Any,
        workflow_id: str,
        interceptor: Any = None,
        activity: str = ATTENTION_ACTIVITY,
    ) -> None:
        if not (self.MODEL_NAME and self.FRAMEWORK and self.METRIC_REFERENCE):
            raise TypeError(
                f"{type(self).__name__} must set MODEL_NAME, FRAMEWORK and "
                f"METRIC_REFERENCE"
            )
        self.model = model
        self.workflow_id = workflow_id
        self.activity = activity
        self.interceptor = interceptor if interceptor is not None else _default_interceptor()

    # -- subclass hooks -----------------------------------------------------

    def collect(self, **kwargs: Any) -> list[AttentionAxis]:
        """Return the axes captured for one batch.

        Called by :meth:`emit`, which forwards its keyword arguments here. A
        torch subclass typically drains buffers filled by forward hooks; a
        Keras subclass typically runs an extraction model over ``inputs``.
        """
        raise NotImplementedError

    def workflow_conf(self, params: dict[str, Any] | None) -> dict[str, Any]:
        """Extra identity to record on the workflow. Override as needed."""
        return dict(params) if params else {}

    def close(self) -> None:
        """Release anything :meth:`collect` depends on. Override as needed."""

    # -- shared behaviour ---------------------------------------------------

    def base_metadata(self) -> dict[str, Any]:
        """Identity carried on every emitted task."""
        return {
            "model": self.MODEL_NAME,
            "framework": self.FRAMEWORK,
            "metric": "attention_weights",
            "metric_reference": self.METRIC_REFERENCE,
        }

    def emit(
        self,
        sample_ids: Sequence[str],
        metadata: dict[str, Any] | None = None,
        **collect_kwargs: Any,
    ) -> list[str]:
        """Capture one batch and emit it. Returns the task ids written.

        Callers get the ids back so a test -- or a downstream step -- can
        assert on exactly what was written rather than scraping the store.
        """
        axes = self.collect(**collect_kwargs)
        if not axes:
            raise RuntimeError(
                f"{type(self).__name__}.collect() returned no axes; call emit() "
                f"after a forward pass, and check the model is in eval mode"
            )

        meta = self.base_metadata()
        if metadata:
            meta.update(metadata)

        return emit_batch(
            self.interceptor,
            workflow_id=self.workflow_id,
            sample_ids=sample_ids,
            axes=[axis.as_emit_axis() for axis in axes],
            metadata=meta,
            activity=self.activity,
        )

    def send_workflow(self, params: dict[str, Any] | None = None) -> None:
        """Record model identity on the workflow, once per run.

        The vLLM connector sends tokenizer identity so token ids can be decoded
        later. The analogue here is whatever a reader needs to interpret a bare
        attention vector: the feature ordering, the checkpoint, the vocabulary.
        """
        conf = {
            "model": self.MODEL_NAME,
            "framework": self.FRAMEWORK,
            "metric_reference": self.METRIC_REFERENCE,
        }
        conf.update(self.workflow_conf(params))
        self.interceptor.send_model_workflow(self.workflow_id, conf)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _default_interceptor() -> Any:
    """Resolve Flowcept's vLLM interceptor.

    Imported lazily and in one place: this package must stay importable with
    neither Flowcept nor vLLM installed, which is what the unit tests rely on.
    """
    from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor

    return VLLMInterceptor.get_instance()
