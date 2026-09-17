# SPDX-License-Identifier: Apache-2.0
"""Emit attention from drug-response models as Flowcept provenance.

The vLLM half of this package exists because decode attention is *destroyed* as
it is computed: FlashAttention tiles the softmax and discards the scores,
queries never enter the KV cache, and ``torch.compile`` erases
``Attention.forward``. Recovering it needs a backend override plus a
recomputation of ``q.K^T`` against the paged cache.

Drug-response models need none of that. Paccmann MCA and HiDRA are single-pass
encoder-regressors: eager mode, no KV cache, no autoregression. **The attention
weights are already materialised.** Paccmann even returns them and then throws
them away. So the whole apparatus collapses to: catch what is already falling on
the floor, and hand it to the same interceptor the vLLM connector uses.

What this module is
-------------------
The framework-agnostic middle. It takes attention that has already been
captured -- as plain lists of floats -- and emits it through
``VLLMInterceptor``. It imports neither torch nor tensorflow, so it is
importable and testable in a base install with neither present.

The two framework-specific capture paths live in ``drp_paccmann.py`` (torch
forward hooks) and ``drp_hidra.py`` (Keras named layers). Both are
*zero-source-edit*: neither model repository is patched, matching this
project's existing stance on vLLM.

Shape of the record
-------------------
One Flowcept task per sample, mirroring the vLLM connector's
``{req_id}:g{group}`` convention:

    <sample_id>:g0    the drug / molecule axis
    <sample_id>:g1    the gene / pathway axis

Each carries the same ``attn_sum`` / ``attn_peak`` field names the vLLM payload
uses, so one downstream consumer reads both without branching on producer.

What does not carry over
------------------------
The vLLM records have a *time* axis: attention is captured per decode step, so
``matrix_shape`` is ``[G, k]`` and turnover between steps is meaningful. These
models emit one prediction from one forward pass, so ``G == 1``. Everything
step-indexed in the vLLM payload -- ``topk_residual``, ``decode_steps_dropped``,
``wins_first_step`` -- has no counterpart and is absent rather than faked.

The quantity also differs and the metadata says so. Paccmann attention is
additive/Bahdanau (``tanh(W_r.ref + W_c.ctx) -> Linear -> softmax``); HiDRA's is
a learned softmax gate over pathway members. Neither is ``softmax(q.K^T/sqrt d)``.
``metric_reference`` records which, so records are not silently compared across
producers measuring different things.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "ATTENTION_ACTIVITY",
    "emit_batch",
    "emit_sample",
    "reduce_heads",
]


# Distinct from the vLLM connector's activity so both can write into one store
# without their records being conflated. The interceptor's own docstring notes
# `activity` exists for exactly this.
ATTENTION_ACTIVITY = "drp_attention"


def _as_float_list(values: Any) -> list[float]:
    """Coerce a 1-D tensor/array/sequence to a plain list of floats.

    Accepts anything with ``.tolist()`` (torch, numpy, TF eager) or any plain
    sequence, so this module never has to import a framework to normalise what
    a framework handed it.
    """
    if hasattr(values, "detach"):  # torch, possibly requires_grad
        values = values.detach()
    if hasattr(values, "cpu"):  # torch on device
        values = values.cpu()
    if hasattr(values, "numpy"):  # torch / TF eager tensor
        values = values.numpy()
    if hasattr(values, "tolist"):  # numpy array
        values = values.tolist()
    out = [float(v) for v in values]
    if not out:
        raise ValueError("attention vector is empty")
    return out


def reduce_heads(per_head: Sequence[Any]) -> dict[str, list[float]]:
    """Reduce a list of per-head attention vectors to sum and peak series.

    ``per_head`` is one vector per (layer, head) pair, each of equal length
    ``T``. Returns the two reductions the vLLM payload also carries:

    ``attn_sum``
        Elementwise sum across heads. Proportional to the mean, so it is the
        distribution-shaped view: it says how much total attention a position
        received.
    ``attn_peak``
        Elementwise max across heads. Kept *as well as* the sum because a mean
        over many heads dilutes -- if a small number of heads carry the
        behaviour of interest, averaging them against the rest buries the
        signal. This is the same reasoning as ``val_all_max`` vs
        ``val_all_avg`` in the vLLM connector.
    ``attn_argmax_head``
        Which head supplied the peak, as a float so the series stays
        homogeneous. Free: the max reduction has to find it anyway.

    A single head is a legitimate input (HiDRA's gates are single-headed): the
    sum and the peak then coincide, which is correct rather than degenerate.
    """
    if not per_head:
        raise ValueError("per_head is empty; nothing to reduce")

    rows = [_as_float_list(h) for h in per_head]
    width = len(rows[0])
    for i, r in enumerate(rows):
        if len(r) != width:
            raise ValueError(
                f"per-head vectors must be equal length; head 0 has {width}, "
                f"head {i} has {len(r)}"
            )

    attn_sum = [0.0] * width
    attn_peak = [float("-inf")] * width
    argmax = [0.0] * width
    for h, row in enumerate(rows):
        for j, v in enumerate(row):
            attn_sum[j] += v
            if v > attn_peak[j]:
                attn_peak[j] = v
                argmax[j] = float(h)

    return {
        "attn_sum": attn_sum,
        "attn_peak": attn_peak,
        "attn_argmax_head": argmax,
    }


def emit_sample(
    interceptor: Any,
    workflow_id: str,
    sample_id: str,
    series: dict[str, list[float]],
    input_ids: Sequence[int] | None,
    metadata: dict[str, Any],
    group: int = 0,
    activity: str = ATTENTION_ACTIVITY,
) -> str:
    """Emit one axis of one sample. Returns the task id used.

    ``group`` follows the vLLM connector's ``:g<n>`` suffix so a store can hold
    several axes per sample without collision.
    """
    task_id = f"{sample_id}:g{group}"
    interceptor.capture_request(
        workflow_id=workflow_id,
        request_id=task_id,
        series=series,
        # The interceptor derives num_prompt_tokens from this. These models have
        # no token ids on the gene axis, so an explicit empty list is honest:
        # the series length still records the width.
        prompt_token_ids=list(input_ids) if input_ids is not None else [],
        metadata=metadata,
        activity=activity,
    )
    return task_id


def emit_batch(
    interceptor: Any,
    workflow_id: str,
    sample_ids: Sequence[str],
    axes: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    activity: str = ATTENTION_ACTIVITY,
) -> list[str]:
    """Emit every axis of every sample in a batch.

    ``axes`` is one dict per axis, each with:

    ``name``
        Axis label, recorded in metadata (e.g. ``"smiles"``, ``"gene"``).
    ``per_head``
        A list, one entry per sample, of per-head attention vectors for that
        sample. A single-head axis passes a one-element list per sample.
    ``input_ids`` (optional)
        One id sequence per sample, when the axis is tokenised.

    Returns every task id emitted, in order. The caller gets these back so a
    test -- or a downstream step -- can assert on exactly what was written
    rather than scraping the buffer.
    """
    n = len(sample_ids)
    for axis in axes:
        got = len(axis["per_head"])
        if got != n:
            raise ValueError(
                f"axis {axis.get('name')!r} has {got} samples, expected {n}"
            )

    emitted: list[str] = []
    for i, sample_id in enumerate(sample_ids):
        for group, axis in enumerate(axes):
            series = reduce_heads(axis["per_head"][i])
            meta = dict(metadata)
            meta["axis"] = axis.get("name", f"g{group}")
            meta["n_heads"] = len(axis["per_head"][i])
            meta["axis_width"] = len(series["attn_sum"])
            # No decode loop: one forward pass, one distribution. Stated so a
            # consumer reading both producers can tell the records apart.
            meta.setdefault("decode_steps_recorded", 1)

            ids = axis.get("input_ids")
            emitted.append(
                emit_sample(
                    interceptor,
                    workflow_id=workflow_id,
                    sample_id=sample_id,
                    series=series,
                    input_ids=ids[i] if ids is not None else None,
                    metadata=meta,
                    group=group,
                    activity=activity,
                )
            )
    return emitted
