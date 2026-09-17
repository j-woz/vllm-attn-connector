# SPDX-License-Identifier: Apache-2.0
"""Checks for the drug-response emitter. No GPU, no vLLM, no torch, no TF.

The framework-specific capture paths (``drp_paccmann``, ``drp_hidra``) need
their frameworks and a trained checkpoint, so they are exercised separately in
``drp_smoke.py``. What is testable anywhere is the part in between: the head
reduction and the shape of what gets handed to the interceptor. That is what
decides whether a record is correct, so it is worth testing without needing a
model to produce one.

Run either way::

    python tests/test_drp_emit.py     # verbose, prints every case
    pytest tests/test_drp_emit.py     # same checks, quiet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_attn_connector.drp_emit import (
    ATTENTION_ACTIVITY,
    emit_batch,
    reduce_heads,
)

VERBOSE = __name__ == "__main__"


def say(msg):
    if VERBOSE:
        print(msg)


class FakeInterceptor:
    """Records calls instead of shipping them. The interceptor's own contract
    is Flowcept's to test; what matters here is what we pass it."""

    def __init__(self):
        self.calls = []
        self.workflows = []

    def capture_request(self, **kw):
        self.calls.append(kw)

    def send_model_workflow(self, workflow_id, conf, parent_workflow_id=None):
        self.workflows.append((workflow_id, conf))


# --------------------------------------------------------------------------
# reduce_heads
# --------------------------------------------------------------------------


def test_sum_and_peak_differ_when_one_head_dominates():
    """The reason both reductions exist.

    One head spikes at a position the others ignore. The sum dilutes it; the
    peak preserves it. If these ever agree on such input, the max reduction has
    silently become a mean and attribution signal is being averaged away.
    """
    per_head = [
        [0.0, 0.9, 0.0],  # head 0 spikes at position 1
        [0.1, 0.0, 0.1],
        [0.1, 0.0, 0.1],
    ]
    out = reduce_heads(per_head)

    assert out["attn_peak"][1] == 0.9
    assert abs(out["attn_sum"][1] - 0.9) < 1e-9
    # Position 1's *mean* is 0.3, well below its peak: the dilution is real.
    assert out["attn_sum"][1] / 3 < out["attn_peak"][1]
    assert out["attn_argmax_head"][1] == 0.0
    say("  ok  sum and peak diverge when a single head carries the signal")


def test_argmax_head_identifies_the_winner():
    per_head = [
        [0.1, 0.1],
        [0.2, 0.9],  # head 1 wins position 1
        [0.8, 0.3],  # head 2 wins position 0
    ]
    out = reduce_heads(per_head)
    assert out["attn_argmax_head"] == [2.0, 1.0]
    assert out["attn_peak"] == [0.8, 0.9]
    say("  ok  argmax head tracks the peak")


def test_single_head_sum_equals_peak():
    """HiDRA's gates are single-headed. Sum == peak is correct there, not a bug."""
    out = reduce_heads([[0.2, 0.5, 0.3]])
    assert out["attn_sum"] == out["attn_peak"] == [0.2, 0.5, 0.3]
    assert out["attn_argmax_head"] == [0.0, 0.0, 0.0]
    say("  ok  single head: sum == peak")


def test_ragged_heads_rejected():
    try:
        reduce_heads([[0.1, 0.2], [0.3]])
    except ValueError as e:
        assert "equal length" in str(e)
        say("  ok  ragged per-head input rejected")
    else:
        raise AssertionError("expected ValueError on ragged input")


def test_empty_rejected():
    for bad, label in [([], "no heads"), ([[]], "empty vector")]:
        try:
            reduce_heads(bad)
        except ValueError:
            say(f"  ok  rejected: {label}")
        else:
            raise AssertionError(f"expected ValueError for {label}")


def test_accepts_objects_with_tolist():
    """Frameworks hand us tensors, not lists. The emitter must not import a
    framework to normalise one."""

    class FakeTensor:
        def __init__(self, v):
            self._v = v

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self._v

    out = reduce_heads([FakeTensor([0.1, 0.9])])
    assert out["attn_peak"] == [0.1, 0.9]
    say("  ok  tensor-like inputs coerced without importing a framework")


# --------------------------------------------------------------------------
# emit_batch
# --------------------------------------------------------------------------


def test_one_task_per_sample_per_axis():
    fake = FakeInterceptor()
    ids = emit_batch(
        fake,
        workflow_id="wf-1",
        sample_ids=["s0", "s1"],
        axes=[
            {"name": "smiles", "per_head": [[[0.5, 0.5]], [[0.1, 0.9]]]},
            {"name": "gene", "per_head": [[[1.0]], [[1.0]]]},
        ],
        metadata={"model": "test"},
    )
    # 2 samples x 2 axes
    assert len(fake.calls) == 4
    assert ids == ["s0:g0", "s0:g1", "s1:g0", "s1:g1"]
    say("  ok  one task per (sample, axis), ids follow the :g<n> convention")


def test_metadata_records_axis_and_width():
    fake = FakeInterceptor()
    emit_batch(
        fake,
        workflow_id="wf-1",
        sample_ids=["s0"],
        axes=[{"name": "smiles", "per_head": [[[0.1, 0.2, 0.7], [0.3, 0.3, 0.4]]]}],
        metadata={"model": "test"},
    )
    meta = fake.calls[0]["metadata"]
    assert meta["axis"] == "smiles"
    assert meta["n_heads"] == 2
    assert meta["axis_width"] == 3
    # No decode loop in these models; the record should say so rather than
    # leave a consumer to assume a time axis that is not there.
    assert meta["decode_steps_recorded"] == 1
    say("  ok  metadata carries axis, head count, width, and G=1")


def test_series_field_names_match_the_vllm_payload():
    """A downstream consumer should not need to branch on producer."""
    fake = FakeInterceptor()
    emit_batch(
        fake,
        workflow_id="wf-1",
        sample_ids=["s0"],
        axes=[{"name": "gene", "per_head": [[[0.4, 0.6]]]}],
        metadata={},
    )
    series = fake.calls[0]["series"]
    assert "attn_sum" in series and "attn_peak" in series
    say("  ok  emits attn_sum / attn_peak, as the vLLM connector does")


def test_activity_is_distinct_from_vllm():
    """Both producers write into one store; conflating them would be silent."""
    fake = FakeInterceptor()
    emit_batch(
        fake,
        workflow_id="wf-1",
        sample_ids=["s0"],
        axes=[{"name": "gene", "per_head": [[[1.0]]]}],
        metadata={},
    )
    assert fake.calls[0]["activity"] == ATTENTION_ACTIVITY == "drp_attention"
    say("  ok  activity is drp_attention, not the vLLM activity")


def test_sample_count_mismatch_rejected():
    fake = FakeInterceptor()
    try:
        emit_batch(
            fake,
            workflow_id="wf-1",
            sample_ids=["s0", "s1"],
            axes=[{"name": "gene", "per_head": [[[1.0]]]}],  # only 1 sample
            metadata={},
        )
    except ValueError as e:
        assert "expected 2" in str(e)
        # Nothing should have been emitted before the failure.
        assert fake.calls == []
        say("  ok  axis/sample-count mismatch rejected before anything is written")
    else:
        raise AssertionError("expected ValueError on count mismatch")


def test_input_ids_passed_through_when_present():
    fake = FakeInterceptor()
    emit_batch(
        fake,
        workflow_id="wf-1",
        sample_ids=["s0"],
        axes=[
            {"name": "smiles", "per_head": [[[0.5, 0.5]]], "input_ids": [[7, 9]]},
            {"name": "gene", "per_head": [[[1.0]]]},
        ],
        metadata={},
    )
    assert fake.calls[0]["prompt_token_ids"] == [7, 9]
    # The gene axis has no tokens; an empty list is honest, the width is still
    # recorded in metadata.
    assert fake.calls[1]["prompt_token_ids"] == []
    say("  ok  token ids forwarded on tokenised axes, empty elsewhere")


def _main():
    print("reduce_heads")
    test_sum_and_peak_differ_when_one_head_dominates()
    test_argmax_head_identifies_the_winner()
    test_single_head_sum_equals_peak()
    test_ragged_heads_rejected()
    test_empty_rejected()
    test_accepts_objects_with_tolist()
    print("\nemit_batch")
    test_one_task_per_sample_per_axis()
    test_metadata_records_axis_and_width()
    test_series_field_names_match_the_vllm_payload()
    test_activity_is_distinct_from_vllm()
    test_sample_count_mismatch_rejected()
    test_input_ids_passed_through_when_present()
    print("\nall checks passed")


if __name__ == "__main__":
    _main()
