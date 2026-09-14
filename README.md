# vllm-attn-connector

Captures **exact per-decode-step attention over the prompt** from a running vLLM
engine and emits it as **Flowcept provenance**.

For each decode step `t` and prompt token `j` it records

```
a[t, j] = softmax( q_t · K_promptᵀ / √d )[j]
```

reduced over layers and heads, per KV cache group. The result answers "which
prompt tokens did *this* generated token attend to", per token, rather than
"which prompt tokens mattered overall".

**No vLLM source is patched and no attention kernel is modified.**

## Repository layout

```
src/vllm_attn_connector/
    connector.py    KVConnector: recomputes q.Kt, reduces, selects, emits
    probe.py        attention-backend override that copies the decode query
    kernels.py      Triton q.Kt against the paged cache, plus a torch reference
    layout.py       KV cache layout resolution across vLLM backends
tests/
    test_aggregations.py   pure torch, no GPU or vLLM needed
    e2e_smoke.py           real engine, asserts 38 properties of the output
```

> This repository previously held `vllm-kvnorm`, a connector that scored
> cache-only KV norms. That approach was retired: statistics computable from the
> cache alone are request-invariant, so they cannot explain request-specific
> behaviour. The query is the missing ingredient, and it is never cached — hence
> the backend probe here. The kvnorm history remains in this repository's log.

## How it works

The scores are **recomputed, not extracted**. FlashAttention tiles the softmax
and discards the intermediate scores, so there is nothing to read back. But `K`
persists in the paged cache, and a single decode row is a matvec — cheap enough
to redo on a side stream.

Queries are the missing ingredient: they are never cached. A small backend
override copies them during the forward pass; everything else happens in the
connector.

```
step N
  bind_connector_metadata()   connector learns this step's requests
  start_load_kv()             emit anything that finished last step
  <forward>
      impl.forward(...)       PROBE: if max_query_len==1, copy q   [blocking, ~16 KB/layer]
                              then super().forward() unchanged
  wait_for_save()             CONNECTOR: q·Kᵀ, softmax, reduce      [overlapped, side stream]
```

| stage | cost | blocking |
|---|---|---|
| probe copies `q` | `heads × head_size × n_reqs` per layer | yes, sub-microsecond |
| `q·Kᵀ`, softmax, reduce | one matvec per request per layer | no — side stream |
| `event.synchronize()` at finish | one event per request | yes, once |

Two implementation notes that are easy to get wrong if you fork this:

**The probe overrides the backend impl, not the `Attention` module.**
`Attention.forward` is traced through by `torch.compile`; only the custom op
`unified_attention_with_output` survives as a runtime node, and it dispatches
`self.impl.forward(...)` dynamically. An `nn.Module` forward hook would not fire
under CUDA graphs.

**Decode only.** In decode each request contributes exactly one query token, so
batch row *i* is request slot *i* — no `query_start_loc` parsing and no
token-to-request mapping is needed. Prefill is skipped; see
[Limitations](#limitations).

## Install

```bash
ROOT=$(cd .. && pwd)
export PYTHONPATH="$ROOT/vllm-attn-connector/src:$ROOT/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
```

Flowcept must come from a checkout that contains
`flowceptor/adapters/vllm/`.

Requires a CUDA device and Triton. Tested against vLLM 0.27.x with
`VLLM_ENABLE_V1_MULTIPROCESSING=0`; with multiprocessing enabled the emitted
records leave the process and Flowcept needs a reachable message queue.

## Quickstart

```python
import vllm_attn_connector
# MUST precede engine construction: _cached_get_attn_backend is @cache'd, so the
# backend class is resolved once and never re-read.
assert vllm_attn_connector.install_probe()

from flowcept import Flowcept
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

with Flowcept("vllm", workflow_id="my-run", workflow_name="my_run"):
    llm = LLM(
        model="Qwen/Qwen3-4B-Instruct-2507",
        kv_transfer_config=KVTransferConfig(
            kv_connector="AttnConnector",
            kv_connector_module_path="vllm_attn_connector",
            kv_role="kv_producer",
            kv_connector_extra_config={"workflow_id": "my-run"},
        ),
    )
    llm.generate(["..."], SamplingParams(max_tokens=64))
```

Use `kv_role="kv_producer"`, not `"kv_both"` — this connector never loads KV,
and declaring it a consumer makes the scheduler defer block frees.

## Configuration

All keys go in `kv_connector_extra_config`.

| key | default | meaning |
|---|---|---|
| `workflow_id` | — | parent workflow to attach records to |
| `max_steps` | `0` | decode steps recorded per request; `0` = every one |
| `top_pct` | `10` | percent of positions kept within each segment |
| `chunk_size` | `32` | segment width when no ranges are declared; `0` = one segment |
| `layer_stats` | `false` | opt-in per-head diagnostics; roughly +50% cost |

By default **every decode token is recorded**. Step-indexed buffers start at 64
rows and double as needed, so unbounded capture costs no more than a bounded one
for short generations, and long generations are not silently truncated. Set a
positive `max_steps` to cap memory explicitly; steps beyond the cap are counted
in `decode_steps_dropped` rather than lost quietly.

```json
{
  "kv_connector": "AttnConnector",
  "kv_connector_module_path": "vllm_attn_connector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
      "workflow_id": "my-run",
      "top_pct": 10,
      "chunk_size": 32
  }
}
```

### Segments: fixed or variable

Storing the full `[decode_steps, prompt_tokens]` matrix does not scale — it is
linear in prompt length and grows without bound on long contexts. Instead the
prompt is **partitioned into segments** and the strongest `top_pct` percent of
positions is kept *within each one*.

Selecting per segment rather than over the whole prompt is the point: a global
top-X% concentrates wherever the distribution happens to peak and can leave
entire regions with no stored entry at all, which makes those regions invisible
downstream. Per-segment selection guarantees every region is represented, with
a floor of one entry per segment however short it is.

Segments come from one of two places, and **the output is identical either way**:

**Fixed** (default) — uniform blocks of `chunk_size` tokens. Use when the prompt
has no structure you can declare.

**Variable** — ranges you declare per request, one per document, tool output,
retrieved chunk, or whatever unit you want statistics for:

```python
SamplingParams(
    max_tokens=256,
    extra_args={"kv_transfer_params": {"ranges": [[131, 496], [496, 548], [578, 765]]}},
)
```

Ranges are clamped to the prompt, sorted, and the **gaps between them become
segments too**, so the partition stays complete and `topk_residual` stays exact.
Overlaps are resolved by truncation. Malformed input is ignored with a warning
rather than raised — provenance capture must never fail a generation request.

Whichever mode produced a record, the partition is emitted as `segments`, so a
consumer runs the same per-segment aggregation without knowing or caring which
was used. `metadata.segment_mode` says which it was.

Entries kept per step is `sum(max(1, round(top_pct/100 × len(seg))))` over
segments. Note the round-and-floor: with `top_pct` small and segments short, the
floor of 1 dominates and the effective rate is higher than `top_pct`.

**Full capture** — `chunk_size=1, top_pct=100` makes every position its own
segment and keeps all of them. That is the replacement for the dense mode this
connector used to have: same information, same field names, no separate code
path.

## Emitted fields

Per request, per KV cache group, under `<req>:g<group>`. Write `G` for decode
steps recorded, `T` for prefill tokens scored, `k` for entries kept per step.

**Sparse mode** (default, `top_pct > 0`):

| field | shape | meaning |
|---|---|---|
| `topk_pos` | `[G, k]` int | retained prefill positions, ascending; `-1` pads the ragged final chunk |
| `val_all_max` | `[G, k]` | attention at those positions, **max** over all (layer, head) |
| `val_all_avg` | `[G, k]` | attention at those positions, **mean** over all (layer, head) |
| `topk_head` | `[G, k]` int | which `layer × num_query_heads + head` supplied the max |
| `topk_residual` | `[G, 1]` | mean-aggregation mass that selection discarded |
| `segments` | `[n_seg, 3]` int | the partition, as `(lo, hi, keep)` |
| `attn_sum` | `[T]` | column sum of the mean row, over **all** positions |
| `attn_peak` | `[T]` | max over (layer, head, decode step), over **all** positions |

`segments` covers `[0, prompt_len_scored)` with no gaps or overlaps, so every
retained position falls in exactly one segment and a `searchsorted` on the
segment starts maps positions to segments.

`attn_sum` and `attn_peak` cover every position, including those selection
dropped, so whole-prompt totals remain available.

### Why both a max and a mean

They answer different questions and neither substitutes for the other.

`val_all_max` is what selection runs on and what tends to be useful for
attributing a generated token to a prompt span. A mean over all (layer, head)
pairs dilutes: if a small number of heads carry the retrieval behaviour,
averaging them against hundreds that do not will bury the signal.

`val_all_avg` is a true probability distribution over positions, which is what
makes `topk_residual` meaningful — it accounts for exactly the mass selection
threw away, so `sum(val_all_avg) + topk_residual == 1` per step. A max cannot
express that, because maxima do not sum to anything.

`topk_head` is free: the max reduction has to identify the winning head anyway.

### Position 0 is never selected

The first prompt token is the attention sink — in decoder-only LLMs it absorbs a
large, roughly content-independent share of attention
([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)). Letting it win a slot
would spend one of the `per_chunk` entries on a token that carries no
information about what the model is doing.

Its mass remains fully visible in `attn_sum` and `attn_peak`, which cover all
positions. If you are consuming `topk_pos` you do not need to mask the sink
yourself: it is never there.

## Output size

Per request, per KV cache group:

```
k · G · 4      (positions + 2 values + head id)
  +  G         (residual)
  +  2·T       (attn_sum, attn_peak)
  +  3·n_seg   (the partition)
```

**Size does not depend on the number of heads, layers, or head_size.** Those are
reduced before anything is stored; they drive *compute*, not output.

| variable | meaning | effect on size |
|---|---|---|
| `T` | prefill tokens scored (`prompt_len_scored`) | `2T`, from `attn_sum`/`attn_peak` |
| `G` | decode steps recorded: all of them, or `max_steps` if capped | linear |
| `k` | entries kept per step | linear (sparse only) |
| `n_groups` | KV cache groups: 1 uniform, 2 for some hybrid models | linear |
| `H`, `L`, `head_size` | heads, layers, head dim | **none** |
| `FLOAT_DECIMALS`, JSONL | 6 dp text ≈ 2.3× float32 | constant factor |

With the default `max_steps=0`, `G` is the full generation length, so output
grows with how much the model actually produces. A positive `max_steps` bounds
it. The `k` factor is under your direct control via `top_pct` and the segment
widths, which is what keeps long prompts affordable. `metadata` also carries `matrix_shape`, `matrix_fields`,
`decode_steps_recorded`, `decode_steps_nonfinite`, `restarts`,
`prompt_len_scored`, `chunk_size`, `per_chunk` and `no_entry_sentinel`.

## Performance

Cost is dominated by recomputing `q·Kᵀ`, which re-reads the prompt's keys from
the paged cache once per decode step per layer. It is memory-bandwidth bound,
so it scales with `prompt_tokens × layers × heads × head_size` and is largely
insensitive to what you do with the result afterwards.

Practical consequences:

- **Selection does not reduce compute**, only output size. Keeping 10% of
  positions costs the same as keeping all of them. Choose `top_pct` and the
  segmentation for storage and for the granularity you want statistics at, not
  for speed.
- **Longer prompts cost proportionally more**, since the whole prompt's `K` is
  re-read every step.
- **Cost is per decode step**, so it scales with generation length. `max_steps`
  caps that if you need a ceiling; by default there is none.
- **Per-head detail is expensive.** Anything that has to carry per-head state
  through the reduction — such as `layer_stats` — costs substantially more than
  the reduced statistics, because the reduction is what keeps the inner loop
  cheap.

On a single consumer GPU with a 4B model and prompts of a few thousand tokens,
end-to-end generation slowdown is high single digit percent, and recording
every decode token rather than a capped prefix does not measurably change it.
Measure on your own workload before budgeting: the ratio depends on prompt
length, generation length and how much headroom the model leaves on your
device.

Two measurement pitfalls, both of which produced wrong numbers during
development: interleave capture and no-capture runs **in one session**, since
GPU clock and thermal state drift between sessions by more than the effect
being measured; and discard the first run of a session, which pays warmup the
others do not.

## Validation

```bash
# no GPU, no vLLM: streaming reduction and segment planning vs references
pytest tests/test_aggregations.py

# end to end against a real engine
python tests/e2e_smoke.py
python tests/e2e_smoke.py --ranges                        # variable segments
python tests/e2e_smoke.py --chunk-size 1 --top-pct 100    # full capture
python tests/e2e_smoke.py --top-pct 25 --chunk-size 64
```

See [`tests/README.md`](tests/README.md) for what each covers, and for what is
*not* covered — preemption and tensor parallelism both lack tests.

The e2e asserts 32 properties. The load-bearing ones:

- **`sum(val_all_avg) + topk_residual == 1`** at every step, so the output is a
  genuine probability distribution with the discarded mass accounted for
  exactly, not a plausible-looking artifact.
- **The partition is complete**: segments cover the prompt with no gaps or
  overlaps, and every declared range appears verbatim as a segment.
- **The attention sink is reproduced** at two to three orders of magnitude above
  the median position. Recovering a known property of the model is evidence the
  recomputation is reading real keys.
- **Attention changes between decode steps.** Under selection this shows up as
  turnover in the retained position set (adjacent-step Jaccard well below 1);
  under full capture it is measured as total-variation distance between
  consecutive steps. If attention did not move, per-step capture would be
  redundant with `attn_sum` and this package would have no purpose.
- **Every segment is represented in every step**, the invariant that
  per-segment selection exists to provide.

The Triton kernel matches a pure-torch reference to ~1e-08 across full, partial
and single-key sequences.

## Limitations

- **Prefill attention is not captured.** Only decode queries are scored;
  attention *between* prompt tokens during prefill is never computed. If the
  content behind an answer was assembled into a late prompt position during
  prefill, decode attention points at that position rather than at the original
  source. Capturing it would require streaming during prefill, since prefill
  queries are not cached either.
- **Attention magnitude is not signed.** A head attending strongly to a token
  may be suppressing it as easily as using it. High attention means "this token
  was consulted", not "this token was used affirmatively".
- **Reduced over layers and heads.** The output cannot tell you *which* layer
  produced a score, only which head supplied the maximum. `layer_stats` exposes
  per-head detail at significant cost.
- **Segments are frozen at the first decode step** and derived from the prompt
  length settled at that point. Ranges are read from the request's first
  appearance only; they describe the prompt, which does not change.
- **Very uneven ranges cost memory.** Selection pads segments to the widest one,
  so one range far larger than the rest inflates a `[n_seg, max_width]`
  scratch buffer. Even-ish segments, or a `chunk_size` grid, avoid this.
- **Scoring cannot currently be restricted to a subset of layers.** Since cost
  is dominated by re-reading `K` per layer, this is the most promising available
  speedup and is not yet implemented.
- **Sliding-window and other non-full attention groups** are scored against the
  positions actually in cache for that group; `scored_attention` and
  `attention_window` in the metadata record which regime applied.
- **Preemption drops partial records.** If a request is preempted and restarted,
  steps recorded before the restart are discarded rather than stitched, and
  `restarts` is incremented.
- **`layout.py` is vendored**, duplicated with `vllm-kvnorm`. If a third package
  needs it, extract a shared dependency instead of copying again.
