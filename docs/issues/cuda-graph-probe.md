# The query probe never fired on a replayed CUDA graph, so served runs recorded zeros

## Symptom

Serving with `vllm serve` and CUDA graphs left on, which is the default,
produced `decode_attention` records that passed every structural check and
carried no attention signal. `attn_sum`, `val_all_max` and `val_all_avg` summed
to exactly 0.0 on every record, `topk_pos` was populated (top-k over an all-zero
row), and `topk_residual` was 1.0 on every step. Nothing errored and nothing
warned.

Reported as https://github.com/spotter-ai-genesis/vllm-attn-connector/issues/2.
Observed on granite-4.2-30b (jobs 3164638, 3164747) and reproduced on
facebook/opt-125m. The same runs with `--enforce-eager` or
`--compilation-config '{"cudagraph_mode":"NONE"}'` gave correct, mass-conserving
values (jobs 3164843, 3164850, 3164857, 3164887).

## Cause

`_capture` ran from `_ProbedImpl.forward`, which `install()` registers as the
backend's impl class so that it sits inside the traced graph. That is what makes
the probe survive `torch.compile`, and the README said so. It does not make the
probe survive CUDA graphs. A graph records the kernels one execution of the
trace produced, and a replay runs those kernels and no Python. `REGISTRY.put`
is a plain dict write, so it ran once, while vLLM captured the graph, and never
again.

vLLM 0.28 defaults to `cudagraph_mode=FULL_AND_PIECEWISE`, which captures a full
graph per decode batch size, so under `vllm serve` every decode step is a
replay. `_WorkerSide.score_step` then looked the layer up with
`REGISTRY.get(layer_name)`, got `None`, and skipped the layer with a bare
`continue`. Every layer of every step took that branch, so each request still
emitted a complete record with a zero matrix in it.

The connector also cleared the registry in `wait_for_save` on every step, which
is correct for a live capture and exactly wrong for a recorded one.

`tests/e2e_smoke.py` built its engine with `enforce_eager=True`, so the test
suite never ran the path that was broken.

## What changed

The probe no longer tries to re-register a tensor per step. Each layer owns one
persistent buffer and `_capture` copies into it. Under capture that copy becomes
a recorded kernel, so every replay of that graph refreshes the buffer with no
Python involved. The buffer is never cleared between steps.

Freshness is tracked instead of being implied by "the registry was rebuilt this
step":

* A graph write counts for any step that replays that graph. The key is
  `(cudagraph_runtime_mode, batch_descriptor)`, which is the pair vLLM's own
  `CudagraphDispatcher` keys graphs by, and both the capturing forward and every
  replay carry it in the forward context. `BatchDescriptor` is a frozen
  dataclass, so the capture-time key and the replay-time key compare equal.
* A live write counts only for the step it ran on, via a generation counter the
  connector bumps in `wait_for_save` where the `clear()` used to be.

One buffer per layer and not one per graph: the capture sizes sum to roughly
thirty times the largest one, so a clone per graph would have cost about 5 GB of
extra GPU memory on granite-4.2-30b for no benefit, since each graph writes only
the rows it owns and the connector reads only the rows the current step filled.
`_configure_registry` sizes the buffers from `max_cudagraph_capture_size` when
the worker connector is built, which is before `compile_or_warm_up_model` runs.

Three supporting changes:

* `score_step` resolves every layer's queries before it scores anything and
  calls `_warn_no_queries` when a decode step has none, rate limited to powers
  of two. The per-request count is emitted as `decode_steps_unscored`. The
  silent `continue` is what let this run for weeks, and a well-formed zero
  record can no longer be produced without a warning next to it.
* The decode-row sanity check no longer compares against the probe's row count.
  Under a replayed graph that number is the padded capture size and says
  nothing about the current step. It now compares the batch's own token count,
  from the scheduler metadata, against the decode requests the connector
  recognises.
* Scoring takes a private copy of the rows it needs and holds the next forward
  only until that copy lands. The buffers are persistent now, so without this
  the next forward would rewrite them in place while the side stream was still
  reading them.

`tests/e2e_smoke.py` takes a `--cuda-graphs` flag that drops `enforce_eager`,
and `tests/test_aggregations.py` covers the key derivation and the freshness
rules with no GPU.

## Limits of the fix

`cudagraph_mode=FULL` without `PIECEWISE` captures one mixed prefill+decode
graph per size and dispatches pure decode batches to it. Those graphs are
captured with `max_query_len != 1`, so the probe skips them and no copy is
recorded. Decode steps under that setting are still unscored, but they now warn
and count instead of emitting zeros. `FULL_AND_PIECEWISE` (the default),
`FULL_DECODE_ONLY`, `PIECEWISE` and `NONE` all capture queries.

## Evidence

