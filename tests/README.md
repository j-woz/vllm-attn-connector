# tests

Two suites with very different requirements.

## `test_aggregations.py` — no GPU, no vLLM, no model

Pure-torch checks of the two things that decide what gets emitted:

* the **streaming reduction** the connector performs layer by layer (so the
  `[L, H, T]` tensor is never materialised) equals the dense definition, across
  varying layer/head/prompt shapes;
* **segment planning and selection** — that the partition is complete, ordered
  and non-overlapping, that every segment gets exactly its quota, that declared
  ranges survive verbatim with gaps filled, and that position 0 is never
  selected.

It reads `_segment_plan`, `_segment_index` and `_segmented_topk` out of
`connector.py` by parsing the source, because importing the module requires
vLLM and these functions do not.

```bash
python tests/test_aggregations.py     # verbose, prints every case
pytest tests/test_aggregations.py     # same checks, quiet
```

## `e2e_smoke.py` — needs a GPU and a real engine

Builds an actual vLLM engine with the connector installed, generates, and
asserts 38 properties of the emitted provenance. Not a pytest module: it takes
CLI flags to exercise each configuration, and one engine build per run is too
slow to parametrise.

```bash
ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PWD/src:$ROOT/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python tests/e2e_smoke.py                                   # defaults
python tests/e2e_smoke.py --ranges                          # variable segments
python tests/e2e_smoke.py --chunk-size 1 --top-pct 100      # full capture
python tests/e2e_smoke.py --top-pct 25 --chunk-size 64      # coarser segments
python tests/e2e_smoke.py --max-steps 32                    # bounded capture
```

Exit status is the number of failed assertions, so these compose in a loop.

The load-bearing assertions are described in the root README under
[Validation](../README.md#validation). The one worth restating: **the attention
sink is reproduced at three orders of magnitude above the median position**.
Recovering a known property of decoder-only LLMs is what tells you the
recomputed scores are real rather than plausible-looking.

## Not covered

**Preemption.** The connector detects restarts by watching
`num_computed_tokens` go backwards, then drops partial records rather than
stitching them. There is no test for it; the path has only ever been exercised
incidentally, by a real run that happened to preempt.

**Tensor parallelism.** `_reduce_row` combines per-rank slices across ranks, and
has never run with `tp_size > 1`.
