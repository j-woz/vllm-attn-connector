# tests

Three suites with very different requirements.

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

python tests/e2e_smoke.py                                   # CUDA graphs ON, as served
python tests/e2e_smoke.py --enforce-eager                   # graphs off
python tests/e2e_smoke.py --ranges                          # variable segments
python tests/e2e_smoke.py --chunk-size 1 --top-pct 100      # full capture
python tests/e2e_smoke.py --top-pct 25 --chunk-size 64      # coarser segments
python tests/e2e_smoke.py --max-steps 32                    # bounded capture
```

Exit status is the number of failed assertions, so these compose in a loop.

**Graphs are on by default here on purpose.** `vllm serve` runs with CUDA
graphs, and the probe has to survive graph replay; an earlier version did not,
and because the test suite forced `enforce_eager=True` it never noticed. Records
came out structurally perfect and numerically all-zero.

The load-bearing assertions are described in the root README under
[Validation](../README.md#validation). Two worth restating: **`attn_sum` totals
one softmax unit per decode step**, which is what catches a capture that never
happened, and **the attention sink is reproduced at three orders of magnitude
above the median position**, which is what tells you the recomputed scores are
real rather than plausible-looking.

## `e2e_example.py` — needs a GPU and a real engine

The other end-to-end test, on the workload the connector was built for: a
multi-round conversation where each turn depends on earlier ones and **the
model's own answers are the history**, so a wrong answer propagates. It
exercises what `e2e_smoke.py` does not — a prompt that grows every turn (953 →
3540 tokens over five rounds), declared ranges that grow with it (3 → 19), and
several captures from one engine.

The chain is `bio_example.csv`: five rounds of a synthetic phenotyping study
from the OPAL benchmark generator, with columns `chain_id, round_no, task_type,
prompt, tool_ranges, correct_answer, explanation`.

```bash
ROOT=$(cd ../.. && pwd)
export PYTHONPATH="$PWD/src:$ROOT/flowcept/src"
export FLOWCEPT_SETTINGS_PATH="$ROOT/flowcept/agent_sandbox/settings.yaml"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

python tests/e2e_example.py                      # Qwen3-4B AWQ, the default
python tests/e2e_example.py --cuda-graphs        # needs >6 GB: 4B + graph pools
python tests/e2e_example.py --model <hf-id>      # any instruct model
```

Asserted: records are emitted for every turn, `segment_mode` is `variable`,
every declared range survives verbatim as a segment, the prompt and the range
count grow monotonically, and kept mass + `topk_residual` = 1 per turn.

Reported but **not** asserted: how many rounds the model got right, where the
first mistake fell, and how many later rounds inherited it. A low score is a
fact about the model, not a bug in the connector. Wrong answers are matched
against the shortcuts named in the CSV's `explanation` column, so a failure that
corresponds to "ignored the quality gate" is distinguishable from noise.

### Two ways to get the ranges wrong

The CSV ships a `tool_ranges` column and it is **not** usable as-is. The test
recomputes spans from the live prompt in `declare_ranges()`. Neither mistake
fails loudly; both just attribute attention to the wrong text.

**The tokenizer differs.** Those spans are indices under the generator's regex
tokenizer, which is far coarser than a BPE vocabulary — on this chain the model
emits 1.24×–1.43× more tokens for the same prompt, and the ratio varies with how
numeric the text is. Used verbatim, a block's start lands 15 to 143 tokens away
from the block.

**History shifts every index.** CSV spans are relative to a round's own prompt.
As turn *N* of a conversation, everything moves by the system prompt, the
previous rounds, the previous answers and the chat template's markers: +965
tokens by round 2, +2969 by round 5. The shift also depends on *whose* answers
the history holds — replaying with the model's own replies instead of ground
truth moved it by +1 token at round 3 and +6 at round 5, since the answers
tokenize to different lengths. Offsets must be derived from the exact string
passed to the tokenizer on that turn, not precomputed per round.

## Not covered

**CUDA graphs in `e2e_example.py`.** It runs eager by default because a 4B model
plus graph pools exceeds a 6 GB card. Graph replay is covered by
`e2e_smoke.py`, which runs graphs-on by default on a 0.5B model; pass
`--cuda-graphs` here on a larger card to cover both at once.

**Tensor parallelism.** `_reduce_row` combines per-rank slices, and has never
run with `tp_size > 1`.

**Preemption.** The connector detects restarts by watching
`num_computed_tokens` go backwards, then drops partial records rather than
stitching them. There is no test for it; the path has only ever been exercised
incidentally, by a real run that happened to preempt.


**Long chains.** `bio_example.csv` stops at five rounds because the sixth needs
~4.5k tokens of context and the default `--max-model-len` is 4096. A longer
chain needs the window raised, and on a 6 GB card that trades against
`--gpu-memory-utilization`.
