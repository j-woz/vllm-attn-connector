# vllm-kvnorm

Captures **one importance score per token** from vLLM's paged KV cache at the
moment a request finishes, and emits it as **Flowcept provenance** — after all its KV has been written, but before its
blocks are returned to the block pool.

The score is the PagedEviction proxy (Chitty-Venkata et al., *Findings of EACL
2026*, [arXiv:2509.04377](https://arxiv.org/abs/2509.04377)), Algorithm 1,
averaged over KV heads and layers:

```
S_i = mean over layers and kv heads of  ||V_i||_2 / ||K_i||_2
```

High `S` means important: per Devoto et al. 2024
([arXiv:2406.11430](https://arxiv.org/abs/2406.11430)) a key's L2 norm is
*inversely* proportional to that token's cumulative attention score.

This avoids attention scores entirely, so **no FlashAttention kernel changes are
needed** — which is the whole reason the metric exists.

## Design

Implemented as an **out-of-tree `KVConnector`**. vLLM is not modified at all;
`kv_connector_module_path` loads the class from this package
(`KVConnectorFactory.get_connector_class`).

`K` and `V` for a token are written once and never change, so a token can be
scored in the step that writes it. That is the whole design, and it is what lets
the connector avoid ever holding on to KV blocks:

```
step N   schedule()        -> metadata: which requests wrote how many tokens
         start_load_kv()   -> emit any request that finished, after waiting on
                              the event covering its last scoring launch
         <forward>         -> writes this step's KV
         wait_for_save()   -> launch scoring for exactly this step's new tokens,
                              on a side CUDA stream, into a per-request buffer
```

By the time a request finishes, every one of its tokens has already been scored.
Its blocks are never read again, so there is nothing to pin, nothing to release,
and no way for capture to withhold memory from the pool.

Scoring runs on a side stream ordered after the forward that wrote the KV
(`stream.wait_stream`), so it overlaps the *next* forward instead of delaying
this one. A request that finishes while its scoring is still in flight is held
back by a CUDA event until it completes.

Three details worth knowing:

- Emission happens in `start_load_kv`, scoring in `wait_for_save`. That split is
  deliberate: `wait_for_save` is skipped on the no-forward path
  (`kv_connector_no_forward` passes `wait_for_save=False`), which is correct for
  scoring -- no forward means no new tokens -- but emission must still run there,
  and `start_load_kv` always does.
- Completion is reported via `build_connector_worker_meta`, not `get_finished`.
  `get_finished` asserts the request is still in `Scheduler.requests`, which is
  false here because the request tears down normally.
- Only the first KV cache group is scored.

### Model-agnostic

Nothing is hard-coded to FlashAttention. vLLM's per-layer KV tensor always has
the *logical* shape from `AttentionBackend.get_kv_cache_shape()`; the NHD/HND
choice only permutes strides. `layout.py` dispatches on that logical shape and
produces copy-free canonical views `(num_blocks, block_size, num_kv_heads,
head_size)`, and the Triton kernel reads `.stride()` off the tensors. Supported:

| Logical shape | Label |
|---|---|
| `(num_blocks, num_kv_heads, block_size, head_size + head_size_v)` | `fused_head_major` (FlashAttention) |
| `(num_blocks, block_size, num_kv_heads, head_size + head_size_v)` | `fused_token_major` |
| `(2, num_blocks, block_size, num_kv_heads, head_size)` | `split_token_major` |
| `(2, num_blocks, num_kv_heads, block_size, head_size)` | `split_head_major` |

Layers that don't match (Mamba/linear-attention state, MLA latents, quantised
caches) are skipped with a warning; the rest still work.

## Install

Assumes vLLM (with Triton) is already installed. Two more things must be
importable:

- **this package**, `vllm_kvnorm`;
- **Flowcept**, including its `vllm` adapter
  (`flowcept/src/flowcept/flowceptor/adapters/vllm/`). A released Flowcept will
  not have that adapter yet, so it must come from the local checkout.

Pick whichever fits what you are doing.

**Not editing either package** — install them:

```bash
pip install ../flowcept .
```

**Editing either package** — put them on the path instead, so changes take
effect immediately:

```bash
ROOT=$(cd .. && pwd)
export PYTHONPATH="$ROOT/vllm-kvnorm/src:$ROOT/flowcept/src"
```

`PYTHONPATH` is inherited by vLLM's EngineCore subprocess, so the connector
resolves there too. Avoid a plain `pip install` while editing: it snapshots the
source, so you would silently keep testing the previously installed copy. (An
editable install, `pip install -e`, also avoids that if you prefer it.)

If `transformers` fails to import due to a torchaudio/CUDA version mismatch,
`pip uninstall torchaudio` — vLLM only needs it for audio models.

## Usage

Scores are emitted through Flowcept's `vllm` adapter
(`flowcept.flowceptor.adapters.vllm.VLLMInterceptor`), so they land in
Flowcept's normal MQ/Mongo pipeline alongside the rest of your provenance.

```python
from flowcept import Flowcept
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

with Flowcept("vllm", workflow_id="my-experiment", workflow_name="my_experiment"):
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        kv_transfer_config=KVTransferConfig(
            kv_connector="KVNormConnector",
            kv_connector_module_path="vllm_kvnorm",
            kv_role="kv_producer",
            kv_connector_extra_config={"workflow_id": "my-experiment"},
        ),
    )
    llm.generate(["The capital of France is"], SamplingParams(max_tokens=24))
```

> **Use `"kv_role": "kv_producer"`, not `"kv_both"`.** This connector never
> loads KV, so it is not a consumer. Declaring it one sets
> `KVTransferConfig.is_kv_consumer`, which combined with async scheduling (on by
> default) makes the scheduler set `defer_block_free = True`
> (`scheduler.py:155`). Preempted blocks then go to a deferred queue instead of
> straight back to the pool, so the preemption loop gets no immediate relief and
> cascades: on a saturated pool this workload preempted **19** times with
> `kv_both` versus **3** with `kv_producer` -- the same as running with no
> connector at all. Capture is identical either way.

`kv_connector_extra_config` takes exactly one optional key, `workflow_id`: the
*parent* workflow this run nests under. There is nothing else to configure.

Two things to know:

- **A live MQ (Redis) is required** for the default multiprocess setup. vLLM runs
  its scheduler in a separate EngineCore process, so the interceptor publishes
  from there, exactly as Flowcept's Dask adapter does from workers. Setting
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` puts everything in one process and removes
  the requirement — that is what the tests and the example do.
- **The run registers its own workflow, it does not reuse yours.** Flowcept
  records a given workflow once, so reusing the caller's id would silently drop
  the model configuration. Join on `parent_workflow_id`.

## Output format

Two Flowcept record types. One **workflow** per run, carrying everything needed
to interpret the tasks -- in particular the tokenizer, without which
`prompt_token_ids` cannot be decoded:

```jsonc
{
  "type": "workflow",
  "workflow_id": "vllm-kvnorm-daea0462c1ca",
  "parent_workflow_id": "my-experiment",
  "name": "facebook/opt-125m",
  "conf": {
    "model": "facebook/opt-125m", "tokenizer": "facebook/opt-125m",
    "tokenizer_mode": "auto", "trust_remote_code": false,
    "dtype": "torch.float16", "max_model_len": 512,
    "architectures": ["OPTForCausalLM"], "tensor_parallel_size": 1
  }
}
```

One **task** per finished request:

```jsonc
{
  "type": "task", "subtype": "kv_token_importance",
  "task_id": "0-9100089a:g0",
  "workflow_id": "vllm-kvnorm-daea0462c1ca",
  "activity_id": "kv_token_importance", "status": "FINISHED",

  "used": {
    "request_id": "0-9100089a",
    "prompt_token_ids": [2, 133, 812, 9, 1470, 16],   // "</s>The capital of France is"
    "num_prompt_tokens": 6, "num_computed_tokens": 29
  },
  "generated": {
    "score": [0.0132, 0.1332, 0.1630, ...]            // one float per token, in order
  },
  "custom_metadata": {
    "metric": "pagedeviction_v_over_k_l2",
    "metric_reference": "arXiv:2509.04377 Algorithm 1",
    "kv_cache_group_id": 0, "num_layers": 12, "num_kv_heads": 12,
    "num_tokens": 29, "finished_at": 1787003963.9
  }
}
```

`score[i]` is the importance of token `i`, in sequence order. There are exactly
`num_computed_tokens` of them.

Decode the prompt with the tokenizer recorded on the workflow:

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(workflow["conf"]["tokenizer"])
tok.convert_ids_to_tokens(task["used"]["prompt_token_ids"])
```

`num_computed_tokens` is the number of positions that actually have KV, which is
**`len(prompt) + len(output) - 1`**: the final sampled token is never fed back
through the model, so no K/V is ever written for it. `prompt_token_ids` labels
only the first `num_prompt_tokens` scores; the rest are generated tokens, whose
ids are not captured.

One task per `(request, kv_cache_group)`. Models with a single attention type --
most models -- have one group, hence one task per request.

### Size

One float per token, rounded to 6 decimals, is **~10 bytes/token**, plus ~6
bytes per prompt token — around 1 KB per request for a short generation,
including all metadata. A 128K
context request is ~1.3 MB regardless of model size, since layers and heads are
reduced away on GPU. This is why there is no top-K or reduction knob: the output
does not scale with model depth or width.

## Tests

Tests live in the sibling experiments repo, `../experiments/vllm-kvnorm/`:

```bash
cd ../experiments/vllm-kvnorm
pytest test_kernels.py -q             # kernel + layout units
pytest test_incremental_scoring.py -q # step-by-step scoring == one-shot
python e2e_smoke.py                   # real vLLM run + record schema
python test_preemption.py             # preemption does not perturb the capture
```

`test_incremental_scoring.py` is the load-bearing one. It drives
`_WorkerSide.score_step` with synthetic metadata and asserts that accumulating a
sequence across steps reproduces a single pass over the whole thing -- under
chunked prefill, incrementally arriving blocks, preemption restarts and
concurrent requests.

That property is where the bugs are. An early version passed the per-layer
output buffer straight to the kernel, which *stores* rather than accumulates, so
only the last layer survived and every score was wrong by roughly 100%. Nothing
end-to-end caught it, because those checks only compared the connector against
itself. Reintroducing that bug fails 6 of the 8 cases here.

Validated on `facebook/opt-125m` (MHA, 12 layers, 12 KV heads) and
`Qwen/Qwen2.5-0.5B-Instruct` (GQA, 24 layers, 2 KV heads).

## Tensor parallelism

**One task per request, not one per rank.**

vLLM creates a worker-role connector on *every* TP rank (`gpu_worker.py:662`),
and each rank holds only its shard of the KV heads
(`num_kv_heads // tp_size`, replicated when `tp_size` exceeds the head count).
So each rank can only compute a *partial* score, and left alone they would all
emit a task under the same `task_id`.

The connector therefore all-reduces before emitting:

```python
score = tensor_model_parallel_all_reduce(score) / tp_size   # every rank
if tp_rank != 0:
    continue                                                # only rank 0 writes
```

Averaging the per-rank shard means recovers the mean over all heads. That holds
in both sharding regimes: with `tp_size <= num_kv_heads` the shards partition the
heads, and with `tp_size > num_kv_heads` vLLM replicates each head the same
number of times, so the average stays uniform over distinct heads.

The collective runs on every rank *before* the rank-0 check, so all ranks reach
it. This is safe because connector metadata is broadcast from the scheduler, so
every rank iterates the same requests in the same order.

Consequences:

- exactly one record per request, identical to the TP=1 output;
- the score is the true all-head mean, not a shard mean;
- no per-rank output files.

`test_kernels.py` verifies the reduction maths for `tp_size` of 1, 2, 4 and for
the replicated case, by simulating shards over the head axis — no multi-GPU
needed. The all-reduce itself is exercised only under a real multi-GPU run,
which has not been tested here (single-GPU machine).

## Preemption is a non-issue

`Scheduler._preempt_request` frees a request's blocks directly
(`scheduler.py:1352`), bypassing the connector hook. That is fine, and requires
no handling:

- preemption resets `num_computed_tokens` to 0 and requeues the request, which is
  then **fully recomputed**;
- the connector keys off `num_computed_tokens`, so it simply re-scores the
  recomputed tokens over the same buffer positions -- the operation is
  idempotent;
- so nothing is missed, and nothing is double counted.

`test_preemption.py` forces this (8 prompts, 10 blocks → 13 preemptions) and
confirms one record per request, not one per preemption, covering the full final
sequence. `test_incremental_scoring.py` covers the restart arithmetic directly.

## Known limitations

- **Quantised KV caches (FP8/NVFP4) are skipped.** Norms would need scale
  plumbing to be meaningful.
- **MLA layouts are skipped** — there is no separate V to take a norm of.
- **Only the first KV cache group is scored.** Fine for the single-attention-type
  models that make up most of the field; hybrid models capture only their first
  group.
- **In-flight score buffers grow with sequence length** — one float per token per
  live request. Negligible in absolute terms (~4 KB per 1k-token request), but it
  is held for the request's lifetime rather than a couple of steps.
- **With `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the last batch is not emitted.**
  A request is reported finished one scheduler step after it ends; with
  multiprocessing enabled the EngineCore loop keeps stepping on its own while
  `has_pending_push_work()` is true, but in-process nothing drives it once
  `generate()` returns. Issue one more trivial request to flush. This does not
  affect the default (multiprocessing on) or a served endpoint.
