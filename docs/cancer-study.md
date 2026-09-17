# Attention provenance for cancer drug-response models

A worked, end-to-end study: take two published drug-response models, capture
what they attend to at inference, record it as provenance, and turn that
recorded provenance into an evaluation benchmark in the OPAL format.

Everything below was run on a laptop (M1 Max, no GPU) and every number is
reproduced from a real model run. Where something is synthetic, it says so.

---

## Part 1 — The basics

Skip to Part 2 if you already know what attention and provenance are.

### The scientific problem

SPOTTER-AI is a cybersecurity framework for scientific AI workflows. The threat
it addresses is **integrity**, not confidentiality: not "who read my data" but
"has something in this pipeline been corrupted, and if a result looks wrong,
which input caused it".

That second question is *attribution*, and it is hard. A model consumes
thousands of inputs and emits one number. When that number is wrong, nothing in
the output says which input was responsible.

### What attention is

Most modern models weight their inputs. Reading a sentence, a language model
weights some words more than others when producing each output word. Reading a
tumour's gene-expression profile, a drug-response model weights some genes more
than others when predicting sensitivity.

Those weights are called **attention**, and they are computed internally on
every forward pass. Capture them and you have a per-prediction record of which
inputs the model actually consulted.

Attention is not a causal explanation, and the limits matter:

- **Not signed.** Strong attention on a gene means "this was consulted", not
  "this drove the answer upward". A model can attend to something in order to
  suppress it.
- **Not free of artifacts.** Some positions absorb weight structurally
  regardless of content. In language models it is the first token, the
  "attention sink" ([arXiv:2309.17453](https://arxiv.org/abs/2309.17453)); we
  found the exact analogue in Paccmann, discussed below.

With those caveats it is still the most direct attribution signal available,
and crucially it is *already computed* — capture costs almost nothing.

### What provenance is

Provenance is the record of how a result came to be: which inputs, which code,
which parameters, which intermediate steps. **Flowcept** (ORNL) is the
provenance system used here. It captures runtime lineage from ML and scientific
workflows and stores it queryably.

The pairing is the point. Attention says *which inputs mattered for this
prediction*; provenance says *which prediction, from which run, on which data*.
Neither alone supports attribution after the fact.

### The models

Both predict **drug response**: given a cancer cell line's molecular profile
and a compound, predict how sensitive that line is to that compound. The
measure is **AUC** — area under the dose-response curve, where lower means more
sensitive.

| | Paccmann MCA | HiDRA |
|---|---|---|
| inputs | gene expression + compound SMILES | gene expression by KEGG pathway + compound fingerprint |
| framework | PyTorch | TensorFlow / Keras |
| attention | 16 heads over compound tokens; 1 head over 2,128 genes | 1 head over 186 pathways; 1 per pathway over its genes |
| structure | bimodal encoder-regressor | hierarchical (gene level, then pathway level) |

Both are curated by [JDACS4C-IMPROVE](https://github.com/JDACS4C-IMPROVE), a
DOE/NCI project that standardised many drug-response models behind one
interface (`improvelib`). That shared interface is why one capture design
serves both, and why a third model would be cheap to add.

### Why these, and why now

The lineage is direct. *Probing Decision Boundaries in Cancer Data Using Noise
Injection and Counterfactual Analysis* (Jain, Shah, Mohd-Yusof, Wozniak,
Brettin, Xia, Stevens — **CAFCW/SC 2021**) did label flipping, correlated and
uncorrelated label noise, Gaussian feature noise at varying rates, an
abstaining classifier, and counterfactual feature attribution over 60,483
features, on the CANDLE NT3 benchmark.

That paper asked: *when you corrupt cancer data, what happens to the decision
boundary, and can you attribute the damage?* It answered it with counterfactual
perturbation — expensive, because it re-runs the model many times.

This work asks the same question with a cheaper instrument. The model already
computes an attribution signal on every forward pass; it is simply discarded.
Capturing it costs one hook. Where CAFCW 2021 perturbed inputs to infer
importance, this reads the importance the model itself assigned — and records
it as provenance so it survives the run.

Neither replaces the other. Perturbation measures causal effect; attention
measures consultation. The interesting experiment is whether they agree, and
where they diverge.

---

## Part 2 — What was built

### The starting point

`vllm-attn-connector` captures exact per-decode-step attention from a running
vLLM engine and emits it as Flowcept provenance. It is hard work, because in an
LLM the attention is **destroyed as it is computed**: FlashAttention tiles the
softmax and discards the scores, decode queries never enter the KV cache, and
`torch.compile` erases the module boundary a hook would attach to. Recovering
it needs a backend override plus recomputation of `q·Kᵀ` against the paged
cache.

Drug-response models need none of that. They are single-pass, eager-mode, no KV
cache, no autoregression. **The attention is already materialised.** Paccmann
even returns it in a dict — and the caller throws it away
(`test_paccmann.py:161` binds `pred_dict` and never reads it).

So the whole apparatus collapses to: catch what is already falling on the
floor, and hand it to the same interceptor the vLLM connector uses.

### Design: three modules, one emitter

```
drp_emit.py      framework-agnostic: reduce over heads, emit to Flowcept
drp_paccmann.py  torch forward hooks
drp_hidra.py     Keras named-layer extraction
drp_chains.py    read provenance back out, build OPAL-format chains
```

Three capture paths rather than one abstraction, because the mechanisms are
genuinely different — and the differences are instructive:

| | mechanism | why |
|---|---|---|
| vLLM | attention backend override | `torch.compile` + CUDA graphs defeat module hooks |
| Paccmann | `register_forward_hook` | eager and uncompiled, so the hook that fails in vLLM works here |
| HiDRA | second Keras `Model` over shared weights | every attention layer is explicitly `name=`d |

**No model source is patched.** That matches the project's existing stance on
vLLM, and it means the capture cannot be blamed for a change in results.

### Adding a third model

`AttentionCapture` owns everything downstream of "attention is in hand":
resolving the interceptor, assembling metadata, reducing over heads, emitting,
recording model identity. A new model supplies three constants and one method:

```python
class MyModelCapture(AttentionCapture):
    MODEL_NAME = "MyModel"
    FRAMEWORK = "torch"
    METRIC_REFERENCE = "mymodel_scaled_dot_product"

    def collect(self, **kwargs) -> list[AttentionAxis]:
        return [AttentionAxis(name="gene", per_sample_heads=...)]
```

Duo is the obvious next one — it already runs on the same `improvelib`
interface as the other two.

Chain task types extend the same way, through a registry rather than an
if-ladder:

```python
@register_task("my_task")
def _build(ctx: ChainContext) -> Round | None:
    return Round(...)          # or None to skip when inputs are unavailable
```

`ChainContext` carries the per-chain state — predictions, QC flags, panel
roles, captured attention, and the answers from earlier rounds — so a new task
needs no change to `build_chains`.

### What gets recorded

One Flowcept task per sample per axis, following the connector's `:g<n>`
convention:

```
<sample_id>:g0    compound / SMILES axis
<sample_id>:g1    gene axis
```

Each carries three series, using the same field names the vLLM payload uses so
one consumer reads both producers without branching:

| field | meaning |
|---|---|
| `attn_sum` | elementwise sum across heads — the distribution-shaped view |
| `attn_peak` | elementwise max across heads |
| `attn_argmax_head` | which head supplied the peak |

**Why both a sum and a peak.** A mean over many heads dilutes. If a small
number of heads carry the behaviour of interest, averaging them against the
rest buries the signal. The peak preserves it. This is the same reasoning as
`val_all_max` vs `val_all_avg` in the vLLM connector, and it is why the
Paccmann hook attaches to `ContextAttentionLayer` rather than reading
`prediction_dict`: the model averages its 16 heads at `paccmann.py:273-279` and
destroys exactly the structure worth keeping.

### What does not carry over from vLLM

Stated rather than faked:

- **No time axis.** vLLM records attention per decode step, so turnover between
  steps is meaningful. These models produce one prediction from one forward
  pass: `decode_steps_recorded = 1`. Everything step-indexed in the vLLM
  payload — `topk_residual`, `decode_steps_dropped`, `wins_first_step` — has no
  counterpart and is absent.
- **A different quantity.** Paccmann attention is additive/Bahdanau
  (`tanh(W_r·ref + W_c·ctx) → Linear → softmax`); HiDRA's is a learned softmax
  gate. Neither is `softmax(qKᵀ/√d)`. Each record carries a
  `metric_reference` string so numbers from different producers are never
  silently pooled.

---

## Part 3 — Running it

### Environment

Both models must be trained first. **`docs/model-setup.md`** is the full path —
environment, the 1.4 GB benchmark download, the patches each model needs, and
the two silent failure modes to check for before trusting any number.

```bash
conda activate spotter          # python 3.13, torch 2.14, tensorflow 2.21
pip install pymongo redis       # for the provenance store
```

### Provenance store

Flowcept ships with every backing store disabled, so records go to an in-memory
buffer and vanish. For real persistence:

```bash
docker run -d --name flowcept_mongo -p 27017:27017 mongo:7.0
docker run -d --name flowcept_redis -p 6379:6379 redis
```

> **Note.** `mongo:latest` will not start on recent Docker kernels
> (`MongoDB cannot start: Linux kernel versions 6.19 and newer has a known
> incompatibility`, SERVER-121912). Pin `mongo:7.0`.

Then enable `mongodb`, `mq`, and `kv_db` in a settings file and point
`FLOWCEPT_SETTINGS_PATH` at it. Flowcept's `full-online` profile sets exactly
these. All three are required together — with `mq.enabled` but `kv_db` off,
startup fails with a `check_safe_stops` error.

### The pipeline, via the CLI

Three stages. Only the first needs a model; once attention is recorded,
nothing downstream loads one again.

```bash
# 1. capture -> provenance store
python -m vllm_attn_connector.drp_cli capture-paccmann \
    --model-dir /tmp/pmca_out3 --data-dir /tmp/pmca_ml2 \
    --workflow-id pmca-attention-v1 --write-features /tmp/genes.txt

# 2. check what landed
python -m vllm_attn_connector.drp_cli inspect --workflow-id pmca-attention-v1
#   tasks : 742
#   axes  : {'smiles': 371, 'gene': 371}

# 3. store -> chains
python -m vllm_attn_connector.drp_cli chains \
    --workflow-id pmca-attention-v1 \
    --predictions preds.csv --features /tmp/genes.txt \
    --out data/cancer_opal_chains.csv
```

`capture-hidra` is the TensorFlow equivalent.

### Capture, as a library

```python
from flowcept import Flowcept
from vllm_attn_connector.drp_paccmann import PaccmannCapture

with Flowcept("vllm", workflow_id=WF, workflow_name="paccmann"):
    with PaccmannCapture(model, workflow_id=WF) as cap:
        cap.send_workflow(params)
        with torch.no_grad():
            model(smiles, gep)
        cap.emit(sample_ids=cell_lines)
```

HiDRA is the same shape, with `HidraCapture(model, workflow_id=WF)` and
`cap.emit(inputs=X, sample_ids=...)`.

### Read it back

```python
from vllm_attn_connector.drp_chains import load_attention_from_store

attn = load_attention_from_store(
    workflow_id="pmca-attention-v1",
    feature_names=gene_labels,
    axis="gene",
)
```

### Tests

```bash
pytest tests/                                        # 27, no GPU/vLLM/torch/TF
python tests/drp_smoke.py --paccmann /tmp/pmca_out3  # real checkpoint, CPU
python tests/drp_smoke.py --hidra    /tmp/hidra_out
```

The smoke tests assert the load-bearing properties: distributions really are
softmax outputs (sum to ~1 per head), heads disagree (so per-head structure
survived), and **predictions are bit-identical with capture installed** —
provenance must observe, not perturb.

---

## Part 4 — Results

### Model baselines

Short runs, enough to prove the pipeline. Not tuned, not comparable to
published numbers.

| model | epochs | val PCC | test PCC | test R² |
|---|---|---|---|---|
| Duo (reference) | full | 0.8929 | 0.8996 | 0.8085 |
| HiDRA | 2 | 0.7516 | 0.7667 | 0.5320 |
| Paccmann MCA | 3 | 0.7702 | 0.7865 | 0.3133 |

CCLE split 0 throughout: 7,608 train / 952 val / 951 test, 371 cell lines,
24 compounds.

### Capture, verified

| check | Paccmann | HiDRA |
|---|---|---|
| heads captured | 16 (architecture: `multiheads=[4,4,4,4]`) | 1 pathway-level |
| axis widths | 560 tokens, 2,128 genes | 186 pathways |
| distributions sum to ~1/head | yes | yes |
| predictions unchanged by capture | yes, bit-identical | yes |
| model source patched | none | none |

Persistence confirmed against MongoDB: 6/6 tasks written and queried back with
attention intact, then 371 cell lines captured in one workflow.

### The attention sink, found not assumed

On the gene axis, **PARN is the top-attended gene on all 371 cell lines**, at
0.9988 of the weight. Second place is RPUSD3 at 0.0009 — three orders of
magnitude down.

This is the drug-response analogue of the LLM attention sink: a position that
absorbs weight structurally, near-independently of content. It carries no
sample-specific information, so an attribution claim naming PARN is vacuous.

It was not designed into the benchmark. It was discovered by looking at real
captured output, and it then became the natural distractor for the attention
task — the same role position 0 plays in the vLLM connector, excluded for the
same reason.

---

## Part 5 — The cancer OPAL chains

### What OPAL is

`data/opal_chains.csv` is SPOTTER's tool-grounded reasoning benchmark: 100
multi-round investigations over plant phenotyping, 430 rows, 8 task types. Each
row is one question. The prompt inlines the tool output it needs and demands a
single exact line back (`ANSWER: 40`).

It tests **rule-following under traps**, not arithmetic. From a real
explanation field:

> *"the step is t100->t200, not the adjacent column t150 which is a pilot
> cell... ablations: stepping to the next column instead of the next design
> level gives RIL 369; ranking without the panel table gives Esp49, which is
> observational"*

The naive answer is wrong. Only a model that consults the design table and the
panel-role table gets it right, and the `explanation` column records each wrong
path and what it produces.

Rounds chain: round 2 says *"starting from the dose you identified in round
1"*, so an early error propagates.

### The cancer counterpart

`data/cancer_opal_chains.csv` — 24 chains, 96 rows, identical 7 columns.

| round | task | the trap |
|---|---|---|
| 1 | `sensitivity_ranking` | a QC gate voids the apparent winner |
| 2 | `panel_restricted_rank` | an observational line outranks the core panel |
| 3 | `attention_attribution` | the housekeeping sink dominates the weights |
| 4 | `prediction_audit` | — (tolerance check) |

Round 3 exists only because of the capture work. It asks *which gene did the
model attend to when making this prediction*, and the tool output is real
captured attention read back out of MongoDB.

### What is real and what is not

**Real:** every AUC is a Paccmann prediction on a real CCLE cell line. Every
attention value is what the model actually computed, read from the provenance
store — not passed in memory. That round trip is deliberate: an in-process dict
would produce identical chains even if persistence were broken, so it would
prove nothing.

**Synthetic:** the QC flags and panel-role assignments. CCLE ships no
per-prediction QC table, and the benchmark needs distractors to be worth
anything. They are derived deterministically from a seed and recorded in the
explanation, so a reviewer can see which rows were marked and why. The
measurements they gate are real.

### tool_ranges is deliberately empty

The ranges in the original look like `(db_lookup[303:429])`. They are **token**
offsets, not character offsets — measured ratio ~3:1 chars per unit, and all
391 multi-range rows tile perfectly contiguously.

Bogdan confirmed two dependencies: the tokenizer of the model under evaluation,
and how much prior-round history is prepended (chains are evaluated with either
ground-truth history, for per-prompt accuracy, or the model's own answers, for
per-chain accuracy).

Neither is known here, so the column is left empty rather than guessed.
`add_tool_ranges(rows, tokenize=..., history=...)` fills it once you supply a
tokenizer. Writing character offsets would produce a file that looks right and
scores wrong.

### How the ground truth is checked

`tests/test_drp_chains.py` re-derives every answer by **parsing the rendered
prompt**, never by consulting the objects that produced it. If the generator
and the renderer ever disagree, that fails the build. All 96 rows verified:
24/24 round-1 answers match the table, 24/24 attention rounds avoid the sink.

---

## Part 6 — Four bugs found in Paccmann

Running a published model on current libraries surfaced four defects. Two are
routine; two would corrupt any result produced with them.

Fixes are on a fork, since upstream `JDACS4C-IMPROVE/Paccmann_MCA` has been
inactive since 2024-09:

    github.com/rajeeja/Paccmann_MCA   branch rajeeja/py313-fixes

### 1. Library incompatibilities (routine)

`np.float` removed in numpy 1.24; pytoda 1.1 now *raises* on the `device=`
argument rather than warning; predictions stack to `(N,1)` but `pearsonr`
requires 1-D.

### 2. The SMILES language pickle (routine)

The curated `.pkl` is a bare `SMILESLanguage` carrying only a vocabulary.
pytoda 1.1 moved every transform flag onto `SMILESTokenizer`, so the dataset
died on `'SMILESLanguage' object has no attribute 'canonical'`.

Fixing it exposed a second issue: pytoda logs *"smiles_language value takes
preference"*, meaning the language object overrides the dataset's `augment=`
argument. Upstream turns augmentation off for val/test, but a single shared
language object silently augmented them anyway. Separate objects for train and
eval; **val MSE improved 0.0188 → 0.0143**.

### 3. Gene expression was silently all zeros (serious)

The omics loader reads `cancer_gene_expression.tsv` expecting a three-row
header (Ensembl / Entrez / Gene_Symbol). In `drp_data_v0.2.0` that file has a
**single** header row of gene symbols. So the loader consumed two cell lines as
header rows, every gene name parsed as a float, the 2,128-gene lookup matched
nothing, and the model was fed an **all-zero expression matrix**.

Nothing raised. Training converged. Inference ran. Scores came out.

The only symptom was validation and test disagreeing wildly — val R² 0.37
against test R² −3.19 — which is what prompted the investigation. After the
fix both agree and **test PCC went 0.687 → 0.787**.

Worth checking any other IMPROVE model pairing a v0.1.0-era loader with v0.2.0
data. The failure is silent by construction.

### 4. Inference was not reproducible (serious)

Found only because a re-run disagreed with itself.

The CCLE SMILES contain six tokens the curated vocabulary lacks (`c`, `n`, `o`,
`[nH]`, `[nH+]`, `[Cl-]`), so the language grows 83 → 89 when the dataset is
built. pytoda assigns those indices from a **set difference**:

```python
tokens_counter.keys() - self.token_to_index.keys()
```

Set iteration order is not stable across processes, so `[Cl-]` gets index 85 in
one run and 86 in the next — and those indices are looked up in a *trained*
embedding.

Three invocations of one unchanged inference command:

```
test PCC   0.787   0.491   0.787
```

Same checkpoint, same data, same command. **Any Paccmann number produced
without controlling for this is a coin flip**, including the ones in this
document before the fix. Sorting the unseen tokens makes it repeatable; three
runs now agree to every digit.

Repeatable is not correct: those six tokens were never trained whatever index
they land on. The real fix is extending the curated vocabulary or routing
unknowns to a dedicated `<UNK>` slot.

---

## Part 7 — Limits, and what comes next

### What this does not show

- **Attention is not causation.** High weight means consulted, not used
  affirmatively. Comparing attention attribution against CAFCW-style
  perturbation is the experiment that would establish how far it can be
  trusted.
- **The models are undertrained.** 2 and 3 epochs. The pipeline is proven; the
  numbers are not benchmarks.
- **The QC and panel distractors are synthetic.** Real chains need a real QC
  table, which means a data source that ships one.
- **One prediction, one distribution.** No temporal structure, so nothing here
  exercises the part of the vLLM design that tracks attention over time.
- **Not run at scale.** Everything is CPU, single node, hundreds of samples.

### Next

1. **`tool_ranges`** — needs the evaluation tokenizer from Bogdan, then
   `add_tool_ranges` completes the chains.
2. **Human verification** — Bogdan's request was explicitly for chains checked
   by a human. The ground truth is machine-verified against the tables; the
   *framing* has not been reviewed by an oncologist.
3. **More task types** — the plant benchmark has 8, this has 4. Missing
   analogues include modality coverage, variance regime, and literature audit.
4. **Poisoning** — `notes/injection-plan.md` in the Spotter-AI working
   directory covers the injection pipeline (M2.1). Attention capture gives it
   an attribution channel: if a poisoned gene is what the model attends to,
   provenance names it.
5. **A third model** — Duo already runs on the same `improvelib` interface.

---

## Reference

### Documents

| file | contents |
|---|---|
| `docs/model-setup.md` | training Paccmann and HiDRA from scratch, and the traps |
| `docs/cancer-study.md` | this document: design, results, findings |
| `docs/run-log.md` | verbatim transcript of one full pipeline run |

### Files

```
src/vllm_attn_connector/
  drp_emit.py       reduction + emit, no framework imports
  drp_capture.py    AttentionCapture base: shared emit/metadata/workflow
  drp_paccmann.py   torch forward hooks
  drp_hidra.py      Keras named-layer extraction
  drp_chains.py     provenance -> OPAL chains
  drp_cli.py        capture / inspect / chains subcommands
tests/
  test_drp_emit.py    reduction and record shape
  test_drp_chains.py  ground truth re-derived from prompts
  drp_smoke.py        real checkpoints, CPU
data/
  opal_chains.csv         Bogdan's plant benchmark (430 rows)
  cancer_opal_chains.csv  this work (96 rows)
```

### Repositories

| repo | branch | contents |
|---|---|---|
| `j-woz/vllm-attn-connector` | `main` | connector, capture, chains |
| `rajeeja/Paccmann_MCA` | `rajeeja/py313-fixes` | the four fixes |
| `spotter-ai-genesis/flowcept` | `main` | Flowcept **with** the vLLM adapter |

> `ORNL/flowcept` `main` has **no** vLLM adapter. Use the
> `spotter-ai-genesis` fork, or the connector cannot emit.

### Provenance

Justin Wozniak's `Transcript.adoc` in this repository scopes the Paccmann
analogue (Exchange 8) and is the direct antecedent of the capture work here.
The CAFCW/SC 2021 noise-injection paper is the scientific antecedent.
