# Run log — full pipeline, 2026-09-17

A complete end-to-end execution: lint, unit tests, both models' capture smoke
tests, capture into a real provenance store, chain generation read back out of
that store, independent validation of every generated answer, and a
reproducibility check.

Everything below is verbatim. Nothing is elided except progress bars and
TensorFlow's startup chatter.

The point of recording it this way is that several claims in
`docs/cancer-study.md` are only meaningful if they can be re-run — "the chains
come from the provenance store" is a design claim until a fresh workflow
produces a byte-identical file, which R9 is.

---

## R0 — Environment

```
2026-09-17T18:24:24Z
macOS 26.7 / arm64 / no GPU

python      3.13.15
torch       2.14.0
tensorflow  2.21.0
numpy       2.5.3    pandas 3.0.5
cuda        False    mps True

flowcept_mongo  mongo:7.0  Up
flowcept_redis  redis      Up
```

> `mongo:latest` will not start on this Docker kernel
> (`MongoDB cannot start: Linux kernel versions 6.19 and newer has a known
> incompatibility`, SERVER-121912). Pin `mongo:7.0`.

Flowcept settings: `mongodb`, `mq` and `kv_db` all enabled,
`db_flush_mode: online` — Flowcept's `full-online` profile. All three are
required together; with `mq` on and `kv_db` off, startup fails on
`check_safe_stops`.

---

## R1 — Lint

```bash
ruff check src/vllm_attn_connector/drp_*.py tests/test_drp*.py tests/drp_smoke.py
```
```
All checks passed!
```

Scoped to the files added here. The four pre-existing modules (`connector.py`,
`probe.py`, `kernels.py`, `layout.py`) carry 21 warnings and are deliberately
left untouched.

---

## R2 — Unit tests

No GPU, no vLLM, no torch, no TensorFlow, no model.

```bash
python -m pytest tests/ -q
```
```
...........................                     [100%]
27 passed in 2.11s
```

13 checks on the emitter and chain builder, 12 on chain ground truth and
`tool_ranges`, plus the 2 pre-existing aggregation suites.

---

## R3 — Paccmann capture, real checkpoint

```bash
python tests/drp_smoke.py --paccmann /tmp/pmca_out3 --data /tmp/pmca_ml2
```
```
Paccmann MCA
  ok  captured 16 SMILES heads, architecture implies 16
  ok  predictions identical with capture installed (observation, not perturbation)
  ok  one task per (sample, axis): 8 == 4x2
  ok  task ids carry the :g<n> group suffix
  ok  task ids are unique
  ok  series carries attn_sum / attn_peak / attn_argmax_head
  ok  every captured distribution sums to ~1 per head (real softmax output)
  ok  attn_peak >= mean at every position
  ok  emit returned 8 task ids
  ok  both the smiles and gene axes are emitted
  ok  smiles axis width == smiles_padding_length (560)
  ok  gene axis width == number_of_genes (2128)
  ok  heads disagree -> per-head structure survived (not pre-averaged)
  ok  different heads win at different positions
  ok  workflow records model identity

all checks passed
```

The two load-bearing lines are *predictions identical* — provenance must observe,
not perturb — and *heads disagree*, which proves the hook caught the per-head
vectors before `paccmann.py:273` averages them away.

---

## R4 — HiDRA capture, real checkpoint

```bash
python tests/drp_smoke.py --hidra /tmp/hidra_out --data /tmp/hidra_ml
```
```
HiDRA
  ok  predictions identical with capture installed
  ok  discovered 186 pathways, geneset has 186
  ok  one task per (sample, axis): 4 == 4x1
  ok  task ids carry the :g<n> group suffix
  ok  task ids are unique
  ok  series carries attn_sum / attn_peak / attn_argmax_head
  ok  every captured distribution sums to ~1 per head (real softmax output)
  ok  attn_peak >= mean at every position
  ok  emit returned 4 task ids
  ok  pathway axis width == pathway count
  ok  pathway attention is single-headed
  ok  workflow carries the pathway order (the decoder for the vectors)
  ok  gene level opt-in emits 748 == 4x(1+186)

all checks passed
```

The last line is why gene-level capture is opt-in: 4 samples become 748 records.

---

## R5 — Capture Paccmann into the store

```bash
FLOWCEPT_SETTINGS_PATH=.../settings.yaml \
python -m vllm_attn_connector.drp_cli capture-paccmann \
    --model-dir /tmp/pmca_out3 --data-dir /tmp/pmca_ml2 \
    --workflow-id run-pmca-final --write-features /tmp/run_genes.txt
```
```
capturing 371 samples, 2087 genes (+41 padded)
wrote feature labels -> /tmp/run_genes.txt
captured 371 samples to workflow 'run-pmca-final'
```

`+41 padded` matters. 41 of the 2,128 genes the model expects are absent from
the expression file, so the tensor is zero-padded. `--write-features` emits
labels describing the tensor *actually fed* — real genes followed by `__PAD_n__`
— because labelling those positions with real gene names would shift every gene
by 41 and mislabel the lot.

That was a real bug, caught by a length check in `load_attention_from_store`
rather than by inspection.

---

## R6 — Capture HiDRA into the same store

```bash
python -m vllm_attn_connector.drp_cli capture-hidra \
    --model-dir /tmp/hidra_out --data-dir /tmp/hidra_ml \
    --workflow-id run-hidra-final --limit 8
```
```
captured 8 samples over 186 pathways to workflow 'run-hidra-final'
```

---

## R7 — Inspect both workflows

```bash
python -m vllm_attn_connector.drp_cli inspect --workflow-id run-pmca-final
python -m vllm_attn_connector.drp_cli inspect --workflow-id run-hidra-final
```
```
workflow      : run-pmca-final
tasks         : 742
axes          : {'smiles': 371, 'gene': 371}
model         : Paccmann_MCA (torch)
metric_ref    : paccmann_mca_context_attention_additive
series        : ['attn_argmax_head', 'attn_peak', 'attn_sum']

workflow      : run-hidra-final
tasks         : 8
axes          : {'pathway': 8}
model         : HiDRA (tensorflow)
metric_ref    : hidra_hierarchical_attention_softmax_gate
series        : ['attn_argmax_head', 'attn_peak', 'attn_sum']
```

Two frameworks, two attention mechanisms, one store, one field vocabulary — and
`metric_reference` keeps them distinguishable. The numbers are *not* the same
measurement: Paccmann's attention is additive/Bahdanau, HiDRA's is a learned
softmax gate, and the vLLM connector's is `softmax(qKᵀ/√d)`. Pooling them would
be wrong, and the record says so.

Store totals: 756 tasks, 2 workflows.

---

## R8 — Build chains from the store

```bash
python -m vllm_attn_connector.drp_cli chains \
    --workflow-id run-pmca-final \
    --predictions /tmp/chain_predictions.csv \
    --features /tmp/run_genes.txt \
    --out /tmp/run_chains.csv --n-chains 24
```
```
read attention for 371 samples from the provenance store
wrote 24 chains / 96 rows -> /tmp/run_chains.csv
```

No model is loaded in this step. The attention comes out of MongoDB.

---

## R9 — Reproducibility

```bash
md5 -q /tmp/run_chains.csv data/cancer_opal_chains.csv
```
```
d79d6579720c2b537001477859fa45e6
d79d6579720c2b537001477859fa45e6
```

**Byte-identical.** A fresh capture into a fresh workflow, read back out and
regenerated, reproduces the committed file exactly. This is what makes "the
chains are built from recorded provenance" a checkable claim rather than an
architectural aspiration — and it also confirms the capture refactor
(`AttentionCapture` base, task registry) changed no behaviour.

---

## R10 — Independent ground-truth validation

Every answer re-derived by **parsing the rendered prompt**, never by consulting
the objects that produced it. If the generator and renderer ever disagree, this
catches it.

```
rows validated        : 96/96
answered the sink     : 0 (must be 0)
tool_ranges empty     : True
task types            : {'sensitivity_ranking': 24, 'panel_restricted_rank': 24,
                         'attention_attribution': 24, 'prediction_audit': 24}
```

`answered the sink: 0` is the interesting one. PARN absorbs 0.9988 of the gene
attention on all 371 cell lines — three orders of magnitude above second place
— so the naive answer to every attention round is PARN, and it is always wrong.
That distractor was discovered in real captured output, not designed in.

`tool_ranges empty: True` is intentional. The offsets are token offsets,
dependent on both the evaluation tokenizer and how much prior-round history is
prepended. Neither is known here, so the column stays empty rather than being
filled with plausible-looking wrong numbers.

---

## R11 — Determinism

```bash
for i in 1 2 3; do
  python Paccmann_MCA_infer_improve.py \
      --input_data_dir /tmp/pmca_ml2 --input_model_dir /tmp/pmca_out3 \
      --output_dir /tmp/det_$i | grep -oE "'pcc': [0-9.]+"
done
```
```
'pcc': 0.787166047199573
'pcc': 0.787166047199573
'pcc': 0.787166047199573
```

Before the fix in `rajeeja/Paccmann_MCA`, this same command produced
**0.787, 0.491, 0.787** — pytoda assigns indices to six unseen SMILES tokens
from a set difference, whose iteration order varies per process, and those
indices hit a trained embedding.

This check belongs in any run that produces a number worth quoting. Everything
above it — the baselines, the chain AUCs, the attention itself — is unreliable
until these three lines agree.

---

## Summary

| run | what | result |
|---|---|---|
| R1 | lint (new files) | clean |
| R2 | unit tests | 27 passed |
| R3 | Paccmann capture smoke | 15/15 |
| R4 | HiDRA capture smoke | 13/13 |
| R5 | capture 371 samples → store | 742 tasks |
| R6 | capture 8 samples → store | 8 tasks |
| R7 | inspect both workflows | both intact, distinguishable |
| R8 | chains from store | 24 chains / 96 rows |
| R9 | reproducibility | byte-identical |
| R10 | ground truth | 96/96, 0 sink answers |
| R11 | determinism | 3/3 identical |

### What this run does not establish

- **The models are undertrained** — 2 and 3 epochs. The pipeline is proven; the
  numbers are not benchmarks.
- **HiDRA capture used random inputs.** R6 fed `standard_normal` arrays, so the
  pathway attention it recorded is structurally valid but not biologically
  meaningful. Only the Paccmann capture (R5) used real expression data.
  *Fixed after this run — see the addendum below.*
- **Attention is not causation.** High weight means consulted, not used
  affirmatively.
- **Single node, CPU, hundreds of samples.** Nothing here is a scale test.


---

## Addendum, same day — HiDRA on real data

R6 above fed random arrays. `capture-hidra` now reads the preprocessed tables
and assembles inputs the way `MultiGenerator` does at train time.

```bash
python -m vllm_attn_connector.drp_cli capture-hidra \
    --model-dir /tmp/hidra_out --data-dir /tmp/hidra_ml \
    --workflow-id hidra-real-v2 --limit 64
```
```
captured 64 samples over 186 pathways to workflow 'hidra-real-v2'
```

Two bugs surfaced immediately, both silent, both of the kind this project keeps
producing — output that passes every structural check and is wrong.

**`send_workflow` was being dropped.** It re-registered the caller's *own*
workflow id, and Flowcept records a given workflow once, so the second
registration vanished — taking `pathway_order` with it. The 186-wide vectors
were therefore undecodable: position 41 meant nothing. It now files under
`<workflow_id>:conf` with the caller's id as parent.

**`load_attention_from_store` collapsed samples.** It split task ids on the
first `:`, but HiDRA ids a prediction `<cell_line>::<compound>` — the same line
attends differently to different drugs. 64 predictions became 29, silently
keeping whichever arrived last. Now strips only a trailing `:g<n>`.

With both fixed, the attention decodes to named pathways:

```
ACH-000046::Drug_1005   KEGG_FC_GAMMA_R_MEDIATED_PHAGOCYTOSIS   0.0104
ACH-000046::Drug_1507   KEGG_AXON_GUIDANCE                      0.0111
ACH-000046::Drug_490    KEGG_MELANOGENESIS                      0.0103
ACH-000052::Drug_1040   KEGG_HYPERTROPHIC_CARDIOMYOPATHY_HCM    0.0096
```

28 distinct pathways win across 64 predictions. Note the first three rows: one
cell line, three compounds, three different winning pathways -- which is why
the id has to name the pair.

**But read those with care.** Uniform over 186 pathways is 0.0054, and the
observed maximum is 0.0113 — **2.1x uniform**. Compare Paccmann's gene axis,
where PARN takes 0.9988, three orders of magnitude above second place. After 2
epochs HiDRA has barely learned to discriminate pathways, so the ranking above
is weak evidence at best. Worth re-checking at 20 epochs.

## Addendum — tool_ranges

R10 recorded `tool_ranges empty: True` as intentional, pending a tokenizer.
`add_tool_ranges` now computes them, using `return_offsets_mapping` over the
whole text rather than summing token counts of substrings — which was the
previous implementation and was wrong, because a BPE merge can span a join.

Verified against gpt2 on all 24 chains:

```
rows with ranges : 96 / 96
total spans      : 192
spans decoded    : 192/192 start at their own >>> block
round 1          : (db_lookup[623:795],qc_lookup[796:895])
round 2          : (db_lookup[1025:1233],design_lookup[1234:1384])
```

Round 2's offsets sit past round 1's, because history mode indexes the
accumulated conversation rather than the round alone.

The shipped `data/cancer_opal_chains.csv` still has the column empty, which is
correct: the spans are only meaningful against the tokenizer of whichever model
is being evaluated.
