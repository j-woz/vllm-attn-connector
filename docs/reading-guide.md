# Reading guide

What to read to understand the models, the framework they sit in, and what this
repository does with them. Ordered so each item makes sense given the ones
before it.

Every DOI below was resolved against Crossref; every repository link was
fetched. Where something does not exist — a Flowcept paper, a HiDRA citation in
its own README — that is stated rather than papered over.

---

## 1. Start here — one hour

If you read three things, read these.

**PaccMann** — [Toward Explainable Anticancer Compound Sensitivity Prediction
via Multimodal Attention-Based Convolutional
Encoders](https://doi.org/10.1021/acs.molpharmaceut.9b00520)
Manica, Oskooei, Born, Subramanian, Sáez-Rodríguez, Rodríguez Martínez.
*Molecular Pharmaceutics* 16(12), 2019.

Note the first word of the title. Explainability is the *stated goal*, and the
mechanism is attention — which is exactly what this repository captures. Read
the multimodal contextual attention section and you will recognise
`ContextAttentionLayer` in the code.

**HiDRA** — [HiDRA: Hierarchical Network for Drug Response Prediction with
Attention](https://doi.org/10.1021/acs.jcim.1c00706)
Jin & Nam. *Journal of Chemical Information and Modeling* 61(8), 2021.

Same idea, different shape: attention over KEGG pathways rather than molecule
tokens, and hierarchical — genes within a pathway, then pathways against each
other. The pathway level is what makes it interesting for attribution, because
"which biology mattered" is directly readable.

> The IMPROVE HiDRA README does **not** cite this paper. The link above was
> resolved from Crossref; the original authors' code is
> [GIST-CSBL/HiDRA](https://github.com/GIST-CSBL/HiDRA).

**Attention sinks** — [Efficient Streaming Language Models with Attention
Sinks](https://arxiv.org/abs/2309.17453), Xiao et al., 2023.

Read section 3 only. It establishes that some positions absorb attention
structurally, independent of content. We found the exact analogue in Paccmann:
PARN takes 0.9988 of the gene-axis weight on all 371 cell lines. That single
observation is why the attention task in our benchmark has a distractor at all.

---

## 2. The two models, in code

Reading order that matches how the capture works.

| | Paccmann MCA | HiDRA |
|---|---|---|
| original | [PaccMann/paccmann_predictor](https://github.com/PaccMann/paccmann_predictor) | [GIST-CSBL/HiDRA](https://github.com/GIST-CSBL/HiDRA) |
| IMPROVE curation | [JDACS4C-IMPROVE/Paccmann_MCA](https://github.com/JDACS4C-IMPROVE/Paccmann_MCA) | [JDACS4C-IMPROVE/HiDRA](https://github.com/JDACS4C-IMPROVE/HiDRA) |
| **runnable fork** | [rajeeja/Paccmann_MCA](https://github.com/rajeeja/Paccmann_MCA) `rajeeja/py313-fixes` | upstream + our patch |

The lines that matter, if you want to see where attention lives:

```
paccmann_predictor/models/paccmann.py:244-253   16 per-head alphas built
paccmann_predictor/models/paccmann.py:273-279   ...then averaged away
paccmann_predictor/utils/layers.py:210-236      ContextAttentionLayer.forward
test_paccmann.py:161                            pred_dict bound, never read

hidra_utils.py:91-92     per-pathway gene attention (186 softmax layers)
hidra_utils.py:111-112   Sample_Attention_Softmax, the pathway level
```

That Paccmann pair is the whole argument for hooking the layer rather than
reading the output dict: the per-head structure exists at line 253 and is
destroyed at 273.

`pytoda` is Paccmann's data layer and the source of two of the four bugs we
hit: [docs](https://paccmann.github.io/paccmann_datasets/api/pytoda.html),
[repo](https://github.com/PaccMann/paccmann_datasets).

---

## 3. The framework the models live in

**IMPROVE** — [JDACS4C-IMPROVE](https://github.com/JDACS4C-IMPROVE),
[docs v0.1.0](https://jdacs4c-improve.github.io/docs/v0.1.0)

A DOE/NCI effort that wrapped many drug-response models behind one interface
(`improvelib`) with one data layout, so they can be compared. That shared
interface is why one capture design serves both models here, and why adding Duo
would be cheap.

Two practical cautions, both learned the hard way:

- The API shifted between v0.1.0 and current `develop`. Loader classes the
  models still import were deleted; config keys were renamed. Our fork vendors
  the three v0.1.0 modules rather than pinning the whole library back.
- The docs describe v0.1.0. `develop` differs.

**Benchmark data** —
[CSA `drp_data_v0.2.0`](https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/)
(1.4 GB). Cell lines, compounds, response, and the splits. Must be v0.2.0;
older copies lack `improve_sample_id`.

**CCLE**, the cell lines themselves —
[Next-generation characterization of the Cancer Cell Line
Encyclopedia](https://doi.org/10.1038/s41586-019-1186-3), *Nature* 569, 2019.
Worth skimming for what `ACH-000956` actually denotes.

**KEGG pathways**, which HiDRA's hierarchy is built on —
[genome.jp/kegg/pathway.html](https://www.genome.jp/kegg/pathway.html). The
186 pathways in `geneset.gmt` come from here.

---

## 4. Provenance

**Flowcept** — [ORNL/flowcept](https://github.com/ORNL/flowcept),
[readthedocs](https://flowcept.readthedocs.io/),
[flowcept.org](https://flowcept.org)

Captures runtime lineage from ML and scientific workflows and stores it
queryably. Start with `docs/architecture.rst` and `docs/prov_capture.rst` in
the repo.

> I could not find a peer-reviewed Flowcept paper via Crossref. The
> documentation is the primary source. If one exists, it should replace this
> line.

> **Use the fork, not ORNL.** `ORNL/flowcept` `main` has **no vLLM adapter**.
> [spotter-ai-genesis/flowcept](https://github.com/spotter-ai-genesis/flowcept)
> has `flowceptor/adapters/vllm/`, without which the connector cannot emit.

**PROV**, the W3C model provenance systems descend from —
[PROV Primer](https://www.w3.org/TR/prov-primer/). Half an hour, and it makes
"entity / activity / agent" vocabulary legible.

---

## 5. This work

Read in this order:

1. **`docs/model-setup.md`** — nothing to two trained models. Environment, the
   1.4 GB download, the patches, and the two silent failure modes to check
   before trusting a number.
2. **`docs/cancer-study.md`** — the design and the findings. Part 1 covers the
   basics if attention and provenance are new.
3. **`docs/run-log.md`** — one full run, verbatim, so the claims are checkable.
4. **`Transcript.adoc`** (repo root) — Justin Wozniak's session tracing the vLLM
   data path from attention kernel to MongoDB document. **Exchange 8** scopes
   the Paccmann analogue and is the direct antecedent of this work.

Then the code, in dependency order:

```
drp_emit.py      reduction and emit; imports no framework
drp_capture.py   AttentionCapture base; shared metadata and workflow handling
drp_paccmann.py  torch forward hooks
drp_hidra.py     Keras named-layer extraction
drp_chains.py    provenance -> OPAL chains
drp_cli.py       capture / inspect / chains
```

`tests/test_drp_chains.py` is worth reading as documentation: it re-derives
every benchmark answer by parsing the rendered prompt, so it states the ground
truth rules more precisely than prose does.

---

## 6. Why this project exists

**The SPOTTER-AI proposal** (`proposal/` in the Spotter-AI working directory) —
the framing. Integrity of scientific AI workflows: not "who read my data" but
"has this been corrupted, and which input caused the wrong answer".

**Direct precedent** — *Probing Decision Boundaries in Cancer Data Using Noise
Injection and Counterfactual Analysis*, Jain, Shah, Mohd-Yusof, **Wozniak**,
Brettin, Xia, Stevens — CAFCW/SC 2021.

Label flipping, correlated and uncorrelated label noise, Gaussian feature
noise, an abstaining classifier, and counterfactual attribution over 60,483
features on CANDLE NT3. Same question as this work — *when cancer data is
corrupted, can you attribute the damage?* — answered by perturbing inputs and
re-running.

This work asks it with a cheaper instrument: the model already computes an
attribution signal on every forward pass and throws it away. Neither replaces
the other. Perturbation measures causal effect; attention measures
consultation. **Whether they agree is the experiment neither has run yet**, and
it is the most interesting thing available from here.

**OPAL** — Bogdan Nicolae's `data/opal_chains.csv` in this repository. 100
multi-round investigations over plant phenotyping, 430 rows, 8 task types.
There is no paper; the CSV is the specification. Read three or four rows'
`explanation` fields and the design becomes obvious: the naive answer is always
wrong, and the explanation records each wrong path.

---

## 7. Background, if attention is unfamiliar

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et
  al., 2017. Section 3.2 only, for scaled dot-product attention.
- [Neural Machine Translation by Jointly Learning to Align and
  Translate](https://arxiv.org/abs/1409.0473) — Bahdanau et al., 2014. This is
  the *additive* attention Paccmann actually uses, which is why our records
  carry a `metric_reference` rather than assuming one formula.
- [Attention is not Explanation](https://arxiv.org/abs/1902.10186) — Jain &
  Wallace, 2019, and the reply, [Attention is not not
  Explanation](https://arxiv.org/abs/1908.04626), Wiegreffe & Pinter, 2019.

Read both sides of that exchange. It is the standing argument about exactly
what this repository records, and it is the reason the docs say attention means
"this input was consulted" rather than "this input caused the answer".
