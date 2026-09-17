# Setting up Paccmann MCA and HiDRA

Both models must be trained before attention can be captured from them. This is
the path that was actually walked, on macOS 26.7 / arm64 / Python 3.13, with no
GPU. Every command below was run; the traps called out are ones that were hit,
not ones imagined.

Expect roughly an hour end to end, most of it downloading 1.4 GB of benchmark
data.

---

## 0. Why this is non-trivial

Both models are curated by [JDACS4C-IMPROVE](https://github.com/JDACS4C-IMPROVE)
and were last touched in 2024. Since then numpy removed `np.float`, pandas 3.0
changed `apply(axis=1)`, Keras 3 stopped accepting DataFrames, and pytoda 1.1
moved half its API. Neither model runs on a current environment unpatched.

Worse, two of the failures are **silent** — the model trains, infers, and emits
plausible scores while being fed zeros, or scores differently on every run. Both
are covered below.

---

## 1. Environment

```bash
conda create -n spotter python=3.13
conda activate spotter

pip install torch numpy pandas scikit-learn scipy pyarrow pyyaml   # shared
pip install tensorflow==2.21.0                                     # HiDRA
pip install pytoda                                                 # Paccmann (pulls rdkit)
pip install pymongo redis                                          # provenance store
```

TensorFlow and torch coexist here without a numpy downgrade — worth checking
after install, since pip will happily resolve them apart:

```bash
python -c "import torch, tensorflow, numpy; print(torch.__version__, tensorflow.__version__, numpy.__version__)"
# 2.14.0 2.21.0 2.5.3
```

### improvelib — do not install from PyPI

The PyPI release (0.1.0) is missing five functions the models need:
`get_x_data`, `get_all_response_data`, `get_features_in_y_data`,
`determine_transform`, `transform_data`. Install the develop checkout editable:

```bash
git clone -b develop https://github.com/JDACS4C-IMPROVE/IMPROVE.git repos/IMPROVE
cd repos/IMPROVE
git apply ../../patches/improve-pyproject-toml-fix.patch
pip install -e .
```

> **Trap.** Upstream `pyproject.toml` has malformed TOML in the `authors` table
> — entries like `{ "Name <email>" }` with no `name =` key — so `pip install -e .`
> dies with `TOMLDecodeError`. The patch rewrites them as proper tables.

---

## 2. Benchmark data

Must be `drp_data_v0.2.0`. Older copies lack the `improve_sample_id` column and
fail preprocessing.

```bash
mkdir -p data && cd data
wget --cut-dirs=8 -P ./ -nH -np -m --reject "index.html*,index.*" \
  https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
```

Expected layout (1.4 GB):

```
data/csa_data/raw_data/
  x_data/   cancer_gene_expression.tsv   drug_mordred.tsv
            drug_ecfp4_nbits512.tsv      drug_SMILES.tsv   drug_info.tsv
  y_data/   response.tsv
  splits/   CCLE_split_0_{train,val,test}.txt
```

The two models use **different** drug featurisations — HiDRA wants
`drug_ecfp4_nbits512.tsv`, Paccmann wants `drug_SMILES.tsv`. Both are in the
mirror above but are easy to miss if you copy an older local dataset.

---

## 3. HiDRA

```bash
git clone -b develop https://github.com/JDACS4C-IMPROVE/HiDRA.git repos/HiDRA
cd repos/HiDRA
git apply ../../patches/hidra-py313-pandas3-keras3.patch
```

The patch fixes three independent breaks:

| break | symptom |
|---|---|
| pandas 3.0 | `df.apply(zscore, axis=1)` returns a Series of arrays, not a DataFrame → `ValueError: Columns must be same length as key` |
| Keras 3 | generator yields DataFrames; optree calls `.dtype` on each leaf → `'DataFrame' object has no attribute 'dtype'`. Also rejects a bare `list` of inputs |
| Keras 3 loading | `load_model` on the legacy `.h5` → `Could not deserialize 'keras.metrics.mse'` |

Then run the three stages:

```bash
python HiDRA_preprocess_improve.py \
    --input_dir  ../../data/csa_data/raw_data \
    --output_dir /tmp/hidra_ml

python HiDRA_train_improve.py \
    --input_dir /tmp/hidra_ml --output_dir /tmp/hidra_out --epochs 2

python HiDRA_infer_improve.py \
    --input_data_dir /tmp/hidra_ml \
    --input_model_dir /tmp/hidra_out --output_dir /tmp/hidra_infer
```

`geneset.gmt` ships in the repo — 186 KEGG pathways, of which 5,003 genes are
present in the expression file. Preprocess writes `geneset.json`, which the
capture code needs.

**Expected (2 epochs, CCLE split 0):**

| split | MSE | RMSE | PCC | SCC | R² |
|---|---|---|---|---|---|
| val | 0.0122 | 0.1105 | 0.7516 | 0.5370 | 0.4875 |
| test | 0.0127 | 0.1129 | 0.7667 | 0.5604 | 0.5320 |

Train loss falls 0.1135 → 0.0215, so the model is learning and simply
undertrained. Default is 20 epochs; use that before comparing against anything.

---

## 4. Paccmann MCA

### Use the fork, and the right branch

```bash
git clone -b rajeeja/py313-fixes \
    git@github.com:rajeeja/Paccmann_MCA.git repos/Paccmann_MCA
```

Two things to know:

**Branch.** Upstream `develop` still imports the retired `improve` + `candle`
API. `framework-api` is the branch that uses `improvelib`, matching HiDRA and
Duo. The fork branches from `framework-api`.

**Fork.** Upstream `JDACS4C-IMPROVE/Paccmann_MCA` has been inactive since
2024-09 and does not carry the four fixes below. The fork does, including the
vendored `_vendor_improvelib/` the model cannot run without.

### Run it

```bash
cd repos/Paccmann_MCA

python Paccmann_MCA_preprocess_improve.py \
    --input_dir  ../../data/csa_data/raw_data \
    --output_dir /tmp/pmca_ml

python Paccmann_MCA_train_improve.py \
    --input_dir /tmp/pmca_ml --output_dir /tmp/pmca_out --epochs 3

python Paccmann_MCA_infer_improve.py \
    --input_data_dir /tmp/pmca_ml \
    --input_model_dir /tmp/pmca_out --output_dir /tmp/pmca_infer
```

Preprocess downloads `Data_MCA.zip` (20 MB: `2128_genes.pkl`,
`smiles_language_chembl_gdsc_ccle.pkl`) into
`improve_output/supplemental_data/`. That directory is gitignored — it is
downloaded, not source — and the capture CLI needs the path to
`2128_genes.pkl` from it.

**Expected (3 epochs, CCLE split 0):**

| split | MSE | RMSE | PCC | SCC | R² |
|---|---|---|---|---|---|
| val | 0.0173 | 0.1316 | 0.7702 | 0.5130 | 0.2439 |
| test | 0.0187 | 0.1367 | 0.7865 | 0.5619 | 0.3133 |

8,170,017 parameters. R² is low because the run is short; PCC is already 0.79,
so the ranking is largely right and the scale is not yet calibrated.

### The four fixes, and why two of them matter

Routine:

1. **Library API drift** — `np.float` removed in numpy 1.24; pytoda 1.1 *raises*
   on `device=`; predictions stack to `(N,1)` but `pearsonr` needs 1-D.
2. **SMILES language** — the curated pickle is a bare `SMILESLanguage`; pytoda
   1.1 wants a `SMILESTokenizer`. Fixing it exposed that pytoda takes the
   augment flag from the *language object*, overriding the dataset argument, so
   validation was being silently augmented. Val MSE 0.0188 → 0.0143.

Serious — both silent:

3. **Gene expression was all zeros.** The v0.1.0 omics loader expects a
   three-row header (Ensembl / Entrez / Gene_Symbol); `drp_data_v0.2.0` has a
   single row of gene symbols. The loader ate two cell lines as header, every
   gene name parsed as a float, the 2,128-gene lookup matched nothing, and the
   model trained on an all-zero matrix. Nothing raised. The only tell was val R²
   0.37 against test R² −3.19. After the fix, test PCC 0.687 → **0.787**.

4. **Inference was not reproducible.** The CCLE SMILES contain six tokens the
   curated vocabulary lacks, so the language grows 83 → 89. pytoda assigns those
   indices from a *set difference*, whose iteration order is not stable across
   processes, so `[Cl-]` gets 85 one run and 86 the next — into a trained
   embedding. Three runs of one unchanged command:

   ```
   test PCC   0.787   0.491   0.787
   ```

   Sorting the unseen tokens makes it repeatable. **Any Paccmann benchmark
   produced without this fix is a coin flip.**

   Repeatable is not correct: those tokens were never trained whatever index
   they land on. The real fix is extending the curated vocabulary or routing
   unknowns to `<UNK>`.

> If you are running any other IMPROVE model that pairs a v0.1.0-era loader with
> v0.2.0 data, check for bug 3. It is silent by construction.

---

## 5. Verify before capturing

Paccmann, three times — the numbers must be identical:

```bash
for i in 1 2 3; do
  python Paccmann_MCA_infer_improve.py \
      --input_data_dir /tmp/pmca_ml --input_model_dir /tmp/pmca_out \
      --output_dir /tmp/pmca_check_$i 2>&1 | grep -oE "'pcc': [0-9.]+"
done
```

If they differ, fix 4 is missing. Everything downstream — baselines, chains,
attention attribution — is unreliable until they agree.

Then the capture smoke tests, which need neither GPU nor vLLM:

```bash
cd repos/vllm-attn-connector
python tests/drp_smoke.py --paccmann /tmp/pmca_out --data /tmp/pmca_ml
python tests/drp_smoke.py --hidra    /tmp/hidra_out --data /tmp/hidra_ml
```

Both should print `all checks passed`. Those assert the distributions really are
softmax outputs, that heads disagree, and that predictions are bit-identical
with capture installed.

---

## 6. Next

With both models trained, `docs/cancer-study.md` covers capturing their
attention into Flowcept and turning it into OPAL-format evaluation chains.
`docs/run-log.md` is a full transcript of one such run.
