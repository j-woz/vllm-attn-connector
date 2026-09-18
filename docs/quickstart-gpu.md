# Quickstart — cancer OPAL chains on a GPU box

Copy-pasteable, start to finish: two trained drug-response models, their
attention captured into a provenance store, and an OPAL-format benchmark built
out of it.

Everything here was executed on CPU (macOS/arm64, no GPU). The commands are
unchanged on a GPU box — both frameworks auto-detect the device — but the
**timings below are CPU timings** and the GPU notes are marked as untested
where they are.

Budget roughly 2 hours, most of it a 1.4 GB download and two training runs.

---

## What actually uses the GPU

Worth knowing before you allocate a node:

| stage | device | note |
|---|---|---|
| preprocess | CPU | pandas and file I/O; a GPU does nothing here |
| **train** | **GPU** | `get_device()` returns cuda if available (`paccmann_predictor/utils/utils.py:5`); Keras likewise |
| infer | GPU | same auto-detection |
| **capture** | either | our code is device-agnostic; it reads tensors the model already produced |
| chains | CPU | reads MongoDB, loads no model |

So the GPU matters for training at realistic epoch counts. The capture and
chain work does not need one — which is the point, and why all of this was
developed on a laptop.

---

## 0. Prerequisites

```bash
# Docker, for the provenance store
docker --version

# CUDA visible to torch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

---

## 1. Environment

```bash
conda create -n spotter python=3.13 -y
conda activate spotter

pip install torch numpy pandas scikit-learn scipy pyarrow pyyaml
pip install tensorflow==2.21.0      # HiDRA
pip install pytoda                  # Paccmann; pulls rdkit
pip install pymongo redis           # provenance store
pip install pytest ruff             # tests and lint
```

Check both frameworks coexist — pip will happily resolve them into a numpy
conflict:

```bash
python -c "import torch, tensorflow, numpy; print(torch.__version__, tensorflow.__version__, numpy.__version__)"
# 2.14.0 2.21.0 2.5.3
```

> **GPU note, untested here.** `pip install tensorflow==2.21.0` gives the CPU
> build on some platforms. If `tf.config.list_physical_devices('GPU')` is empty,
> install `tensorflow[and-cuda]`. Torch wheels from the default index are CUDA
> builds already.

---

## 2. Workspace and IMPROVE

```bash
mkdir -p ~/spotter && cd ~/spotter
mkdir -p repos data

git clone -b develop https://github.com/JDACS4C-IMPROVE/IMPROVE.git repos/IMPROVE
```

IMPROVE's `pyproject.toml` has malformed TOML in its `authors` table, so an
editable install fails with `TOMLDecodeError`. Fix it in place:

```bash
cd repos/IMPROVE
python - <<'EOF'
import re
p = "pyproject.toml"
s = open(p).read()
s = re.sub(r'\{\s*"([^"<]+?)\s*<([^>]+)>"\s*\}', r'{ name = "\1", email = "\2" }', s)
s = re.sub(r'\{\s*"([^"<]+?)"\s*\}', r'{ name = "\1" }', s)
open(p, "w").write(s)
EOF
pip install -e .
cd ~/spotter
```

Do **not** `pip install improvelib` — the PyPI 0.1.0 release is missing five
functions the models call.

---

## 3. Benchmark data (1.4 GB)

```bash
cd ~/spotter/data
wget --cut-dirs=8 -P ./ -nH -np -m --reject "index.html*,index.*" \
  https://web.cels.anl.gov/projects/IMPROVE_FTP/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data/
cd ~/spotter

# must list drug_SMILES.tsv (Paccmann) and drug_ecfp4_nbits512.tsv (HiDRA)
ls data/csa_data/raw_data/x_data/
```

Must be `drp_data_v0.2.0`. Older copies lack `improve_sample_id` and fail
preprocessing.

---

## 4. Provenance store

```bash
docker run -d --name flowcept_mongo -p 27017:27017 mongo:7.0
docker run -d --name flowcept_redis -p 6379:6379 redis
docker ps    # both Up
```

> `mongo:latest` will not start on kernels ≥ 6.19 (SERVER-121912). Pin 7.0.

Flowcept, and the fork that has the vLLM adapter:

```bash
git clone https://github.com/spotter-ai-genesis/flowcept.git repos/flowcept
pip install omegaconf msgpack orjson rich
```

> **Use this fork.** `ORNL/flowcept` `main` has no `flowceptor/adapters/vllm/`,
> without which nothing can be emitted.

Settings — all three of `mongodb`, `mq`, `kv_db` must be on together, or
startup fails on `check_safe_stops`:

```bash
mkdir -p ~/spotter/flowcept-local
python - <<'EOF'
import os, re
src = os.path.expanduser("~/spotter/repos/flowcept/resources/sample_settings.yaml")
s = open(src).read()
s = s.replace("  mongodb:\n    enabled: false", "  mongodb:\n    enabled: true")
s = re.sub(r"(mq:\n  enabled: )false", r"\1true", s, count=1)
s = re.sub(r"(kv_db:\n  enabled: )false", r"\1true", s, count=1)
s = s.replace("db_flush_mode: offline", "db_flush_mode: online")
out = os.path.expanduser("~/spotter/flowcept-local/settings.yaml")
open(out, "w").write(s)
print("wrote", out)
EOF

export FLOWCEPT_SETTINGS_PATH=~/spotter/flowcept-local/settings.yaml
export PYTHONPATH=~/spotter/repos/flowcept/src:~/spotter/repos/vllm-attn-connector/src
```

---

## 5. The models

```bash
cd ~/spotter

git clone -b rajeeja/py313-fixes \
    git@github.com:rajeeja/Paccmann_MCA.git repos/Paccmann_MCA

git clone -b develop https://github.com/JDACS4C-IMPROVE/HiDRA.git repos/HiDRA
git clone git@github.com:j-woz/vllm-attn-connector.git repos/vllm-attn-connector

cd repos/HiDRA
git apply ../vllm-attn-connector/patches/hidra-py313-pandas3-keras3.patch 2>/dev/null \
  || echo "patch lives in the Spotter-AI working dir; see docs/model-setup.md"
cd ~/spotter
```

Two things that are easy to get wrong:

- **Paccmann: use the fork and that branch.** Upstream `develop` imports the
  retired `improve` + `candle` API, and upstream carries none of the four fixes
  — including the one where gene expression silently arrives as all zeros.
- **HiDRA needs the patch.** Three independent breaks under pandas 3 / Keras 3.

---

## 6. Train

`--epochs` below are the values used for the recorded baselines. On a GPU,
raise them: HiDRA's default is 20, Paccmann's config says 200.

### HiDRA

```bash
cd ~/spotter/repos/HiDRA

python HiDRA_preprocess_improve.py \
    --input_dir ~/spotter/data/csa_data/raw_data \
    --output_dir /tmp/hidra_ml

python HiDRA_train_improve.py \
    --input_dir /tmp/hidra_ml --output_dir /tmp/hidra_out --epochs 20

python HiDRA_infer_improve.py \
    --input_data_dir /tmp/hidra_ml \
    --input_model_dir /tmp/hidra_out --output_dir /tmp/hidra_infer
```

### Paccmann

```bash
cd ~/spotter/repos/Paccmann_MCA

python Paccmann_MCA_preprocess_improve.py \
    --input_dir ~/spotter/data/csa_data/raw_data \
    --output_dir /tmp/pmca_ml

python Paccmann_MCA_train_improve.py \
    --input_dir /tmp/pmca_ml --output_dir /tmp/pmca_out --epochs 200

python Paccmann_MCA_infer_improve.py \
    --input_data_dir /tmp/pmca_ml \
    --input_model_dir /tmp/pmca_out --output_dir /tmp/pmca_infer
```

Preprocess pulls `Data_MCA.zip` (20 MB) into
`improve_output/supplemental_data/`. Note that path — capture needs
`2128_genes.pkl` from it.

> **Checkpointing caveat.** `train_paccmann.py` only best-val checkpoints inside
> `if epoch >= 50`. Below 51 epochs it saves the *last* epoch, not the best. At
> 200 this is fine; at 3 it is not.

### Verify before going further

```bash
cd ~/spotter/repos/Paccmann_MCA
for i in 1 2 3; do
  python Paccmann_MCA_infer_improve.py \
      --input_data_dir /tmp/pmca_ml --input_model_dir /tmp/pmca_out \
      --output_dir /tmp/det_$i 2>&1 | grep -oE "'pcc': [0-9.]+"
done
```

All three must be identical. If they differ, you are on upstream rather than
the fork: pytoda assigns indices to six unseen SMILES tokens from a set
difference whose order varies per process, and those indices hit a trained
embedding. Unfixed, the same command gave **0.787, 0.491, 0.787**. Every number
downstream is a coin flip until these agree.

---

## 7. Capture attention

```bash
cd ~/spotter/repos/vllm-attn-connector
export FLOWCEPT_SETTINGS_PATH=~/spotter/flowcept-local/settings.yaml
export PYTHONPATH=src:~/spotter/repos/flowcept/src

python -m vllm_attn_connector.drp_cli capture-paccmann \
    --model-dir /tmp/pmca_out \
    --data-dir  /tmp/pmca_ml \
    --paccmann-repo ~/spotter/repos/Paccmann_MCA \
    --genes ~/spotter/repos/Paccmann_MCA/improve_output/supplemental_data/Data/2128_genes.pkl \
    --workflow-id pmca-gpu-v1 \
    --write-features /tmp/pmca_genes.txt

python -m vllm_attn_connector.drp_cli capture-hidra \
    --model-dir /tmp/hidra_out \
    --data-dir  /tmp/hidra_ml \
    --workflow-id hidra-gpu-v1 \
    --limit 8
```

Expect:

```
capturing 371 samples, 2087 genes (+41 padded)
captured 371 samples to workflow 'pmca-gpu-v1'
captured 8 samples over 186 pathways to workflow 'hidra-gpu-v1'
```

`+41 padded` is expected: 41 of the 2,128 genes the model wants are absent from
the expression file, so the tensor is zero-padded. `--write-features` emits
labels describing the tensor *actually fed*, which is what stops every gene
name shifting by 41.

> **`capture-hidra` currently feeds random inputs**, so its attention is
> structurally valid but not biologically meaningful. Fine for checking the
> plumbing; not a result. Only the Paccmann path uses real expression data.

Check what landed:

```bash
python -m vllm_attn_connector.drp_cli inspect --workflow-id pmca-gpu-v1
python -m vllm_attn_connector.drp_cli inspect --workflow-id hidra-gpu-v1
```

```
workflow      : pmca-gpu-v1
tasks         : 742
axes          : {'smiles': 371, 'gene': 371}
model         : Paccmann_MCA (torch)
metric_ref    : paccmann_mca_context_attention_additive
```

---

## 8. Build the chains

```bash
python -m vllm_attn_connector.drp_cli predictions \
    --ml-dir    /tmp/pmca_ml \
    --infer-dir /tmp/pmca_infer \
    --out       /tmp/pmca_preds.csv

python -m vllm_attn_connector.drp_cli chains \
    --workflow-id pmca-gpu-v1 \
    --predictions /tmp/pmca_preds.csv \
    --features    /tmp/pmca_genes.txt \
    --out         /tmp/cancer_opal_chains.csv \
    --n-chains 24
```

```
wrote 951 predictions -> /tmp/pmca_preds.csv
read attention for 371 samples from the provenance store
wrote 24 chains / 96 rows -> /tmp/cancer_opal_chains.csv
```

The `predictions` step exists because `*_infer_improve.py` writes only
`auc_true,auc_pred` and drops the identifying columns; the join key is row
order against the preprocessed y-data.

No model is loaded to build chains. The attention comes out of MongoDB.

---

## 9. Verify

```bash
pytest tests/ -q                                   # 27 passed
python tests/drp_smoke.py --paccmann /tmp/pmca_out --data /tmp/pmca_ml
python tests/drp_smoke.py --hidra    /tmp/hidra_out --data /tmp/hidra_ml
```

Both smoke tests print `all checks passed`, asserting the distributions really
are softmax outputs, heads disagree (so per-head structure survived the hook),
and predictions are bit-identical with capture installed.

Then check the generated chains:

```bash
python - <<'EOF'
import csv, collections
rows = list(csv.DictReader(open("/tmp/cancer_opal_chains.csv")))
print("rows       :", len(rows))
print("chains     :", len({r["chain_id"] for r in rows}))
print("tasks      :", dict(collections.Counter(r["task_type"] for r in rows)))
print("ranges empty:", all(not (r["tool_ranges"] or "").strip() for r in rows))
EOF
```

```
rows       : 96
chains     : 24
tasks      : {'sensitivity_ranking': 24, 'panel_restricted_rank': 24,
              'attention_attribution': 24, 'prediction_audit': 24}
ranges empty: True
```

With longer training your AUC values will differ from the committed
`data/cancer_opal_chains.csv` — the chains are built from *your* model's
predictions. The structure and the row count should match.

---

## 10. What is still missing

**`tool_ranges` is empty and must stay that way until a tokenizer is chosen.**
The offsets are token offsets and depend on both the evaluation tokenizer and
how much prior-round history is prepended:

```python
from vllm_attn_connector.drp_chains import add_tool_ranges
rows = add_tool_ranges(rows, tokenize=enc.encode, history=True)
```

`history=True` accumulates prior rounds the way an evaluation harness would;
`False` scopes each round to itself. Bogdan's two modes — ground-truth history
for per-prompt accuracy, the model's own answers for per-chain accuracy —
correspond to what you feed that accumulation.

**Human verification.** The ground truth is machine-verified against the inlined
tables (96/96 re-derived by parsing the rendered prompts), but the clinical
framing has had no oncologist review, which is what Bogdan actually asked for.

**QC flags and panel roles are synthetic.** CCLE ships no per-prediction QC
table and the benchmark needs distractors. They are seeded and recorded in each
`explanation`. The measurements they gate are real.

---

## One-shot script

Assumes sections 1–5 are done.

```bash
#!/usr/bin/env bash
set -euo pipefail

export FLOWCEPT_SETTINGS_PATH=~/spotter/flowcept-local/settings.yaml
export PYTHONPATH=~/spotter/repos/vllm-attn-connector/src:~/spotter/repos/flowcept/src
PACC=~/spotter/repos/Paccmann_MCA
HIDRA=~/spotter/repos/HiDRA
CONN=~/spotter/repos/vllm-attn-connector
DATA=~/spotter/data/csa_data/raw_data

cd "$HIDRA"
python HiDRA_preprocess_improve.py --input_dir "$DATA" --output_dir /tmp/hidra_ml
python HiDRA_train_improve.py --input_dir /tmp/hidra_ml --output_dir /tmp/hidra_out --epochs 20
python HiDRA_infer_improve.py --input_data_dir /tmp/hidra_ml \
    --input_model_dir /tmp/hidra_out --output_dir /tmp/hidra_infer

cd "$PACC"
python Paccmann_MCA_preprocess_improve.py --input_dir "$DATA" --output_dir /tmp/pmca_ml
python Paccmann_MCA_train_improve.py --input_dir /tmp/pmca_ml --output_dir /tmp/pmca_out --epochs 200
python Paccmann_MCA_infer_improve.py --input_data_dir /tmp/pmca_ml \
    --input_model_dir /tmp/pmca_out --output_dir /tmp/pmca_infer

cd "$CONN"
python -m vllm_attn_connector.drp_cli capture-paccmann \
    --model-dir /tmp/pmca_out --data-dir /tmp/pmca_ml \
    --paccmann-repo "$PACC" \
    --genes "$PACC/improve_output/supplemental_data/Data/2128_genes.pkl" \
    --workflow-id pmca-gpu-v1 --write-features /tmp/pmca_genes.txt

python -m vllm_attn_connector.drp_cli capture-hidra \
    --model-dir /tmp/hidra_out --data-dir /tmp/hidra_ml \
    --workflow-id hidra-gpu-v1 --limit 8

python -m vllm_attn_connector.drp_cli predictions \
    --ml-dir /tmp/pmca_ml --infer-dir /tmp/pmca_infer --out /tmp/pmca_preds.csv

python -m vllm_attn_connector.drp_cli chains \
    --workflow-id pmca-gpu-v1 --predictions /tmp/pmca_preds.csv \
    --features /tmp/pmca_genes.txt --out /tmp/cancer_opal_chains.csv --n-chains 24

python -m vllm_attn_connector.drp_cli inspect --workflow-id pmca-gpu-v1
```

---

## See also

| document | contents |
|---|---|
| `docs/model-setup.md` | the four Paccmann bugs and the three HiDRA breaks, in detail |
| `docs/cancer-study.md` | design, results, what attention and provenance are |
| `docs/run-log.md` | verbatim transcript of a full run, with expected output |
| `docs/reading-guide.md` | the papers behind the models |
