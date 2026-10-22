# Patches

Two upstream repositories do not run on a current Python environment. These
patches are the minimum needed to make them run; neither changes model
behaviour.

They live here because `docs/model-setup.md` and `docs/quickstart-gpu.md`
reference them, and until now they existed only in an untracked working
directory — so anyone following those docs reached a `git apply` on a path
that did not exist.

| patch | applies to | what it fixes |
|---|---|---|
| `improve-pyproject-toml-fix.patch` | `JDACS4C-IMPROVE/IMPROVE` @ `develop` | malformed TOML in the `authors` table; `pip install -e .` dies with `TOMLDecodeError`. Packaging metadata only. |
| `hidra-py313-pandas3-keras3.patch` | `JDACS4C-IMPROVE/HiDRA` @ `develop` | three independent breaks under pandas 3 / Keras 3 — see below. |

Apply from inside the cloned repository:

```bash
cd repos/IMPROVE && git apply /path/to/patches/improve-pyproject-toml-fix.patch
cd repos/HiDRA  && git apply /path/to/patches/hidra-py313-pandas3-keras3.patch
```

`git apply --check <patch>` first if you want to confirm without writing.
Both were re-verified against fresh clones on 2026-10-22.

## What the HiDRA patch changes

1. **pandas 3** — `df.apply(zscore, axis=1)` now returns a Series of arrays
   instead of expanding to a DataFrame, so assigning it back raised
   `ValueError: Columns must be same length as key`. Replaced with a
   vectorized `zscore(..., axis=1)` on the ndarray.
2. **Keras 3** — the data generator yielded DataFrames; Keras flattens inputs
   with optree and calls `.dtype` on each leaf (`'DataFrame' object has no
   attribute 'dtype'`), and separately rejects a bare `list` of inputs as a
   nested structure. Now yields a tuple of float32 ndarrays.
3. **Keras 3 loading** — `load_model` on the legacy `.h5` failed with
   `Could not deserialize 'keras.metrics.mse'`. Inference only calls
   `predict()`, so it loads with `compile=False`.

## There is no Paccmann patch here, deliberately

Paccmann needs four fixes — including one where gene expression silently
arrives as all zeros — but they are **commits on the fork**, not patches:

    github.com/rajeeja/Paccmann_MCA  branch rajeeja/py313-fixes

Clone that branch and you have them. The repository is public; no invitation
is required. Applying a patch on top would fail, because the changes are
already in the tree.
