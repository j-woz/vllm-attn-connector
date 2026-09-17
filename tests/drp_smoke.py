# SPDX-License-Identifier: Apache-2.0
"""End-to-end capture against real trained drug-response models.

Unlike ``e2e_smoke.py`` this needs **no GPU and no vLLM** -- both models run on
CPU in seconds. It does need a trained checkpoint for whichever model you ask
for, and the corresponding framework.

    python tests/drp_smoke.py --paccmann /tmp/pmca_out3 --data /tmp/pmca_ml2
    python tests/drp_smoke.py --hidra    /tmp/hidra_out --data /tmp/hidra_ml
    python tests/drp_smoke.py --paccmann ... --hidra ... --data ...   # both

Asserts the properties that decide whether a record is trustworthy:

* attention is captured at all, with the head count the architecture implies
* every distribution sums to ~1 -- these are softmax outputs, so anything else
  means the wrong tensor was grabbed
* one Flowcept task per (sample, axis), ids following the ``:g<n>`` convention
* ``attn_peak >= mean`` everywhere, with strict inequality somewhere on a
  multi-head axis (otherwise the max reduction has collapsed into a mean)
* the model's own predictions are unchanged by capture being installed --
  provenance must observe, not perturb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FAILURES: list[str] = []


def check(cond: bool, label: str) -> None:
    if cond:
        print(f"  ok  {label}")
    else:
        print(f"  FAIL {label}")
        FAILURES.append(label)


class RecordingInterceptor:
    """Captures what would be shipped, so the smoke test can assert on it
    without requiring a live Flowcept store."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.workflows: list[tuple] = []

    def capture_request(self, **kw):
        self.calls.append(kw)

    def send_model_workflow(self, workflow_id, conf, parent_workflow_id=None):
        self.workflows.append((workflow_id, conf))


def _shared_assertions(rec: RecordingInterceptor, n_samples: int, n_axes: int) -> None:
    check(len(rec.calls) == n_samples * n_axes,
          f"one task per (sample, axis): {len(rec.calls)} == {n_samples}x{n_axes}")

    ids = [c["request_id"] for c in rec.calls]
    check(all(":g" in i for i in ids), "task ids carry the :g<n> group suffix")
    check(len(set(ids)) == len(ids), "task ids are unique")

    for c in rec.calls:
        s = c["series"]
        if not {"attn_sum", "attn_peak", "attn_argmax_head"} <= set(s):
            check(False, "series carries attn_sum / attn_peak / attn_argmax_head")
            return
    check(True, "series carries attn_sum / attn_peak / attn_argmax_head")

    # Softmax outputs. On a single-headed axis the sum is the distribution
    # itself; on a k-headed axis it is k distributions added, so it sums to ~k.
    ok = True
    for c in rec.calls:
        total = sum(c["series"]["attn_sum"])
        heads = c["metadata"]["n_heads"]
        if abs(total - heads) > 0.05 * max(1, heads):
            ok = False
            print(f"       {c['request_id']}: sum={total:.4f}, expected ~{heads}")
    check(ok, "every captured distribution sums to ~1 per head (real softmax output)")

    ok = True
    for c in rec.calls:
        s, p = c["series"]["attn_sum"], c["series"]["attn_peak"]
        h = c["metadata"]["n_heads"]
        if any(pk < (sm / h) - 1e-6 for sm, pk in zip(s, p)):
            ok = False
    check(ok, "attn_peak >= mean at every position")


def run_paccmann(model_dir: str, data_dir: str) -> None:
    print("\nPaccmann MCA")
    import pickle

    import torch

    sys.path.insert(0, str(Path.home() / "Work/Spotter-AI/repos/Paccmann_MCA"))
    from paccmann_predictor.models import MODEL_FACTORY

    from vllm_attn_connector.drp_paccmann import PaccmannCapture

    with open(Path(model_dir) / "final_params.pickle", "rb") as handle:
        params = pickle.load(handle)
    model = MODEL_FACTORY["mca"](dict(params))
    model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location="cpu"))
    model.eval()

    bs = 4
    torch.manual_seed(0)
    smiles = torch.randint(0, params["smiles_vocabulary_size"],
                           (bs, params["smiles_padding_length"]))
    gep = torch.randn(bs, params["number_of_genes"])

    # Provenance must not perturb the thing it observes.
    with torch.no_grad():
        baseline, _ = model(smiles, gep)

    rec = RecordingInterceptor()
    with PaccmannCapture(model, workflow_id="wf-paccmann", interceptor=rec) as cap:
        cap.send_workflow(params)
        with torch.no_grad():
            preds, _ = model(smiles, gep)
        n_heads = cap.n_smiles_heads
        ids = cap.emit(sample_ids=[f"pm-{i}" for i in range(bs)])

    expected_heads = sum(params["multiheads"])
    check(n_heads == expected_heads,
          f"captured {n_heads} SMILES heads, architecture implies {expected_heads}")
    check(torch.allclose(baseline, preds, atol=1e-6),
          "predictions identical with capture installed (observation, not perturbation)")

    # SMILES (16 heads) + gene (1 head)
    _shared_assertions(rec, n_samples=bs, n_axes=2)
    check(len(ids) == bs * 2, f"emit returned {len(ids)} task ids")

    smiles_calls = [c for c in rec.calls if c["metadata"]["axis"] == "smiles"]
    gene_calls = [c for c in rec.calls if c["metadata"]["axis"] == "gene"]
    check(len(smiles_calls) == bs and len(gene_calls) == bs,
          "both the smiles and gene axes are emitted")
    check(all(c["metadata"]["axis_width"] == params["smiles_padding_length"]
              for c in smiles_calls),
          f"smiles axis width == smiles_padding_length ({params['smiles_padding_length']})")
    check(all(c["metadata"]["axis_width"] == params["number_of_genes"]
              for c in gene_calls),
          f"gene axis width == number_of_genes ({params['number_of_genes']})")

    # The reason per-head capture exists: hooking before the mean at
    # paccmann.py:273-279 must preserve head-to-head disagreement.
    c = smiles_calls[0]
    h = c["metadata"]["n_heads"]
    strict = any(pk > (sm / h) + 1e-6
                 for sm, pk in zip(c["series"]["attn_sum"], c["series"]["attn_peak"]))
    check(strict, "heads disagree -> per-head structure survived (not pre-averaged)")
    check(len(set(c["series"]["attn_argmax_head"])) > 1,
          "different heads win at different positions")

    check(len(rec.workflows) == 1 and rec.workflows[0][1]["model"] == "Paccmann_MCA",
          "workflow records model identity")


def run_hidra(model_dir: str, data_dir: str) -> None:
    print("\nHiDRA")
    import json
    import os
    import warnings

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    warnings.filterwarnings("ignore")

    import numpy as np
    from tensorflow.keras.models import load_model

    from vllm_attn_connector.drp_hidra import HidraCapture

    model = load_model(str(Path(model_dir) / "model.h5"), compile=False)
    with open(Path(data_dir) / "geneset.json") as handle:
        geneset = json.load(handle)

    rec = RecordingInterceptor()
    cap = HidraCapture(model, workflow_id="wf-hidra", interceptor=rec)
    cap.send_workflow({"modelpath": str(Path(model_dir) / "model.h5")})

    bs = 4
    rng = np.random.default_rng(0)
    # One input per pathway, plus the drug vector: the layout MultiGenerator yields.
    inputs = [rng.standard_normal((bs, len(geneset[p]))).astype("float32")
              for p in cap.pathway_names]
    drug_width = model.inputs[-1].shape[-1]
    inputs.append(rng.standard_normal((bs, drug_width)).astype("float32"))

    baseline = model.predict(inputs, verbose=0)
    ids = cap.emit(inputs=inputs, sample_ids=[f"hd-{i}" for i in range(bs)])
    after = model.predict(inputs, verbose=0)

    check(np.allclose(baseline, after, atol=1e-6),
          "predictions identical with capture installed")
    check(len(cap.pathway_names) == len(geneset),
          f"discovered {len(cap.pathway_names)} pathways, geneset has {len(geneset)}")

    _shared_assertions(rec, n_samples=bs, n_axes=1)
    check(len(ids) == bs, f"emit returned {len(ids)} task ids")
    check(all(c["metadata"]["axis_width"] == len(cap.pathway_names)
              for c in rec.calls),
          "pathway axis width == pathway count")
    check(all(c["metadata"]["n_heads"] == 1 for c in rec.calls),
          "pathway attention is single-headed")

    wf = rec.workflows[0][1]
    check(wf["model"] == "HiDRA" and len(wf["pathway_order"]) == len(cap.pathway_names),
          "workflow carries the pathway order (the decoder for the vectors)")

    # Opt-in gene level: one axis per pathway, so the record count multiplies.
    rec2 = RecordingInterceptor()
    cap2 = HidraCapture(model, workflow_id="wf-hidra-genes",
                        interceptor=rec2, include_gene_level=True)
    cap2.emit(inputs=inputs, sample_ids=[f"hd-{i}" for i in range(bs)])
    expected = bs * (1 + len(cap.pathway_names))
    check(len(rec2.calls) == expected,
          f"gene level opt-in emits {len(rec2.calls)} == {bs}x(1+{len(cap.pathway_names)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paccmann", help="Paccmann output dir (model.pt, final_params.pickle)")
    ap.add_argument("--hidra", help="HiDRA output dir (model.h5)")
    ap.add_argument("--data", help="preprocessed data dir", default="")
    args = ap.parse_args()

    if not args.paccmann and not args.hidra:
        ap.error("give --paccmann and/or --hidra")

    if args.paccmann:
        run_paccmann(args.paccmann, args.data)
    if args.hidra:
        run_hidra(args.hidra, args.data)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
