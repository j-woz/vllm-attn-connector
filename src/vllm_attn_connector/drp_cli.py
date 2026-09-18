# SPDX-License-Identifier: Apache-2.0
"""Command line for the drug-response capture and chain pipeline.

Three subcommands, matching the three stages::

    python -m vllm_attn_connector.drp_cli capture-paccmann \\
        --model-dir /tmp/pmca_out3 --data-dir /tmp/pmca_ml2 \\
        --workflow-id pmca-attention-v1

    python -m vllm_attn_connector.drp_cli chains \\
        --workflow-id pmca-attention-v1 \\
        --predictions /tmp/preds.csv --features /tmp/genes.txt \\
        --out data/cancer_opal_chains.csv

    python -m vllm_attn_connector.drp_cli inspect \\
        --workflow-id pmca-attention-v1

``capture-*`` needs the model's framework and a checkpoint. ``chains`` and
``inspect`` need only a reachable provenance store, which is the point: once
attention is recorded, nothing downstream has to load a model again.

Persistence is off in Flowcept's defaults, so records go to an in-memory buffer
and vanish. For a real store, run MongoDB and Redis and point
``FLOWCEPT_SETTINGS_PATH`` at a settings file with ``mongodb``, ``mq`` and
``kv_db`` all enabled -- Flowcept's ``full-online`` profile sets exactly those
three, and it fails at startup if only some are on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main"]

DEFAULT_MONGO_URI = "mongodb://localhost:27017"
DEFAULT_BATCH = 64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _read_lines(path: str) -> list[str]:
    """Read newline-delimited labels, ignoring blanks."""
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_predictions(path: str):
    import pandas as pd

    frame = pd.read_csv(path)
    required = {"cell_line", "drug", "auc_true", "auc_pred"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(
            f"{path} is missing required column(s): {', '.join(sorted(missing))}"
        )
    return frame


def _flowcept(workflow_id: str, name: str):
    """Open a Flowcept session, or a no-op if Flowcept is unavailable."""
    from flowcept import Flowcept

    return Flowcept("vllm", workflow_id=workflow_id, workflow_name=name)


# ---------------------------------------------------------------------------
# capture-paccmann
# ---------------------------------------------------------------------------


def cmd_capture_paccmann(args: argparse.Namespace) -> int:
    import pickle

    import pandas as pd
    import torch

    from .drp_paccmann import PaccmannCapture

    model_dir, data_dir = Path(args.model_dir), Path(args.data_dir)
    with open(model_dir / "final_params.pickle", "rb") as handle:
        params = pickle.load(handle)

    sys.path.insert(0, str(Path(args.paccmann_repo).expanduser()))
    from paccmann_predictor.models import MODEL_FACTORY

    model = MODEL_FACTORY["mca"](dict(params))
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu"))
    model.eval()

    with open(Path(args.genes).expanduser(), "rb") as handle:
        gene_list = pickle.load(handle)

    expression = pd.read_csv(data_dir / "gene_expression.csv", index_col=0)
    samples = [s for s in expression.index if s in set(expression.index)]
    if args.limit:
        samples = samples[: args.limit]

    present = [g for g in gene_list if g in expression.columns]
    n_pad = len(gene_list) - len(present)
    # Labels must describe the tensor actually fed. Genes absent from the
    # expression file are zero-padded, and mislabelling those positions would
    # silently shift every gene name by the number of missing entries.
    labels = present + [f"__PAD_{i}__" for i in range(n_pad)]

    torch.manual_seed(args.seed)
    print(f"capturing {len(samples)} samples, {len(present)} genes (+{n_pad} padded)")

    with (
        _flowcept(args.workflow_id, "paccmann_attention"),
        PaccmannCapture(model, workflow_id=args.workflow_id) as cap,
    ):
        cap.send_workflow(params)
        for start in range(0, len(samples), args.batch_size):
            chunk = samples[start : start + args.batch_size]
            gep = torch.tensor(
                expression.loc[chunk, present].to_numpy(), dtype=torch.float32
            )
            if n_pad:
                gep = torch.nn.functional.pad(gep, (0, n_pad))
            smiles = torch.randint(
                0, params["smiles_vocabulary_size"],
                (len(chunk), params["smiles_padding_length"]),
            )
            with torch.no_grad():
                model(smiles, gep)
            cap.emit(sample_ids=chunk)

    if args.write_features:
        Path(args.write_features).write_text("\n".join(labels))
        print(f"wrote feature labels -> {args.write_features}")

    print(f"captured {len(samples)} samples to workflow {args.workflow_id!r}")
    return 0


# ---------------------------------------------------------------------------
# capture-hidra
# ---------------------------------------------------------------------------


def cmd_capture_hidra(args: argparse.Namespace) -> int:
    import numpy as np
    from tensorflow.keras.models import load_model

    from .drp_hidra import HidraCapture

    model = load_model(str(Path(args.model_dir) / "model.h5"), compile=False)
    with open(Path(args.data_dir) / "geneset.json") as handle:
        geneset = json.load(handle)

    with _flowcept(args.workflow_id, "hidra_attention"):
        cap = HidraCapture(
            model,
            workflow_id=args.workflow_id,
            include_gene_level=args.gene_level,
        )
        cap.send_workflow({"modelpath": str(Path(args.model_dir) / "model.h5")})

        n = args.limit or 8
        rng = np.random.default_rng(args.seed)
        inputs = [
            rng.standard_normal((n, len(geneset[p]))).astype("float32")
            for p in cap.pathway_names
        ]
        inputs.append(
            rng.standard_normal((n, model.inputs[-1].shape[-1])).astype("float32")
        )
        cap.emit(inputs=inputs, sample_ids=[f"hidra-{i}" for i in range(n)])

    print(
        f"captured {n} samples over {len(cap.pathway_names)} pathways "
        f"to workflow {args.workflow_id!r}"
    )
    return 0


# ---------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------


def cmd_predictions(args: argparse.Namespace) -> int:
    """Join an inference run into the table the chain builder needs.

    ``*_infer_improve.py`` writes only ``auc_true,auc_pred`` -- the identifying
    columns are dropped, so the predictions cannot be joined back to a cell
    line or compound without the preprocessed y-data they came from. Row order
    is the join key: both files are written from the same test set in the same
    order.
    """
    import pandas as pd

    ydata = pd.read_csv(args.ml_dir + "/test_y_data.csv")
    preds = pd.read_csv(args.infer_dir + "/test_y_data_predicted.csv")
    if len(ydata) != len(preds):
        raise SystemExit(
            f"row count mismatch: {len(ydata)} in test_y_data.csv vs "
            f"{len(preds)} in test_y_data_predicted.csv -- these must come "
            f"from the same run"
        )

    joined = pd.concat(
        [ydata.reset_index(drop=True), preds.reset_index(drop=True)], axis=1
    )[["cell_line", "drug", "auc_true", "auc_pred"]]
    joined.to_csv(args.out, index=False)
    print(f"wrote {len(joined)} predictions -> {args.out}")
    return 0


def cmd_chains(args: argparse.Namespace) -> int:
    from .drp_chains import build_chains, load_attention_from_store, write_csv

    features = _read_lines(args.features)
    predictions = _load_predictions(args.predictions)

    attention = load_attention_from_store(
        workflow_id=args.workflow_id,
        feature_names=features,
        axis=args.axis,
        mongo_uri=args.mongo_uri,
    )
    if not attention:
        raise SystemExit(
            f"no attention found for workflow {args.workflow_id!r} on axis "
            f"{args.axis!r}. Was the store enabled during capture?"
        )
    print(f"read attention for {len(attention)} samples from the provenance store")

    chains = build_chains(
        predictions,
        attention_by_line=attention,
        n_chains=args.n_chains,
        rounds_per_chain=args.rounds,
        seed=args.seed,
    )
    rows = write_csv(chains, args.out)
    print(f"wrote {len(chains)} chains / {rows} rows -> {args.out}")
    return 0


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


def cmd_inspect(args: argparse.Namespace) -> int:
    from pymongo import MongoClient

    db = MongoClient(args.mongo_uri)["flowcept"]
    tasks = list(db["tasks"].find({"workflow_id": args.workflow_id}))
    if not tasks:
        print(f"no tasks for workflow {args.workflow_id!r}")
        return 1

    by_axis: dict[str, int] = {}
    for task in tasks:
        axis = (task.get("custom_metadata") or {}).get("axis", "?")
        by_axis[axis] = by_axis.get(axis, 0) + 1

    first = tasks[0]
    meta = first.get("custom_metadata") or {}
    print(f"workflow      : {args.workflow_id}")
    print(f"tasks         : {len(tasks)}")
    print(f"axes          : {by_axis}")
    print(f"model         : {meta.get('model')} ({meta.get('framework')})")
    print(f"metric_ref    : {meta.get('metric_reference')}")
    print(f"series        : {sorted((first.get('generated') or {}).keys())}")
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drp_cli", description=__doc__.split("\n")[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pm = sub.add_parser("capture-paccmann", help="capture Paccmann MCA attention")
    pm.add_argument("--model-dir", required=True)
    pm.add_argument("--data-dir", required=True)
    pm.add_argument("--workflow-id", required=True)
    pm.add_argument("--paccmann-repo", default="~/Work/Spotter-AI/repos/Paccmann_MCA")
    pm.add_argument(
        "--genes",
        default="~/Work/Spotter-AI/repos/Paccmann_MCA/improve_output/"
        "supplemental_data/Data/2128_genes.pkl",
    )
    pm.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    pm.add_argument("--limit", type=int, default=0, help="cap samples (0 = all)")
    pm.add_argument("--seed", type=int, default=0)
    pm.add_argument("--write-features", help="write the feature labels here")
    pm.set_defaults(func=cmd_capture_paccmann)

    hd = sub.add_parser("capture-hidra", help="capture HiDRA attention")
    hd.add_argument("--model-dir", required=True)
    hd.add_argument("--data-dir", required=True)
    hd.add_argument("--workflow-id", required=True)
    hd.add_argument("--gene-level", action="store_true")
    hd.add_argument("--limit", type=int, default=8)
    hd.add_argument("--seed", type=int, default=0)
    hd.set_defaults(func=cmd_capture_hidra)

    pr = sub.add_parser(
        "predictions", help="join inference output into a chain-ready table"
    )
    pr.add_argument("--ml-dir", required=True, help="preprocess output dir")
    pr.add_argument("--infer-dir", required=True, help="inference output dir")
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=cmd_predictions)

    ch = sub.add_parser("chains", help="build OPAL chains from stored provenance")
    ch.add_argument("--workflow-id", required=True)
    ch.add_argument("--predictions", required=True)
    ch.add_argument("--features", required=True)
    ch.add_argument("--out", required=True)
    ch.add_argument("--axis", default="gene")
    ch.add_argument("--n-chains", type=int, default=24)
    ch.add_argument("--rounds", type=int, default=4)
    ch.add_argument("--seed", type=int, default=0)
    ch.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ch.set_defaults(func=cmd_chains)

    ins = sub.add_parser("inspect", help="summarise a workflow in the store")
    ins.add_argument("--workflow-id", required=True)
    ins.add_argument("--mongo-uri", default=DEFAULT_MONGO_URI)
    ins.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
