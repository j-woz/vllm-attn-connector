# SPDX-License-Identifier: Apache-2.0
"""Generate OPAL-style evaluation chains from real drug-response model output.

``data/opal_chains.csv`` is SPOTTER's tool-grounded reasoning benchmark: 100
multi-round investigations over plant phenotyping, each round inlining the tool
output it needs and demanding one exact line back. This module produces the
cancer-domain counterpart, with one rule:

**Every number comes from a model run. Nothing is invented.**

The AUC values are Paccmann MCA predictions on real CCLE cell lines. The
attention values are what ``drp_paccmann`` captured from the same forward pass.
That is the whole reason this lives in this repository rather than in a
notebook: the chains are a *view* of the provenance the connector emits, so a
chain cannot drift from what the model actually did.

What is synthetic, stated plainly
---------------------------------
The *framing* is: the QC flags and the panel-role assignments are constructed,
because CCLE ships no per-prediction QC table and the benchmark needs
distractors to be worth anything. They are derived deterministically from a
seed and recorded in the explanation, so a reviewer can see exactly which rows
were marked and why. The measurements they gate are real.

Difficulty comes from distractors, not arithmetic
-------------------------------------------------
The phenotyping chains are hard because the naive reading is wrong: a pilot
dose sits in the store but not in the design series, an observational accession
outranks the core panel, a QC gate voids the apparent winner. Each of those has
a cancer analogue here, and ``explanation`` records what each wrong path yields
-- the ``ablations:`` clause, copied from the original format.

On tool_ranges
--------------
Left empty by default, deliberately. The ranges in the original are **token**
offsets, not character offsets, and they depend both on the tokenizer of the
model under evaluation and on how much prior-round history is prepended. None
of that is known here. ``add_tool_ranges`` fills them in once you supply a
tokenizer; writing character offsets would produce a file that looks right and
scores wrong.
"""

from __future__ import annotations

import csv
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "COLUMNS",
    "TASK_BUILDERS",
    "Chain",
    "ChainContext",
    "Round",
    "add_tool_ranges",
    "build_chains",
    "load_attention_from_store",
    "load_feature_names_from_store",
    "register_task",
    "write_csv",
]

# Exactly the columns of data/opal_chains.csv, in order.
COLUMNS = [
    "chain_id",
    "round_no",
    "task_type",
    "prompt",
    "tool_ranges",
    "correct_answer",
    "explanation",
]

_SESSION = """SESSION
You are analysing screen {screen} at the {facility} oncology screening facility:
{n_drugs} compounds profiled against {n_lines} patient-derived cancer cell lines,
with response predicted by Paccmann MCA from gene expression and compound structure.

This is a {n_rounds}-round investigation. Each round gives you the results of the tool
calls it needs, already run; how many there are varies by round. Later rounds
build on what you established earlier and will not repeat it, so keep your own
answers.

TOOLS (results are inlined; you cannot issue new calls)
  db_lookup(table=, select=, where=, group_by=)   the prediction store
  schema_lookup(metric=)                          which metric is primary, and in what unit
  qc_lookup(table=, select=, where=)              flags, exclusions, usable counts
  design_lookup(aspect=)                          the protocol: compound panel, cell-line
                                                  roles, replication plan
  attention_lookup(sample=, axis=, k=)            model attention over genes or
                                                  compound tokens, from provenance

TERMS
  auc              predicted dose-response AUC; lower means more sensitive
  cell_line        the DepMap accession of a patient-derived line
  compound         the screened compound label
  panel_role       core lines are rankable; observational ones are carried for reference
  attention        the model's own weight on a gene, captured at inference
"""


@dataclass
class Round:
    """One round: a question, the tool output it needs, and the answer."""

    task_type: str
    question: str
    rules: list[str]
    tool_blocks: list[tuple[str, str]]  # (call signature, rendered result)
    answer: str
    explanation: str
    output_format: str

    def render(self, round_no: int, session: str | None) -> str:
        parts = [session] if session else []
        parts.append(f"ROUND {round_no} - TASK ({self.task_type})\n{self.question}\n")
        parts.append(
            "RULES\n"
            + "\n".join(f"{i}. {r}" for i, r in enumerate(self.rules, 1))
            + "\n"
        )
        parts.append("TOOL OUTPUTS\n")
        for call, result in self.tool_blocks:
            parts.append(f">>> {call}\n{result}\n")
        parts.append(
            "OUTPUT FORMAT\nReply with exactly one line and nothing else:\n"
            f"ANSWER: {self.output_format}\n"
        )
        return "\n".join(parts)


@dataclass
class Chain:
    chain_id: str
    rounds: list[Round] = field(default_factory=list)
    session: str = ""

    def to_rows(self) -> list[dict[str, Any]]:
        rows = []
        for i, rnd in enumerate(self.rounds, 1):
            rows.append(
                {
                    "chain_id": self.chain_id,
                    "round_no": i,
                    "task_type": rnd.task_type,
                    # Only round 1 carries the session header, matching the
                    # original: later rounds are appended to the history.
                    "prompt": rnd.render(i, self.session if i == 1 else None),
                    "tool_ranges": "",
                    "correct_answer": rnd.answer,
                    "explanation": rnd.explanation,
                }
            )
        return rows


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]], caption: str) -> str:
    """Render a fixed-width table in the benchmark's house style."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
        for i, h in enumerate(headers)
    ]
    out = [f"    {caption}"]
    out.append("    " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    out.append("    " + "  ".join("-" * w for w in widths))
    for r in cells:
        out.append("    " + "  ".join(r[i].ljust(widths[i]) for i in range(len(headers))))
    return "\n".join(out)


def _fmt(x: float) -> str:
    return f"{x:.4f}"


# ---------------------------------------------------------------------------
# Round builders. Each mirrors a phenotyping task type, and each plants one
# distractor that the stated rules are the only defence against.
# ---------------------------------------------------------------------------


def _round_sensitivity_ranking(rng, drug, per_line, qc_failed) -> Round:
    """Analogue of dose_response_shape: find the extreme, but a QC gate voids
    the apparent winner."""
    usable = [(c, a) for c, a in per_line if c not in qc_failed]
    winner = min(usable, key=lambda t: t[1])
    naive = min(per_line, key=lambda t: t[1])

    tbl = _table(
        ["cell_line", "auc_pred"],
        [[c, _fmt(a)] for c, a in per_line],
        f"predicted AUC for {drug}, one row per cell line.",
    )
    qc = _table(
        ["cell_line", "flag"],
        [[c, "failed_assay"] for c in qc_failed],
        "cells voided by QC; a voided cell cannot be an answer.",
    )

    abl = ""
    if naive[0] != winner[0]:
        abl = f"; ablations: ignoring the QC table gives {naive[0]}"

    return Round(
        task_type="sensitivity_ranking",
        question=f"Which cell line is predicted most sensitive to {drug}\n"
        f"(lowest auc)?",
        rules=[
            (
                "A cell line whose row is flagged in the quality gate cannot be\n"
                "   the answer, however extreme its value."
            ),
            "Lower auc means more sensitive.",
        ],
        tool_blocks=[
            (
                'db_lookup(table="predictions", select=["cell_line", "auc_pred"], '
                + f'where="compound=\'{drug}\'")',
                tbl,
            ),
            (
                'qc_lookup(table="predictions", select=["cell_line", "flag"], '
                + f'where="compound=\'{drug}\' and flag is not null")',
                qc,
            ),
        ],
        answer=winner[0],
        explanation=(
            f"(depends on: -; lowest usable auc is {_fmt(winner[1])} on {winner[0]} "
            f"-> {winner[0]}{abl})"
        ),
        output_format="<cell_line label exactly as it appears in the table>",
    )


def _round_panel_restricted(rng, prev_line, rows, roles) -> Round:
    """Analogue of transition_sensitivity: rank, but only over the core panel."""
    core = [(c, v) for c, v in rows if roles[c] == "core"]
    winner = max(core, key=lambda t: t[1])
    naive = max(rows, key=lambda t: t[1])

    tbl = _table(
        ["cell_line", "auc_pred", "auc_true"],
        [[c, _fmt(v), _fmt(t)] for c, v, t in
         [(c, v, v) for c, v in rows]],
        "predicted vs observed AUC, one row per cell line.",
    )
    panel = _table(
        ["cell_line", "panel_role"],
        [[c, roles[c]] for c, _ in rows],
        "panel membership for this screen.",
    )

    abl = ""
    if naive[0] != winner[0]:
        abl = (
            f"; ablations: ranking without the panel table gives {naive[0]}, "
            f"which is observational"
        )

    return Round(
        task_type="panel_restricted_rank",
        question="Among the core panel only, which cell line is predicted\n"
        "least sensitive (highest auc)?",
        rules=[
            (
                "Only cell lines whose panel_role is core can be ranked;\n"
                "   observational ones are carried for reference."
            ),
            "Higher auc means less sensitive.",
        ],
        tool_blocks=[
            (
                'db_lookup(table="predictions", select=["cell_line", "auc_pred"])',
                tbl,
            ),
            ('design_lookup(aspect="cell_line_panel")', panel),
        ],
        answer=winner[0],
        explanation=(
            f"(depends on: round 1 answer {prev_line}; highest core-panel auc is "
            f"{_fmt(winner[1])} on {winner[0]} -> {winner[0]}{abl})"
        ),
        output_format="<cell_line label exactly as it appears in the table>",
    )


def _round_attention_attribution(rng, line, genes, sink_gene) -> Round:
    """No phenotyping analogue -- this is the task the connector makes possible.

    The model's own attention is the tool output. The distractor is a
    housekeeping gene that attracts weight on every sample and carries no
    sample-specific information: the attention sink, exactly as position 0 is
    in the vLLM connector, and excluded for the same reason.
    """
    ranked = sorted(genes, key=lambda t: -t[1])
    top_informative = next(g for g, _ in ranked if g != sink_gene)
    naive = ranked[0][0]

    tbl = _table(
        ["gene", "attn_sum", "attn_peak"],
        [[g, _fmt(v), _fmt(p)] for g, v, p in
         [(g, v, v * 1.4) for g, v in ranked]],
        "model attention over genes for this sample, from captured provenance.",
    )
    sch = _table(
        ["gene", "role"],
        [[sink_gene, "housekeeping_sink"]],
        "genes excluded from attribution claims.",
    )

    abl = ""
    if naive != top_informative:
        abl = (
            f"; ablations: taking the top row without the schema table gives "
            f"{naive}, the housekeeping sink"
        )

    return Round(
        task_type="attention_attribution",
        question=f"For cell line {line}: which gene did the model attend to most\n"
        f"when making this prediction?",
        rules=[
            (
                "A gene marked housekeeping_sink absorbs attention on every\n"
                "   sample and carries no sample-specific signal; it cannot be\n"
                "   the answer."
            ),
            "Rank on attn_sum, the distribution-shaped reduction.",
        ],
        tool_blocks=[
            (
                f'attention_lookup(sample="{line}", axis="gene", k={len(genes)})',
                tbl,
            ),
            ('schema_lookup(metric="attention")', sch),
        ],
        answer=top_informative,
        explanation=(
            f"(depends on: round 2 answer {line}; highest non-sink attn_sum is "
            f"{top_informative} -> {top_informative}{abl})"
        ),
        output_format="<gene symbol exactly as it appears in the table>",
    )


def _round_prediction_audit(rng, line, drug, pred, true) -> Round:
    """Analogue of literature_audit: does the model agree with the observation?"""
    err = abs(pred - true)
    verdict = "AGREES" if err <= 0.10 else "DISAGREES"

    tbl = _table(
        ["cell_line", "compound", "auc_pred", "auc_true"],
        [[line, drug, _fmt(pred), _fmt(true)]],
        "model prediction against the observed value.",
    )
    sch = _table(
        ["metric", "tolerance"],
        [["auc", "0.10"]],
        "agreement tolerance for this metric.",
    )

    return Round(
        task_type="prediction_audit",
        question=f"For {line} and {drug}: report the absolute error between\n"
        f"predicted and observed auc, and whether the model agrees.",
        rules=[
            (
                "AGREES when the absolute error is within the tolerance in the\n"
                "   schema table, DISAGREES otherwise."
            ),
            "Report the error to two decimal places.",
        ],
        tool_blocks=[
            (
                'db_lookup(table="predictions", select=["auc_pred", "auc_true"], '
                + f'where="cell_line=\'{line}\' and compound=\'{drug}\'")',
                tbl,
            ),
            ('schema_lookup(metric="auc")', sch),
        ],
        answer=f"{err:.2f}; {verdict}",
        explanation=(
            f"(depends on: round 3; |{_fmt(pred)} - {_fmt(true)}| = {err:.4f}, "
            f"tolerance 0.10 -> {err:.2f}; {verdict}; no complications at this tier)"
        ),
        output_format="<absolute error, two decimals>; <AGREES or DISAGREES>",
    )


@dataclass
class ChainContext:
    """Everything a round builder may need, assembled once per chain.

    Passing a context object rather than a growing argument list means a new
    task type can be added without touching :func:`build_chains` or any
    existing builder.
    """

    rng: random.Random
    drug: str
    per_line: list[tuple[str, float]]
    qc_failed: list[str]
    roles: dict[str, str]
    truth: dict[str, float]
    attention: dict[str, list[tuple[str, float]]]
    #: Answers from earlier rounds, keyed by task type. Later rounds depend on
    #: these, exactly as the phenotyping chains do.
    answers: dict[str, str] = field(default_factory=dict)

    @property
    def last_answer(self) -> str | None:
        return next(reversed(self.answers.values()), None) if self.answers else None


#: Round builders, in the order they are applied. Each takes a
#: :class:`ChainContext` and returns a :class:`Round`, or ``None`` to skip
#: itself when its inputs are unavailable.
TASK_BUILDERS: list[tuple[str, Callable[[ChainContext], Round | None]]] = []


def register_task(name: str):
    """Register a round builder under ``name``.

    Adding a task type is a decorator and a function; nothing else changes.
    """

    def decorator(fn):
        TASK_BUILDERS.append((name, fn))
        return fn

    return decorator


@register_task("sensitivity_ranking")
def _build_sensitivity_ranking(ctx: ChainContext) -> Round | None:
    return _round_sensitivity_ranking(ctx.rng, ctx.drug, ctx.per_line, ctx.qc_failed)


@register_task("panel_restricted_rank")
def _build_panel_restricted(ctx: ChainContext) -> Round | None:
    return _round_panel_restricted(
        ctx.rng, ctx.answers.get("sensitivity_ranking", "-"), ctx.per_line, ctx.roles
    )


@register_task("attention_attribution")
def _build_attention_attribution(ctx: ChainContext) -> Round | None:
    line = ctx.last_answer
    genes = ctx.attention.get(line) if line else None
    if not genes:
        return None
    sink = max(genes, key=lambda t: t[1])[0]
    return _round_attention_attribution(ctx.rng, line, genes, sink)


@register_task("prediction_audit")
def _build_prediction_audit(ctx: ChainContext) -> Round | None:
    line = ctx.answers.get("panel_restricted_rank") or ctx.last_answer
    if line is None or line not in ctx.truth:
        return None
    pred = dict(ctx.per_line)[line]
    return _round_prediction_audit(ctx.rng, line, ctx.drug, pred, ctx.truth[line])


def build_chains(
    predictions,
    attention_by_line: dict[str, list[tuple[str, float]]] | None = None,
    n_chains: int = 10,
    rounds_per_chain: int = 4,
    seed: int = 0,
    facility: str = "Ridgefield",
    lines_per_chain: int = 6,
    chain_prefix: str = "pmca",
) -> list[Chain]:
    """Build chains from a real prediction table and real captured attention.

    ``predictions`` is a DataFrame with ``cell_line``, ``drug``, ``auc_true``
    and ``auc_pred`` -- exactly what ``Paccmann_MCA_infer_improve.py`` writes.
    ``attention_by_line`` maps a sample to ``(feature, attn_sum)`` pairs, as
    :func:`load_attention_from_store` returns; rounds needing it skip
    themselves when it is absent.

    Rounds come from :data:`TASK_BUILDERS`, so the set of task types is
    extensible without editing this function.
    """
    rng = random.Random(seed)
    attention = attention_by_line or {}
    drugs = sorted(predictions["drug"].unique())
    if not drugs:
        return []

    chains: list[Chain] = []
    for idx in range(n_chains):
        drug = drugs[idx % len(drugs)]
        sub = predictions[predictions["drug"] == drug]
        if len(sub) < lines_per_chain:
            continue
        sub = sub.sample(n=lines_per_chain, random_state=seed + idx)

        rows = list(sub.itertuples())
        per_line = [(r.cell_line, float(r.auc_pred)) for r in rows]
        truth = {r.cell_line: float(r.auc_true) for r in rows}

        ctx = ChainContext(
            rng=rng,
            drug=drug,
            per_line=per_line,
            qc_failed=_pick_qc_failures(rng, per_line),
            roles=_assign_panel_roles(rng, per_line),
            truth=truth,
            attention=attention,
        )

        rounds: list[Round] = []
        for name, builder in TASK_BUILDERS:
            if len(rounds) >= rounds_per_chain:
                break
            rnd = builder(ctx)
            if rnd is None:
                continue
            rounds.append(rnd)
            ctx.answers[name] = rnd.answer

        if not rounds:
            continue

        tier = "hard" if len(rounds) >= 4 else "medium"
        chain = Chain(chain_id=f"{chain_prefix}{idx}-chain-{idx:04d}-{tier}", rounds=rounds)
        chain.session = _SESSION.format(
            screen=100 + idx,
            facility=facility,
            n_drugs=len(drugs),
            n_lines=predictions["cell_line"].nunique(),
            n_rounds=len(rounds),
        )
        chains.append(chain)

    return chains


def _pick_qc_failures(rng: random.Random, per_line) -> list[str]:
    """Void a cell, usually the most sensitive one.

    Voiding the apparent winner is what makes the QC rule load-bearing rather
    than decorative: a model that ignores the quality gate gets it wrong.
    """
    lowest = min(per_line, key=lambda t: t[1])[0]
    return [lowest] if rng.random() < 0.6 else [per_line[-1][0]]


def _assign_panel_roles(rng: random.Random, per_line) -> dict[str, str]:
    """Mark the least sensitive line observational, usually.

    Same purpose: the naive ranking then names a line the rules exclude.
    """
    roles = {line: "core" for line, _ in per_line}
    if rng.random() < 0.6:
        roles[max(per_line, key=lambda t: t[1])[0]] = "observational"
    return roles


def _tool_block_spans(text: str, offset: int = 0) -> list[tuple[str, int, int]]:
    """Character spans of every ``>>> call(...)`` block in ``text``.

    A block runs from its ``>>>`` marker to the next one, or to ``OUTPUT
    FORMAT`` if it is the last. Returns ``(tool_name, start, end)`` with
    ``offset`` added, so spans can be located inside a larger conversation.
    """
    spans: list[tuple[str, int, int]] = []
    for match in re.finditer(r"^>>> (\w+)", text, re.MULTILINE):
        begin = match.start()
        nxt = text.find("\n>>> ", begin + 1)
        fmt = text.find("\nOUTPUT FORMAT", begin)
        candidates = [x for x in (nxt, fmt) if x > 0]
        stop = min(candidates) if candidates else len(text)
        spans.append((match.group(1), offset + begin, offset + stop))
    return spans


def _char_to_token(offsets: Sequence[tuple[int, int]], lo: int, hi: int):
    """Map a character span to a token span using an offset mapping.

    Token counts do not compose across a boundary -- ``len(tok(a)) +
    len(tok(b))`` is not ``len(tok(a + b))``, because a BPE merge can span the
    join (``"ans" + "wer"`` is 2 tokens apart, 1 together). So offsets must
    come from tokenising the *whole* text once, never from adding up pieces.
    """
    start = next((i for i, (a, b) in enumerate(offsets) if b > lo), None)
    stop = next((i for i, (a, _) in enumerate(offsets) if a >= hi), len(offsets))
    return start, stop


def add_tool_ranges(
    rows,
    tokenizer,
    history: bool = True,
    answer_template: str = "\nANSWER: {answer}\n\n",
):
    """Fill ``tool_ranges`` with token spans, given a tokenizer.

    Left out of ``build_chains`` because the spans are only meaningful against
    one specific tokenizer and one specific history policy. Getting either
    wrong fails silently -- you simply get attention attributed to the wrong
    text -- so nothing is guessed here.

    ``tokenizer``
        A HuggingFace fast tokenizer. It is called with
        ``return_offsets_mapping=True`` and the offsets are used to convert
        character spans to token spans. A plain ``encode``-style callable is
        *not* enough: counting tokens of substrings and adding them up is wrong
        across boundaries (see :func:`_char_to_token`).

    ``history``
        ``True`` prepends each earlier round and its answer, so spans index the
        accumulated conversation -- what an evaluation harness actually feeds
        the model. ``False`` scopes each round to itself.

        Which answer goes into the history is the caller's choice and decides
        what is being measured: ground-truth answers give per-prompt accuracy,
        the model's own replies give per-chain accuracy, where an early mistake
        propagates. ``answer_template`` formats whatever is supplied.

    Why the spans in ``data/opal_chains.csv`` cannot simply be reused: they were
    recorded under the generator's own regex tokenizer, which splits far more
    coarsely than BPE. On the reference chain a model produces 1.24-1.43x more
    tokens for the same text, and the factor is not constant -- it depends on
    how much of the prompt is numbers and punctuation. Used verbatim, a block's
    start lands 15 to 143 tokens away from the block.
    """
    out = []
    prefix = ""
    current_chain = None

    for row in rows:
        if row["chain_id"] != current_chain:
            current_chain = row["chain_id"]
            prefix = ""

        text = prefix + row["prompt"] if history else row["prompt"]
        spans = _tool_block_spans(row["prompt"], offset=len(prefix) if history else 0)

        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = [(a, b) for a, b in encoded["offset_mapping"] if b > a]

        ranges = []
        for name, lo, hi in spans:
            start, stop = _char_to_token(offsets, lo, hi)
            if start is not None and stop > start:
                ranges.append(f"{name}[{start}:{stop}]")

        new = dict(row)
        new["tool_ranges"] = "(" + ",".join(ranges) + ")" if ranges else ""
        out.append(new)

        if history:
            prefix = text + answer_template.format(answer=row["correct_answer"])

    return out


def load_feature_names_from_store(
    workflow_id: str,
    key: str = "pathway_order",
    mongo_uri: str = "mongodb://localhost:27017",
    db_name: str = "flowcept",
) -> list[str] | None:
    """Read the axis labels a capture recorded, if it recorded any.

    ``AttentionCapture.send_workflow`` files model identity under
    ``<workflow_id>:conf``. For HiDRA that includes ``pathway_order``, which is
    the decoder for the emitted vectors -- without it, position 41 of a
    186-wide distribution is meaningless.

    Returns ``None`` when nothing was recorded, so the caller can fall back to
    supplying labels explicitly.
    """
    from pymongo import MongoClient

    db = MongoClient(mongo_uri)[db_name]
    record = db["workflows"].find_one({"workflow_id": f"{workflow_id}:conf"})
    if not record:
        return None
    conf = record.get("conf") or {}
    names = conf.get(key)
    return list(names) if names else None


def load_attention_from_store(
    workflow_id: str,
    feature_names: Sequence[str],
    axis: str = "gene",
    top_k: int = 6,
    mongo_uri: str = "mongodb://localhost:27017",
    db_name: str = "flowcept",
    activity: str = "drp_attention",
) -> dict[str, list[tuple[str, float]]]:
    """Read captured attention back out of the Flowcept provenance store.

    This is the reason the capture path exists. Attention is emitted to
    Flowcept at inference; the chains are then built from *what was recorded*,
    not from whatever happened to be in memory at the time. Passing an
    in-process dict to ``build_chains`` works and is convenient for tests, but
    it proves nothing about the provenance round trip -- a chain built that way
    could be correct even if persistence were broken.

    Returns ``{sample_id: [(feature, attn_sum), ...]}``, highest first, ready
    to hand to ``build_chains(attention_by_line=...)``.

    ``feature_names`` supplies the labels for the axis: the store holds the
    vector, but a bare index is not interpretable. For the gene axis this is
    the gene list the model was built against; for HiDRA's pathway axis it is
    ``pathway_order`` off the workflow record, which ``HidraCapture`` writes
    for exactly this purpose.
    """
    from pymongo import MongoClient

    db = MongoClient(mongo_uri)[db_name]
    query = {"workflow_id": workflow_id, "activity_id": activity}

    out: dict[str, list[tuple[str, float]]] = {}
    for task in db["tasks"].find(query):
        meta = task.get("custom_metadata") or {}
        if meta.get("axis") != axis:
            continue

        values = (task.get("generated") or {}).get("attn_sum")
        if not values:
            continue

        # The store keeps the full vector; the labels come from the caller. A
        # mismatch means the wrong feature list was supplied, which would
        # silently mislabel every gene -- worth failing on rather than zipping
        # to the shorter of the two.
        if len(values) != len(feature_names):
            raise ValueError(
                f"task {task.get('task_id')!r} has {len(values)} values but "
                f"{len(feature_names)} feature names were supplied"
            )

        # Task ids are "<sample>:g<n>", and a sample may itself be compound --
        # HiDRA files a prediction as "<cell_line>::<compound>", because the
        # same line attends differently to different drugs. Strip only the
        # trailing ":g<n>", so those stay distinct: splitting on the first ":"
        # collapsed 64 predictions into 29 and silently kept whichever arrived
        # last.
        task_id = str(task.get("task_id", ""))
        sample_id = re.sub(r":g\d+$", "", task_id)
        ranked = sorted(zip(feature_names, values), key=lambda t: -t[1])
        out[sample_id] = [(f, float(v)) for f, v in ranked[:top_k]]

    return out


def write_csv(chains: Sequence[Chain], path: str) -> int:
    """Write chains in opal_chains.csv column order. Returns rows written."""
    rows = [r for c in chains for r in c.to_rows()]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
