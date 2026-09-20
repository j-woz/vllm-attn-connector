# SPDX-License-Identifier: Apache-2.0
"""Checks for the cancer chain generator. No model, no framework needed.

The point of a benchmark is that its ground truth is right. A chain whose
``correct_answer`` disagrees with its own inlined tool output is worse than no
chain at all -- it silently penalises a model that reasons correctly. So the
load-bearing test here re-derives every answer by **parsing the rendered
prompt**, never by consulting the objects that produced it. If the generator
and the renderer ever disagree, that is exactly the bug this catches.

    python tests/test_drp_chains.py
    pytest tests/test_drp_chains.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vllm_attn_connector.drp_chains import (
    COLUMNS,
    add_tool_ranges,
    build_chains,
)

VERBOSE = __name__ == "__main__"


def say(msg):
    if VERBOSE:
        print(msg)


class FakeFrame:
    """Enough of a DataFrame for the generator, without requiring pandas."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return FakeCol([r[key] for r in self.rows])
        return FakeFrame([r for r, keep in zip(self.rows, key.values) if keep])

    def __getattr__(self, name):
        if name in ("cell_line", "drug", "auc_true", "auc_pred"):
            return FakeCol([r[name] for r in self.rows])
        raise AttributeError(name)

    def sample(self, n, random_state=0):
        import random

        rng = random.Random(random_state)
        return FakeFrame(rng.sample(self.rows, min(n, len(self.rows))))

    def itertuples(self):
        class Row:
            def __init__(self, d):
                self.__dict__.update(d)

        return [Row(r) for r in self.rows]

    @property
    def iloc(self):
        outer = self

        class ILoc:
            def __getitem__(self, i):
                class Row:
                    def __init__(self, d):
                        self.__dict__.update(d)

                return Row(outer.rows[i])

        return ILoc()


class FakeCol:
    def __init__(self, values):
        self.values_list = values

    def unique(self):
        seen, out = set(), []
        for v in self.values_list:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out

    def nunique(self):
        return len(set(self.values_list))

    def __eq__(self, other):
        return FakeMask([v == other for v in self.values_list])

    def __iter__(self):
        return iter(self.values_list)


class FakeMask:
    def __init__(self, values):
        self.values = values


def _fixture():
    """Six cell lines x two compounds, with values chosen so the distractors
    actually bite: the most sensitive line is one that QC can void, and the
    least sensitive is one that can be marked observational."""
    lines = [f"ACH-{i:06d}" for i in range(6)]
    rows = []
    for d, base in [("Drug_A", 0.30), ("Drug_B", 0.60)]:
        for i, ln in enumerate(lines):
            rows.append(
                {
                    "cell_line": ln,
                    "drug": d,
                    "auc_pred": base + i * 0.07,
                    "auc_true": base + i * 0.07 + 0.02,
                }
            )
    attn = {
        ln: [
            ("SINKGENE", 0.90),  # housekeeping sink: always top
            ("BRCA1", 0.04 + i * 0.001),
            ("TP53", 0.03),
            ("EGFR", 0.02),
        ]
        for i, ln in enumerate(lines)
    }
    return FakeFrame(rows), attn


def _rows():
    preds, attn = _fixture()
    chains = build_chains(preds, attention_by_line=attn, n_chains=2, seed=0)
    return [r for c in chains for r in c.to_rows()]


# --------------------------------------------------------------------------
# structure
# --------------------------------------------------------------------------


def test_columns_match_opal_exactly():
    assert COLUMNS == [
        "chain_id", "round_no", "task_type", "prompt",
        "tool_ranges", "correct_answer", "explanation",
    ]
    for r in _rows():
        assert set(r) == set(COLUMNS)
    say("  ok  columns match opal_chains.csv exactly")


def test_session_header_only_on_round_one():
    """Later rounds are appended to a running history, so repeating the header
    would both waste context and desynchronise token offsets."""
    for r in _rows():
        if r["round_no"] == 1:
            assert r["prompt"].startswith("SESSION")
        else:
            assert "SESSION" not in r["prompt"]
    say("  ok  SESSION header on round 1 only")


def test_every_round_is_self_contained():
    for r in _rows():
        p = r["prompt"]
        for section in ("TASK (", "RULES", "TOOL OUTPUTS", "OUTPUT FORMAT", "ANSWER:"):
            assert section in p, f"round {r['round_no']} missing {section}"
        assert ">>> " in p, "no tool call rendered"
    say("  ok  every round carries task, rules, tool output and answer format")


def test_rounds_numbered_consecutively():
    from collections import defaultdict

    by_chain = defaultdict(list)
    for r in _rows():
        by_chain[r["chain_id"]].append(r["round_no"])
    for cid, nums in by_chain.items():
        assert nums == list(range(1, len(nums) + 1)), f"{cid}: {nums}"
    say("  ok  round numbers are consecutive from 1")


def test_explanations_declare_dependency_and_ablations():
    rows = _rows()
    for r in rows:
        assert r["explanation"].startswith("(depends on:")
    # Ablations are the record of what each wrong path yields. Not every round
    # has a live distractor, but the chain as a whole must.
    assert any("ablations:" in r["explanation"] for r in rows)
    say("  ok  explanations declare dependencies, chain records ablations")


# --------------------------------------------------------------------------
# ground truth -- re-derived from the rendered prompt
# --------------------------------------------------------------------------


def _parse_table(prompt, call_prefix):
    """Pull (first column, float columns) out of a rendered tool block."""
    start = prompt.find(">>> " + call_prefix)
    assert start != -1, f"no tool block for {call_prefix}"
    nxt = prompt.find(">>> ", start + 1)
    end = nxt if nxt != -1 else prompt.find("OUTPUT FORMAT", start)
    block = prompt[start:end]

    rows = []
    for line in block.split("\n"):
        line = line.strip()
        if not line or line.startswith((">>>", "-")) or line.endswith("."):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        # Keep every numeric cell and drop the non-numeric ones, rather than
        # requiring the whole row to parse: some tables carry a second label
        # column (cell_line, compound, auc_pred, auc_true) and demanding all
        # floats would silently skip those rows entirely.
        vals = []
        for x in parts[1:]:
            try:
                vals.append(float(x))
            except ValueError:
                pass
        if not vals:
            continue
        rows.append((parts[0], vals))
    return rows


def test_round1_answer_is_the_lowest_usable_auc():
    """Re-derive: lowest auc among lines NOT in the QC table."""
    for r in _rows():
        if r["task_type"] != "sensitivity_ranking":
            continue
        p = r["prompt"]
        data = _parse_table(p, "db_lookup")

        qc_start = p.find(">>> qc_lookup")
        qc_block = p[qc_start:p.find("OUTPUT FORMAT", qc_start)]
        failed = {ln for ln, _ in data if re.search(rf"^\s*{re.escape(ln)}\s+failed_assay",
                                                    qc_block, re.MULTILINE)}

        usable = [(ln, v[0]) for ln, v in data if ln not in failed]
        expected = min(usable, key=lambda t: t[1])[0]
        assert r["correct_answer"] == expected, (
            f"{r['chain_id']} r1: said {r['correct_answer']}, table says {expected}"
        )
        assert r["correct_answer"] not in failed, "answer is a QC-failed line"
    say("  ok  round 1 answer == lowest usable auc, and never a QC-failed line")


def test_round2_answer_is_highest_core_auc():
    """Re-derive: highest auc among core-panel lines only."""
    for r in _rows():
        if r["task_type"] != "panel_restricted_rank":
            continue
        p = r["prompt"]
        data = _parse_table(p, "db_lookup")

        d_start = p.find(">>> design_lookup")
        panel = p[d_start:p.find("OUTPUT FORMAT", d_start)]
        core = {ln for ln, _ in data
                if re.search(rf"^\s*{re.escape(ln)}\s+core\s*$", panel, re.MULTILINE)}

        cands = [(ln, v[0]) for ln, v in data if ln in core]
        expected = max(cands, key=lambda t: t[1])[0]
        assert r["correct_answer"] == expected
        assert r["correct_answer"] in core, "answer is an observational line"
    say("  ok  round 2 answer == highest core-panel auc, never observational")


def test_round3_never_answers_the_attention_sink():
    """The sink dominates by construction; answering it means the schema rule
    was ignored. This is the cancer analogue of 'position 0 is never selected'."""
    n = 0
    for r in _rows():
        if r["task_type"] != "attention_attribution":
            continue
        n += 1
        p = r["prompt"]
        sink_match = re.search(r"^\s*(\S+)\s+housekeeping_sink", p, re.MULTILINE)
        assert sink_match, "no sink declared"
        sink = sink_match.group(1)
        assert r["correct_answer"] != sink, "answered the housekeeping sink"

        data = _parse_table(p, "attention_lookup")
        expected = next(g for g, _ in sorted(data, key=lambda t: -t[1][0]) if g != sink)
        assert r["correct_answer"] == expected
    assert n > 0, "no attention rounds generated"
    say(f"  ok  round 3 skips the attention sink and picks the top informative gene ({n} rounds)")


def test_round4_error_and_verdict_agree_with_the_table():
    for r in _rows():
        if r["task_type"] != "prediction_audit":
            continue
        p = r["prompt"]
        data = _parse_table(p, "db_lookup")
        assert data, "no prediction row"
        # row is [cell_line, compound, auc_pred, auc_true]; the compound token
        # is non-numeric so only the two floats survive the parse
        vals = data[0][1]
        pred, true = vals[-2], vals[-1]
        err, verdict = r["correct_answer"].split("; ")
        assert abs(float(err) - abs(pred - true)) < 0.005
        assert verdict == ("AGREES" if abs(pred - true) <= 0.10 else "DISAGREES")
    say("  ok  round 4 error and verdict re-derive from the table")


def test_answers_are_single_line():
    for r in _rows():
        assert "\n" not in r["correct_answer"]
    say("  ok  every answer is a single line")


# --------------------------------------------------------------------------
# tool_ranges
# --------------------------------------------------------------------------


def test_tool_ranges_empty_until_a_tokenizer_is_supplied():
    """Guessing offsets would produce a file that looks right and scores wrong."""
    for r in _rows():
        assert r["tool_ranges"] == ""
    say("  ok  tool_ranges left empty rather than guessed")


class FakeTokenizer:
    """Whitespace tokenizer with an offset mapping, like a HF fast tokenizer.

    ``add_tool_ranges`` needs real character offsets, not a token count: token
    counts do not compose across a boundary, so adding up the lengths of
    substrings gives the wrong answer. This mimics the interface it relies on.
    """

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets, pos = [], 0
        for word in text.split():
            start = text.index(word, pos)
            offsets.append((start, start + len(word)))
            pos = start + len(word)
        out = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def test_add_tool_ranges_produces_ordered_spans():
    """Bogdan: ranges are tokenizer- and history-dependent. Whatever the
    tokenizer, spans must be ordered, non-empty, and non-overlapping."""
    rows = _rows()
    toks = add_tool_ranges(rows, FakeTokenizer(), history=False)

    multi = 0
    for r in toks:
        if not r["tool_ranges"]:
            continue
        spans = [(int(a), int(b)) for a, b in
                 re.findall(r"\[(\d+):(\d+)\]", r["tool_ranges"])]
        assert spans, "range string produced but unparseable"
        assert all(a < b for a, b in spans), "empty or inverted span"
        if len(spans) > 1:
            multi += 1
            assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1)), \
                f"overlapping spans in {r['chain_id']} r{r['round_no']}"
    assert multi > 0, "no multi-range rows to check ordering on"
    say(f"  ok  token spans are ordered and non-overlapping ({multi} multi-range rows)")


def test_spans_actually_cover_the_tool_block():
    """The whole point: the span must land on the tool block, not near it.

    Reusing the CSV's own indices under a different tokenizer puts a block's
    start 15-143 tokens away. This asserts the recomputed spans do not.
    """
    rows = _rows()
    tok = FakeTokenizer()
    toks = add_tool_ranges(rows, tok, history=False)

    checked = 0
    for original, row in zip(rows, toks):
        if not row["tool_ranges"]:
            continue
        words = original["prompt"].split()
        for name, lo, hi in re.findall(r"(\w+)\[(\d+):(\d+)\]", row["tool_ranges"]):
            lo, hi = int(lo), int(hi)
            covered = " ".join(words[lo:hi])
            assert covered.startswith(">>>"), \
                f"span does not start at the marker: {covered[:40]!r}"
            assert name in covered.split("(")[0], \
                f"span {name} does not contain its own call: {covered[:60]!r}"
            checked += 1
    assert checked > 0
    say(f"  ok  every span starts at its >>> marker and contains its call ({checked})")


def test_history_mode_shifts_offsets_forward():
    """With history prepended, round N's offsets must sit past round N-1's --
    they index the accumulated conversation, not the round alone."""
    rows = _rows()
    tok = FakeTokenizer()
    solo = add_tool_ranges(rows, tok, history=False)
    hist = add_tool_ranges(rows, tok, history=True)

    def first(r):
        m = re.search(r"\[(\d+):", r["tool_ranges"])
        return int(m.group(1)) if m else None

    shifted = 0
    for s, h in zip(solo, hist):
        if s["round_no"] == 1:
            assert first(s) == first(h), "round 1 has no history to shift past"
        elif first(s) is not None and first(h) is not None:
            assert first(h) > first(s)
            shifted += 1
    assert shifted > 0
    say(f"  ok  history mode shifts later rounds forward ({shifted} rounds)")


def test_history_answer_choice_is_the_callers():
    """Ground-truth history measures per-prompt accuracy; the model's own
    answers measure per-chain accuracy. Both must be expressible."""
    rows = _rows()
    tok = FakeTokenizer()
    truth = add_tool_ranges(rows, tok, history=True)
    longer = add_tool_ranges(
        rows, tok, history=True,
        answer_template="\nANSWER: {answer} plus a much longer reply\n\n",
    )

    def first(r):
        m = re.search(r"\[(\d+):", r["tool_ranges"])
        return int(m.group(1)) if m else None

    moved = sum(
        1 for a, b in zip(truth, longer)
        if a["round_no"] > 1 and first(a) is not None and first(b) is not None
        and first(b) > first(a)
    )
    assert moved > 0, "a longer history must push later rounds further out"
    say(f"  ok  answer text changes later offsets ({moved} rounds)")


def _main():
    print("structure")
    test_columns_match_opal_exactly()
    test_session_header_only_on_round_one()
    test_every_round_is_self_contained()
    test_rounds_numbered_consecutively()
    test_explanations_declare_dependency_and_ablations()
    print("\nground truth (re-derived from the rendered prompt)")
    test_round1_answer_is_the_lowest_usable_auc()
    test_round2_answer_is_highest_core_auc()
    test_round3_never_answers_the_attention_sink()
    test_round4_error_and_verdict_agree_with_the_table()
    test_answers_are_single_line()
    print("\ntool_ranges")
    test_tool_ranges_empty_until_a_tokenizer_is_supplied()
    test_add_tool_ranges_produces_ordered_spans()
    test_spans_actually_cover_the_tool_block()
    test_history_mode_shifts_offsets_forward()
    test_history_answer_choice_is_the_callers()
    print("\nall checks passed")


if __name__ == "__main__":
    _main()
