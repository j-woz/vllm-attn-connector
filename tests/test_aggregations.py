#!/usr/bin/env python3
"""Unit checks for the per-position head reduction, no GPU or vLLM needed.

The connector reduces each decode step's [L, H, T] attention to two [T] vectors
plus a winner id, accumulating layer by layer so the full tensor is never
materialised. These tests replicate that streaming reduction and compare it
against the obvious dense computation.

Also pins the two facts that decided the emitted schema:

  * a max over the top K heads equals the max over all heads identically, so
    such a column could never carry information;
  * a mean over the top K heads *is* distinct from both emitted aggregations --
    it is a real statistic that was left out on cost grounds, not because it is
    redundant. The test keeps that distinction checkable.

Run: python experiments/vllm-attn-connector/test_aggregations.py
"""
from __future__ import annotations

import sys

import torch


def streaming(layers: list[torch.Tensor]):
    """What score_step does: one layer at a time, O(T) state."""
    T = layers[0].shape[1]
    acc = torch.zeros(T, dtype=torch.float32)
    run_max = torch.full((T,), float("-inf"), dtype=torch.float32)
    owner = torch.full((T,), -1, dtype=torch.int64)
    for li, probs in enumerate(layers):
        acc += probs.mean(0)
        lmax, hidx = probs.max(0)
        better = lmax > run_max
        owner = torch.where(better, li * probs.shape[0] + hidx, owner)
        run_max = torch.maximum(run_max, lmax)
    return {"val_all_max": run_max,
            "val_all_avg": acc / len(layers),
            "topk_head": owner}


def dense(layers: list[torch.Tensor]):
    """The definition, with the whole [L*H, T] tensor in memory."""
    a = torch.cat(layers, 0)
    v, i = a.max(0)
    return {"val_all_max": v, "val_all_avg": a.mean(0), "topk_head": i}


def case(name, L, H, T, mk):
    layers = [mk(H, T) for _ in range(L)]
    got, want = streaming(layers), dense(layers)
    worst = 0.0
    for key in ("val_all_max", "val_all_avg"):
        d = (got[key] - want[key]).abs().max().item()
        worst = max(worst, d)
        assert d < 1e-6, f"{name}/{key}: streaming != dense, max diff {d:g}"
    # The winner is compared by the value it carries: ties are broken
    # arbitrarily and a tied swap is not an error.
    a = torch.cat(layers, 0)
    gv = torch.gather(a, 0, got["topk_head"].clamp_min(0).unsqueeze(0))[0]
    dv = (gv - want["val_all_max"]).abs().max().item()
    assert dv < 1e-6, f"{name}/topk_head: winner does not carry the max ({dv:g})"
    exact = (got["topk_head"] == want["topk_head"]).float().mean().item()
    print(f"  ok  {name:<34} L={L:<3} H={H:<4} T={T:<5} "
          f"max|diff|={worst:.2e}  winner id exact {exact:.0%}")


def test_reduction() -> None:
    """Streaming layer-by-layer reduction must equal the dense definition."""
    torch.manual_seed(0)
    rnd = lambda H, T: torch.rand(H, T)
    print("streaming layer-by-layer reduction == dense reduction")
    case("uniform random", 36, 32, 512, rnd)
    case("single layer", 1, 32, 128, rnd)
    case("single head", 8, 1, 64, rnd)
    case("one hot per column", 12, 8, 64,
         lambda H, T: torch.nn.functional.one_hot(
             torch.randint(0, H, (T,)), H).T.float())
    case("softmax rows (realistic)", 36, 32, 384,
         lambda H, T: torch.softmax(torch.randn(H, T) * 3, dim=1))

    print("\nwhy there is no val_topk_max column")
    for kh in (1, 2, 3, 8, 64):
        a = torch.rand(1152, 97)
        assert torch.equal(a.topk(kh, dim=0).values[0], a.max(0).values), kh
        print(f"  ok  K={kh:<4} max over top-K == max over all heads, all 97 positions")

    print("\nwhy mean-over-top-K is distinct, yet not emitted")
    a = torch.rand(1152, 97)
    lo, mid, hi = a.mean(0), a.topk(3, dim=0).values.mean(0), a.max(0).values
    assert (lo <= mid + 1e-6).all() and (mid <= hi + 1e-6).all()
    strict = ((lo < mid - 1e-6) & (mid < hi - 1e-6)).float().mean().item()
    print(f"  ok  all_avg <= topk_avg <= all_max, strict at {strict:.0%} of positions")
    print("      -> a real third statistic, but it is only interpretable with the")
    print("         K head identities alongside it, and carrying per-head state")
    print("         through the reduction is what makes capture expensive.")
    print("         Only the winner (topk_head) is emitted; it is free.")


def main() -> int:
    test_reduction()
    print()
    test_segments()
    print()
    test_group_block_ids()
    print()
    test_graph_key()
    print()
    test_registry_freshness()
    print("\nall checks passed")
    return 0


# --------------------------------------------------------------------------
# Query registry keying and freshness. The probe half of the CUDA-graph fix,
# and the part that has no GPU in it.
# --------------------------------------------------------------------------

def _load_probe():
    """Pull QueryRegistry and graph_key_of out of probe.py without importing it.

    probe.py imports vllm.logger at module scope; these two do not need it.
    """
    import ast as _ast
    import pathlib
    import threading
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/vllm_attn_connector/probe.py").read_text()
    tree = _ast.parse(src)
    want = {"QueryRegistry", "graph_key_of", "_capturing"}
    mod = _ast.Module(body=[n for n in tree.body
                            if isinstance(n, (_ast.ClassDef, _ast.FunctionDef))
                            and n.name in want],
                      type_ignores=[])

    class _Log:
        def warning(self, *a, **k):
            pass

    ns = {"torch": torch, "threading": threading, "logger": _Log(), "Any": object}
    exec(compile(mod, "<probe>", "exec"), ns)
    missing = want - set(ns)
    assert not missing, f"probe.py no longer defines {sorted(missing)}"
    return ns["QueryRegistry"], ns["graph_key_of"]


class _Desc:
    """Stands in for vllm.forward_context.BatchDescriptor: frozen, so it is
    hashable and compares by value, which is what makes a capture-time key and
    a replay-time key the same key."""

    __slots__ = ("num_tokens", "uniform")

    def __init__(self, num_tokens, uniform=True):
        object.__setattr__(self, "num_tokens", num_tokens)
        object.__setattr__(self, "uniform", uniform)

    def __eq__(self, other):
        return (isinstance(other, _Desc) and other.num_tokens == self.num_tokens
                and other.uniform == self.uniform)

    def __hash__(self):
        return hash((self.num_tokens, self.uniform))

    def __repr__(self):
        return f"_Desc({self.num_tokens}, {self.uniform})"


class _Mode:
    def __init__(self, name, value):
        self.name, self.value = name, value

    def __eq__(self, other):
        return isinstance(other, _Mode) and other.value == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return self.name


NONE, PIECEWISE, FULL = _Mode("NONE", 0), _Mode("PIECEWISE", 1), _Mode("FULL", 2)


class _Ctx:
    def __init__(self, mode, desc):
        self.cudagraph_runtime_mode, self.batch_descriptor = mode, desc


def test_graph_key() -> None:
    _, graph_key_of = _load_probe()
    print("cudagraph key derived from the forward context")

    assert graph_key_of(None) is None
    assert graph_key_of(_Ctx(NONE, _Desc(7, False))) is None, \
        "CUDAGraphMode.NONE means no graph will be replayed"
    assert graph_key_of(_Ctx(FULL, None)) is None
    print("  ok  no graph in play -> None, so the eager path keeps one entry")

    k = graph_key_of(_Ctx(FULL, _Desc(16)))
    assert k == graph_key_of(_Ctx(FULL, _Desc(16))), \
        "capture and replay of one graph must produce equal keys"
    assert hash(k) == hash(graph_key_of(_Ctx(FULL, _Desc(16))))
    print("  ok  the same graph gives the same key at capture and at replay")

    assert k != graph_key_of(_Ctx(FULL, _Desc(32))), "one graph per padded size"
    assert k != graph_key_of(_Ctx(PIECEWISE, _Desc(16))), \
        "piecewise and full are different graphs at the same size"
    print("  ok  padded size and runtime mode both separate graphs")


def test_registry_freshness() -> None:
    QueryRegistry, _ = _load_probe()
    print("query registry: persistent buffers, per-step freshness")

    reg = QueryRegistry()
    reg.configure(8)
    q = torch.arange(8 * 2 * 3, dtype=torch.float32).reshape(8, 2, 3)

    # --- eager: fresh only for the step it ran on -------------------------
    reg.put("l0", q, 3, None)
    assert reg.get("l0", None) is not None, "a live write is readable this step"
    assert torch.equal(reg.get("l0", None)[:3], q[:3])
    reg.end_step()
    assert reg.get("l0", None) is None, \
        "a live write must not survive into the next step"
    print("  ok  eager write is fresh for its own step and stale after it")

    # --- graph: the recorded copy makes it fresh on every replay ----------
    gk = ("FULL", 16)
    reg.put("l0", q, 8, gk)
    for _ in range(50):
        reg.end_step()
    assert reg.get("l0", gk) is not None, \
        "a graph's copy kernel reruns on every replay; the key must stay valid"
    assert reg.get("l0", ("FULL", 32)) is None, \
        "a different graph's key must not read this buffer"
    assert reg.get("l0", None) is None, "an eager step must not read a graph write"
    assert reg.get("nosuch", gk) is None
    print("  ok  graph key stays fresh across steps; other keys do not match")

    # --- the buffer is the same object, which is what replay depends on ---
    before = reg.get("l0", gk)
    reg.put("l0", q * 2, 8, gk)
    after = reg.get("l0", gk)
    assert before.data_ptr() == after.data_ptr(), \
        "the buffer must be written in place, not replaced"
    assert torch.equal(after, q * 2)
    print("  ok  writes land in the same buffer, in place")

    # --- configure() sizes it once, so capture never has to grow it -------
    assert reg.get("l0", gk).shape[0] == 8, "configure(8) sizes the buffer to 8"
    reg.put("l0", q, 2, gk)
    assert reg.get("l0", gk).data_ptr() == after.data_ptr(), \
        "a smaller batch must not reallocate"
    print("  ok  a smaller batch reuses the buffer")

    # --- growing after capture costs the graphs recorded against the old --
    small = QueryRegistry()
    small.put("l0", q, 2, gk)
    assert small.get("l0", gk) is not None
    gk2 = ("FULL", 64)
    small.put("l0", q, 8, gk2)
    assert small.get("l0", gk) is None, \
        "graphs recorded against the old buffer no longer write the new one"
    assert small.get("l0", gk2) is not None
    print("  ok  a buffer that has to grow drops the keys it invalidates")


# --------------------------------------------------------------------------
# Per-KV-cache-group block tables.
# --------------------------------------------------------------------------

def test_group_block_ids() -> None:
    """Each scored group must be handed its OWN group's block ids.

    StepRequest.new_block_ids carries one block-id list per KV cache group, in
    kv_cache_groups order. Ground truth from a dump of Qwen/Qwen3.8-27B on vLLM
    0.28.0 (deploy/runs/kvnorm/dump-groups-3044670.out): 4 groups, 0-2 MambaSpec
    (48 GDN layers, num_kv_heads=None) and 3 FullAttentionSpec (16 layers,
    num_kv_heads=4, head_size=256, block_size=784). Only group 3 resolves an
    attention layout, so the scored gids are [3] and index 0 is a Mamba table.
    """
    fn = _load_fns({"_group_block_ids"})["_group_block_ids"]
    print("per-group block tables")

    qwen = ([11, 12], [21, 22], [31, 32], [41, 42, 43])
    assert fn(qwen, [3]) == [[41, 42, 43]], "hybrid: must read the attention group"
    assert fn(qwen, [0]) == [[11, 12]], "index 0 is the Mamba table (the old bug)"
    print("  ok  hybrid Qwen3.8-27B shape: gid 3 -> group 3's blocks, not group 0's")

    dense = ([5, 6, 7],)
    assert fn(dense, [0]) == [[5, 6, 7]], "dense: single group at index 0"
    print("  ok  dense single group unchanged")

    assert fn(qwen, [1, 3]) == [[21, 22], [41, 42, 43]], "scored order preserved"
    assert fn(qwen, [3, 1]) == [[41, 42, 43], [21, 22]], "scored order preserved"
    print("  ok  several scored groups keep scored order, each with its own list")

    assert fn(qwen, [9]) == [[]], "gid past the tuple -> nothing to extend"
    assert fn((), [0]) == [[]], "empty tuple -> nothing to extend"
    print("  ok  out-of-range and empty tuples degrade to no blocks")

    out = fn(qwen, [3])
    out[0].append(99)
    assert qwen[3] == [41, 42, 43], "result must not alias the scheduler's lists"
    print("  ok  returned lists are copies, so extending never mutates the metadata")


# --------------------------------------------------------------------------
# Segment planning and selection. Imported from the connector source directly:
# the module needs vLLM to import, these two functions do not.
# --------------------------------------------------------------------------

def _load_fns(want: set[str]) -> dict:
    """Pull named top-level functions out of connector.py without importing it."""
    import ast as _ast
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/vllm_attn_connector/connector.py").read_text()
    tree = _ast.parse(src)
    mod = _ast.Module(body=[n for n in tree.body
                            if isinstance(n, _ast.FunctionDef) and n.name in want],
                      type_ignores=[])
    ns = {"torch": torch, "NO_ENTRY": -1}
    exec(compile(mod, "<connector>", "exec"), ns)
    missing = want - set(ns)
    assert not missing, f"connector.py no longer defines {sorted(missing)}"
    return ns


def _load_segment_fns():
    ns = _load_fns({"_segment_plan", "_segment_index", "_segmented_topk"})
    return ns["_segment_plan"], ns["_segment_index"], ns["_segmented_topk"]


def test_segments() -> None:
    plan_fn, index_fn, topk_fn = _load_segment_fns()
    torch.manual_seed(0)

    def run(n_keys, chunk, pct, ranges, label):
        plan = plan_fn(n_keys, chunk, pct, ranges)
        idx, live, want, total = index_fn(plan, "cpu")
        kmax = max(k for _, _, k in plan)
        row = torch.rand(n_keys)
        pos, val = topk_fn(row, idx, live, want, total, kmax)
        p = [int(x) for x in pos.tolist() if x >= 0]

        # the partition is complete, ordered and non-overlapping
        assert plan[0][0] == 0 and plan[-1][1] == n_keys, f"{label}: not covering"
        for (a, b, _), (c, d, _) in zip(plan, plan[1:]):
            assert b == c, f"{label}: gap/overlap at {b}!={c}"
        # positions are legal, unique, ascending, never the sink
        assert all(0 < x < n_keys for x in p), f"{label}: out of range or sink"
        assert len(set(p)) == len(p), f"{label}: duplicates"
        assert p == sorted(p), f"{label}: not ascending"
        # each segment got exactly its quota (minus the sink it cannot use)
        for lo, hi, keepn in plan:
            got = sum(1 for x in p if lo <= x < hi)
            cap = min(keepn, (hi - lo) - (1 if lo == 0 else 0))
            assert got == cap, f"{label}: segment [{lo},{hi}) got {got}, wanted {cap}"
        # values match the row at the positions claimed
        for x, v in zip(pos.tolist(), val.tolist()):
            if x >= 0:
                assert abs(row[x].item() - v) < 1e-6, f"{label}: value mismatch"
        print(f"  ok  {label:<44} segs={len(plan):<4} k={total}")
        return plan

    print("segment planning and selection")
    run(200, 32, 10, None, "fixed, chunk 32, 10%")
    run(200, 1, 100, None, "fixed, chunk 1, 100% (every position)")
    run(200, 512, 10, None, "fixed, chunk wider than prompt")
    run(37, 32, 10, None, "fixed, ragged final chunk")
    run(200, 32, 10, [(10, 60), (80, 95), (120, 190)], "variable, 3 ranges + gaps")
    run(200, 32, 10, [(0, 200)], "variable, one range spanning all")
    run(200, 32, 10, [(0, 50), (50, 200)], "variable, adjacent, no gaps")
    run(200, 32, 10, [(150, 400)], "variable, range past the prompt end")
    run(200, 32, 10, [(20, 80), (50, 120)], "variable, overlapping (truncated)")
    run(200, 32, 0, None, "top_pct=0 -> floor of 1 per segment")

    # fixed mode must reproduce the uniform grid exactly
    plan = plan_fn(200, 32, 10, None)
    grid = [(i, min(i + 32, 200)) for i in range(0, 200, 32)]
    assert [(lo, hi) for lo, hi, _ in plan] == grid, "grid mismatch"
    print("  ok  fixed mode reproduces the uniform grid")

    # declaring ranges that happen to be the grid == fixed mode
    same = plan_fn(200, 32, 10, [(i, min(i + 32, 200)) for i in range(0, 200, 32)])
    assert same == plan, "declaring the grid as ranges should equal fixed mode"
    print("  ok  declaring the grid as ranges is identical to fixed mode")


if __name__ == "__main__":
    sys.exit(main())
