# Known issues, and what is fixed where

Write-ups Jaime Cernuda produced against GH200 hardware, plus a note on the one
overlap that is *not* resolved.

| file | bug | status |
|---|---|---|
| `hybrid-group-block-table.md` | scored group got group 0's block table on hybrid models | fixed on `main` (`e513426`) |
| `query-head-count.md` | query head count read 4 instead of 24 on Qwen3.8-27B | fixed on `main` (`4a6bb44`) |
| — | Triton rejected a non-power-of-two block size (784) | fixed on `main` (`ba8b9e8`) |

All three were cherry-picked from `fix/hybrid-group-aware-block-table`. They
are independent of issue #2 and touch code no other fix touches, so they apply
cleanly.

## The unresolved overlap: issue #2

**Two people fixed the same bug, differently, within hours of each other, and
only one fix is on `main`.**

Issue #2 is that under CUDA graphs the probe's Python does not run on replay, so
every served record carried zero attention while passing every structural check.

| | Bogdan (on `main`, `8282506`) | Jaime (on the branch, `d35a1db`+`7ef41d4`) |
|---|---|---|
| freshness signal | `ran_eagerly` flag set by `note_ran()` | step generation + `torch.cuda.is_current_stream_capturing()` |
| buffer sizing | one per layer at `max_num_seqs` | one per layer, plus per-graph recorded-row tracking |
| reasoning recorded | inline comments | `docs/issues/` write-up citing GH200 job 3166664 |

They conflict across four files -- `probe.py` alone has four mutually exclusive
hunks, roughly 170 lines. Neither can be validated without a GPU and a served
vLLM engine, so **the choice is not being made here.** Jaime's version reasons
explicitly about every cudagraph mode vLLM ships; Bogdan's is simpler and
already shipped. That is a judgement for the two of them.

The branch remains available on this fork as
`fix/hybrid-group-aware-block-table` with its history intact.

## Why these bugs are worth reading even if you never hit them

Every one of them is silent. The wrong block table returns a well-formed record
with zero retained positions. The wrong head count scores 4 of 24 heads and
still emits a valid softmax over real keys. The CUDA graph bug conserves mass,
because 0 + 1 == 1.

That is the recurring shape of failure in this codebase, and the reason the
tests assert on values rather than on shapes.
