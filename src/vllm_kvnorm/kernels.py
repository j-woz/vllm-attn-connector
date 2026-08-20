# SPDX-License-Identifier: Apache-2.0
"""Per-token importance scoring over vLLM's paged KV cache.

Implements the PagedEviction proxy (Chitty-Venkata et al., Findings of EACL 2026;
arXiv:2509.04377), Algorithm 1, token mode, averaged over KV heads:

    S_i = mean_h ( ||V_i,h||_2 / ||K_i,h||_2 )

High ``S`` means important: per Devoto et al. 2024 (arXiv:2406.11430) a key's L2
norm is inversely proportional to that token's cumulative attention score.

One program per token reduces the whole ``(kv_heads, head_size)`` tile to a
single float, so the caller never materialises per-head intermediates.
"""

from __future__ import annotations

import torch

try:
    from vllm.triton_utils import tl, triton

    HAS_TRITON = True
except Exception:  # pragma: no cover - exercised only outside vLLM
    try:
        import triton
        import triton.language as tl

        HAS_TRITON = True
    except Exception:
        triton = None  # type: ignore[assignment]
        tl = None  # type: ignore[assignment]
        HAS_TRITON = False

_EPS = 1e-6


if HAS_TRITON:

    @triton.jit
    def _score_kernel(
        k_ptr, v_ptr, block_table_ptr, out_ptr,
        start_token,
        k_sb, k_sn, k_sh, k_sd,
        v_sb, v_sn, v_sh, v_sd,
        BLOCK_SIZE: tl.constexpr,
        NUM_HEADS: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HEAD_SIZE_V: tl.constexpr,
        PAD_H: tl.constexpr,
        PAD_D: tl.constexpr,
        PAD_DV: tl.constexpr,
    ):
        tok = tl.program_id(0) + start_token
        # Paged indirection: token -> logical block -> physical block id.
        block_id = tl.load(block_table_ptr + tok // BLOCK_SIZE).to(tl.int64)
        offset = tok % BLOCK_SIZE

        h = tl.arange(0, PAD_H)[:, None]
        h_ok = tl.arange(0, PAD_H) < NUM_HEADS

        d = tl.arange(0, PAD_D)[None, :]
        k = tl.load(
            k_ptr + block_id * k_sb + offset * k_sn + h * k_sh + d * k_sd,
            mask=(h < NUM_HEADS) & (d < HEAD_SIZE),
            other=0.0,
        ).to(tl.float32)

        dv = tl.arange(0, PAD_DV)[None, :]
        v = tl.load(
            v_ptr + block_id * v_sb + offset * v_sn + h * v_sh + dv * v_sd,
            mask=(h < NUM_HEADS) & (dv < HEAD_SIZE_V),
            other=0.0,
        ).to(tl.float32)

        k_norm = tl.sqrt(tl.sum(k * k, axis=1))
        v_norm = tl.sqrt(tl.sum(v * v, axis=1))
        # Epsilon inlined: a @triton.jit body cannot read module globals.
        ratio = v_norm / tl.maximum(k_norm, 1e-6)

        tl.store(out_ptr + tok, tl.sum(tl.where(h_ok, ratio, 0.0)) / NUM_HEADS)


def token_scores(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    *,
    out: torch.Tensor | None = None,
    start_token: int = 0,
) -> torch.Tensor:
    """Score tokens ``[start_token, num_tokens)`` for one layer.

    Args:
        k_view: ``(num_blocks, block_size, num_kv_heads, head_size)`` view.
        v_view: ``(num_blocks, block_size, num_kv_heads, head_size_v)`` view.
        block_table: 1-D tensor of physical block ids, on device.
        num_tokens: Exclusive upper bound of the token range.
        out: Optional ``(>=num_tokens,)`` float32 buffer to accumulate into,
            letting a caller score a sequence incrementally across steps.
        start_token: Inclusive lower bound of the token range.
    """
    _, block_size, num_heads, head_size = k_view.shape
    head_size_v = v_view.shape[3]
    count = num_tokens - start_token
    if out is None:
        out = torch.empty(num_tokens, dtype=torch.float32, device=k_view.device)
    if count <= 0:
        return out

    if not HAS_TRITON:
        ref = token_scores_torch(
            k_view, v_view, block_table, num_tokens, start_token=start_token
        )
        out[start_token:num_tokens] = ref
        return out

    _score_kernel[(count,)](
        k_view, v_view, block_table, out,
        start_token,
        *k_view.stride(), *v_view.stride(),
        BLOCK_SIZE=block_size,
        NUM_HEADS=num_heads,
        HEAD_SIZE=head_size,
        HEAD_SIZE_V=head_size_v,
        PAD_H=triton.next_power_of_2(num_heads),
        PAD_D=triton.next_power_of_2(head_size),
        PAD_DV=triton.next_power_of_2(head_size_v),
    )
    return out


def token_scores_torch(
    k_view: torch.Tensor,
    v_view: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    *,
    start_token: int = 0,
) -> torch.Tensor:
    """Pure-torch reference for :func:`token_scores`.

    Returns just the ``[start_token, num_tokens)`` slice.
    """
    block_size = k_view.shape[1]
    tok = torch.arange(start_token, num_tokens, device=k_view.device)
    blk = block_table.to(torch.long)[tok // block_size]
    off = tok % block_size

    k = torch.linalg.vector_norm(k_view[blk, off].float(), dim=-1)
    v = torch.linalg.vector_norm(v_view[blk, off].float(), dim=-1)
    return (v / k.clamp_min(_EPS)).mean(dim=-1)
