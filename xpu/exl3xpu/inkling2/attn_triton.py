"""Inkling 2 attention on XPU (Triton): flash attention with a banded relative-position bias and a per-query sink.

score(q, k) = q.k * scale + table[q, h, dist]   if 0 <= dist < extent   (dist = qpos - kpos)
            = q.k * scale                       if dist >= extent       (global layers only)
            = -inf                              if dist < 0, kpos < 0, or (SWA) dist >= window
p = exp(score) / (exp(sink[q, h]) + sum exp(score))       (the sink has no value)

Ragged batch: queries of request b are rows q_start[b] : q_start[b+1]; their keys are K[kv_loc[kv_start[b]:kv_start[b+1]]]
with positions kv_pos[...] (paged pool, ring buffer or a contiguous scratch -- anything addressable by row index).
Long key ranges are split across programs (flash-decoding) and merged with log-sum-exp.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, KV_LOC, KV_POS, Q_POS, Q_REQ, KV_START, TABLE, SINK,
    O_ACC, O_M, O_L,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh,
    stride_tt, stride_th, stride_ot, stride_oh, stride_os, stride_mt, stride_mh,
    scale, k_scale, v_scale, window, n_q, split_len,
    H_PER_KV: tl.constexpr, D: tl.constexpr, EXTENT: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, IS_SWA: tl.constexpr, KV_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    hk = h // H_PER_KV
    rows = pid_m * BM + tl.arange(0, BM)
    rmask = rows < n_q
    dd = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + h * stride_qh + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.bfloat16)
    qpos = tl.load(Q_POS + rows, mask=rmask, other=-1)
    req = tl.load(Q_REQ + rows, mask=rmask, other=0)
    # all rows of a block belong to one request (caller pads per request)
    b = tl.min(tl.where(rmask, req, 2147483647), axis=0)
    ks = tl.load(KV_START + b)
    ke = tl.load(KV_START + b + 1)
    lo = ks + sp * split_len
    hi = tl.minimum(ke, lo + split_len)
    qmax = tl.max(tl.where(rmask, qpos, -1), axis=0)
    qmin = tl.min(tl.where(rmask, qpos, 2147483647), axis=0)
    if sp == 0:
        m_i = tl.load(SINK + rows * stride_mt + h * stride_mh, mask=rmask, other=0.0).to(tl.float32)
        l_i = tl.where(rmask, 1.0, 0.0)
    else:
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    for start in range(lo, hi, BN):
        cols = start + tl.arange(0, BN)
        cmask = cols < hi
        loc = tl.load(KV_LOC + cols, mask=cmask, other=0)
        kpos = tl.load(KV_POS + cols, mask=cmask, other=-1)
        kmin = tl.min(tl.where(cmask & (kpos >= 0), kpos, 2147483647), axis=0)
        kmax = tl.max(tl.where(cmask, kpos, -1), axis=0)
        skip = (kmin > qmax) | (kmax < 0)
        if IS_SWA:
            skip = skip | (qmin - kmax >= window)
        if not skip:
            k = tl.load(K + loc[:, None] * stride_kt + hk * stride_kh + dd[None, :], mask=cmask[:, None], other=0.0)
            v = tl.load(V + loc[:, None] * stride_vt + hk * stride_vh + dd[None, :], mask=cmask[:, None], other=0.0)
            if KV_FP8:
                k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
                v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)
            s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * scale
            dist = qpos[:, None] - kpos[None, :]
            valid = (dist >= 0) & (kpos[None, :] >= 0) & cmask[None, :] & rmask[:, None]
            if IS_SWA:
                valid = valid & (dist < window)
            near = valid & (dist < EXTENT)
            if (qmax - kmin) >= 0:
                if (qmin - kmax) < EXTENT:
                    bias = tl.load(TABLE + rows[:, None] * stride_tt + h * stride_th + tl.minimum(tl.maximum(dist, 0), EXTENT - 1),
                                   mask=near, other=0.0).to(tl.float32)
                    s = s + bias
            s = tl.where(valid, s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            m_i = m_new
    o_off = rows[:, None] * stride_ot + h * stride_oh + sp * stride_os + dd[None, :]
    tl.store(O_ACC + o_off, acc, mask=rmask[:, None])
    ml = rows * (stride_ot // D) + h * (stride_oh // D) + sp
    tl.store(O_M + ml, m_i, mask=rmask)
    tl.store(O_L + ml, l_i, mask=rmask)


def inkling2_attention(
    q: torch.Tensor,            # [Tq, H, D]
    k: torch.Tensor,            # [N, Hkv, D] (rows addressable by kv_loc)
    v: torch.Tensor,
    kv_loc: torch.Tensor,       # [Lsum] int32/int64 row index into k/v
    kv_pos: torch.Tensor,       # [Lsum] int32 position of each key (-1 = empty)
    kv_start: torch.Tensor,     # [B+1] int32 offsets into kv_loc per request
    q_pos: torch.Tensor,        # [Tq] int32
    q_req: torch.Tensor,        # [Tq] int32 request index of each query row
    table: torch.Tensor,        # [Tq, H, extent] relative bias (any float dtype)
    sink: torch.Tensor,         # [Tq, H] fp32
    *, window: int = 0, scale: float | None = None, k_scale: float = 1.0, v_scale: float = 1.0,
    max_kv: int | None = None, block_m: int = 16, block_n: int = 64, split_len: int = 4096,
) -> torch.Tensor:
    """Returns [Tq, H, D] fp32. Query rows of one request must be contiguous; blocks never mix requests
    because the caller pads each request's rows to a multiple of block_m (see pad_rows) -- or uses block_m=1."""
    Tq, H, D = q.shape
    Hkv = k.shape[1]
    extent = table.shape[-1]
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    if max_kv is None:
        max_kv = int((kv_start[1:] - kv_start[:-1]).max().item()) if kv_start.numel() > 1 else 0
    n_split = max(1, triton.cdiv(max_kv, split_len))
    acc = torch.empty(Tq, H, n_split, D, device=q.device, dtype=torch.float32)
    m = torch.empty(Tq, H, n_split, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    fp8 = k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    grid = (triton.cdiv(Tq, block_m), H, n_split)
    _attn_fwd[grid](
        q, k, v, kv_loc, kv_pos, q_pos, q_req, kv_start, table, sink,
        acc, m, l,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        table.stride(0), table.stride(1), acc.stride(0), acc.stride(1), acc.stride(2), sink.stride(0), sink.stride(1),
        scale, k_scale, v_scale, window, Tq, split_len,
        H_PER_KV=H // Hkv, D=D, EXTENT=extent, BM=block_m, BN=block_n, IS_SWA=window > 0, KV_FP8=fp8,
    )
    if n_split == 1:
        return acc[:, :, 0] / l[:, :, 0, None]
    mm = m.max(-1, keepdim=True).values
    mm = torch.where(torch.isinf(mm), torch.zeros_like(mm), mm)
    w = torch.exp(m - mm)
    return (acc * w[..., None]).sum(2) / (l * w).sum(-1)[..., None]


def reference_attention(q, k, v, kv_pos, q_pos, table, sink, window=0, scale=None):
    """Dense reference for one request. q [T,H,D], k/v [L,Hkv,D] fp32-able, kv_pos [L], q_pos [T]."""
    T, H, D = q.shape
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    g = H // k.shape[1]
    K = k.float().repeat_interleave(g, 1)
    V = v.float().repeat_interleave(g, 1)
    s = torch.einsum("thd,lhd->thl", q.float(), K) * scale
    dist = q_pos[:, None].long() - kv_pos[None, :].long()
    valid = (dist >= 0) & (kv_pos[None, :] >= 0)
    if window:
        valid &= dist < window
    extent = table.shape[-1]
    near = valid & (dist < extent)
    idx = dist.clamp(0, extent - 1)
    bias = torch.gather(table.float(), 2, idx[:, None, :].expand(T, H, -1))
    s = s + torch.where(near[:, None, :], bias, torch.zeros_like(bias))
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    s = torch.cat([s, sink.float()[:, :, None]], -1)
    p = s.softmax(-1)[..., :-1]
    return torch.einsum("thl,lhd->thd", p, V)


def pad_rows(counts: list[int], block_m: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Rows of each request start on a block_m boundary. Returns (src index per padded row or -1, req per padded row, n_pad)."""
    src, req = [], []
    off = 0
    for b, c in enumerate(counts):
        n = -(-c // block_m) * block_m if c else 0
        src += list(range(off, off + c)) + [-1] * (n - c)
        req += [b] * n
        off += c
    return torch.tensor(src, device=device), torch.tensor(req, device=device, dtype=torch.int32), len(src)


def attention_ragged(q, k, v, kv_loc, kv_pos, kv_start, q_pos, q_counts, table, sink, *, block_m=16, **kw):
    """q rows grouped by request (q_counts[b] rows each). Pads per request so no block mixes requests."""
    src, req, n = pad_rows(q_counts, block_m, q.device)
    take = src.clamp(min=0)
    qp = q[take]
    pos = torch.where(src >= 0, q_pos[take], torch.full_like(q_pos[take], -1)).to(torch.int32)
    out = inkling2_attention(qp, k, v, kv_loc, kv_pos, kv_start, pos, req, table[take], sink[take].float(), block_m=block_m, **kw)
    return out[src >= 0]
