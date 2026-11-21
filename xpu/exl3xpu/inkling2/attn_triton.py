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


# ---- graph-capturable decode / verify attention ---------------------------------------------------------------
# Every request has N_Q query rows (1 = decode, draft width = verify), rows b*N_Q .. b*N_Q+N_Q-1.
# PAGED (global layers): key j of request b lives at pool row req_to_token[slot_b, j], position j, j < kv_len[b].
# RING  (SWA layers):    key j lives at row slot_b*RING + j with position ring_pos[slot_b, j] (-1 = empty), j < RING.
# Keys are split into NSPLIT equal parts per request (chunk = ceil(len / NSPLIT)); results merged by LSE.
@triton.jit
def _attn_dec(
    Q, K, V, R2T, SLOTS, KVLEN, RPOS, Q_POS, TABLE, SINK, O_ACC, O_M, O_L,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh, stride_r2t,
    stride_tt, stride_th, stride_st, stride_sh,
    scale, k_scale, v_scale, window, ring,
    N_Q: tl.constexpr, H: tl.constexpr, H_PER_KV: tl.constexpr, D: tl.constexpr, EXTENT: tl.constexpr,
    NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PAGED: tl.constexpr, KV_FP8: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    hk = h // H_PER_KV
    r = tl.arange(0, BM)
    rmask = r < N_Q
    rows = b * N_Q + r
    dd = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + h * stride_qh + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.bfloat16)
    qpos = tl.load(Q_POS + rows, mask=rmask, other=-1)
    slot = tl.load(SLOTS + b).to(tl.int64)
    if PAGED:
        L = tl.load(KVLEN + b)
    else:
        L = ring
    chunk = (L + NSPLIT - 1) // NSPLIT
    chunk = (chunk + BN - 1) // BN * BN
    lo = sp * chunk
    hi = tl.minimum(L, lo + chunk)
    if sp == 0:
        m_i = tl.load(SINK + rows * stride_st + h * stride_sh, mask=rmask, other=0.0).to(tl.float32)
        l_i = tl.where(rmask, 1.0, 0.0)
    else:
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qmax = tl.max(tl.where(rmask, qpos, -1), axis=0)
    for start in range(lo, hi, BN):
        cols = start + tl.arange(0, BN)
        cmask = cols < hi
        if PAGED:
            loc = tl.load(R2T + slot * stride_r2t + cols, mask=cmask, other=0).to(tl.int64)
            kpos = tl.where(cmask, cols, -1)
        else:
            loc = slot * ring + cols
            kpos = tl.load(RPOS + slot * ring + cols, mask=cmask, other=-1)
        k = tl.load(K + loc[:, None] * stride_kt + hk * stride_kh + dd[None, :], mask=cmask[:, None], other=0.0)
        v = tl.load(V + loc[:, None] * stride_vt + hk * stride_vh + dd[None, :], mask=cmask[:, None], other=0.0)
        if KV_FP8:
            k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
            v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * scale
        dist = qpos[:, None] - kpos[None, :]
        valid = (dist >= 0) & (kpos[None, :] >= 0) & rmask[:, None]
        if window > 0:
            valid = valid & (dist < window)
        near = valid & (dist < EXTENT)
        bias = tl.load(TABLE + rows[:, None] * stride_tt + h * stride_th + tl.minimum(tl.maximum(dist, 0), EXTENT - 1),
                       mask=near, other=0.0).to(tl.float32)
        s = tl.where(valid, s + bias, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
        m_i = m_new
    o = (rows[:, None] * H + h) * NSPLIT + sp
    tl.store(O_ACC + o * D + dd[None, :], acc, mask=rmask[:, None])
    ml = (rows * H + h) * NSPLIT + sp
    tl.store(O_M + ml, m_i, mask=rmask)
    tl.store(O_L + ml, l_i, mask=rmask)


def attention_decode(q, k, v, slots, q_pos, table, sink, *, n_q: int, paged: bool, r2t=None, kv_len=None,
                     ring_pos=None, ring: int = 0, window: int = 0, nsplit: int = 8, k_scale=1.0, v_scale=1.0,
                     block_n: int = 64):
    """q [bs*n_q, H, D]; returns [bs*n_q, H, D] fp32. No host syncs; static shapes (graph-capturable)."""
    T, H, D = q.shape
    bs = slots.shape[0]
    acc = torch.empty(T, H, nsplit, D, device=q.device, dtype=torch.float32)
    m = torch.empty(T, H, nsplit, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    fp8 = k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    dummy = slots
    _attn_dec[(bs, H, nsplit)](
        q, k, v, r2t if paged else dummy, slots, kv_len if paged else dummy, ring_pos if not paged else dummy, q_pos,
        table, sink, acc, m, l,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), r2t.stride(0) if paged else 0,
        table.stride(0), table.stride(1), sink.stride(0), sink.stride(1),
        1.0 / math.sqrt(D), k_scale, v_scale, window, ring,
        N_Q=n_q, H=H, H_PER_KV=H // k.shape[1], D=D, EXTENT=table.shape[-1], NSPLIT=nsplit,
        BM=max(16, triton.next_power_of_2(n_q)), BN=block_n, PAGED=paged, KV_FP8=fp8,
    )
    mm = m.max(-1, keepdim=True).values
    mm = torch.where(torch.isinf(mm), torch.zeros_like(mm), mm)
    w = torch.exp(m - mm)
    return (acc * w[..., None]).sum(2) / (l * w).sum(-1)[..., None]


# ---- fused short convolution for decode / verify ---------------------------------------------------------------
# y[t, c] = (x[t, c] + sum_j w[c, j] * h_j[t, c] + sum_j tap[t, head(c), 3-j] * h_j[t, c]) * out_scale
# h_j[t] = input at position pos[t] - (3 - j) of row t's slot (j = 3 is x itself), from the position ring if its stored
# position matches, else 0. Writes x into ring[slot, pos % RC] first (the ring position table is updated by the caller).
@triton.jit
def _conv_dec(X, W, TAP, RING, RPOS, SLOT, POS, Y,
              C, c0, stride_x, stride_rs, stride_rr, stride_y, stride_tap, out_scale,
              RC: tl.constexpr, NC: tl.constexpr, HEAD: tl.constexpr, DYN: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    cs = cb * BC + tl.arange(0, BC)
    cm = cs < C
    slot = tl.load(SLOT + t).to(tl.int64)
    p = tl.load(POS + t)
    x = tl.load(X + t * stride_x + cs, mask=cm, other=0.0).to(tl.float32)
    acc = x * tl.load(W + cs * NC + (NC - 1), mask=cm, other=0.0)
    if DYN:
        acc += x * tl.load(TAP + t * stride_tap + (cs // HEAD) * NC + 0, mask=cm, other=0.0)
    for j in tl.static_range(NC - 1):
        pj = p - (NC - 1 - j)
        idx = ((pj % RC) + RC) % RC
        ok = (tl.load(RPOS + slot * RC + idx) == pj) & (pj >= 0)
        h = tl.load(RING + slot * stride_rs + idx * stride_rr + c0 + cs, mask=cm & ok, other=0.0)
        acc += h * tl.load(W + cs * NC + j, mask=cm, other=0.0)
        if DYN:
            acc += h * tl.load(TAP + t * stride_tap + (cs // HEAD) * NC + (NC - 1 - j), mask=cm, other=0.0)
    tl.store(Y + t * stride_y + cs, (x + acc) * out_scale, mask=cm)


@triton.jit
def _conv_store(X, RING, SLOT, POS, C, c0, stride_x, stride_rs, stride_rr, RC: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    cs = cb * BC + tl.arange(0, BC)
    cm = cs < C
    slot = tl.load(SLOT + t).to(tl.int64)
    p = tl.load(POS + t)
    idx = ((p % RC) + RC) % RC
    tl.store(RING + slot * stride_rs + idx * stride_rr + c0 + cs, tl.load(X + t * stride_x + cs, mask=cm, other=0.0).to(tl.float32), mask=cm)


def conv_decode(x, w32, tap, ring, rpos, slot, pos, c0: int, n_conv: int = 4):
    """x [T, C]; w32 [C, n_conv] fp32 (oldest..current); tap [T, heads*n_conv] or None; ring [S, RC, NCH] fp32;
    rpos [S, RC] int64. Returns fp32 [T, C]. Two launches: store (so verify rows see each other), then conv."""
    T, C = x.shape
    RC = ring.shape[1]
    y = torch.empty(T, C, device=x.device, dtype=torch.float32)
    BC = 256
    grid = (T, triton.cdiv(C, BC))
    _conv_store[grid](x, ring, slot, pos, C, c0, x.stride(0), ring.stride(0), ring.stride(1), RC=RC, BC=BC)
    dyn = tap is not None
    heads = tap.shape[1] // n_conv if dyn else 1
    _conv_dec[grid](x, w32, tap if dyn else x, ring, rpos, slot, pos, y, C, c0, x.stride(0), ring.stride(0), ring.stride(1),
                    y.stride(0), tap.stride(0) if dyn else 0, 1.0 / math.sqrt((2 if dyn else 1) * n_conv + 1),
                    RC=RC, NC=n_conv, HEAD=C // heads, DYN=dyn, BC=BC)
    return y


# v2: one program per (request, kv head, split) covering the H_PER_KV query heads x N_Q rows (K/V read once per kv head);
# bias table read only for key blocks inside the band (dist < EXTENT).
@triton.jit
def _attn_dec2(
    Q, K, V, R2T, SLOTS, KVLEN, RPOS, Q_POS, TABLE, SINK, O_ACC, O_M, O_L,
    stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh, stride_r2t,
    stride_tt, stride_th, stride_st, stride_sh,
    scale, k_scale, v_scale, window, ring,
    N_Q: tl.constexpr, H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, EXTENT: tl.constexpr,
    NSPLIT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, PAGED: tl.constexpr, KV_FP8: tl.constexpr,
):
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    r = tl.arange(0, BM)
    rmask = r < G * N_Q
    gi = r // N_Q                      # head within group
    qi = r % N_Q                       # query row of the request
    h = hk * G + gi
    rows = b * N_Q + qi
    dd = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + h[:, None] * stride_qh + dd[None, :], mask=rmask[:, None], other=0.0).to(tl.bfloat16)
    qpos = tl.load(Q_POS + rows, mask=rmask, other=-1)
    slot = tl.load(SLOTS + b).to(tl.int64)
    if PAGED:
        L = tl.load(KVLEN + b)
    else:
        L = ring
    chunk = (L + NSPLIT - 1) // NSPLIT
    chunk = (chunk + BN - 1) // BN * BN
    lo = sp * chunk
    hi = tl.minimum(L, lo + chunk)
    if sp == 0:
        m_i = tl.load(SINK + rows * stride_st + h * stride_sh, mask=rmask, other=0.0).to(tl.float32)
        l_i = tl.where(rmask, 1.0, 0.0)
    else:
        m_i = tl.full([BM], -float("inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    qmin = tl.min(tl.where(rmask, qpos, 2147483647), axis=0)
    for start in range(lo, hi, BN):
        cols = start + tl.arange(0, BN)
        cmask = cols < hi
        if PAGED:
            loc = tl.load(R2T + slot * stride_r2t + cols, mask=cmask, other=0).to(tl.int64)
            kpos = tl.where(cmask, cols, -1)
        else:
            loc = slot * ring + cols
            kpos = tl.load(RPOS + slot * ring + cols, mask=cmask, other=-1)
        k = tl.load(K + loc[:, None] * stride_kt + hk * stride_kh + dd[None, :], mask=cmask[:, None], other=0.0)
        v = tl.load(V + loc[:, None] * stride_vt + hk * stride_vh + dd[None, :], mask=cmask[:, None], other=0.0)
        if KV_FP8:
            k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
            v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * scale
        dist = qpos[:, None] - kpos[None, :]
        valid = (dist >= 0) & (kpos[None, :] >= 0) & rmask[:, None]
        if window > 0:
            valid = valid & (dist < window)
        kmax = tl.max(tl.where(cmask, kpos, -1), axis=0)
        if qmin - kmax < EXTENT:
            near = valid & (dist < EXTENT)
            bias = tl.load(TABLE + rows[:, None] * stride_tt + h[:, None] * stride_th + tl.minimum(tl.maximum(dist, 0), EXTENT - 1),
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
    o = (rows[:, None] * H + h[:, None]) * NSPLIT + sp
    tl.store(O_ACC + o * D + dd[None, :], acc, mask=rmask[:, None])
    ml = (rows * H + h) * NSPLIT + sp
    tl.store(O_M + ml, m_i, mask=rmask)
    tl.store(O_L + ml, l_i, mask=rmask)


def attention_decode2(q, k, v, slots, q_pos, table, sink, *, n_q: int, paged: bool, r2t=None, kv_len=None,
                      ring_pos=None, ring: int = 0, window: int = 0, nsplit: int = 8, k_scale=1.0, v_scale=1.0,
                      block_n: int = 32):
    T, H, D = q.shape
    bs = slots.shape[0]
    Hkv = k.shape[1]
    G = H // Hkv
    acc = torch.empty(T, H, nsplit, D, device=q.device, dtype=torch.float32)
    m = torch.empty(T, H, nsplit, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    fp8 = k.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    dummy = slots
    _attn_dec2[(bs, Hkv, nsplit)](
        q, k, v, r2t if paged else dummy, slots, kv_len if paged else dummy, ring_pos if not paged else dummy, q_pos,
        table, sink, acc, m, l,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), r2t.stride(0) if paged else 0,
        table.stride(0), table.stride(1), sink.stride(0), sink.stride(1),
        1.0 / math.sqrt(D), k_scale, v_scale, window, ring,
        N_Q=n_q, H=H, G=G, D=D, EXTENT=table.shape[-1], NSPLIT=nsplit,
        BM=max(16, triton.next_power_of_2(G * n_q)), BN=block_n, PAGED=paged, KV_FP8=fp8,
    )
    mm = m.max(-1, keepdim=True).values
    mm = torch.where(torch.isinf(mm), torch.zeros_like(mm), mm)
    w = torch.exp(m - mm)
    return (acc * w[..., None]).sum(2) / (l * w).sum(-1)[..., None]


def attention_decode_sycl(q, k, v, slots, q_pos, rrel, proj, sink, *, n_q: int, paged: bool, r2t=None, kv_len=None,
                          ring_pos=None, ring: int = 0, window: int = 0, nsplit: int = 8, k_scale=1.0, v_scale=1.0):
    """SYCL flash-decoding (exl3xpu/_inkling2.so decode_attn). Relative bias computed in-kernel:
    bias[row, h, dist] = rrel[row, h, :] . proj[:, dist] for dist < proj.shape[1]."""
    from .moe_triton import ops
    T, H, D = q.shape
    acc = torch.empty(T, H, nsplit, D, device=q.device, dtype=torch.float32)
    m = torch.empty(T, H, nsplit, device=q.device, dtype=torch.float32)
    l = torch.empty_like(m)
    ops().decode_attn(q.to(torch.bfloat16).contiguous(), k, v, slots, kv_len if paged else None, r2t if paged else None,
                      None if paged else ring_pos, q_pos.to(torch.int32), rrel.float().contiguous(), proj, sink.float(),
                      acc, m, l, n_q, ring, window, float(k_scale), float(v_scale))
    if nsplit == 1:
        return acc[:, :, 0] / l[:, :, 0, None]
    mm = m.max(-1, keepdim=True).values
    w = torch.exp(m - mm)
    return (acc * w[..., None]).sum(2) / (l * w).sum(-1)[..., None]


# single-launch conv: inputs at positions >= the request's first position in this batch come from X (rows of the
# same request), older ones from the ring; each program also stores its own input into the ring (no read/write overlap
# while N_Q + 3 <= RC).
@triton.jit
def _conv_dec1(X, W, TAP, RING, RPOS, SLOT, POS, Y,
               C, c0, stride_x, stride_rs, stride_rr, stride_y, stride_tap, out_scale,
               RC: tl.constexpr, NC: tl.constexpr, HEAD: tl.constexpr, DYN: tl.constexpr, NQ: tl.constexpr, BC: tl.constexpr):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    cs = cb * BC + tl.arange(0, BC)
    cm = cs < C
    slot = tl.load(SLOT + t).to(tl.int64)
    p = tl.load(POS + t)
    t0 = (t // NQ) * NQ
    p0 = tl.load(POS + t0)
    x = tl.load(X + t * stride_x + cs, mask=cm, other=0.0).to(tl.float32)
    acc = x * tl.load(W + cs * NC + (NC - 1), mask=cm, other=0.0)
    if DYN:
        acc += x * tl.load(TAP + t * stride_tap + (cs // HEAD) * NC + 0, mask=cm, other=0.0)
    for j in tl.static_range(NC - 1):
        pj = p - (NC - 1 - j)
        in_batch = pj >= p0
        idx = ((pj % RC) + RC) % RC
        ok = (tl.load(RPOS + slot * RC + idx) == pj) & (pj >= 0)
        hb = tl.load(X + (t - (p - pj)) * stride_x + cs, mask=cm & in_batch, other=0.0).to(tl.float32)
        hr = tl.load(RING + slot * stride_rs + idx * stride_rr + c0 + cs, mask=cm & ok & (pj < p0), other=0.0)
        h = tl.where(in_batch, hb, hr)
        acc += h * tl.load(W + cs * NC + j, mask=cm, other=0.0)
        if DYN:
            acc += h * tl.load(TAP + t * stride_tap + (cs // HEAD) * NC + (NC - 1 - j), mask=cm, other=0.0)
    tl.store(Y + t * stride_y + cs, (x + acc) * out_scale, mask=cm)
    tl.store(RING + slot * stride_rs + (((p % RC) + RC) % RC) * stride_rr + c0 + cs, x, mask=cm)


def conv_decode1(x, w32, tap, ring, rpos, slot, pos, c0: int, n_q: int, n_conv: int = 4):
    T, C = x.shape
    RC = ring.shape[1]
    y = torch.empty(T, C, device=x.device, dtype=torch.float32)
    BC = 256
    dyn = tap is not None
    heads = tap.shape[1] // n_conv if dyn else 1
    _conv_dec1[(T, triton.cdiv(C, BC))](x, w32, tap if dyn else x, ring, rpos, slot, pos, y, C, c0, x.stride(0), ring.stride(0),
                                       ring.stride(1), y.stride(0), tap.stride(0) if dyn else 0,
                                       1.0 / math.sqrt((2 if dyn else 1) * n_conv + 1),
                                       RC=RC, NC=n_conv, HEAD=C // heads, DYN=dyn, NQ=n_q, BC=BC)
    return y


# ---- prefill (extend) attention: one request, contiguous K/V ---------------------------------------------------
# q [n, H, D] at positions qpos (increasing), K/V [L, Hkv, D] with key positions kpos [L] (increasing, -1 = empty).
# Grid (cdiv(n, BM), H). Key blocks outside the causal / window range are skipped; the bias table [n, H, E] is read only
# for blocks that intersect the band dist < E.
@triton.jit
def _attn_prefill(Q, K, V, QPOS, KPOS, TABLE, SINK, O,
                  n, L, stride_qt, stride_qh, stride_kt, stride_kh, stride_vt, stride_vh, stride_tt, stride_th,
                  stride_st, stride_sh, stride_ot, stride_oh, scale, window,
                  H_PER_KV: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // H_PER_KV
    rows = pm * BM + tl.arange(0, BM)
    rm = rows < n
    dd = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * stride_qt + h * stride_qh + dd[None, :], mask=rm[:, None], other=0.0).to(tl.bfloat16)
    qpos = tl.load(QPOS + rows, mask=rm, other=-1)
    qmax = tl.max(qpos, 0)
    qmin = tl.min(tl.where(rm, qpos, 2147483647), 0)
    m_i = tl.load(SINK + rows * stride_st + h * stride_sh, mask=rm, other=0.0).to(tl.float32)
    l_i = tl.where(rm, 1.0, 0.0)
    acc = tl.zeros([BM, D], tl.float32)
    for start in range(0, L, BN):
        cols = start + tl.arange(0, BN)
        cm = cols < L
        kpos = tl.load(KPOS + cols, mask=cm, other=-1)
        kmin = tl.min(tl.where(cm & (kpos >= 0), kpos, 2147483647), 0)
        kmax = tl.max(tl.where(cm, kpos, -1), 0)
        live = (kmin <= qmax) & (kmax >= 0)
        if window > 0:
            live = live & (qmin - kmax < window)
        if live:
            k = tl.load(K + cols[:, None] * stride_kt + hk * stride_kh + dd[None, :], mask=cm[:, None], other=0.0).to(tl.bfloat16)
            s = tl.dot(q, tl.trans(k)) * scale
            dist = qpos[:, None] - kpos[None, :]
            valid = (dist >= 0) & (kpos[None, :] >= 0) & rm[:, None]
            if window > 0:
                valid = valid & (dist < window)
            if qmin - kmax < E:
                near = valid & (dist < E)
                s += tl.load(TABLE + rows[:, None] * stride_tt + h * stride_th + tl.minimum(tl.maximum(dist, 0), E - 1),
                             mask=near, other=0.0).to(tl.float32)
            s = tl.where(valid, s, -1e30)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)
            l_i = l_i * alpha + tl.sum(p, 1)
            v = tl.load(V + cols[:, None] * stride_vt + hk * stride_vh + dd[None, :], mask=cm[:, None], other=0.0).to(tl.bfloat16)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
            m_i = m_new
    tl.store(O + rows[:, None] * stride_ot + h * stride_oh + dd[None, :], acc / l_i[:, None], mask=rm[:, None])


def attention_prefill(q, k, v, qpos, kpos, table, sink, *, window: int = 0, block_m: int = 64, block_n: int = 64,
                      num_warps: int = 8):
    n, H, D = q.shape
    L = k.shape[0]
    o = torch.empty(n, H, D, device=q.device, dtype=torch.float32)
    _attn_prefill[(triton.cdiv(n, block_m), H)](
        q, k, v, qpos, kpos, table, sink, o, n, L,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1), table.stride(0), table.stride(1),
        sink.stride(0), sink.stride(1), o.stride(0), o.stride(1), 1.0 / math.sqrt(D), window,
        H_PER_KV=H // k.shape[1], D=D, E=table.shape[-1], BM=block_m, BN=block_n, num_warps=num_warps)
    return o


# v2: block pointers (2D block loads on Xe), contiguous key positions kpos = kpos0 + j, queries qpos = qpos0 + i.
@triton.jit
def _attn_prefill2(Q, K, V, TABLE, SINK, O,
                   n, L, qpos0, kpos0, stride_qt, stride_kt, stride_vt, stride_tt, stride_th,
                   stride_st, stride_sh, stride_ot, scale, window, kv_cols,
                   H_PER_KV: tl.constexpr, D: tl.constexpr, E: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    pm = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // H_PER_KV
    rows = pm * BM + tl.arange(0, BM)
    rm = rows < n
    qb = tl.make_block_ptr(Q, shape=(n, kv_cols * H_PER_KV), strides=(stride_qt, 1), offsets=(pm * BM, h * D),
                           block_shape=(BM, D), order=(1, 0))
    q = tl.load(qb, boundary_check=(0,)).to(tl.bfloat16)
    qpos = qpos0 + rows
    qmax = qpos0 + tl.minimum(pm * BM + BM, n) - 1
    qmin = qpos0 + pm * BM
    m_i = tl.load(SINK + rows * stride_st + h * stride_sh, mask=rm, other=0.0).to(tl.float32)
    l_i = tl.where(rm, 1.0, 0.0)
    acc = tl.zeros([BM, D], tl.float32)
    # key range: causal end, window start
    hi = tl.minimum(L, qmax - kpos0 + 1)
    lo = 0
    if window > 0:
        lo = tl.maximum(0, qmin - window + 1 - kpos0)
    lo = lo // BN * BN
    kb = tl.make_block_ptr(K, shape=(L, kv_cols), strides=(stride_kt, 1), offsets=(lo, hk * D), block_shape=(BN, D), order=(1, 0))
    vb = tl.make_block_ptr(V, shape=(L, kv_cols), strides=(stride_vt, 1), offsets=(lo, hk * D), block_shape=(BN, D), order=(1, 0))
    for start in range(lo, hi, BN):
        k = tl.load(kb, boundary_check=(0,)).to(tl.bfloat16)
        s = tl.dot(q, tl.trans(k)) * scale
        kpos = kpos0 + start + tl.arange(0, BN)
        dist = qpos[:, None] - kpos[None, :]
        valid = (dist >= 0) & ((start + tl.arange(0, BN)) < L)[None, :] & rm[:, None]
        if window > 0:
            valid = valid & (dist < window)
        if qmin - (kpos0 + start + BN - 1) < E:
            near = valid & (dist < E)
            s += tl.load(TABLE + rows[:, None] * stride_tt + h * stride_th + tl.minimum(tl.maximum(dist, 0), E - 1),
                         mask=near, other=0.0).to(tl.float32)
        s = tl.where(valid, s, -1e30)
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.where(valid, tl.exp(s - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(vb, boundary_check=(0,)).to(tl.bfloat16)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
        kb = tl.advance(kb, (BN, 0))
        vb = tl.advance(vb, (BN, 0))
    ob = tl.make_block_ptr(O, shape=(n, kv_cols * H_PER_KV), strides=(stride_ot, 1), offsets=(pm * BM, h * D),
                           block_shape=(BM, D), order=(1, 0))
    tl.store(ob, acc / l_i[:, None], boundary_check=(0,))


def attention_prefill2(q, k, v, qpos0: int, kpos0: int, table, sink, *, window: int = 0, block_m: int = 64,
                       block_n: int = 64, num_warps: int = 8):
    """q [n, H, D] (rows contiguous, positions qpos0..), k/v [L, Hkv, D] contiguous rows, positions kpos0.."""
    n, H, D = q.shape
    L, Hkv, _ = k.shape
    q = q.contiguous(); k = k.contiguous(); v = v.contiguous()
    o = torch.empty(n, H, D, device=q.device, dtype=torch.float32)
    _attn_prefill2[(triton.cdiv(n, block_m), H)](
        q, k, v, table, sink, o, n, L, qpos0, kpos0, q.stride(0), k.stride(0), v.stride(0), table.stride(0), table.stride(1),
        sink.stride(0), sink.stride(1), o.stride(0), 1.0 / math.sqrt(D), window, Hkv * D,
        H_PER_KV=H // Hkv, D=D, E=table.shape[-1], BM=block_m, BN=block_n, num_warps=num_warps)
    return o


def attention_prefill_esimd(q, k, v, qpos0: int, kpos0: int, table, sink, *, window: int = 0, qtw: int = 8):
    """ESIMD/DPAS prefill attention (exl3xpu/_inkling2.so prefill_attn). q [n, H, 128], k/v [L, Hkv, 128] contiguous
    positions; table bf16 [n, H, E] or None; sink fp32 [n, H] or None. Returns fp32 [n, H, 128]."""
    from .moe_triton import ops
    n, H, D = q.shape
    o = torch.empty(n, H, D, device=q.device, dtype=torch.float32)
    ops().prefill_attn(q.to(torch.float16).contiguous(), k.to(torch.float16).contiguous(), v.to(torch.float16).contiguous(), o,
                       table.to(torch.bfloat16).contiguous() if table is not None else None,
                       sink.float().contiguous() if sink is not None else None, int(qpos0), int(kpos0), int(window), int(qtw))
    return o
