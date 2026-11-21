"""Fused small ops for Inkling 2 decode on XPU (Triton). Each replaces 5-25 torch launches.

rms_norm(x, w)            : y = x * rsqrt(mean(x^2) + eps) * w                       (1 launch)
mix(buf, idx, cur, q, ...) : depth attention over S states (blocks in buf[idx], then cur): values RMS-normalized,
                            scores = q . v, softmax over states, out[j] = sum_s p[j,s] v_s; optional fused RMSNorm with
                            per-output weights (attn_norm / q,k,v input norms)                    (1 launch)
route(logits, bias, k)    : sigmoid, + bias for selection, top-k (k=8 of 128), weights renormalized * scale   (1 launch)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel(X, W, Y, D, stride_x, stride_y, eps, HAS_W: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    acc = tl.zeros([BD], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        x = tl.load(X + t * stride_x + d, mask=d < D, other=0.0).to(tl.float32)
        acc += x * x
    inv = tl.rsqrt(tl.sum(acc, 0) / D + eps)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        m = d < D
        x = tl.load(X + t * stride_x + d, mask=m, other=0.0).to(tl.float32) * inv
        if HAS_W:
            x = x * tl.load(W + d, mask=m, other=0.0).to(tl.float32)
        tl.store(Y + t * stride_y + d, x, mask=m)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, device=x.device, dtype=torch.float32)
    _rms_kernel[(x2.shape[0],)](x2, w if w is not None else x2, y, shape[-1], x2.stride(0), y.stride(0), eps,
                                HAS_W=w is not None, BD=1024)
    return y.view(shape)


@triton.jit
def _load_states(BUF, IDX, CUR, t, d, dm, S, stride_bs, stride_bt, stride_ct, SMAX: tl.constexpr):
    s_ar = tl.arange(0, SMAX)
    bidx = tl.load(IDX + s_ar, mask=s_ar < S, other=0)
    is_cur = s_ar == S
    vb = tl.load(BUF + bidx[:, None] * stride_bs + t * stride_bt + d[None, :], mask=(s_ar < S)[:, None] & dm[None, :], other=0.0)
    vc = tl.load(CUR + t * stride_ct + d[None, :] + 0 * s_ar[:, None], mask=is_cur[:, None] & dm[None, :], other=0.0)
    return tl.where(is_cur[:, None], vc, vb).to(tl.float32)        # [SMAX, BD]


@triton.jit
def _mix_kernel(BUF, IDX, CUR, Q, OUT, NW, S, D, stride_bs, stride_bt, stride_ct, stride_qj, stride_oj, stride_ot,
                stride_nw, eps, NQ: tl.constexpr, NQP: tl.constexpr, SMAX: tl.constexpr, NORM: tl.constexpr, BD: tl.constexpr):
    t = tl.program_id(0)
    s_ar = tl.arange(0, SMAX)
    smask = s_ar < S + 1
    jq = tl.arange(0, NQP)
    inv = tl.zeros([SMAX], tl.float32)
    dots = tl.zeros([SMAX, NQP], tl.float32)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dm = d < D
        v = _load_states(BUF, IDX, CUR, t, d, dm, S, stride_bs, stride_bt, stride_ct, SMAX)
        inv += tl.sum(v * v, 1)
        qv = tl.load(Q + jq[:, None] * stride_qj + d[None, :], mask=(jq < NQ)[:, None] & dm[None, :], other=0.0).to(tl.float32)
        dots += tl.dot(v, tl.trans(qv), input_precision="ieee") if False else tl.sum(v[:, None, :] * qv[None, :, :], 2)
    inv = tl.rsqrt(inv / D + eps)
    scores = tl.where(smask[:, None], dots * inv[:, None], -1e30)
    mx = tl.max(scores, 0)
    p = tl.where(smask[:, None], tl.exp(scores - mx[None, :]), 0.0)
    pw = p / tl.sum(p, 0)[None, :] * inv[:, None]                     # weights on the raw states [SMAX, NQP]
    if NORM:
        acc2 = tl.zeros([NQP], tl.float32)
        for d0 in range(0, D, BD):
            d = d0 + tl.arange(0, BD)
            dm = d < D
            v = _load_states(BUF, IDX, CUR, t, d, dm, S, stride_bs, stride_bt, stride_ct, SMAX)
            o = tl.sum(v[:, None, :] * pw[:, :, None], 0)                # [NQP, BD]
            acc2 += tl.sum(o * o, 1)
        onorm = tl.rsqrt(acc2 / D + eps)
    for d0 in range(0, D, BD):
        d = d0 + tl.arange(0, BD)
        dm = d < D
        v = _load_states(BUF, IDX, CUR, t, d, dm, S, stride_bs, stride_bt, stride_ct, SMAX)
        o = tl.sum(v[:, None, :] * pw[:, :, None], 0)                    # [NQP, BD]
        om = (jq < NQ)[:, None] & dm[None, :]
        if NORM:
            o = o * onorm[:, None] * tl.load(NW + jq[:, None] * stride_nw + d[None, :], mask=om, other=0.0).to(tl.float32)
        tl.store(OUT + jq[:, None] * stride_oj + t * stride_ot + d[None, :], o, mask=om)


_INT_DUMMY = {}


def _int_dummy(dev):
    t = _INT_DUMMY.get(dev)
    if t is None:
        t = _INT_DUMMY[dev] = torch.zeros(1, dtype=torch.int32, device=dev)
    return t


def mix(buf: torch.Tensor, idx: torch.Tensor, cur: torch.Tensor, q: torch.Tensor, norm_w: torch.Tensor | None = None,
        eps: float = 1e-6) -> torch.Tensor:
    """buf [NB, T, D] fp32 block store; idx [S] int32 slots of the blocks in order; cur [T, D]; q [NQ, D] fp32 (scaled
    by D^-0.5); norm_w [NQ, D] or None. Returns [NQ, T, D] fp32."""
    NQ, D = q.shape
    T = cur.shape[0]
    S = idx.shape[0]
    out = torch.empty(NQ, T, D, device=cur.device, dtype=torch.float32)
    cur = cur if cur.stride(-1) == 1 else cur.contiguous()
    if S == 0:
        idx = _int_dummy(cur.device)
    _mix_kernel[(T,)](buf, idx, cur, q, out, norm_w if norm_w is not None else q, S, D,
                      buf.stride(0), buf.stride(1), cur.stride(0), q.stride(0), out.stride(0), out.stride(1),
                      norm_w.stride(0) if norm_w is not None else 0, eps,
                      NQ=NQ, NQP=max(2, triton.next_power_of_2(NQ)), SMAX=max(2, triton.next_power_of_2(S + 1)),
                      NORM=norm_w is not None, BD=256, num_warps=16)
    return out


@triton.jit
def _route_kernel(LOGITS, BIAS, IDS, W, E, scale, K: tl.constexpr, BE: tl.constexpr):
    t = tl.program_id(0)
    e = tl.arange(0, BE)
    em = e < E
    p = tl.sigmoid(tl.load(LOGITS + t * E + e, mask=em, other=-1e30).to(tl.float32))
    sel = tl.where(em, p + tl.load(BIAS + e, mask=em, other=0.0), -1e30)
    tot = 0.0
    ks = tl.arange(0, K)
    ids = tl.zeros([K], tl.int32)
    ws = tl.zeros([K], tl.float32)
    for i in tl.static_range(K):
        m = tl.max(sel, 0)
        am = tl.min(tl.where(sel == m, e, BE), 0)
        wi = tl.sum(tl.where(e == am, p, 0.0), 0)
        ids = tl.where(ks == i, am, ids)
        ws = tl.where(ks == i, wi, ws)
        tot += wi
        sel = tl.where(e == am, -1e30, sel)
    tl.store(IDS + t * K + ks, ids)
    tl.store(W + t * K + ks, ws / tot * scale)


def route(logits: torch.Tensor, bias: torch.Tensor, k: int, scale: float):
    T, E = logits.shape
    ids = torch.empty(T, k, device=logits.device, dtype=torch.int32)
    w = torch.empty(T, k, device=logits.device, dtype=torch.float32)
    _route_kernel[(T,)](logits.contiguous(), bias, ids, w, E, scale, K=k, BE=triton.next_power_of_2(E))
    return ids, w


@triton.jit
def _route_gemv_kernel(X, WR, BIAS, IDS, W, E, D, stride_x, scale, K: tl.constexpr, BE: tl.constexpr, BK: tl.constexpr):
    """router logits (fp32 weights [E, D]) + sigmoid + top-k + renormalize, one program per token."""
    t = tl.program_id(0)
    e = tl.arange(0, BE)
    em = e < E
    acc = tl.zeros([BE], tl.float32)
    for k0 in range(0, D, BK):
        kk = k0 + tl.arange(0, BK)
        km = kk < D
        x = tl.load(X + t * stride_x + kk, mask=km, other=0.0).to(tl.float32)
        w = tl.load(WR + e[:, None] * D + kk[None, :], mask=em[:, None] & km[None, :], other=0.0)
        acc += tl.sum(w * x[None, :], 1)
    p = tl.sigmoid(acc)
    sel = tl.where(em, p + tl.load(BIAS + e, mask=em, other=0.0), -1e30)
    tot = 0.0
    ks = tl.arange(0, K)
    ids = tl.zeros([K], tl.int32)
    ws = tl.zeros([K], tl.float32)
    for i in tl.static_range(K):
        m = tl.max(sel, 0)
        am = tl.min(tl.where(sel == m, e, BE), 0)
        wi = tl.sum(tl.where(e == am, p, 0.0), 0)
        ids = tl.where(ks == i, am, ids)
        ws = tl.where(ks == i, wi, ws)
        tot += wi
        sel = tl.where(e == am, -1e30, sel)
    tl.store(IDS + t * K + ks, ids)
    tl.store(W + t * K + ks, ws / tot * scale)


def route_gemv(x: torch.Tensor, wr: torch.Tensor, bias: torch.Tensor, k: int, scale: float):
    T, D = x.shape
    E = wr.shape[0]
    x = x if x.stride(-1) == 1 else x.contiguous()
    ids = torch.empty(T, k, device=x.device, dtype=torch.int32)
    w = torch.empty(T, k, device=x.device, dtype=torch.float32)
    _route_gemv_kernel[(T,)](x, wr, bias, ids, w, E, D, x.stride(0), scale, K=k, BE=triton.next_power_of_2(E), BK=64,
                             num_warps=16)
    return ids, w
