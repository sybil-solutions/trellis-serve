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


@triton.jit
def _situ_kernel(H, A, n, I, stride_h, stride_a, gcap, ucap, BI: tl.constexpr):
    """h [n, 2I] interleaved (gate 2i, up 2i+1) -> act bf16 [n, I] = gcap tanh(g/gcap) sigmoid(g) * ucap tanh(u/ucap)."""
    t = tl.program_id(0)
    ib = tl.program_id(1)
    i = ib * BI + tl.arange(0, BI)
    m = i < I
    g = tl.load(H + t * stride_h + 2 * i, mask=m, other=0.0).to(tl.float32)
    u = tl.load(H + t * stride_h + 2 * i + 1, mask=m, other=0.0).to(tl.float32)
    tg = 2.0 * tl.sigmoid(2.0 * g / gcap) - 1.0
    tu = 2.0 * tl.sigmoid(2.0 * u / ucap) - 1.0
    a = gcap * tg * tl.sigmoid(g) * (ucap * tu)
    tl.store(A + t * stride_a + i, a.to(tl.bfloat16), mask=m)


def situ(h: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    n, I2 = h.shape
    I = I2 // 2
    a = torch.empty(n, I, device=h.device, dtype=torch.bfloat16)
    BI = 256
    _situ_kernel[(n, triton.cdiv(I, BI))](h, a, n, I, h.stride(0), a.stride(0), gate_cap, up_cap, BI=BI)
    return a


@triton.jit
def _attn_prep_kernel(QRGS, K, V, POS, QSC, RSC, QO, RO, GO, SO, KO, VO,
                      stride_x, stride_k, stride_v, qk_thr, rel_thr, alpha_log, floor_log, eps,
                      H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, DR: tl.constexpr, TAU: tl.constexpr):
    """one program per token: q (scale, L2 clip, tau) -> bf16; r (scale, clip, tau) fp32; sigmoid(gate); sink;
    k (L2 clip) / v (RMS, no gamma) -> bf16 rows ready for the ring / pool."""
    t = tl.program_id(0)
    if TAU:
        p = tl.load(POS + t).to(tl.float32)
        tau = 1.0 + alpha_log * tl.log(tl.maximum((p + 1.0) / floor_log, 1.0))
    else:
        tau = 1.0
    h = tl.arange(0, 32)
    hm = h < H
    d = tl.arange(0, D)
    q = tl.load(QRGS + t * stride_x + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
    q = q * tl.load(QSC + h[:, None] * D + d[None, :], mask=hm[:, None], other=0.0)
    qn = tl.sqrt(tl.sum(q * q, 1)) + 1e-6
    q = q * tl.minimum(qk_thr / qn, 1.0)[:, None] * tau
    tl.store(QO + t * H * D + h[:, None] * D + d[None, :], q.to(tl.bfloat16), mask=hm[:, None])
    dr = tl.arange(0, DR)
    r = tl.load(QRGS + t * stride_x + H * D + h[:, None] * DR + dr[None, :], mask=hm[:, None], other=0.0)
    r = r * tl.load(RSC + h[:, None] * DR + dr[None, :], mask=hm[:, None], other=0.0)
    rn = tl.sqrt(tl.sum(r * r, 1)) + 1e-6
    r = r * tl.minimum(rel_thr / rn, 1.0)[:, None] * tau
    tl.store(RO + t * H * DR + h[:, None] * DR + dr[None, :], r, mask=hm[:, None])
    go = H * D + H * DR
    for c0 in tl.static_range(0, H * D, 1024):
        c = c0 + tl.arange(0, 1024)
        g = tl.load(QRGS + t * stride_x + go + c, mask=c < H * D, other=0.0)
        tl.store(GO + t * H * D + c, tl.sigmoid(g), mask=c < H * D)
    so = go + H * D
    sk = tl.load(QRGS + t * stride_x + so + h, mask=hm, other=0.0)
    tl.store(SO + t * H + h, sk, mask=hm)
    hk = tl.arange(0, HKV)
    k = tl.load(K + t * stride_k + hk[:, None] * D + d[None, :])
    kn = tl.sqrt(tl.sum(k * k, 1)) + 1e-6
    k = k * tl.minimum(qk_thr / kn, 1.0)[:, None]
    tl.store(KO + t * HKV * D + hk[:, None] * D + d[None, :], k.to(tl.bfloat16))
    v = tl.load(V + t * stride_v + hk[:, None] * D + d[None, :])
    v = v * tl.rsqrt(tl.sum(v * v, 1) / D + eps)[:, None]
    tl.store(VO + t * HKV * D + hk[:, None] * D + d[None, :], v.to(tl.bfloat16))


def attn_prep(qrgs, k, v, pos, q_scale, r_scale, *, H, HKV, D, DR, qk_thr, rel_thr, tau_alpha=None, tau_floor=None, eps=1e-6):
    T = qrgs.shape[0]
    dev = qrgs.device
    qo = torch.empty(T, H, D, device=dev, dtype=torch.bfloat16)
    ro = torch.empty(T, H, DR, device=dev, dtype=torch.float32)
    go = torch.empty(T, H * D, device=dev, dtype=torch.float32)
    so = torch.empty(T, H, device=dev, dtype=torch.float32)
    ko = torch.empty(T, HKV, D, device=dev, dtype=torch.bfloat16)
    vo = torch.empty(T, HKV, D, device=dev, dtype=torch.bfloat16)
    tau = tau_alpha is not None
    _attn_prep_kernel[(T,)](qrgs, k, v, pos, q_scale, r_scale, qo, ro, go, so, ko, vo,
                            qrgs.stride(0), k.stride(0), v.stride(0), qk_thr, rel_thr,
                            tau_alpha if tau else 0.0, tau_floor if tau else 1.0, eps,
                            H=H, HKV=HKV, D=D, DR=DR, TAU=tau, num_warps=8)
    return qo, ro, go, so, ko, vo


@triton.jit
def _gather_f16_kernel(SRC, LOC, DST, n_cols, stride_s, BC: tl.constexpr):
    r = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BC + tl.arange(0, BC)
    m = c < n_cols
    row = tl.load(LOC + r).to(tl.int64)
    x = tl.load(SRC + row * stride_s + c, mask=m, other=0.0).to(tl.float32)
    tl.store(DST + r.to(tl.int64) * n_cols + c, x.to(tl.float16), mask=m)


def gather_f16(src: torch.Tensor, loc: torch.Tensor) -> torch.Tensor:
    """rows src[loc] (any float dtype incl. fp8, [N, ...] contiguous rows) -> fp16 [len(loc), ...] without an fp8 copy."""
    n_cols = src[0].numel()
    out = torch.empty((loc.numel(), *src.shape[1:]), device=src.device, dtype=torch.float16)
    BC = 1024
    _gather_f16_kernel[(loc.numel(), triton.cdiv(n_cols, BC))](src, loc, out, n_cols, src.stride(0), BC=BC)
    return out


@triton.jit
def _q8_dequant_kernel(Q, S, W, K, BK: tl.constexpr):
    n = tl.program_id(0)
    kb = tl.program_id(1)
    k = kb * BK + tl.arange(0, BK)
    m = k < K
    q = tl.load(Q + n.to(tl.int64) * K + k, mask=m, other=0).to(tl.float32)
    s = tl.load(S + n.to(tl.int64) * (K // 32) + k // 32, mask=m, other=0.0).to(tl.float32)
    tl.store(W + n.to(tl.int64) * K + k, (q * s).to(tl.bfloat16), mask=m)


def q8_dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """int8 [1, N, K] + fp16 scales [1, N, K/32] -> bf16 [N, K] (transient prefill weight)."""
    _, N, K = q.shape
    w = torch.empty(N, K, device=q.device, dtype=torch.bfloat16)
    _q8_dequant_kernel[(N, triton.cdiv(K, 1024))](q, s, w, K, BK=1024)
    return w
