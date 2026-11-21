"""Inkling 2 routed experts on XPU (Triton), straight from the checkpoint's native formats.

NVFP4 (layers 0-19): w uint8 [E, N, K/2] (low nibble = even column), scale fp8-e4m3 [E, N, K/16], alpha[e] = amax/denominator.
MXFP8 (layers 20-23): w fp8-e4m3 [E, N, K], scale uint8 e8m0 [E, N, K/32].
w13 rows interleave gate (even) / up (odd); activation SiTU: 4 tanh(g/4) sigmoid(g) * 25 tanh(u/25).

One grouped kernel serves decode (BM=16), verify and prefill (BM=64): pairs (token, slot) are sorted by expert and
padded per expert to BM (moe_align, on device, no host sync); each program dequantizes one [BN, BK] weight tile
in registers and multiplies it with up to BM activation rows.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _e2m1(q):
    # q: int32 nibble 0..15 -> fp32
    e = (q >> 1) & 3
    m = (q & 1).to(tl.float32)
    mag = tl.where(e == 0, m * 0.5, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    return tl.where((q & 8) != 0, -mag, mag)


@triton.jit
def _e4m3(b):
    # b: int32 byte -> fp32 (e4m3fn, no inf)
    e = (b >> 3) & 15
    m = (b & 7).to(tl.float32)
    mag = tl.where(e == 0, m * 0.001953125, (1.0 + m * 0.125) * tl.exp2((e - 7).to(tl.float32)))
    return tl.where((b & 128) != 0, -mag, mag)


@triton.jit
def _moe_kernel(
    A, W, S, ALPHA, OUT, SORTED, EXPERT_OF_BLOCK, TOPW,
    n_pairs, N, K, a_row_div,
    stride_am, stride_we, stride_wn, stride_se, stride_sn, stride_om,
    NVFP4: tl.constexpr, MUL_TOPW: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(EXPERT_OF_BLOCK + pid_m)
    if e < 0:
        return
    pairs = tl.load(SORTED + pid_m * BM + tl.arange(0, BM))
    pmask = pairs < n_pairs
    arow = tl.where(pmask, pairs // a_row_div, 0)
    ns = pid_n * BN + tl.arange(0, BN)
    nmask = ns < N
    acc = tl.zeros([BM, BN], tl.float32)
    wbase = W + e.to(tl.int64) * stride_we + ns[:, None].to(tl.int64) * stride_wn
    sbase = S + e.to(tl.int64) * stride_se + ns[:, None].to(tl.int64) * stride_sn
    if NVFP4:
        HB: tl.constexpr = BK // 2
        for k0 in range(0, K, BK):
            kb = k0 // 2 + tl.arange(0, HB)                                   # byte columns
            byt = tl.load(wbase + kb[None, :], mask=nmask[:, None], other=0).to(tl.int32)
            sc = _e4m3(tl.load(sbase + (kb[None, :] // 8), mask=nmask[:, None], other=0).to(tl.int32))
            lo = (_e2m1(byt & 15) * sc).to(tl.bfloat16)                       # columns k0 + 2i
            hi = (_e2m1(byt >> 4) * sc).to(tl.bfloat16)                       # columns k0 + 2i + 1
            xe = tl.load(A + arow[:, None] * stride_am + (k0 + 2 * tl.arange(0, HB))[None, :], mask=pmask[:, None], other=0.0)
            xo = tl.load(A + arow[:, None] * stride_am + (k0 + 2 * tl.arange(0, HB) + 1)[None, :], mask=pmask[:, None], other=0.0)
            acc += tl.dot(xe.to(tl.bfloat16), tl.trans(lo))
            acc += tl.dot(xo.to(tl.bfloat16), tl.trans(hi))
        acc *= tl.load(ALPHA + e)
    else:
        for k0 in range(0, K, BK):
            kk = k0 + tl.arange(0, BK)
            w = _e4m3(tl.load(wbase + kk[None, :], mask=nmask[:, None], other=0).to(tl.int32))
            sc = tl.exp2(tl.load(sbase + (kk[None, :] // 32), mask=nmask[:, None], other=127).to(tl.float32) - 127.0)
            x = tl.load(A + arow[:, None] * stride_am + kk[None, :], mask=pmask[:, None], other=0.0)
            acc += tl.dot(x.to(tl.bfloat16), tl.trans((w * sc).to(tl.bfloat16)))
    if MUL_TOPW:
        acc *= tl.load(TOPW + pairs, mask=pmask, other=0.0)[:, None]
    tl.store(OUT + pairs[:, None].to(tl.int64) * stride_om + ns[None, :], acc.to(OUT.dtype.element_ty),
             mask=pmask[:, None] & nmask[None, :])


def moe_align(topk_ids: torch.Tensor, n_experts: int, bm: int):
    """Sort (token, slot) pairs by expert, pad each expert's group to bm. Static-shape, no host sync.
    Returns sorted pair ids [max_blocks*bm] (pad = n_pairs) and expert of each block [max_blocks] (-1 = empty)."""
    n_pairs = topk_ids.numel()
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts = torch.zeros(n_experts, dtype=torch.int64, device=flat.device).scatter_add_(0, flat, torch.ones_like(flat))
    padded = (counts + bm - 1) // bm * bm
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    rank = torch.arange(n_pairs, device=flat.device) - (torch.cumsum(counts, 0) - counts)[flat[order]]
    max_blocks = min(n_pairs, n_experts) + n_pairs // bm + 1
    sorted_ids = torch.full((max_blocks * bm,), n_pairs, dtype=torch.int32, device=flat.device)
    sorted_ids[starts[flat[order]] + rank] = order.to(torch.int32)
    blk_start = torch.arange(max_blocks, device=flat.device) * bm
    ends = torch.cumsum(padded, 0)
    eob = torch.searchsorted(ends, blk_start, right=True)
    eob = torch.where(blk_start < ends[-1], eob, torch.full_like(eob, -1)).to(torch.int32)
    return sorted_ids, eob


class Inkling2ExpertsXPU:
    """Holds one layer's raw expert tensors and runs the routed MLP."""

    def __init__(self, w13, s13, w2, s2, alpha13=None, alpha2=None, gate_cap=4.0, up_cap=25.0):
        self.nvfp4 = alpha13 is not None
        self.w13 = w13.view(torch.uint8)
        self.s13 = s13.view(torch.uint8)
        self.w2 = w2.view(torch.uint8)
        self.s2 = s2.view(torch.uint8)
        self.a13 = alpha13.float().contiguous() if self.nvfp4 else torch.ones(w13.shape[0], device=w13.device)
        self.a2 = alpha2.float().contiguous() if self.nvfp4 else torch.ones(w13.shape[0], device=w13.device)
        self.E, self.N13 = w13.shape[0], w13.shape[1]
        self.H = w2.shape[1]
        self.I = self.N13 // 2
        self.gate_cap, self.up_cap = gate_cap, up_cap

    def _gemm(self, a, w, s, alpha, out, sorted_ids, eob, n_pairs, N, K, a_row_div, topw, bm):
        bn, bk = (64, 128) if bm <= 16 else (128, 64)
        grid = (eob.numel(), triton.cdiv(N, bn))
        _moe_kernel[grid](
            a, w, s, alpha, out, sorted_ids, eob, topw if topw is not None else alpha,
            n_pairs, N, K, a_row_div,
            a.stride(0), w.stride(0), w.stride(1), s.stride(0), s.stride(1), out.stride(0),
            NVFP4=self.nvfp4, MUL_TOPW=topw is not None, BM=bm, BN=bn, BK=bk,
        )

    def __call__(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
        T, k = topk_ids.shape
        n_pairs = T * k
        bm = 16 if n_pairs <= 64 else 64
        sorted_ids, eob = moe_align(topk_ids, self.E, bm)
        x = x.to(torch.bfloat16).contiguous()
        h = torch.empty(n_pairs, self.N13, device=x.device, dtype=torch.float32)
        self._gemm(x, self.w13, self.s13, self.a13, h, sorted_ids, eob, n_pairs, self.N13, x.shape[1], k, None, bm)
        g, u = h[:, 0::2], h[:, 1::2]
        act = (self.gate_cap * torch.tanh(g / self.gate_cap) * torch.sigmoid(g) * (self.up_cap * torch.tanh(u / self.up_cap))).to(torch.bfloat16)
        y = torch.empty(n_pairs, self.H, device=x.device, dtype=torch.float32)
        self._gemm(act, self.w2, self.s2, self.a2, y, sorted_ids, eob, n_pairs, self.H, self.I, 1,
                   topk_w.reshape(-1).float().contiguous(), bm)
        return y.view(T, k, self.H).sum(1)


# ---- SYCL path (exl3xpu/_inkling2.so): decode/verify GEMV + prefill dequant-GEMM ------------------------------
_OPS = None
import os as _os
# (variant, rows per sub-group) of inkling2_xpu.moe_gemv2; sweep: tests/t_inkling2_gemv_sweep.py
GEMV_EXPERT = tuple(int(v) for v in _os.environ.get("INKLING2_GEMV_EXPERT", "3,8").split(","))
GEMV_DENSE = tuple(int(v) for v in _os.environ.get("INKLING2_GEMV_DENSE", "4,4").split(","))


def _mod():
    import sys
    return sys.modules[__name__]


def ops():
    global _OPS
    if _OPS is None:
        import os
        torch.ops.load_library(os.environ.get("INKLING2_LIB") or os.path.join(os.path.dirname(__file__), "..", "_inkling2.so"))
        _OPS = torch.ops.inkling2_xpu
    return _OPS


def _situ(h: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    g, u = h[:, 0::2], h[:, 1::2]
    return (gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g) * (up_cap * torch.tanh(u / up_cap))).to(torch.bfloat16)


class Inkling2ExpertsSYCL(Inkling2ExpertsXPU):
    max_gemv_tokens = 8
    dequant_group = 16

    def __call__(self, x, topk_ids, topk_w):
        T, k = topk_ids.shape
        if T <= self.max_gemv_tokens:
            return self.decode(x, topk_ids, topk_w)
        return self.prefill(x, topk_ids, topk_w)

    _pair_plan = {}
    _dummies = {}

    def _dummy_out(self, dev):
        d = self._dummies.get(dev)
        if d is None:
            d = self._dummies[dev] = torch.empty(1, device=dev)
        return d

    def decode(self, x, topk_ids, topk_w):
        T, k = topk_ids.shape
        n_pairs = T * k
        if _os.environ.get("INKLING2_MOE_PAIR", "1") == "1":
            # one (token, expert) pair per block: no sort/align launches; expert of block i = topk_ids.flat[i]
            mb = 1
            sorted_ids = self._pair_plan.get((n_pairs, x.device))
            if sorted_ids is None:
                sorted_ids = torch.arange(n_pairs, dtype=torch.int32, device=x.device)
                self._pair_plan[(n_pairs, x.device)] = sorted_ids
            eob = topk_ids.reshape(-1).to(torch.int32)
        else:
            mb = 1 if T == 1 else 2 if T == 2 else 4 if T <= 4 else 8
            sorted_ids, eob = moe_align(topk_ids, self.E, mb)
        x = x.contiguous() if x.dtype == torch.float32 else x.to(torch.bfloat16).contiguous()
        o = ops()
        ve, rps = _mod().GEMV_EXPERT
        act = torch.empty(n_pairs, self.I, device=x.device, dtype=torch.bfloat16)
        o.moe_gemv2(x, self.w13, self.s13, self.a13 if self.nvfp4 else None, self._dummy_out(x.device), sorted_ids, eob, None,
                    n_pairs, k, mb, ve, rps, 1, self.gate_cap, self.up_cap, act)
        y = torch.empty(n_pairs, self.H, device=x.device, dtype=torch.float32)
        o.moe_gemv2(act, self.w2, self.s2, self.a2 if self.nvfp4 else None, y, sorted_ids, eob,
                    topk_w.reshape(-1).float().contiguous(), n_pairs, 1, mb, ve, rps)
        return y.view(T, k, self.H).sum(1)

    def prefill(self, x, topk_ids, topk_w):
        """grouped by expert: one row gather, per-expert bf16 GEMMs into shared buffers, one fused SiTU, one scatter-add."""
        from .fused import situ
        T, k = topk_ids.shape
        flat = topk_ids.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.zeros(self.E, dtype=torch.int64, device=x.device).scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.int64)).cpu().tolist()
        tok = order // k
        xs = x.to(torch.bfloat16)[tok]                                   # [P, D] rows grouped by expert
        wts = topk_w.reshape(-1)[order].float()
        P = flat.numel()
        active = [e for e in range(self.E) if counts[e]]
        starts, s0 = [0] * self.E, 0
        for e in range(self.E):
            starts[e] = s0; s0 += counts[e]
        o = ops()
        h = torch.empty(P, self.N13, device=x.device, dtype=torch.bfloat16)
        for g0 in range(0, len(active), self.dequant_group):
            grp = active[g0:g0 + self.dequant_group]
            ids = torch.tensor(grp, device=x.device, dtype=torch.int32)
            w13 = torch.empty(len(grp), self.N13, x.shape[1], device=x.device, dtype=torch.bfloat16)
            o.dequant(self.w13, self.s13, self.a13 if self.nvfp4 else None, ids, w13)
            for i, e in enumerate(grp):
                sl = slice(starts[e], starts[e] + counts[e])
                torch.matmul(xs[sl], w13[i].T, out=h[sl])
        del w13
        act = situ(h, self.gate_cap, self.up_cap)
        del h
        y = torch.empty(P, self.H, device=x.device, dtype=torch.bfloat16)
        for g0 in range(0, len(active), self.dequant_group):
            grp = active[g0:g0 + self.dequant_group]
            ids = torch.tensor(grp, device=x.device, dtype=torch.int32)
            w2 = torch.empty(len(grp), self.H, self.I, device=x.device, dtype=torch.bfloat16)
            o.dequant(self.w2, self.s2, self.a2 if self.nvfp4 else None, ids, w2)
            for i, e in enumerate(grp):
                sl = slice(starts[e], starts[e] + counts[e])
                torch.matmul(act[sl], w2[i].T, out=y[sl])
        out = torch.zeros(T, self.H, device=x.device, dtype=torch.float32)
        out.index_add_(0, tok, y.float() * wts[:, None])
        return out
