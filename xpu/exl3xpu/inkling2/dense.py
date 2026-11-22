"""Dense linears for Inkling 2 on XPU: bf16 weights for prefill (XMX GEMM), INT8 group-32 copies (fp16 scales) for
small-M decode/verify through the SYCL GEMV (inkling2_xpu.moe_gemv with one 'expert')."""
from __future__ import annotations

import torch

from . import moe_triton as _mt
from .moe_triton import ops


def quant_q8(w: torch.Tensor, group: int = 32):
    """bf16 [N, K] -> int8 [1, N, K], fp16 scales [1, N, K/group] (symmetric, absmax)."""
    N, K = w.shape
    g = w.float().view(N, K // group, group)
    sc = g.abs().amax(-1).clamp(min=1e-12) / 127.0
    q = torch.round(g / sc[..., None]).clamp(-127, 127).to(torch.int8).view(1, N, K)
    return q, sc.to(torch.float16).view(1, N, K // group)


def quant_fp4(w: torch.Tensor):
    """bf16 [N, K] -> NVFP4-style RTN: uint8 [1, N, K/2] (low nibble = even column), e4m3 block scales [1, N, K/16],
    per-tensor alpha [1] so block scales use the e4m3 range (same layout the expert kernels read)."""
    N, K = w.shape
    g = w.float().view(N, K // 16, 16)
    amax = g.abs().amax(-1)                                         # [N, K/16]
    alpha = (amax.max() / (6.0 * 448.0)).clamp(min=1e-12)
    sc = (amax / (6.0 * alpha)).to(torch.float8_e4m3fn)              # block scale (e4m3)
    scf = sc.float().clamp(min=1e-12) * alpha
    v = g / scf[..., None]                                          # in [-6, 6]
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=w.device)
    mag = (v.abs()[..., None] - lut).abs().argmin(-1)                # nearest e2m1 magnitude
    q = (mag | ((v < 0) & (mag > 0)).to(mag.dtype) << 3).to(torch.uint8).view(N, K)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous().view(1, N, K // 2)
    return packed, sc.view(torch.uint8).view(1, N, K // 16), alpha.view(1).float()


class Q8Linear:
    max_gemv_rows = 8

    def __init__(self, w: torch.Tensor, keep_bf16: bool = True, fp4: bool = False):
        self.N, self.K = w.shape
        self.w = w.to(torch.bfloat16).contiguous() if keep_bf16 else None
        self.alpha = None
        if fp4 and self.K % 16 == 0:
            self.q, self.s, self.alpha = quant_fp4(w)
        else:
            self.q, self.s = quant_q8(w)
        self._align = {}
        self.dequant_prefill = False

    def _plan(self, M: int, device):
        if M not in self._align:
            mb = 1 if M == 1 else 2 if M == 2 else 4 if M <= 4 else 8
            sorted_ids = torch.full((mb,), M, dtype=torch.int32, device=device)
            sorted_ids[:M] = torch.arange(M, dtype=torch.int32, device=device)
            self._align[M] = (sorted_ids, torch.zeros(1, dtype=torch.int32, device=device), mb)
        return self._align[M]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x [M, K] -> fp32 [M, N]"""
        M = x.shape[0]
        if self.w is None and M > self.max_gemv_rows and (self.alpha is not None or not self.dequant_prefill):
            return torch.cat([self(x[i:i + self.max_gemv_rows]) for i in range(0, M, self.max_gemv_rows)])
        if M <= self.max_gemv_rows or (self.w is None and not self.dequant_prefill):
            out = torch.empty(M, self.N, device=x.device, dtype=torch.float32)
            sid, eob, mb = self._plan(M, x.device)
            ops().moe_gemv2(x.contiguous() if x.dtype == torch.float32 else x.to(torch.bfloat16).contiguous(), self.q, self.s, self.alpha, out, sid, eob, None, M, 1, mb, *_mt.GEMV_DENSE)
            return out
        if self.w is None:   # transient bf16 weight from the INT8 copy (no resident bf16 copy kept)
            from .fused import q8_dequant
            return torch.nn.functional.linear(x.to(torch.bfloat16), q8_dequant(self.q, self.s)).float()
        return torch.nn.functional.linear(x.to(torch.bfloat16), self.w).float()
