"""Dense linears for Inkling 2 on XPU: bf16 weights for prefill (XMX GEMM), INT8 group-32 copies (fp16 scales) for
small-M decode/verify through the SYCL GEMV (inkling2_xpu.moe_gemv with one 'expert')."""
from __future__ import annotations

import torch

from .moe_triton import ops


def quant_q8(w: torch.Tensor, group: int = 32):
    """bf16 [N, K] -> int8 [1, N, K], fp16 scales [1, N, K/group] (symmetric, absmax)."""
    N, K = w.shape
    g = w.float().view(N, K // group, group)
    sc = g.abs().amax(-1).clamp(min=1e-12) / 127.0
    q = torch.round(g / sc[..., None]).clamp(-127, 127).to(torch.int8).view(1, N, K)
    return q, sc.to(torch.float16).view(1, N, K // group)


class Q8Linear:
    max_gemv_rows = 8

    def __init__(self, w: torch.Tensor, keep_bf16: bool = True):
        self.N, self.K = w.shape
        self.w = w.to(torch.bfloat16).contiguous() if keep_bf16 else None
        self.q, self.s = quant_q8(w)
        self._align = {}

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
        if M <= self.max_gemv_rows or self.w is None:
            out = torch.empty(M, self.N, device=x.device, dtype=torch.float32)
            sid, eob, mb = self._plan(M, x.device)
            ops().moe_gemv(x.to(torch.bfloat16).contiguous(), self.q, self.s, None, out, sid, eob, None, M, 1, mb)
            return out
        return torch.nn.functional.linear(x.to(torch.bfloat16), self.w).float()
