"""Validate Ornith-sized 4-bit MoE kernels against independent reconstructed weights.

Run in the built XPU environment: python3 xpu/tests/test_moe_k4.py
No model download is required. Exercises shared-USM and device expert pointers,
both activation dtypes, pair/grouped routing, and the above-prefill-threshold path.
"""
import json
import sys
import faulthandler
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "core/src"), str(ROOT / "xpu")]
from trellis_core import reference as ref
from exl3xpu.moe_offload import ExpertStore, pack_expert, planar4

H, I, K, E = 2048, 512, 4, 4
faulthandler.dump_traceback_later(60, repeat=True)
torch.manual_seed(20261115)
torch.set_num_threads(4)
device = torch.device("xpu:0")
experts = []
weights = []
for e in range(E):
    print(f"reference expert {e}", flush=True)
    tensors = {}
    reconstructed = {}
    for role, rows, cols in (("gate", H, I), ("up", H, I), ("down", I, H)):
        tr = torch.randint(-32768, 32768, (rows // 16, cols // 16, 16 * K), dtype=torch.int16)
        assert torch.equal(planar4(tr, K).view(torch.int16), tr)
        tensors[role] = {"trellis": tr, "suh": torch.full((rows,), .05, dtype=torch.float16),
                         "svh": torch.full((cols,), .05, dtype=torch.float16)}
        # Keep the independent bitstream decoder on CPU. Its int64 indexing
        # otherwise triggers a very expensive first-use Intel JIT compilation.
        t = tensors[role]
        print(f"reconstruct {e}/{role}", flush=True)
        reconstructed[role] = ref.weight_orig(t["trellis"], t["suh"], t["svh"], K, 2)
    experts.append(pack_expert(tensors["gate"], tensors["up"], tensors["down"], K))
    weights.append(reconstructed)

store = ExpertStore(H, I, K, E, E, device=device)
print("allocate expert store", flush=True)
store.add_layer("synthetic", torch.stack(experts))
X = store.X
X.moe_set_prefill_min_m(256)
rows = []
for placement in ("host", "device"):
    if placement == "device":
        store.make_resident("synthetic", range(E))
    for dtype in (torch.float16, torch.bfloat16):
        for m in (1, 2, 5, 16, 257, 512):
            x = (torch.randn((m, H)) * .5).to(dtype)
            ids = torch.stack((torch.arange(m) % E, (torch.arange(m) + 1) % E), dim=1).to(torch.int32)
            w = torch.tensor([.3, .7]).expand(m, -1).contiguous()
            dx, di, dw = x.to(device), ids.to(device), w.to(device)
            out = store.forward("synthetic", dx, di, dw).cpu()
            repeated = store.forward("synthetic", dx, di, dw).cpu()
            expected = torch.zeros((m, H))
            for e, ws in enumerate(weights):
                ri, ki = (ids == e).nonzero(as_tuple=True)
                z = x[ri].float()
                z = F.silu(z @ ws["gate"]) * (z @ ws["up"])
                expected.index_add_(0, ri, (z @ ws["down"]) * w[ri, ki, None])
            torch.xpu.synchronize()
            relative_rms = ((out.float() - expected).square().mean() / expected.square().mean()).sqrt().item()
            result = {"placement": placement, "dtype": str(dtype), "M": m,
                      "relative_rms": relative_rms, "deterministic": torch.equal(out, repeated)}
            print(json.dumps(result), flush=True)
            assert torch.isfinite(out).all(), result
            assert relative_rms < .02, result
            assert result["deterministic"], result
            rows.append(result)
print(json.dumps({"status": "pass", "cases": len(rows)}))
