"""DPAS grouped MoE prefill (exl3xpu/_inkling2_moe.so) vs the dequant+oneDNN path, real checkpoint experts.
Correctness (rel err) and interleaved min-of-N timing for T tokens (top-8 of 128, random routing)."""
import sys, time, torch
sys.path.insert(0, "/opt/trellis-serve/xpu")
from safetensors import safe_open
from exl3xpu.inkling2.moe_triton import Inkling2ExpertsSYCL
torch.manual_seed(0)
f = safe_open("/models/Inkling-2-Mini/model-00001-of-00001.safetensors", "pt", device="cpu")
def load(il):
    e = f"model.llm.layers.{il}.mlp.experts."
    g = lambda n: f.get_tensor(e + n).to("xpu")
    if (e + "w13_weight_amax") in f.keys():
        return Inkling2ExpertsSYCL(g("w13_weight"), g("w13_weight_scale"), g("w2_weight"), g("w2_weight_scale"),
                                   g("w13_weight_amax") / g("w13_weight_outer_scale_denominator"), g("w2_weight_amax") / g("w2_weight_outer_scale_denominator"))
    return Inkling2ExpertsSYCL(g("w13_weight"), g("w13_weight_scale"), g("w2_weight"), g("w2_weight_scale"))
cfgs = [(int(a.split(",")[0]), int(a.split(",")[1])) for a in sys.argv[1:]] or [(32, 8)]
for il in (0, 20):
    m = load(il)
    for T in (512, 4096):
        x = (torch.randn(T, 2304, device="xpu") * 0.7).bfloat16()
        logits = torch.randn(T, 128, device="xpu")
        ids = logits.topk(8, -1).indices.int(); tw = torch.rand(T, 8, device="xpu")
        ref = m.prefill_dequant(x, ids, tw)
        for mb, tpw in cfgs:
            y = m.prefill_dpas(x, ids, tw, mb=mb, tw=tpw)
            err = ((y - ref).norm() / ref.norm()).item()
            print(f"L{il} {'nvfp4' if m.nvfp4 else 'mxfp8'} T={T} MB{mb} TW{tpw}: rel err {err:.2e} {'OK' if err < 1e-2 else 'FAIL'}", flush=True)
        # interleaved min-of-N timing
        fns = {"dequant": lambda: m.prefill_dequant(x, ids, tw)}
        for mb, tpw in cfgs:
            fns[f"dpas MB{mb} TW{tpw}"] = (lambda mb=mb, tq=tpw: m.prefill_dpas(x, ids, tw, mb=mb, tw=tq))
        best = {k: 1e9 for k in fns}
        for rep in range(4):
            for k, fn in fns.items():
                torch.xpu.synchronize(); t0 = time.perf_counter(); fn(); torch.xpu.synchronize()
                best[k] = min(best[k], time.perf_counter() - t0)
        flops = 2 * T * 8 * (2304 * 2304 + 2304 * 1152)
        print("   " + "  ".join(f"{k}: {v*1e3:.1f} ms ({flops/v/1e12:.1f} TF)" for k, v in best.items()), flush=True)
    del m; torch.xpu.empty_cache()
