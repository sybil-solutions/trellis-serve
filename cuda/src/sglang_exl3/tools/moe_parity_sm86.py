"""Parity + timing of the two EXL3 MoE paths on one real layer of an EXL3 MoE checkpoint (GPU; run inside the dev
image on the box under the GPU lock):

    python -m sglang_exl3.tools.moe_parity_sm86 /models/Qwen3.6-35B-A3B-EXL3-3.00bpw-H5 --layer 3 --json /w/runs/moe_parity.json

Checks, all on the same inputs (random fp16 activations, zipf-skewed top-k routing, softmax weights):
  (1) SGLang's moe_align_block_size obeys the contract the grouped kernel expects (every slot exactly once, block
      expert ids right, padding = numel, num_post_padded consistent), for every block family;
  (2) grouped kernel (kernels/marlin_moe) and ExLlamaV3 exl3_mgemm (kernels/reference.moe_mgemm) against a float64
      reference of the exact same decoded weights (exllamav3_ext.reconstruct), tokens 1 / 2 / 8 / 64 / 1024;
      gate: grouped error <= 1.25 x mgemm error, all finite;
  (3) CUDA graph capture + replay (fresh input and routing) == eager, bit for bit, both paths;
  (4) per-layer time of both paths at 1 / 2 / 4 / 8 / 16 / 64 / 1024 tokens (CUDA events, graphs for <= 16 tokens).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
from safetensors import safe_open

from ..kernels import marlin_moe, reference
from ..parity.l1_linear import _block_had

_PROJ = ("gate_proj", "up_proj", "down_proj")


def load_layer(model_dir: str, layer: int, num_experts: int, device):
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    probe = [k for k in idx if f".layers.{layer}.mlp.experts.0.gate_proj.trellis" in k]
    pre = probe[0].rsplit("experts.0.", 1)[0]
    handles = {}

    def get(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(model_dir, f), "pt")
        return handles[f].get_tensor(name)

    out = {}
    for p in _PROJ:
        out[p] = tuple([get(f"{pre}experts.{e}.{p}.{s}").contiguous() for e in range(num_experts)] for s in ("trellis", "suh", "svh"))
    cb = "mcg" if f"{pre}experts.0.gate_proj.mcg" in idx else "mul1" if f"{pre}experts.0.gate_proj.mul1" in idx else "3inst"
    return pre, out, {"3inst": 0, "mcg": 1, "mul1": 2}[cb]


def routing(tokens, top_k, num_experts, gen):
    p = 1.0 / torch.arange(1, num_experts + 1, dtype=torch.float64) ** 1.1
    p = p[torch.randperm(num_experts, generator=gen)]
    ids = torch.multinomial(p.expand(tokens, -1), top_k, replacement=False, generator=gen).to(torch.int32)
    w = torch.softmax(torch.randn((tokens, top_k), generator=gen, dtype=torch.float32), dim=-1)
    return ids.cuda(), w.cuda()


def check_align(ids, block, num_experts) -> dict:
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    sorted_ids, expert_ids, num_post = moe_align_block_size(ids, block, num_experts)
    torch.cuda.synchronize()
    numel = ids.numel()
    n_post = int(num_post.item())
    s = sorted_ids[:n_post].cpu()
    e = expert_ids[: n_post // block].cpu()
    flat = ids.reshape(-1).cpu()
    valid = s[s < numel]
    ok_once = valid.numel() == numel and torch.equal(valid.sort().values, torch.arange(numel))
    ok_pad = bool((s[s >= numel] == numel).all())
    ok_exp = bool(all(int(flat[t]) == int(e[i // block]) for i, t in enumerate(s.tolist()) if t < numel))
    return {"block": block, "num_post": n_post, "sorted_len": int(sorted_ids.numel()), "expert_ids_len": int(expert_ids.numel()),
            "each_slot_once": bool(ok_once), "padding_is_numel": ok_pad, "block_expert_ids_right": ok_exp,
            "pass": bool(ok_once and ok_pad and ok_exp)}


def exact_layer(x, ids, wts, t, ref13, ref2, inter):
    tokens, hidden = x.shape
    y = torch.zeros((tokens, hidden), dtype=torch.float64, device="cuda")
    x64 = x.double()
    for e in ids.unique().tolist():
        tok, slot = (ids == e).nonzero(as_tuple=True)
        xe = x64[tok]
        g = _block_had(_block_had(xe * t["gate_proj"][1][e].double()) @ ref13[e, :, :inter].double()) * t["gate_proj"][2][e].double()
        u = _block_had(_block_had(xe * t["up_proj"][1][e].double()) @ ref13[e, :, inter:].double()) * t["up_proj"][2][e].double()
        a = torch.nn.functional.silu(g) * u
        d = _block_had(_block_had(a * t["down_proj"][1][e].double()) @ ref2[e].double()) * t["down_proj"][2][e].double()
        y.index_add_(0, tok, d * wts[tok, slot].double().unsqueeze(1))
    return y


class LazyRef:
    """ref13 / ref2 of exact_layer decoded per expert on demand (exllamav3_ext.reconstruct): the stacked
    references of 512 experts do not fit an 8 GB card. ref[e] / ref[e, rows, cols] like the stacked tensor.
    Not cached (a 2048-token routing touches every expert): exact_layer reads each expert once per projection."""

    def __init__(self, cb, mats):
        self.cb, self.mats = cb, mats

    def _get(self, e):
        return torch.cat([reference.reconstruct(m[e], self.cb) for m in self.mats], dim=1)

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            return self._get(int(idx[0]))[idx[1:]]
        return self._get(int(idx))


class Paths:
    def __init__(self, t, cb, num_experts, inter, bits):
        self.t, self.cb, self.e, self.inter, self.bits = t, cb, num_experts, inter, bits
        self.tables = {p: tuple(reference.pointer_table(list(v)) for v in t[p]) for p in _PROJ}
        self.pack = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], cb)

    def grouped(self, x, ids, wts):
        from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
        block = marlin_moe.moe_block_size(x.shape[0], ids.shape[1], self.e)
        return marlin_moe.run(x, wts, ids, *moe_align_block_size(ids, block, self.e), block, self.pack)

    def mgemm(self, x, ids, wts):
        return reference.moe_mgemm(x, ids, wts, self.tables["gate_proj"], self.tables["up_proj"], self.tables["down_proj"],
                                   self.bits, self.bits, self.inter, self.e, self.cb)


def graph_check(fn, x, ids, wts, gen, num_experts) -> dict:
    """Capture fn on one input, replay on fresh input + routing copied into the captured buffers; compare to eager."""
    xs, ws = x.clone(), wts.clone()
    is_ = ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn(xs, is_, ws)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = fn(xs, is_, ws)
    ids2, w2 = routing(x.shape[0], ids.shape[1], num_experts, gen)
    x2 = (torch.randn(x.shape, generator=gen, dtype=torch.float32) * 0.5).to(x.dtype).cuda()
    xs.copy_(x2); is_.copy_(ids2); ws.copy_(w2)
    g.replay()
    torch.cuda.synchronize()
    replay = y.clone()
    eager = fn(x2, ids2, w2)
    torch.cuda.synchronize()
    same = torch.equal(replay.view(torch.int16), eager.view(torch.int16))
    g.replay(); torch.cuda.synchronize()
    return {"replay_equals_eager": bool(same), "replay_deterministic": bool(torch.equal(y, replay)),
            "max_abs_diff": float((replay.float() - eager.float()).abs().max())}


def timeit(fn, x, ids, wts, iters=50, graph=False) -> float:
    if graph:
        xs, is_, ws = x.clone(), ids.clone(), wts.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(xs, is_, ws); fn(xs, is_, ws)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(xs, is_, ws)
        run = g.replay
    else:
        run = lambda: fn(x, ids, wts)
    for _ in range(5):
        run()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        run()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--json", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.cuda.init()
    gen = torch.Generator().manual_seed(a.seed)
    t0 = time.time()
    pre, t, cb = load_layer(a.model_dir, a.layer, a.experts, "cpu")
    t = {p: tuple([x.cuda() for x in g] for g in t[p]) for p in _PROJ}
    bits = t["gate_proj"][0][0].shape[2] / 16
    hidden, inter = t["gate_proj"][0][0].shape[0] * 16, t["gate_proj"][0][0].shape[1] * 16
    print(f"layer {a.layer} ({pre}): {a.experts} experts, K={bits:g}, cb={cb}, hidden {hidden}, inter {inter}, loaded in {time.time() - t0:.1f}s")
    paths = Paths(t, cb, a.experts, inter, bits)
    if os.environ.get("MOE_PARITY_LAZY_REF", "0") == "1":     # 8 GB cards: decode the references per expert
        ref13, ref2 = LazyRef(cb, (t["gate_proj"][0], t["up_proj"][0])), LazyRef(cb, (t["down_proj"][0],))
    else:
        ref13 = torch.stack([torch.cat([reference.reconstruct(g, cb), reference.reconstruct(u, cb)], dim=1)
                             for g, u in zip(t["gate_proj"][0], t["up_proj"][0])])
        ref2 = torch.stack([reference.reconstruct(d, cb) for d in t["down_proj"][0]])
    out = {"model": a.model_dir, "layer": a.layer, "experts": a.experts, "K": bits, "codebook": cb,
           "device": torch.cuda.get_device_name(), "align": {}, "outputs": {}, "graph": {}, "timing_us": {}}

    # (1) align contract, every block family
    for tokens in (1, 8, 64, 1024):
        ids, _ = routing(tokens, a.top_k, a.experts, gen)
        block = marlin_moe.moe_block_size(tokens, a.top_k, a.experts)
        out["align"][f"{tokens}"] = check_align(ids, block, a.experts)
        print("align", tokens, out["align"][f"{tokens}"])
    for block in marlin_moe.BLOCK_SIZES:
        ids, _ = routing(64, a.top_k, a.experts, gen)
        out["align"][f"64:block{block}"] = check_align(ids, block, a.experts)

    # (2) outputs vs float64
    for tokens in (1, 2, 8, 64, 1024):
        ids, wts = routing(tokens, a.top_k, a.experts, gen)
        x = (torch.randn((tokens, hidden), generator=gen, dtype=torch.float32) * 0.5).to(torch.float16).cuda()
        exact = exact_layer(x, ids, wts, t, ref13, ref2, inter)
        scale = exact.abs().mean().item()
        rel = lambda y: (y.double() - exact).abs().mean().item() / scale
        yg, ym = paths.grouped(x, ids, wts), paths.mgemm(x, ids, wts)
        yb = paths.grouped(x.bfloat16(), ids, wts)
        r = {"err64_grouped": rel(yg), "err64_mgemm": rel(ym), "err64_grouped_bf16_in": rel(yb),
             "finite": bool(torch.isfinite(yg).all() and torch.isfinite(ym).all()),
             "max_rows_per_expert": int(torch.bincount(ids.reshape(-1).long(), minlength=a.experts).max())}
        r["pass"] = r["finite"] and r["err64_grouped"] <= 1.25 * r["err64_mgemm"]
        out["outputs"][str(tokens)] = r
        print("outputs", tokens, r)

    # (3) CUDA graphs
    for tokens in (1, 2, 8):
        ids, wts = routing(tokens, a.top_k, a.experts, gen)
        x = (torch.randn((tokens, hidden), generator=gen, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()
        out["graph"][f"grouped:{tokens}"] = graph_check(paths.grouped, x, ids, wts, gen, a.experts)
        out["graph"][f"mgemm:{tokens}"] = graph_check(paths.mgemm, x, ids, wts, gen, a.experts)
        print("graph", tokens, out["graph"][f"grouped:{tokens}"], out["graph"][f"mgemm:{tokens}"])

    # (4) timing per layer
    for tokens in (1, 2, 4, 8, 16, 64, 1024):
        ids, wts = routing(tokens, a.top_k, a.experts, gen)
        x = (torch.randn((tokens, hidden), generator=gen, dtype=torch.float32) * 0.5).to(torch.bfloat16).cuda()
        graph = tokens <= 16
        r = {"grouped": timeit(paths.grouped, x, ids, wts, graph=graph, iters=50 if tokens <= 64 else 10),
             "mgemm": timeit(paths.mgemm, x, ids, wts, graph=graph, iters=50 if tokens <= 64 else 10), "cuda_graph": graph}
        out["timing_us"][str(tokens)] = r
        print("timing us", tokens, r)
    out["pass"] = (all(v["pass"] for v in out["align"].values()) and all(v["pass"] for v in out["outputs"].values())
                   and all(v["replay_equals_eager"] for v in out["graph"].values()))
    print("MOE_PARITY", "PASS" if out["pass"] else "FAIL")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
