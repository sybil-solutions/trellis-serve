"""Correctness + microbenchmarks of the pointer-table MoE path (kernels/offload_moe) on one REAL layer.

    python -m sglang_exl3.tools.offload_moe_bench /models/turboderp-Qwen3.8-Flash-Next-exl3-3.05bpw_h5_ng5 \
        --layer 3 --json /runs/e1.json [--skip-correctness] [--skip-bench] [--reps 50]

Correctness (all on real expert tensors of the checkpoint):
  decode:  every decoded weight W_hat read back through the pointer-table GEMM (identity rows, no output transform),
           from host-resident AND device-resident records, == exllamav3_ext.reconstruct AND == trellis_core
           reference.reconstruct_inner, bit for bit (int16 view), gate / up / down of sampled experts;
  layer:   pointer-table outputs (all device / all host / mixed) == stacked grouped path (marlin_moe.run), bit for bit;
           error vs float64 of the same decoded weights next to exllamav3 exl3_mgemm's;
  graph:   capture once, rewrite the table (experts move host <-> device) + new inputs, replay == eager, bit for bit.
Benchmarks (median over reps, CUDA events per rep, distinct routings per rep so host data is not L2-resident):
  decode tokens 1/4/16 x {all device, all host (zero-copy), mixed 50/50}; prefill 2048/8192 tokens from a device
  staging buffer; H2D of a whole layer's records into a staging buffer alone and overlapped with the prefill on a
  second stream; zero-copy read bandwidth of plain SM reads of host memory (torch sum over the mapped bank).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import torch

from ..kernels import marlin_moe, offload_moe, reference
from .moe_parity_sm86 import load_layer, exact_layer

_PROJ = ("gate_proj", "up_proj", "down_proj")


FAST_ALIGN = os.environ.get("OFFLOAD_FAST_ALIGN", "0") == "1"


def align(ids, block, e):
    if FAST_ALIGN and ids.numel() <= 4096:
        return offload_moe.align_decode(ids, block, e)
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size
    return moe_align_block_size(ids, block, e, ignore_invalid_expert=True)   # drop sentinel = e


def uniform_routing(tokens, top_k, e, gen, device="cuda"):
    ids = torch.stack([torch.randperm(e, generator=gen)[:top_k] for _ in range(tokens)]).to(torch.int32)
    w = torch.softmax(torch.randn((tokens, top_k), generator=gen), dim=-1)
    return ids.to(device), w.to(device)


def eq16(a, b) -> bool:
    return torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def median_ms(samples):
    return statistics.median(samples)


class Layer:
    def __init__(self, model_dir, layer):
        t0 = time.time()
        self.pre, t, self.cb = load_layer(model_dir, layer, 512, "cpu")
        self.t_cpu = t
        g0 = t["gate_proj"][0][0]
        self.e = len(t["gate_proj"][0])
        self.bits = g0.shape[2] // 16
        self.hidden, self.inter = g0.shape[0] * 16, g0.shape[1] * 16
        self.lay = offload_moe.layout(self.hidden, self.inter, self.bits)
        self.host = offload_moe.build_bank(t["gate_proj"], t["up_proj"], t["down_proj"], self.lay, "host")
        self.dev = self.host.cuda()                                   # device staging / arena holding every expert
        self.host_base = offload_moe.base_address(self.host)
        self.dev_base = self.dev.data_ptr()
        self.offs = self.lay.offsets_tensor("cuda")
        self.pack = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], self.cb)
        self.load_s = time.time() - t0
        ar = torch.arange(self.e, dtype=torch.int64, device="cuda")
        self.bases_dev = self.dev_base + ar * self.lay.record_bytes
        self.bases_host = self.host_base + ar * self.lay.record_bytes
        self.table = offload_moe.new_table(self.e)

    def set_table(self, where: str, host_mask=None):
        if where == "device":
            b = self.bases_dev
        elif where == "host":
            b = self.bases_host
        else:
            b = torch.where(host_mask, self.bases_host, self.bases_dev)
        offload_moe.fill_table_(self.table, b, self.offs)

    def run_ptr(self, x, ids, w):
        block = marlin_moe.moe_block_size(x.shape[0], ids.shape[1], self.e)
        return offload_moe.run(x, w, ids, *align(ids, block, self.e), block, self.table, self.lay, self.cb)

    def run_stack(self, x, ids, w):
        block = marlin_moe.moe_block_size(x.shape[0], ids.shape[1], self.e)
        return marlin_moe.run(x, w, ids, *align(ids, block, self.e), block, self.pack)


def check_decode(L: Layer, experts) -> dict:
    """W_hat through the pointer-table GEMM == exllamav3 reconstruct == core reference, bit for bit."""
    from trellis_core import reference as core_ref
    mod = offload_moe._mod()
    res = {}
    for where in ("host", "device"):
        L.set_table(where)
        for e in experts:
            H, I = L.hidden, L.inter
            eye_h = torch.eye(H, dtype=torch.float16, device="cuda")
            a13 = torch.cat([eye_h, eye_h])                          # two input slabs (gate, up)
            c13 = torch.empty((H, 2 * I), dtype=torch.float16, device="cuda")
            ids = torch.full((H, 1), e, dtype=torch.int32, device="cuda")
            s, ei, p = align(ids, 64, L.e)
            mod.moe_gemm_ptr(a13, c13, L.table, offload_moe.F_W13, -1, L.bits, s, ei, p, 64, I, L.cb)
            eye_i = torch.eye(I, dtype=torch.float16, device="cuda")
            c2 = torch.empty((I, H), dtype=torch.float16, device="cuda")
            ids2 = torch.full((I, 1), e, dtype=torch.int32, device="cuda")
            s2, ei2, p2 = align(ids2, 64, L.e)
            mod.moe_gemm_ptr(eye_i, c2, L.table, offload_moe.F_W2, -1, L.bits, s2, ei2, p2, 64, 0, L.cb)
            got = {"gate_proj": c13[:, :I], "up_proj": c13[:, I:], "down_proj": c2}
            for proj in _PROJ:
                tr = L.t_cpu[proj][0][e].cuda()
                ex = reference.reconstruct(tr, L.cb)
                core = core_ref.reconstruct_inner(tr, L.bits, L.cb)
                g = got[proj]
                res[f"{where}:e{e}:{proj}"] = {
                    "eq_exllamav3": eq16(g, ex), "eq_core_ref": eq16(g, core), "exllamav3_eq_core": eq16(ex, core),
                    "mismatch_vs_exllamav3": int((g.view(torch.int16) != ex.view(torch.int16)).sum())}
    return res


def check_layer(L: Layer, gen) -> dict:
    t_gpu = {p: tuple([x.cuda() for x in g] for g in L.t_cpu[p]) for p in _PROJ}
    tables = {p: tuple(reference.pointer_table(list(v)) for v in t_gpu[p]) for p in _PROJ}
    if os.environ.get("MOE_PARITY_LAZY_REF", "0") == "1":     # 8 GB cards: decode the references per expert
        from .moe_parity_sm86 import LazyRef
        ref13 = LazyRef(L.cb, (t_gpu["gate_proj"][0], t_gpu["up_proj"][0]))
        ref2 = LazyRef(L.cb, (t_gpu["down_proj"][0],))
    else:
        ref13 = torch.stack([torch.cat([reference.reconstruct(g, L.cb), reference.reconstruct(u, L.cb)], dim=1)
                             for g, u in zip(t_gpu["gate_proj"][0], t_gpu["up_proj"][0])])
        ref2 = torch.stack([reference.reconstruct(d, L.cb) for d in t_gpu["down_proj"][0]])
    out = {}
    for tokens in (1, 4, 16, 128, 2048):
        ids, w = uniform_routing(tokens, 10, L.e, gen)
        x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.float16).cuda()
        ys = L.run_stack(x, ids, w)
        r = {}
        for where in ("device", "host", "mixed"):
            L.set_table(where, torch.rand(L.e, generator=gen).cuda() < 0.5)
            yp = L.run_ptr(x, ids, w)
            r[f"{where}_eq_stacked"] = eq16(yp, ys)
        exact = exact_layer(x, ids, w, t_gpu, ref13, ref2, L.inter)
        scale = exact.abs().mean().item()
        rel = lambda y: (y.double() - exact).abs().mean().item() / scale
        r["err64_ptr"] = rel(yp)
        r["err64_exl3_mgemm"] = rel(reference.moe_mgemm(x, ids, w, tables["gate_proj"], tables["up_proj"],
                                                         tables["down_proj"], L.bits, L.bits, L.inter, L.e, L.cb))
        r["max_abs_ptr_vs_f64"] = float((yp.double() - exact).abs().max())
        r["finite"] = bool(torch.isfinite(yp).all())
        out[str(tokens)] = r
        print("layer", tokens, r, flush=True)
    del t_gpu, tables, ref13, ref2
    torch.cuda.empty_cache()
    return out


def check_graph(L: Layer, gen) -> dict:
    out = {}
    for tokens in (1, 8):
        ids, w = uniform_routing(tokens, 10, L.e, gen)
        x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        xs, is_, ws = x.clone(), ids.clone(), w.clone()
        L.set_table("device")
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                L.run_ptr(xs, is_, ws)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = L.run_ptr(xs, is_, ws)
        ok = True
        for trial in range(4):
            ids2, w2 = uniform_routing(tokens, 10, L.e, gen)
            x2 = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
            L.set_table("mixed", torch.rand(L.e, generator=gen).cuda() < (0.25 * trial + 0.1))   # experts move
            xs.copy_(x2); is_.copy_(ids2); ws.copy_(w2)
            g.replay(); torch.cuda.synchronize()
            rep = y.clone()
            ok &= eq16(rep, L.run_ptr(x2, ids2, w2)) and eq16(rep, L.run_stack(x2, ids2, w2))
        out[str(tokens)] = {"replay_eq_eager_and_stacked_after_table_rewrites": bool(ok)}
        print("graph", tokens, out[str(tokens)], flush=True)
    return out


def bench_decode(L: Layer, gen, tokens: int, where: str, reps: int) -> dict:
    """Graph-replayed layer; before every rep (untimed): new routing (distinct experts) + table placement."""
    ids, w = uniform_routing(tokens, 10, L.e, gen)
    x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    L.set_table("device")
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            L.run_ptr(x, ids, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        L.run_ptr(x, ids, w)
    routs = [uniform_routing(tokens, 10, L.e, gen) for _ in range(reps + 5)]
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps + 5)]
    host_frac = []
    for i, (r_ids, r_w) in enumerate(routs):
        ids.copy_(r_ids); w.copy_(r_w)
        if where == "mixed":   # exactly half of this step's distinct experts on the host
            u = r_ids.reshape(-1).unique()
            mask = torch.zeros(L.e, dtype=torch.bool, device="cuda")
            mask[u[torch.randperm(u.numel(), generator=gen)[: u.numel() // 2].cuda()]] = True
            L.set_table("mixed", mask)
            host_frac.append(float(mask[u].float().mean()))
        else:
            L.set_table(where)
        torch.cuda.synchronize()
        ev[i][0].record(); g.replay(); ev[i][1].record()
        torch.cuda.synchronize()
    times = [a.elapsed_time(b) * 1000 for a, b in ev[5:]]
    distinct = tokens * 10 if tokens == 1 else None
    u_mean = statistics.mean(int(r[0].reshape(-1).unique().numel()) for r in routs)
    med = median_ms(times)
    host_experts = u_mean * (1.0 if where == "host" else (0.5 if where == "mixed" else 0.0))
    host_bytes = host_experts * L.lay.record_bytes
    return {"tokens": tokens, "where": where, "median_us": med, "p10_us": sorted(times)[len(times) // 10],
            "p90_us": sorted(times)[len(times) * 9 // 10], "reps": len(times), "distinct_experts_mean": u_mean,
            "host_bytes_per_step": host_bytes,
            "zero_copy_GBps": (host_bytes / (med * 1e-6) / 1e9) if host_bytes else None}


def bench_prefill(L: Layer, gen, tokens: int, reps: int, stacked=False, where="device") -> dict:
    ids, w = uniform_routing(tokens, 10, L.e, gen)
    x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    L.set_table(where)
    fn = L.run_stack if stacked else L.run_ptr
    for _ in range(3):
        fn(x, ids, w)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(x, ids, w); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    med = median_ms(times)
    block = marlin_moe.moe_block_size(tokens, 10, L.e)
    _, _, npost = align(ids, block, L.e)
    return {"tokens": tokens, "path": "stacked" if stacked else f"ptr_{where}", "median_ms": med, "reps": reps,
            "tok_per_s_moe_layer": tokens / (med * 1e-3), "block": block, "moe_blocks": int(npost.item()) // block,
            "host_GBps_if_host": (int(npost.item()) // block) * L.lay.record_bytes / (med * 1e-3) / 1e9 if where == "host" else None}


def bench_copy_overlap(L: Layer, gen, tokens: int, reps: int) -> dict:
    """H2D of the whole layer bank (pinned host -> device staging) alone, prefill alone, and both concurrently
    (copy on a side stream, prefill on the current stream reading the OTHER staging buffer = double buffering)."""
    staging = torch.empty_like(L.dev)
    nbytes = L.host.numel()
    side = torch.cuda.Stream()
    ids, w = uniform_routing(tokens, 10, L.e, gen)
    x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    L.set_table("device")
    for _ in range(2):
        staging.copy_(L.host, non_blocking=True); L.run_ptr(x, ids, w)
    torch.cuda.synchronize()
    E = lambda: torch.cuda.Event(enable_timing=True)
    copy_alone, pre_alone, both_wall, both_copy, both_pre = [], [], [], [], []
    for _ in range(reps):
        with torch.cuda.stream(side):
            a, b = E(), E(); a.record(side); staging.copy_(L.host, non_blocking=True); b.record(side)
        torch.cuda.synchronize(); copy_alone.append(a.elapsed_time(b))
        a, b = E(), E(); a.record(); L.run_ptr(x, ids, w); b.record(); torch.cuda.synchronize()
        pre_alone.append(a.elapsed_time(b))
        start = E(); start.record()
        side.wait_event(start)
        with torch.cuda.stream(side):
            ca, cb = E(), E(); ca.record(side); staging.copy_(L.host, non_blocking=True); cb.record(side)
        pa, pb = E(), E(); pa.record(); L.run_ptr(x, ids, w); pb.record()
        end = E(); torch.cuda.current_stream().wait_event(cb); end.record()
        torch.cuda.synchronize()
        both_wall.append(start.elapsed_time(end)); both_copy.append(ca.elapsed_time(cb)); both_pre.append(pa.elapsed_time(pb))
    m = median_ms
    return {"tokens": tokens, "bytes": nbytes, "copy_alone_ms": m(copy_alone), "copy_alone_GBps": nbytes / m(copy_alone) / 1e6,
            "prefill_alone_ms": m(pre_alone), "overlap_wall_ms": m(both_wall), "overlap_copy_ms": m(both_copy),
            "overlap_copy_GBps": nbytes / m(both_copy) / 1e6, "overlap_prefill_ms": m(both_pre),
            "serial_sum_ms": m(copy_alone) + m(pre_alone), "reps": reps}


def bench_zero_copy_read(L: Layer, reps: int) -> dict:
    """SM-driven zero-copy reads: int32 sum over the mapped host bank (no copy engine), vs DMA memcpy."""
    hv = offload_moe._mod().host_as_cuda(L.host, torch.cuda.current_device()).view(torch.int32)
    nbytes = L.host.numel()
    out = {}
    for name, fn in (("sm_sum_int32", lambda: hv.sum(dtype=torch.int64)),
                     ("dma_memcpy_h2d", lambda: L.dev.copy_(L.host, non_blocking=True))):
        fn(); torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
        out[name] = {"median_ms": median_ms(ts), "GBps": nbytes / median_ms(ts) / 1e6, "bytes": nbytes}
    return out


def check_synthetic_k4(gen, experts=16, hidden=2560, inter=640) -> dict:
    """K=4 must keep working: random K=4 MUL1 trellises (valid bit streams), pointer path == stacked path and the
    decoded W_hat == exllamav3 reconstruct, from host and device records."""
    mk = lambda k, n: torch.randint(-32768, 32767, (k // 16, n // 16, 64), generator=gen, dtype=torch.int16)
    sv = lambda n: ((torch.randint(0, 2, (n,), generator=gen) * 2 - 1).half())
    t = {"gate_proj": ([mk(hidden, inter) for _ in range(experts)], [sv(hidden) for _ in range(experts)], [sv(inter) for _ in range(experts)]),
         "up_proj": ([mk(hidden, inter) for _ in range(experts)], [sv(hidden) for _ in range(experts)], [sv(inter) for _ in range(experts)]),
         "down_proj": ([mk(inter, hidden) for _ in range(experts)], [sv(inter) for _ in range(experts)], [sv(hidden) for _ in range(experts)])}
    cb = 2
    lay = offload_moe.layout(hidden, inter, 4)
    host = offload_moe.build_bank(t["gate_proj"], t["up_proj"], t["down_proj"], lay, "host")
    dev = host.cuda()
    pack = marlin_moe.prepare(t["gate_proj"], t["up_proj"], t["down_proj"], cb)
    offs = lay.offsets_tensor("cuda")
    table = offload_moe.new_table(experts)
    ar = torch.arange(experts, dtype=torch.int64, device="cuda")
    res = {}
    for tokens in (1, 8, 256, 256, 256, 1024):
        ids, w = uniform_routing(tokens, 4, experts, gen)
        x = (torch.randn((tokens, hidden), generator=gen) * 0.01).to(torch.float16).cuda()   # random trellis: keep fp16 finite
        block = marlin_moe.moe_block_size(tokens, 4, experts)
        al = align(ids, block, experts)
        ys = marlin_moe.run(x, w, ids, *al, block, pack)
        for where, base in (("device", dev.data_ptr()), ("host", offload_moe.base_address(host))):
            offload_moe.fill_table_(table, base + ar * lay.record_bytes, offs)
            yp = offload_moe.run(x, w, ids, *al, block, table, lay, cb)
            res[f"{tokens}:{where}_eq_stacked"] = eq16(yp, ys)
            if not eq16(yp, ys):
                d = (yp.view(torch.int16) != ys.view(torch.int16))
                print("K4 mismatch", tokens, where, "block", block, "n_diff", int(d.sum()), "rows", d.any(1).nonzero().flatten()[:20].tolist(),
                      "finite ys/yp", bool(torch.isfinite(ys).all()), bool(torch.isfinite(yp).all()),
                      "max abs diff", float((yp.float() - ys.float()).abs().max()), flush=True)
                for blk in (8, 16, 32, 64):
                    al2 = align(ids, blk, experts)
                    a1 = marlin_moe.run(x, w, ids, *al2, blk, pack); a2 = offload_moe.run(x, w, ids, *al2, blk, table, lay, cb)
                    a3 = marlin_moe.run(x, w, ids, *al2, blk, pack)
                    print("  block", blk, "ptr==stack", eq16(a1, a2), "stack deterministic", eq16(a1, a3), flush=True)
        res[f"{tokens}:finite"] = bool(torch.isfinite(ys).all())
    mod = offload_moe._mod()
    for where, base in (("device", dev.data_ptr()), ("host", offload_moe.base_address(host))):
        offload_moe.fill_table_(table, base + ar * lay.record_bytes, offs)
        for e in (0, experts - 1):
            eye = torch.eye(inter, dtype=torch.float16, device="cuda")
            c2 = torch.empty((inter, hidden), dtype=torch.float16, device="cuda")
            s2, ei2, p2 = align(torch.full((inter, 1), e, dtype=torch.int32, device="cuda"), 64, experts)
            mod.moe_gemm_ptr(eye, c2, table, offload_moe.F_W2, -1, 4, s2, ei2, p2, 64, 0, cb)
            res[f"decode:{where}:e{e}:down_eq_exllamav3"] = eq16(c2, reference.reconstruct(t["down_proj"][0][e].cuda(), cb))
    res["pass"] = all(res.values())
    print("K4", res, flush=True)
    return res


def check_masked(L: Layer, gen) -> dict:
    """Hybrid split contract: slots whose id is E (num_experts) are dropped (not read, not summed), so a CPU / other path can serve
    them. y(ids with -1) vs float64 of the kept slots only (masked slots given weight 0)."""
    t_gpu = {p: tuple([x.cuda() for x in g] for g in L.t_cpu[p]) for p in _PROJ}
    if os.environ.get("MOE_PARITY_LAZY_REF", "0") == "1":     # 8 GB cards: decode the references per expert
        from .moe_parity_sm86 import LazyRef
        ref13 = LazyRef(L.cb, (t_gpu["gate_proj"][0], t_gpu["up_proj"][0]))
        ref2 = LazyRef(L.cb, (t_gpu["down_proj"][0],))
    else:
        ref13 = torch.stack([torch.cat([reference.reconstruct(g, L.cb), reference.reconstruct(u, L.cb)], dim=1)
                             for g, u in zip(t_gpu["gate_proj"][0], t_gpu["up_proj"][0])])
        ref2 = torch.stack([reference.reconstruct(d, L.cb) for d in t_gpu["down_proj"][0]])
    out = {}
    for tokens in (1, 16, 512):
        ids, w = uniform_routing(tokens, 10, L.e, gen)
        x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.float16).cuda()
        drop = torch.rand(ids.shape, generator=gen).cuda() < 0.5
        ids_m = torch.where(drop, torch.full_like(ids, L.e), ids)   # sentinel E (align: ignore_invalid_expert=True)
        L.set_table("mixed", torch.rand(L.e, generator=gen).cuda() < 0.5)
        y = L.run_ptr(x, ids_m, w)
        exact = exact_layer(x, ids, torch.where(drop, torch.zeros_like(w), w), t_gpu, ref13, ref2, L.inter)
        err = (y.double() - exact).abs().mean().item() / exact.abs().mean().item()
        out[str(tokens)] = {"err64": err, "eq_stacked": eq16(y, L.run_stack(x, ids_m, w)), "finite": bool(torch.isfinite(y).all()),
                            "pass": err < 2e-3 and bool(torch.isfinite(y).all())}
        print("masked", tokens, out[str(tokens)], flush=True)
    return out


def check_and_bench_admit(L: Layer, gen, reps: int) -> dict:
    """Fused admission: zero-copy misses written into cache slots by the GEMMs themselves.
    Correctness: output == stacked; every admitted slot's record == the host record byte for byte; re-pointing the
    table to the slots gives the same output again. Timing: graph-replayed 1-token decode, all 10 experts host +
    admitted, vs the same without admission."""
    res = {}
    arena = torch.zeros((L.e, L.lay.record_bytes), dtype=torch.uint8, device="cuda")
    admit = offload_moe.new_table(L.e)
    ok_all = True
    for tokens, host_frac in ((1, 1.0), (8, 1.0), (16, 0.5), (64, 1.0)):
        ids, w = uniform_routing(tokens, 10, L.e, gen)
        x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        u = ids.reshape(-1).unique().long()
        host_mask = torch.zeros(L.e, dtype=torch.bool, device="cuda")
        host_mask[u[torch.randperm(u.numel(), generator=gen)[: int(round(u.numel() * host_frac))].cuda()]] = True
        L.set_table("mixed", host_mask)
        adm = u[host_mask[u]]                                            # admit exactly the host-resident routed experts
        arena.zero_()
        dst = torch.zeros(L.e, dtype=torch.int64, device="cuda")
        dst[adm] = arena.data_ptr() + torch.arange(adm.numel(), device="cuda") * L.lay.record_bytes
        offload_moe.fill_admit_(admit, dst, L.offs)
        block = marlin_moe.moe_block_size(tokens, 10, L.e)
        y = offload_moe.run(x, w, ids, *align(ids, block, L.e), block, L.table, L.lay, L.cb, admit=admit)
        ys = L.run_stack(x, ids, w)
        rec_ok = all(torch.equal(arena[i].cpu(), L.host[int(e)]) for i, e in enumerate(adm.tolist()))
        # untouched slots stay zero
        rest_zero = bool((arena[adm.numel():] == 0).all())
        # now serve the admitted experts from their slots
        bases = torch.where(host_mask, L.bases_host, L.bases_dev)
        bases[adm] = dst[adm]
        offload_moe.fill_table_(L.table, bases, L.offs)
        y2 = L.run_ptr(x, ids, w)
        r = {"eq_stacked": eq16(y, ys), "admitted": int(adm.numel()), "records_exact": bool(rec_ok),
             "other_slots_untouched": rest_zero, "from_slots_eq_stacked": eq16(y2, ys)}
        ok_all &= all(v for k, v in r.items() if k != "admitted")
        res[f"{tokens}tok_host{host_frac}"] = r
        print("admit", tokens, r, flush=True)
    res["pass"] = bool(ok_all)
    # timing: 1 token, all 10 host, admission on vs off
    for use_admit in (False, True):
        tokens = 1
        ids, w = uniform_routing(tokens, 10, L.e, gen)
        x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        L.set_table("host")
        admit.zero_()
        block = marlin_moe.moe_block_size(tokens, 10, L.e)
        fn = lambda: offload_moe.run(x, w, ids, *align(ids, block, L.e), block, L.table, L.lay, L.cb,
                                     admit=admit if use_admit else None)
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(); fn()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ts = []
        ar10 = torch.arange(10, device="cuda")
        for i in range(reps + 5):
            r_ids, r_w = uniform_routing(tokens, 10, L.e, gen)
            ids.copy_(r_ids); w.copy_(r_w)
            if use_admit:
                dst = torch.zeros(L.e, dtype=torch.int64, device="cuda")
                dst[r_ids.reshape(-1).long()] = arena.data_ptr() + ((i % 20) * 10 + ar10) * L.lay.record_bytes
                offload_moe.fill_admit_(admit, dst, L.offs)
            torch.cuda.synchronize()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            if i >= 5:
                ts.append(a.elapsed_time(b) * 1000)
        res[f"decode1_host_admit{int(use_admit)}_median_us"] = median_ms(ts)
        print("admit timing", use_admit, median_ms(ts), flush=True)
    return res


def bench_gather_then_compute(L: Layer, gen, reps: int) -> dict:
    """1-token decode, all 10 experts miss: graph-safe copy-in (SM gather of the 10 host records into cache slots by
    torch.index_select on the mapped bank, table rows re-pointed on device) followed by the device-resident layer."""
    tokens = 1
    hostv = offload_moe._mod().host_as_cuda(L.host, torch.cuda.current_device())
    arena = torch.empty((16, L.lay.record_bytes), dtype=torch.uint8, device="cuda")
    slot_rows = arena[:10]
    ar10 = torch.arange(10, dtype=torch.int64, device="cuda")
    ids, w = uniform_routing(tokens, 10, L.e, gen)
    x = (torch.randn((tokens, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
    L.set_table("host")

    def step():
        miss = ids.reshape(-1).to(torch.int64)
        torch.index_select(hostv, 0, miss, out=slot_rows)                          # PCIe: SM gather
        L.table.index_copy_(0, miss, arena.data_ptr() + ar10.unsqueeze(1) * L.lay.record_bytes + L.offs.unsqueeze(0))
        return L.run_ptr(x, ids, w)

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step(); step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = step()
    ts, ok = [], True
    for i in range(reps + 5):
        r_ids, r_w = uniform_routing(tokens, 10, L.e, gen)
        ids.copy_(r_ids); w.copy_(r_w); L.set_table("host"); torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        if i >= 5:
            ts.append(a.elapsed_time(b) * 1000)
        if i < 3:
            ok &= eq16(y, L.run_stack(x, r_ids, r_w))
    # the gather alone
    gs = []
    for _ in range(reps):
        miss = uniform_routing(1, 10, L.e, gen)[0].reshape(-1).to(torch.int64)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); torch.index_select(hostv, 0, miss, out=slot_rows); b.record(); torch.cuda.synchronize()
        gs.append(a.elapsed_time(b) * 1000)
    nb = 10 * L.lay.record_bytes
    return {"median_us": median_ms(ts), "eq_stacked": bool(ok), "gather_only_us": median_ms(gs),
            "gather_GBps": nb / (median_ms(gs) * 1e-6) / 1e9, "reps": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--json", default=None)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--prefill-reps", type=int, default=20)
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--decode-tokens", default="1,4,16")
    ap.add_argument("--prefill-tokens", default="2048,8192")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list: k4,gather (run just these extra experiments)")
    a = ap.parse_args()
    torch.cuda.init()
    gen = torch.Generator().manual_seed(a.seed)
    only = {o for o in a.only.split(",") if o}
    if "k4" in only:
        r = check_synthetic_k4(gen)
        if a.json:
            json.dump({"k4": r}, open(a.json, "w"), indent=1)
        if only == {"k4"}:
            return
    L = Layer(a.model_dir, a.layer)
    if "masked" in only:
        r = check_masked(L, gen)
        if a.json:
            json.dump({"masked": r}, open(a.json.replace(".json", "_masked.json"), "w"), indent=1)
    if "admit" in only:
        r = check_and_bench_admit(L, gen, a.reps)
        print("ADMIT", "PASS" if r["pass"] else "FAIL", flush=True)
        if a.json:
            json.dump({"admit": r}, open(a.json.replace(".json", "_admit.json"), "w"), indent=1)
    if "decode" in only:
        for tokens in [int(t) for t in a.decode_tokens.split(",") if t]:
            for where in ("device", "host", "mixed"):
                print("decode", bench_decode(L, gen, tokens, where, a.reps), flush=True)
    if "prefill_host" in only:
        for tokens in [int(t) for t in a.prefill_tokens.split(",") if t]:
            for where in ("device", "host"):
                print("prefill", bench_prefill(L, gen, tokens, a.prefill_reps, False, where), flush=True)
    if "fewmiss" in only:
        # K07: 1-token layer, exactly m of the 10 routed experts host-resident (zero-copy), the rest in device slots
        ids, w = uniform_routing(1, 10, L.e, gen)
        x = (torch.randn((1, L.hidden), generator=gen) * 0.5).to(torch.bfloat16).cuda()
        L.set_table("device")
        sd = torch.cuda.Stream(); sd.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(sd):
            L.run_ptr(x, ids, w)
        torch.cuda.current_stream().wait_stream(sd)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            L.run_ptr(x, ids, w)
        res = {}
        for m in (0, 1, 2, 3, 5, 10):
            ts = []
            for r in range(a.reps + 5):
                r_ids, r_w = uniform_routing(1, 10, L.e, gen)
                ids.copy_(r_ids); w.copy_(r_w)
                mask = torch.zeros(L.e, dtype=torch.bool, device="cuda")
                mask[r_ids[0, :m].long()] = True
                L.set_table("mixed", mask)
                torch.cuda.synchronize()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
                if r >= 5:
                    ts.append(e0.elapsed_time(e1) * 1000)
            med = statistics.median(ts)
            res[m] = {"median_us": round(med, 1), "model_us": 5.5 * (10 - m) + 73 * m}
            print("FEWMISS", m, res[m], flush=True)
        base = res[0]["median_us"]
        for m in (1, 2, 3, 5, 10):
            extra = res[m]["median_us"] - base
            print("FEWMISS_GBPS", m, "extra us", round(extra, 1), "effective GB/s", round(m * L.lay.record_bytes / (extra * 1e-6) / 1e9, 2), flush=True)
    if "gather" in only:
        r = bench_gather_then_compute(L, gen, a.reps)
        print("gather_then_compute", r, flush=True)
        if a.json:
            json.dump({"gather": r}, open(a.json.replace(".json", "_gather.json"), "w"), indent=1)
        return
    if only:
        return
    res = {"model": a.model_dir, "layer": a.layer, "prefix": L.pre, "experts": L.e, "K": L.bits, "codebook": L.cb,
           "hidden": L.hidden, "inter": L.inter, "record_bytes": L.lay.record_bytes, "offsets": L.lay.offsets,
           "device": torch.cuda.get_device_name(), "load_s": L.load_s}
    print({k: res[k] for k in ("prefix", "experts", "K", "codebook", "record_bytes", "load_s")}, flush=True)
    if not a.skip_correctness:
        dec = check_decode(L, [0, 1, 257, 511])
        res["decode_bitexact"] = dec
        res["decode_all_eq"] = all(v["eq_exllamav3"] and v["eq_core_ref"] for v in dec.values())
        print("DECODE_BITEXACT", res["decode_all_eq"], flush=True)
        res["layer"] = check_layer(L, gen)
        res["graph"] = check_graph(L, gen)
        res["pass"] = (res["decode_all_eq"] and all(all(v[f"{w}_eq_stacked"] for w in ("device", "host", "mixed")) and v["finite"]
                                                    and v["err64_ptr"] <= v["err64_exl3_mgemm"] for v in res["layer"].values())
                       and all(v["replay_eq_eager_and_stacked_after_table_rewrites"] for v in res["graph"].values()))
        print("OFFLOAD_MOE_CORRECTNESS", "PASS" if res["pass"] else "FAIL", flush=True)
    if not a.skip_bench:
        res["decode"] = []
        for tokens in [int(t) for t in a.decode_tokens.split(",") if t]:
            for where in ("device", "host", "mixed"):
                r = bench_decode(L, gen, tokens, where, a.reps)
                res["decode"].append(r)
                print("decode", r, flush=True)
        res["prefill"] = []
        for tokens in [int(t) for t in a.prefill_tokens.split(",") if t]:
            for stacked in (False, True):
                r = bench_prefill(L, gen, tokens, a.prefill_reps, stacked)
                res["prefill"].append(r)
                print("prefill", r, flush=True)
            r = bench_copy_overlap(L, gen, tokens, a.prefill_reps)
            res.setdefault("copy_overlap", []).append(r)
            print("copy_overlap", r, flush=True)
        res["zero_copy_read"] = bench_zero_copy_read(L, 20)
        print("zero_copy_read", res["zero_copy_read"], flush=True)
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
