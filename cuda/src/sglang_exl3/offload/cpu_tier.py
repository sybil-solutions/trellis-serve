"""CPU expert tier for the GPU-cache offload (SGLANG_EXL3_OFFLOAD_CPU_TIER=1): at decode sizes a share of each layer's
cache misses is computed by an AVX2 thread pool straight from the pinned host records (the same memory the GPU reads
zero-copy), in parallel with the GPU's zero-copy GEMMs for the rest. Only the GPU's share crosses PCIe.

csrc/offload/cpu_tier/: cpu_tier_host.cpp (pool + worker, ported from FreeToken-EXL3's GLM-5.3 CPU tier, reading the
offload record layout), cpu_tier_dev.cu (split / combine kernels, graph-capturable). JIT-built on first use (host code
with -march=native: the tier is built for the CPU it runs on).

Knobs (env):
  SGLANG_EXL3_OFFLOAD_CPU_TIER       1 = on
  SGLANG_EXL3_CPU_TIER_THREADS       pool threads incl. the worker (default: physical cores - 2, at least 1)
  SGLANG_EXL3_CPU_TIER_CPUS          comma list of CPU ids to pin to (default: physical cores first, from the last)
  SGLANG_EXL3_CPU_TIER_MODE          1 = EXACT (GPU-exact fp16 weights, default), 0 = AFFINE
  SGLANG_EXL3_CPU_TIER_VARIANT       1 = row-band forward (default), 0 = block forward
  SGLANG_EXL3_CPU_TIER_POLICY        "t_zc,t_hit,cpu_a,cpu_b,cpu_tok,max_cpu,force_n,publish_empty" (us), see the .cu
  SGLANG_EXL3_CPU_TIER_TIMEOUT_S     GPU wait bound for a CPU job (default 5 s; a timeout drops that partial: logged)
"""
from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_mods: dict = {}


def enabled() -> bool:
    return os.environ.get("SGLANG_EXL3_OFFLOAD_CPU_TIER", "0") == "1"


def _load():
    with _lock:
        if "host" in _mods:
            return _mods["host"], _mods["dev"]
        from torch.utils import cpp_extension
        here = os.path.dirname(os.path.abspath(__file__))
        src = os.path.normpath(os.path.join(here, "..", "..", "..", "csrc", "offload", "cpu_tier"))
        build = os.environ.get("SGLANG_EXL3_JIT_DIR", os.path.join(src, "..", "..", "build", "jit"))
        os.makedirs(build, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        _mods["host"] = cpp_extension.load(
            name="trellis_cpu_tier_host", sources=[os.path.join(src, "cpu_tier_host.cpp")],
            extra_cflags=["-O3", "-std=c++17", "-march=native", "-mavx2", "-mfma", "-mf16c", "-pthread"],
            extra_include_paths=[src], build_directory=build, verbose=False)
        _mods["dev"] = cpp_extension.load(
            name="trellis_cpu_tier_dev", sources=[os.path.join(src, "cpu_tier_dev.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17"], build_directory=build, verbose=False)
        return _mods["host"], _mods["dev"]


def _physical_cpus() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    seen, first, sib = set(), [], []
    for c in allowed:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id") as f:
                core = int(f.read())
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id") as f:
                pk = int(f.read())
        except OSError:
            core, pk = c, 0
        (sib if (pk, core) in seen else first).append(c)
        seen.add((pk, core))
    return first, sib


class CpuTier:
    MAXT = 8   # tokens per CPU job (decode batch)

    def __init__(self, store, cache, top_k: int = 10):
        from ..kernels import offload_moe as om
        host, dev = _load()
        self.host, self.dev = host, dev
        self.store, self.cache, self.E, self.L = store, cache, store.E, store.L
        lay = store.lay
        H, I = lay.hidden, lay.inter
        mod = om._mod()
        alloc = lambda n: mod.host_alloc_mapped(int(n))
        self._ctrl = alloc(16 * 8)
        self._ctrl.zero_()
        self._hx = alloc(self.MAXT * H * 4)
        self._picks = alloc(self.MAXT * top_k * 3 * 4)
        self._hout = alloc(self.MAXT * H * 4)
        dptr = lambda t: int(mod.host_device_ptr(t))
        self.d_ctrl, self.d_hx, self.d_picks, self.d_hout = (dptr(t) for t in (self._ctrl, self._hx, self._picks, self._hout))
        first, sib = _physical_cpus()
        env_cpus = os.environ.get("SGLANG_EXL3_CPU_TIER_CPUS", "")
        if env_cpus:
            cpus = [int(c) for c in env_cpus.split(",") if c.strip()]
        else:
            # leave the lowest cores for SGLang's own processes (scheduler, tokenizer, detokenizer, n-gram readers)
            cpus = list(reversed(first)) + sib
        threads = int(os.environ.get("SGLANG_EXL3_CPU_TIER_THREADS", "0")) or max(1, len(first) - 2)
        threads = min(threads, len(cpus))
        mode = int(os.environ.get("SGLANG_EXL3_CPU_TIER_MODE", "1"))
        variant = int(os.environ.get("SGLANG_EXL3_CPU_TIER_VARIANT", "1"))
        host.tier_init(threads, cpus[:threads], self._ctrl.view(torch.int64), self._hx.view(torch.float32),
                       self._picks.view(torch.int32), self._hout.view(torch.float32), mode, variant, H, I, self.L)
        for l in range(self.L):
            host.tier_add_layer(l, store.bank(l).data_ptr(), self.E, lay.record_bytes, list(lay.offsets))
        dv = cache.slot_of.device
        # policy (us): t_zc per zero-copy miss (1.86 MB over ~11.5 GB/s PCIe3 x16 ~= 162), t_hit, cpu_a fixed per job,
        # cpu_b per CPU expert (auto-calibrated from the worker's timings when p[8] = 1, times safety p[9]), cpu_tok,
        # max_cpu, force_n, publish_empty, auto, safety. Lives in pinned + mapped memory: the worker updates p[3].
        pol = os.environ.get("SGLANG_EXL3_CPU_TIER_POLICY", "162,2,60,350,0.3,10,-1,0")
        vals = [float(v) for v in pol.split(",")]
        vals += [1.0, 1.15][len(vals) - 8:] if len(vals) < 10 else []
        vals[8] = float(os.environ.get("SGLANG_EXL3_CPU_TIER_AUTO", str(vals[8])))
        self._pol = alloc(16 * 4)
        self.policy_host = self._pol.view(torch.float32)
        self.policy_host.zero_()
        self.policy_host[:len(vals)] = torch.tensor(vals, dtype=torch.float32)
        self.policy = mod.host_as_cuda(self._pol, dv.index if dv.index is not None else torch.cuda.current_device()).view(torch.float32)
        host.tier_set_policy(self.policy_host)
        self.freq = torch.zeros((self.L * self.E,), dtype=torch.float32, device=dv)
        self.dseq = torch.zeros((1,), dtype=torch.int64, device=dv)
        self.dflag = torch.zeros((1,), dtype=torch.int64, device=dv)
        self.stats_t = torch.zeros((8,), dtype=torch.int64, device=dv)
        self.timeout_ns = int(float(os.environ.get("SGLANG_EXL3_CPU_TIER_TIMEOUT_S", "5")) * 1e9)
        self.threads, self.cpus, self.mode, self.variant = threads, cpus[:threads], mode, variant
        self.started = False
        logger.info("EXL3 offload CPU tier: %d threads on cpus %s, mode %d, variant %d, policy %s", threads,
                    cpus[:threads], mode, variant, pol)

    def start(self):
        if not self.started:
            self.host.tier_start(self.cpus[0])
            self.started = True

    def split(self, layer: int, x: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """-> ids with the CPU picks dropped (sentinel E); publishes the CPU job. x [T, H], ids int32 [T, k], w fp32."""
        ids = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
        ids = ids.contiguous()
        out = torch.empty_like(ids)
        E = self.E
        self.dev.ct_split(ids, w, x, E, self.cache.slot_of[layer * E:(layer + 1) * E], self.freq[layer * E:(layer + 1) * E],
                          self.policy, self.d_ctrl, self.d_hx, self.d_picks, self.dseq, layer, out, self.dflag, self.stats_t)
        return out

    def combine(self, y: torch.Tensor) -> torch.Tensor:
        self.dev.ct_combine(y, self.dflag, self.d_ctrl, self.d_hout, self.stats_t, self.timeout_ns)
        return y

    def stats(self) -> dict:
        s = self.stats_t.cpu().tolist()
        h = self.host.tier_stats()
        return {"hits": s[0], "misses": s[1], "cpu_picks": s[2], "calls": s[3], "wait_us_mean": round(s[4] / max(1, s[5]) / 1e3, 1),
                "timeouts": s[6], "jobs": s[7], "worker_jobs": h[0], "worker_experts": h[1],
                "worker_busy_us_mean": round(h[4] / max(1, h[0]) / 1e3, 1), "cpu_us_per_expert_ema": h[5] / 1e3,
                "policy": [round(float(v), 1) for v in self.policy_host[:10].tolist()]}


def get(store, cache, top_k: int = 10) -> CpuTier:
    """The process-wide tier (the host pool is a singleton): built on first use, re-bound to `cache` afterwards."""
    t = _mods.get("tier")
    if t is None or t.store is not store:
        if t is not None:
            raise RuntimeError("CPU tier: one expert store per process")
        t = _mods["tier"] = CpuTier(store, cache, top_k)
        t.start()
    t.cache = cache
    t.freq.zero_()
    return t
