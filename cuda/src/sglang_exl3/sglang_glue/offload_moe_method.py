"""Exl3OffloadMoEMethod: SGLang FusedMoE quant method with the routed experts offloaded to pinned host memory and a
global GPU expert cache (kernels/offload_runtime.OffloadRuntime). Selected by Exl3Config.get_quant_method when
SGLANG_EXL3_MOE_OFFLOAD=1 for main-model MoE layers (the MTP draft layer keeps Exl3MoEMethod).

Weights: the expert placeholders are registered like Exl3MoEMethod's (so SGLang's loader finds every expert tensor),
but their loader DISCARDS the tensors: the experts are read once, straight from the shards, by HostExpertStore when
the first layer finishes loading (all layers in one pass). To also skip SGLang's own read of those tensors, filter the
weight iterator with `is_offloaded_expert_key(name)`.

Knobs (env): SGLANG_EXL3_EXPERT_CACHE_GB (slot-pool byte budget, GB) or SGLANG_EXL3_OFFLOAD_SLOTS (slots) or
SGLANG_EXL3_OFFLOAD_CACHE_GB (default 8 GB),
SGLANG_EXL3_OFFLOAD_PREFILL_MIN (tokens from which the staged prefill path is used, default 192 = K05 crossover),
SGLANG_EXL3_OFFLOAD_STATS_EVERY (log per-layer hit rates every N decode forwards of layer 0; 0 = off).
"""
from __future__ import annotations

import logging
import os
import re

import torch
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

from ..kernels import offload_moe as om
from ..kernels.offload_runtime import OffloadRuntime
from ..kernels.offload_store import HostExpertStore
from .moe import Exl3MoEMethod

logger = logging.getLogger(__name__)
_RUNTIMES: dict = {}
_LAYER_RE = re.compile(r"layers\.(\d+)\.")
_KEY_RE = re.compile(r"^(.*layers\.)(\d+)\.mlp\.experts\.\d+\.(gate|up|down)_proj\.")


def offload_enabled() -> bool:
    return os.environ.get("SGLANG_EXL3_MOE_OFFLOAD", "0") in ("1", "gpu_cache")


def elastic_exit_all() -> None:
    """Before a decode forward: every runtime still in prefill mode re-takes its flex chunks (no-op otherwise)."""
    for rt in _RUNTIMES.values():
        rt.elastic_exit()


def elastic_enter_all() -> None:
    """Before an eager (extend / prefill) forward: hand the flex chunks back to torch for its activations."""
    for rt in _RUNTIMES.values():
        rt.elastic_enter()


def offload_stats(reset: bool = False) -> dict:
    """Per-layer counters of every live runtime: decode hits / misses (device counters), prefill copied / resident
    experts, slots, resident slots. Host sync: call outside CUDA-graph capture (e.g. per request for logging)."""
    return {f"{k[0]}@cuda:{k[1]}": rt.stats(reset) for k, rt in _RUNTIMES.items()}


def is_offloaded_expert_key(name: str) -> bool:
    """True for checkpoint / SGLang weight names of main-model routed experts (their data is read by the store)."""
    return offload_enabled() and ".mlp.experts." in name \
        and not name.startswith("mtp.") and "shared_expert" not in name


def _slots_from_env(record_bytes: int, staging_bytes: int = 0) -> int:
    v = os.environ.get("SGLANG_EXL3_EXPERT_CACHE_GB", "")
    if v.strip().lower() == "auto":
        # auto-fit: free VRAM now (weights loaded, before KV / mamba / graphs are sized) minus the prefill staging and
        # a reserve for everything SGLang allocates later (SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB; the default is
        # calibrated on the 3090 serve config: KV 210k fp8, mamba 16, 8k chunks, graphs bs 1-3)
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        reserve = float(os.environ.get("SGLANG_EXL3_EXPERT_CACHE_RESERVE_GB", "8.9")) * 1e9
        nbytes = max(0, free - staging_bytes - reserve)
        logger.info("EXL3 offload: expert cache auto-fit: free %.2f GB of %.2f, staging %.2f GB, reserve %.2f GB -> %.2f GB",
                    free / 1e9, total / 1e9, staging_bytes / 1e9, reserve / 1e9, nbytes / 1e9)
        return int(nbytes // record_bytes)
    if v:                                                          # integration contract: byte budget
        return int(float(v) * 1e9 // record_bytes)
    if os.environ.get("SGLANG_EXL3_OFFLOAD_SLOTS"):
        return int(os.environ["SGLANG_EXL3_OFFLOAD_SLOTS"])
    gb = float(os.environ.get("SGLANG_EXL3_OFFLOAD_CACHE_GB", "8"))
    return int(gb * 1e9 // record_bytes)


def get_runtime(config, num_experts: int, hidden: int, inter: int, bits: int, codebook: int) -> OffloadRuntime:
    """One OffloadRuntime per (model path, device); built (store loaded + pinned, cache allocated) on first use."""
    dev = torch.cuda.current_device()
    key = (config.model_path if hasattr(config, "model_path") else id(config), dev)
    rt = _RUNTIMES.get(key)
    if rt is not None:
        return rt
    layers, prefix = set(), None
    for k in config.modules:
        m = _KEY_RE.match(k + ".")
        if m and not k.startswith("mtp."):
            layers.add(int(m.group(2))); prefix = m.group(1)
    layers = sorted(layers)
    model_dir = getattr(config, "model_path", None) or config.path
    store = HostExpertStore(model_dir, layers, num_experts, hidden, inter, bits, prefix=prefix)
    st = store.load()
    parts = int(os.environ.get("SGLANG_EXL3_OFFLOAD_STAGING_PARTS", "1"))
    staging_bytes = 2 * (-(-num_experts // parts)) * store.lay.record_bytes
    # elastic cache (SGLANG_EXL3_OFFLOAD_ELASTIC_GB): up to that many GB of extra slots, allocated at the first decode
    # from the VRAM left free after SGLang's pools and graphs (minus SGLANG_EXL3_OFFLOAD_ELASTIC_MARGIN_GB) and handed
    # back to torch while a prefill runs (activations + the staging buffers, which are then not resident at all)
    elastic_gb = float(os.environ.get("SGLANG_EXL3_OFFLOAD_ELASTIC_GB", "0") or 0)
    elastic_slots = int(elastic_gb * 1e9 // store.lay.record_bytes)
    slots = _slots_from_env(store.lay.record_bytes, 0 if elastic_slots else staging_bytes)
    rt = OffloadRuntime(store, slots + elastic_slots, codebook,
                        prefill_min_tokens=int(os.environ.get("SGLANG_EXL3_OFFLOAD_PREFILL_MIN", "192")),
                        staging_parts=int(os.environ.get("SGLANG_EXL3_OFFLOAD_STAGING_PARTS", "1")),
                        prefill_subchunk=int(os.environ.get("SGLANG_EXL3_OFFLOAD_PREFILL_SUBCHUNK", str(1 << 30))),
                        elastic_slots=elastic_slots)
    logger.info("EXL3 offload: %d layers x %d experts pinned (%.1f GB) in %.1fs (read wait %.1fs, relayout %.1fs, "
                "register %.1fs); cache %d slots (%.1f GB)", store.L, num_experts, store.L * store.bank_bytes / 1e9,
                st.seconds_total, st.seconds_read_wait, st.seconds_relayout, st.seconds_register, slots,
                slots * store.lay.record_bytes / 1e9)
    _RUNTIMES[key] = rt
    return rt


class Exl3OffloadMoEMethod(Exl3MoEMethod):
    def __init__(self, config, prefix: str, experts: dict):
        super().__init__(config, prefix, experts)
        m = _LAYER_RE.search(prefix)
        if m is None:
            raise ValueError(f"{prefix}: cannot find the layer index for the offloaded MoE")
        self.layer_id = int(m.group(1))
        self._stats_every = int(os.environ.get("SGLANG_EXL3_OFFLOAD_STATS_EVERY", "0"))
        self._calls = 0

    def _make_loader(self, layer, suffix):
        def load(param, loaded_weight, weight_name=None, shard_id=None, expert_id=None, *a, **k):
            layer.exl3_discarded = getattr(layer, "exl3_discarded", 0) + 1      # the store reads the shards itself
        return load

    def process_weights_after_loading(self, layer) -> None:
        n = layer.exl3_num_experts
        k, nn_, twice, codebook = self.infos[("gate", 0)][:4]
        bits = twice // 2
        cb = {"3inst": 0, "mcg": 1, "mul1": 2}[codebook] if isinstance(codebook, str) else int(codebook)
        rt = get_runtime(self.config, n, layer.exl3_hidden, layer.exl3_inter, bits, cb)
        layer.exl3_offload = rt
        layer.exl3_store_index = rt.store.layers.index(self.layer_id)
        for w in ("w13", "w2"):
            for suffix in ("trellis", "suh", "svh", "su", "sv", "mcg", "mul1"):
                if hasattr(layer, f"{w}_{suffix}"):
                    delattr(layer, f"{w}_{suffix}")
        layer.exl3_pack, layer.exl3_tables = None, None
        if hasattr(layer, "exl3_store"):
            del layer.exl3_store
        # warm: the align / per-slot kernels compile here, before any graph capture
        top_k = self.moe_runner_config.top_k if self.moe_runner_config else 10
        x = torch.zeros((1, layer.exl3_hidden), dtype=torch.bfloat16, device="cuda")
        ids = torch.full((1, top_k), n, dtype=torch.int32, device="cuda")     # all dropped: no cache side effects
        w = torch.zeros((1, top_k), dtype=torch.float32, device="cuda")
        rt.forward(layer.exl3_store_index, x, ids, w, force="decode")
        torch.cuda.synchronize()

    def _experts(self, layer, x, topk_ids, topk_weights):
        rt = layer.exl3_offload
        try:
            from sglang.srt.model_executor.runner import get_is_capture_mode
            capture = bool(get_is_capture_mode())
        except Exception:  # pragma: no cover
            capture = False
        # capture mode (or stream capture): decode path only - no host sync, graph-safe
        y = rt.forward(layer.exl3_store_index, x, topk_ids, topk_weights, force="decode" if capture else None)
        if (os.environ.get("SGLANG_EXL3_OFFLOAD_LOG_PREFILL", "1") == "1" and layer.exl3_store_index == 0
                and x.shape[0] >= rt.prefill_min_tokens and not torch.cuda.is_current_stream_capturing()):
            # eager prefill of a new request: report (and reset) the decode hit rate accumulated since the last one
            # (decode runs inside CUDA graphs, where no Python code executes)
            s = rt.stats(reset=True)
            if sum(s["decode_hits"]) + sum(s["decode_misses"]):
                logger.info("EXL3 offload: decode hit rate %.4f since last prefill (%d hits, %d misses), resident %d/%d, "
                            "live slots %d, elastic %s%s", s["decode_hit_rate"], sum(s["decode_hits"]),
                            sum(s["decode_misses"]), s["resident"], s["slots"], s["slots_live"], s["elastic"],
                            (", cpu tier " + str(rt.cpu_tier.stats())) if rt.cpu_tier is not None else "")
        if self._stats_every and layer.exl3_store_index == 0 and x.shape[0] < rt.prefill_min_tokens \
                and not torch.cuda.is_current_stream_capturing():
            self._calls += 1
            if self._calls % self._stats_every == 0:
                s = rt.stats()
                logger.info("EXL3 offload: decode hit rate %.3f, resident %d/%d slots", s["decode_hit_rate"],
                            s["resident"], s["slots"])
        return y
