"""
vLLM out-of-tree quantization plugin: EXL3 (exllamav3 trellis) weights on Intel XPU.

Registered via the `vllm.general_plugins` entry point, so every vLLM process (API server,
engine core, workers) imports it before model construction.
"""
from __future__ import annotations
import json
import os
import re
from typing import Any

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

logger = init_logger("vllm.exl3xpu")

# vLLM fused module -> checkpoint constituents
FUSED = {
    "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}
QKV_IDS = {"q": 0, "k": 1, "v": 2}
CB_IDS = {"3inst": 0, "mcg": 1, "mul1": 2}


def _norm_key(k: str) -> str:
    """Canonical module name shared by checkpoint keys and vLLM prefixes:
       'model.language_model.layers.3.mlp.gate_proj' / 'language_model.model.layers.3.mlp.gate_proj'
           -> 'layers.3.mlp.gate_proj'
       'mtp.layers.0.mlp.gate_proj' / 'model.mtp.layers.0...' -> 'mtp.layers.0.mlp.gate_proj'
       '...lm_head' -> 'lm_head'"""
    parts = k.split(".")
    if "mtp" in parts:
        return "mtp." + ".".join(parts[parts.index("mtp") + 1:])
    m = re.search(r"(layers\.\d+\..*)$", k)
    if m:
        return m.group(1)
    return parts[-1]


@register_quantization_config("exl3")
class Exl3Config(QuantizationConfig):

    def __init__(self, bits: float, head_bits: int, codebook: str, storage: dict | None = None,
                 mtp_bits: int | None = None):
        super().__init__()
        self.bits = bits
        self.mtp_bits = int(mtp_bits or bits)
        self.head_bits = head_bits
        self.codebook = codebook
        self.storage = storage or {}

    def __repr__(self):
        return f"Exl3Config(bits={self.bits}, head_bits={self.head_bits}, codebook={self.codebook})"

    def get_name(self):
        return "exl3"

    def get_supported_act_dtypes(self):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Exl3Config":
        return cls(bits=config.get("bits", 4), head_bits=config.get("head_bits", 6),
                   codebook=config.get("codebook", "mul1"), mtp_bits=config.get("mtp_bits"))

    def _model_file(self, model_name, fname, revision):
        path = os.path.join(model_name, fname)
        if os.path.isfile(path):
            return path
        try:
            from huggingface_hub import hf_hub_download
            return hf_hub_download(model_name, fname, revision=revision)
        except Exception:
            return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision=None):
        # Which modules are EXL3 is decided by the checkpoint itself: a module is EXL3 iff the weight
        # index has '<module>.trellis'. (quantization_config.json's tensor_storage omits the MTP head.)
        idx_path = self._model_file(model_name, "model.safetensors.index.json", revision)
        qc_path = self._model_file(model_name, "quantization_config.json", revision)
        stored = {}
        if qc_path:
            with open(qc_path) as f:
                for key, v in json.load(f).get("tensor_storage", {}).items():
                    if v.get("quant_format") == "exl3":
                        stored[_norm_key(key)] = int(v.get("bits_per_weight", self.bits))
        if idx_path:
            with open(idx_path) as f:
                wmap = json.load(f)["weight_map"]
            for name in wmap:
                if name.endswith(".trellis"):
                    key = _norm_key(name[: -len(".trellis")])
                    default = self.mtp_bits if key.startswith("mtp.") else int(self.bits)
                    self.storage[key] = stored.get(key, default)
        else:
            self.storage.update(stored)
        logger.info("exl3: %d EXL3 modules (%d in MTP head)", len(self.storage),
                    sum(1 for k in self.storage if k.startswith("mtp.")))

    def _bits_for(self, prefix: str) -> int | None:
        parts = prefix.split(".")
        if "visual" in parts or "vision_tower" in parts:
            return None
        key = _norm_key(prefix)
        if not self.storage:
            if key.endswith("in_proj_ba"):
                return None
            return self.head_bits if key == "lm_head" else int(self.bits)
        base, _, leaf = key.rpartition(".")
        members = FUSED.get(leaf, [leaf])
        names = [f"{base}.{p}" if base else p for p in members]
        bits = [self.storage.get(n) for n in names]
        if all(b is None for b in bits):
            return None
        assert all(b == bits[0] for b in bits), f"exl3: mixed quant in fused module {prefix}: {bits}"
        return bits[0]

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if isinstance(layer, LinearBase):
            bits = self._bits_for(prefix)
            if bits is None:
                return UnquantizedLinearMethod()
            return Exl3LinearMethod(self, bits)
        if isinstance(layer, ParallelLMHead):
            bits = self._bits_for(prefix)
            if bits is None:
                return None
            return Exl3LinearMethod(self, bits)
        return None


class Exl3LinearMethod(LinearMethodBase):

    def __init__(self, config: Exl3Config, bits: int):
        self.config = config
        self.K = int(bits)
        self.cb = CB_IDS[config.codebook]

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        k = input_size_per_partition
        n = sum(output_partition_sizes)
        tp_size = getattr(layer, "tp_size", 1)
        tp_rank = getattr(layer, "tp_rank", 0)
        is_lm_head = isinstance(layer, ParallelLMHead)
        input_sharded = (k != input_size)
        output_sharded = (tp_size > 1) if is_lm_head else (n != output_size)
        if input_sharded and output_sharded:
            raise NotImplementedError("exl3: simultaneous input and output sharding is not supported")
        if input_sharded:
            assert input_size == k * tp_size, f"exl3: uneven input TP shard k={k}, input_size={input_size}, tp={tp_size}"
            assert (tp_rank * k) % 128 == 0, f"exl3: input TP offset must be multiple of 128"
        if is_lm_head and output_sharded:
            v0 = layer.shard_indices.org_vocab_start_index
            v1 = layer.shard_indices.org_vocab_end_index
            assert v0 % 128 == 0 and (v1 - v0) % 128 == 0, \
                f"exl3: ParallelLMHead shard [{v0}, {v1}) must be aligned to 128"
        assert k % 128 == 0 and all(s % 128 == 0 for s in output_partition_sizes), \
            f"exl3: dims must be multiples of 128 (k={k}, n={output_partition_sizes})"
        K = self.K
        layer.exl3_sizes = list(output_partition_sizes)
        layer.exl3_offsets = [sum(output_partition_sizes[:i]) for i in range(len(output_partition_sizes) + 1)]
        layer.exl3_groups = set()
        layer.exl3_K = K
        layer.exl3_cb = self.cb
        layer.exl3_tp_size = tp_size
        layer.exl3_tp_rank = tp_rank
        layer.exl3_input_sharded = input_sharded
        layer.exl3_output_sharded = output_sharded
        layer.exl3_is_lm_head = is_lm_head
        layer.exl3_input_size = input_size
        layer.exl3_output_size = output_size

        def reg(name, tensor, loader):
            p = Parameter(tensor, requires_grad=False)
            p.weight_loader = loader
            layer.register_parameter(name, p)

        reg("trellis", torch.empty((k // 16, n // 16, 16 * K), dtype=torch.int16), self._load_trellis(layer))
        reg("suh", torch.empty((len(output_partition_sizes), k), dtype=torch.float16), self._load_suh(layer))
        reg("svh", torch.empty((n,), dtype=torch.float16), self._load_svh(layer))
        if self.config.codebook != "3inst":
            reg(self.config.codebook, torch.empty((), dtype=torch.int32), lambda p, w, *a, **kw: p.data.copy_(w))

    @staticmethod
    def _shards(layer, loaded, sid=None) -> list[int]:
        if sid is None:
            sid = getattr(loaded, "shard_id", None)
        if sid is None:
            return list(range(len(layer.exl3_sizes)))
        if isinstance(sid, str):
            return [QKV_IDS[sid]]
        if isinstance(sid, int):
            return [sid]
        return list(sid)

    def _range(self, layer, loaded, sid=None):
        s = self._shards(layer, loaded, sid)
        assert s == list(range(s[0], s[-1] + 1)), s
        return s, layer.exl3_offsets[s[0]], layer.exl3_offsets[s[-1] + 1]

    @staticmethod
    def _output_shard_slices(layer, s: list[int], sid=None):
        """Return list of (src_start, dst_start, length) for each sub-shard in s when output-sharded."""
        tp_size = layer.exl3_tp_size
        tp_rank = layer.exl3_tp_rank
        is_qkv = hasattr(layer, "num_kv_head_replicas")
        kv_replicas = getattr(layer, "num_kv_head_replicas", 1)
        out = []
        src_off = 0
        for i in s:
            part_len = layer.exl3_sizes[i]
            dst_start = layer.exl3_offsets[i]
            if is_qkv and i in (1, 2):
                eff_tp = tp_size // kv_replicas
                shard_rank = tp_rank // kv_replicas
            else:
                eff_tp = tp_size
                shard_rank = tp_rank
            full_len = part_len * eff_tp
            src_start = src_off + shard_rank * part_len
            assert src_start % 128 == 0 and part_len % 128 == 0 and dst_start % 128 == 0, \
                f"exl3: unaligned TP output slice src={src_start} dst={dst_start} len={part_len}"
            out.append((src_start, dst_start, part_len))
            src_off += full_len
        return out, src_off

    @staticmethod
    def _draft_block_plan(layer, total_vocab_size: int):
        """Return balanced per-rank draft vocab blocks when EXL3_DRAFT_VOCAB is enabled on ParallelLMHead."""
        path = os.environ.get("EXL3_DRAFT_VOCAB")
        if not (layer.exl3_is_lm_head and path and _spec_is_mtp()):
            return None
        plan = getattr(layer, "_exl3_draft_plan", None)
        if plan is not None:
            return plan
        with open(path) as f:
            spec = json.load(f)
        n_total_blocks = total_vocab_size // 128
        blocks = [int(b) for b in spec["blocks"] if int(b) < n_total_blocks]
        tp_size = layer.exl3_tp_size
        tp_rank = layer.exl3_tp_rank
        # Ensure equal number of blocks per rank so tensor_model_parallel_all_gather has uniform shape
        rem = len(blocks) % tp_size
        if rem:
            blocks = blocks[: len(blocks) - rem]
        my_blocks = torch.tensor(blocks[tp_rank::tp_size], dtype=torch.long)
        # Rank-concatenated order matching all_gather(dim=-1)
        all_blocks_ordered = []
        for r in range(tp_size):
            all_blocks_ordered.extend(blocks[r::tp_size])
        all_blocks = torch.tensor(all_blocks_ordered, dtype=torch.long)
        plan = dict(my_blocks=my_blocks, all_blocks=all_blocks, n_total_blocks=n_total_blocks, n_kept=len(blocks))
        layer._exl3_draft_plan = plan
        return plan

    def _load_trellis(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            assert w.dtype == torch.int16 and w.shape[-1] == param.shape[-1], \
                f"exl3: trellis bits mismatch {tuple(w.shape)} vs {tuple(param.shape)}"
            if layer.exl3_input_sharded:
                k_part = param.shape[0]
                k0 = layer.exl3_tp_rank * k_part
                assert w.shape[0] == k_part * layer.exl3_tp_size and w.shape[1] * 16 == b - a, \
                    (w.shape, param.shape, a, b)
                param.data[:, a // 16: b // 16].copy_(w[k0: k0 + k_part])
            elif layer.exl3_is_lm_head and layer.exl3_output_sharded:
                assert w.shape[0] == param.shape[0], (w.shape, param.shape)
                v0 = layer.shard_indices.org_vocab_start_index
                v1 = layer.shard_indices.org_vocab_end_index
                vlen = v1 - v0
                param.data[:, : vlen // 16].copy_(w[:, v0 // 16: v1 // 16])
                if vlen // 16 < param.shape[1]:
                    param.data[:, vlen // 16:].zero_()
                plan = self._draft_block_plan(layer, w.shape[1] * 16)
                if plan is not None:
                    my_b = plan["my_blocks"]
                    tiles = (my_b[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(w.device)
                    layer._exl3_draft_trellis = w.index_select(1, tiles).contiguous()
            elif layer.exl3_output_sharded:
                assert w.shape[0] == param.shape[0], (w.shape, param.shape)
                slices, src_total = self._output_shard_slices(layer, s, sid)
                assert w.shape[1] * 16 == src_total, (w.shape, src_total, s)
                for src_s, dst_s, length in slices:
                    param.data[:, dst_s // 16: (dst_s + length) // 16].copy_(
                        w[:, src_s // 16: (src_s + length) // 16]
                    )
            else:
                assert w.shape[1] * 16 == b - a and w.shape[0] == param.shape[0], (w.shape, a, b)
                param.data[:, a // 16: b // 16].copy_(w)
            layer.exl3_groups.add(tuple(s))
        return f

    def _load_suh(self, layer):
        def f(param, w, sid=None):
            s, _, _ = self._range(layer, w, sid)
            if layer.exl3_input_sharded:
                k_part = param.shape[1]
                k0 = layer.exl3_tp_rank * k_part
                w = w[k0: k0 + k_part]
            for i in s:
                param.data[i].copy_(w)
        return f

    def _load_svh(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            if layer.exl3_is_lm_head and layer.exl3_output_sharded:
                v0 = layer.shard_indices.org_vocab_start_index
                v1 = layer.shard_indices.org_vocab_end_index
                vlen = v1 - v0
                param.data[:vlen].copy_(w[v0:v1])
                if vlen < param.shape[0]:
                    param.data[vlen:].zero_()
                plan = self._draft_block_plan(layer, w.shape[0])
                if plan is not None:
                    my_b = plan["my_blocks"].to(w.device)
                    layer._exl3_draft_svh = w.view(-1, 128).index_select(0, my_b).flatten().contiguous()
            elif layer.exl3_output_sharded:
                slices, src_total = self._output_shard_slices(layer, s, sid)
                assert w.shape[0] == src_total, (w.shape, src_total, s)
                for src_s, dst_s, length in slices:
                    param.data[dst_s: dst_s + length].copy_(w[src_s: src_s + length])
            else:
                param.data[a:b].copy_(w)
        return f

    def process_weights_after_loading(self, layer):
        groups = sorted(layer.exl3_groups) or [tuple(range(len(layer.exl3_sizes)))]
        covered = [i for g in groups for i in g]
        assert covered == list(range(len(layer.exl3_sizes))), f"exl3: incomplete shards {groups}"
        dev = layer.trellis.device
        suh = torch.stack([layer.suh.data[g[0]] for g in groups]).contiguous()
        bounds = [layer.exl3_offsets[g[0]] for g in groups] + [layer.exl3_offsets[-1]]
        n = bounds[-1]
        shard_of_nb = torch.empty(n // 128, dtype=torch.int32)
        for gi in range(len(groups)):
            shard_of_nb[bounds[gi] // 128: bounds[gi + 1] // 128] = gi
        layer.suh = Parameter(suh, requires_grad=False)
        layer.exl3_shard_of_nb = shard_of_nb.to(dev)
        layer.exl3_bounds = bounds
        if isinstance(layer, ParallelLMHead) and os.environ.get("EXL3_DRAFT_VOCAB") and _spec_is_mtp():
            _build_draft_head(layer, os.environ["EXL3_DRAFT_VOCAB"])
        cb = self.cb
        if hasattr(layer, "mcg"):
            cb = 1
        elif hasattr(layer, "mul1"):
            cb = 2
        layer.exl3_cb = cb
        # decide the backend once, outside any traced code: the all-C++ op when the ESIMD library has it
        from . import ops
        E = ops._get_esimd()
        layer.exl3_cpp = bool(E) and hasattr(E, "linear") and bool(E.exl3_supported(layer.exl3_K, cb))

    def apply(self, layer, x, bias=None):
        from . import ops
        if layer.exl3_cpp:
            y = torch.ops.exl3xpu_C.linear(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                           layer.exl3_bounds, layer.exl3_K, layer.exl3_cb,
                                           ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        else:
            y = ops._exl3_linear_py(x, layer.trellis, layer.suh, layer.svh, layer.exl3_shard_of_nb,
                                    layer.exl3_bounds, layer.exl3_K, layer.exl3_cb)
        if bias is not None:
            y = y + bias
        return y

    # lm_head (ParallelLMHead) path
    def embedding(self, layer, input_):
        raise NotImplementedError("exl3: quantized input embeddings are not supported")


# ------------------------------------------------------------------------------------------------
# Pruned-vocabulary draft head for MTP speculative decoding.
# The drafter proposes tokens from a subset of 128-token blocks (EXL3 lm_head columns can only be
# sliced by whole 128-column Hadamard blocks); the target still verifies with the full lm_head, so
# generated text is unchanged. Enabled by EXL3_DRAFT_VOCAB=<draft_vocab.json> (see scripts/draft_vocab.py).

def _spec_is_mtp():
    """The pruned draft head only serves the MTP drafter; with another drafter (dspark, dflash) or none it is
    dead weight in VRAM, so skip it."""
    try:
        from vllm.config import get_current_vllm_config
        sc = get_current_vllm_config().speculative_config
    except Exception:
        return True
    return sc is not None and "mtp" in str(getattr(sc, "method", "")).lower()


def _build_draft_head(layer, path):
    dev = layer.trellis.device
    tp_size = getattr(layer, "exl3_tp_size", 1)
    if tp_size > 1 and hasattr(layer, "_exl3_draft_trellis") and hasattr(layer, "_exl3_draft_svh"):
        plan = layer._exl3_draft_plan
        my_blocks = plan["my_blocks"]
        all_blocks = plan["all_blocks"]
        trellis = layer._exl3_draft_trellis.to(dev).contiguous()
        svh = layer._exl3_draft_svh.to(dev).contiguous()
        del layer._exl3_draft_trellis
        del layer._exl3_draft_svh
        idx = (my_blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
        all_idx = (all_blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
        layer.exl3_draft = dict(
            trellis=trellis, svh=svh, idx=idx, all_idx=all_idx,
            shard=torch.zeros(len(my_blocks), dtype=torch.int32, device=dev),
            bounds=[0, len(my_blocks) * 128],
            tp_size=tp_size,
        )
        logger.info("exl3: MTP draft head TP=%d rank %d uses %d of %d total vocab blocks (total kept %d)",
                    tp_size, layer.exl3_tp_rank, len(my_blocks), plan["n_total_blocks"], plan["n_kept"])
        return
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    n_total_blocks = layer.svh.shape[0] // 128
    blocks = blocks[blocks < n_total_blocks]
    tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
    trellis = layer.trellis.data.index_select(1, tiles).contiguous()
    svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
    idx = (blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
    layer.exl3_draft = dict(trellis=trellis, svh=svh, idx=idx, all_idx=idx,
                            shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
                            bounds=[0, len(blocks) * 128],
                            tp_size=1)
    logger.info("exl3: MTP draft head uses %d of %d vocab blocks (%.1f%% of lm_head)",
                len(blocks), n_total_blocks, 100.0 * len(blocks) / n_total_blocks)


def _patch_mtp_draft_logits():
    try:
        from vllm.model_executor.models import qwen3_5_mtp
    except Exception:
        return
    cls = qwen3_5_mtp.Qwen3_5MTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig = cls.compute_logits
    orig_top = getattr(cls, "get_top_tokens", None)

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return orig(self, hidden_states, spec_step_idx)
        from . import ops
        sub = torch.ops.exl3xpu_C.linear(hidden_states, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
                                         lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        if d.get("tp_size", 1) > 1:
            from vllm.distributed import tensor_model_parallel_all_gather
            sub = tensor_model_parallel_all_gather(sub, dim=-1)
            logits = hidden_states.new_full((hidden_states.shape[0], self.config.vocab_size), float("-inf"))
            logits.index_copy_(1, d["all_idx"], sub)
            return logits
        logits = hidden_states.new_full((hidden_states.shape[0], lm.svh.shape[0]), float("-inf"))
        logits.index_copy_(1, d["idx"], sub)
        return logits[:, : self.config.vocab_size]

    def get_top_tokens(self, hidden_states):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return orig_top(self, hidden_states)
        from . import ops
        sub = torch.ops.exl3xpu_C.linear(hidden_states, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
                                         lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        local_max_vals, local_idx = sub.max(dim=-1)
        global_idx = d["idx"][local_idx]
        tp_size = d.get("tp_size", 1)
        if tp_size <= 1:
            return global_idx
        from vllm.distributed import tensor_model_parallel_all_gather
        local_pair = torch.stack([local_max_vals.float(), global_idx.float()], dim=-1)
        gathered = tensor_model_parallel_all_gather(local_pair, dim=-1)
        gathered = gathered.view(hidden_states.shape[0], tp_size, 2)
        max_rank_idx = gathered[:, :, 0].argmax(dim=-1, keepdim=True)
        return gathered[:, :, 1].gather(dim=-1, index=max_rank_idx).squeeze(-1).to(torch.int64)

    cls.compute_logits = compute_logits
    if orig_top is not None:
        cls.get_top_tokens = get_top_tokens
    cls._exl3_patched = True


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    if os.environ.get("EXL3_DRAFT_VOCAB"):
        _patch_mtp_draft_logits()
    if os.environ.get("EXL3_VLLM_PATCHES", "1") != "0":
        from . import vllm_patches
        vllm_patches.apply_all()
    return None
