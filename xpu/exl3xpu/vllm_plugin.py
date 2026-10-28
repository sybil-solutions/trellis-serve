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
        if key.endswith("output_gate_proj"):
            key = key[: -len("output_gate_proj")] + "gate_proj"
        # DFlash drafter stores a single layer as layers.0, but vLLM names it
        # after the target layer index (e.g. layers.52). Map back to layers.0.
        if key.startswith("layers.") and key not in self.storage:
            parts = key.split(".", 2)
            if len(parts) == 3:
                alt = "layers.0." + parts[2]
                if alt in self.storage:
                    key = alt
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
        orig_sizes = list(output_partition_sizes)
        if isinstance(layer, ParallelLMHead):
            padded_sizes = [((s + 127) // 128) * 128 for s in orig_sizes]
        else:
            assert k % 128 == 0 and all(s % 128 == 0 for s in orig_sizes), \
                f"exl3: dims must be multiples of 128 (k={k}, n={orig_sizes})"
            padded_sizes = list(orig_sizes)
        orig_n = sum(orig_sizes)
        n = sum(padded_sizes)
        if k != input_size or (not isinstance(layer, ParallelLMHead) and n != output_size):
            raise NotImplementedError("exl3: tensor parallel sharding not supported yet (use DP)")
        assert k % 128 == 0
        K = self.K
        layer.exl3_orig_sizes = orig_sizes
        layer.exl3_sizes = padded_sizes
        layer.exl3_orig_offsets = [sum(orig_sizes[:i]) for i in range(len(orig_sizes) + 1)]
        layer.exl3_offsets = [sum(padded_sizes[:i]) for i in range(len(padded_sizes) + 1)]
        layer.exl3_orig_n = orig_n
        layer.exl3_groups = set()
        layer.exl3_K = K
        layer.exl3_cb = self.cb

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

    def _load_trellis(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            assert w.dtype == torch.int16 and w.shape[-1] == param.shape[-1], \
                f"exl3: trellis bits mismatch {tuple(w.shape)} vs {tuple(param.shape)}"
            orig_a = layer.exl3_orig_offsets[s[0]]
            orig_b = layer.exl3_orig_offsets[s[-1] + 1]
            orig_width = orig_b - orig_a
            padded_width = b - a
            assert w.shape[1] * 16 in (orig_width, padded_width), (w.shape, orig_width, padded_width)
            if w.shape[1] * 16 == orig_width and padded_width > orig_width:
                pad_blocks = (padded_width - orig_width) // 16
                w = torch.nn.functional.pad(w, (0, 0, 0, pad_blocks))
            assert w.shape[1] * 16 == padded_width
            assert w.shape[0] == param.shape[0], (w.shape, a, b)
            param.data[:, a // 16: b // 16].copy_(w)
            layer.exl3_groups.add(tuple(s))
        return f

    def _load_suh(self, layer):
        def f(param, w, sid=None):
            s, _, _ = self._range(layer, w, sid)
            for i in s:
                param.data[i].copy_(w)
        return f

    def _load_svh(self, layer):
        def f(param, w, sid=None):
            s, a, b = self._range(layer, w, sid)
            orig_a = layer.exl3_orig_offsets[s[0]]
            orig_b = layer.exl3_orig_offsets[s[-1] + 1]
            orig_width = orig_b - orig_a
            padded_width = b - a
            if w.numel() == orig_width and padded_width > orig_width:
                w = torch.nn.functional.pad(w, (0, padded_width - orig_width))
            assert w.numel() in (orig_width, padded_width), (tuple(w.shape), orig_width, padded_width)
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
        if hasattr(layer, "exl3_orig_n") and y.shape[-1] != layer.exl3_orig_n:
            y = y[..., :layer.exl3_orig_n]
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
    with open(path) as f:
        spec = json.load(f)
    blocks = torch.tensor(spec["blocks"], dtype=torch.long)
    n_total_blocks = layer.svh.shape[0] // 128
    blocks = blocks[blocks < n_total_blocks]
    dev = layer.trellis.device
    tiles = (blocks[:, None] * 8 + torch.arange(8)[None, :]).flatten().to(dev)
    trellis = layer.trellis.data.index_select(1, tiles).contiguous()
    svh = layer.svh.data.view(-1, 128).index_select(0, blocks.to(dev)).flatten().contiguous()
    idx = (blocks[:, None] * 128 + torch.arange(128)[None, :]).flatten().to(dev)
    layer.exl3_draft = dict(trellis=trellis, svh=svh, idx=idx,
                            shard=torch.zeros(len(blocks), dtype=torch.int32, device=dev),
                            bounds=[0, len(blocks) * 128])
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

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        lm = self.lm_head
        d = getattr(lm, "exl3_draft", None)
        if d is None:
            return orig(self, hidden_states, spec_step_idx)
        from . import ops
        sub = torch.ops.exl3xpu_C.linear(hidden_states, d["trellis"], lm.suh, d["svh"], d["shard"], d["bounds"],
                                         lm.exl3_K, lm.exl3_cb, ops.SMALL_M_MAX, ops.RECON_SLICE_N)
        logits = hidden_states.new_full((hidden_states.shape[0], lm.svh.shape[0]), float("-inf"))
        logits.index_copy_(1, d["idx"], sub)
        return logits[:, : self.config.vocab_size]

    cls.compute_logits = compute_logits
    cls._exl3_patched = True


def register():
    """vllm.general_plugins entry point (runs in every vLLM process)."""
    if os.environ.get("EXL3_DRAFT_VOCAB"):
        _patch_mtp_draft_logits()
    if os.environ.get("EXL3_VLLM_PATCHES", "1") != "0":
        from . import vllm_patches
        vllm_patches.apply_all()
    return None


# ---------------------------------------------------------------------------
# DFlash draft models: ensure EXL3 quantization is resolved for the drafter.
# vLLM's get_draft_quant_config returns None when draft_model_config.quantization
# is None, even if the checkpoint's hf_config declares quant_method="exl3".
# Promote it so the Exl3Config is built and the drafter's linear layers are
# quantized to match the EXL3 checkpoint.
# ---------------------------------------------------------------------------
try:
    import vllm.model_executor.models.utils as _draft_utils

    _orig_get_draft_quant_config = _draft_utils.get_draft_quant_config

    def _patched_get_draft_quant_config(vllm_config):
        try:
            dmc = getattr(getattr(vllm_config, "speculative_config", None), "draft_model_config", None)
            if dmc is not None and getattr(dmc, "quantization", None) is None:
                hf = getattr(dmc, "hf_config", None)
                qc = getattr(hf, "quantization_config", None) if hf is not None else None
                if not isinstance(qc, dict):
                    model_name = getattr(dmc, "model", None)
                    if model_name:
                        qpath = os.path.join(model_name, "quantization_config.json")
                        if os.path.isfile(qpath):
                            try:
                                with open(qpath) as f:
                                    qc = json.load(f)
                            except Exception:
                                qc = None
                if isinstance(qc, dict) and qc.get("quant_method") == "exl3":
                    dmc.quantization = "exl3"
                    logger.info("exl3: promoted draft model quantization to exl3")
        except Exception as e:
            logger.warning("exl3: draft quantization promotion failed: %s", e)
        _ret = _orig_get_draft_quant_config(vllm_config)
        return _ret

    _draft_utils.get_draft_quant_config = _patched_get_draft_quant_config
    try:
        import vllm.model_executor.models.qwen3_dflash as _qd
        _qd.get_draft_quant_config = _patched_get_draft_quant_config
    except Exception:
        pass
except Exception as e:
    logger.warning("exl3: could not patch draft quantization resolution: %s", e)
