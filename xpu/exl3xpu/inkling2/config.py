"""HF config facade for Inkling-2-Mini under SGLang 0.5.20.

SGLang sizes its token KV pool by num_hidden_layers; only the global-attention visits live there (8 for Mini), so the
facade reports those. The real text/vision/MTP configs travel as plain dicts (inkling_text / inkling_vision /
inkling_mtp). The checkpoint's modelopt quantization_config is hidden from SGLang: the model reads the native
NVFP4/MXFP8 tensors itself.
"""
from transformers import PretrainedConfig


def n_global_visits(tc: dict) -> int:
    local = set(tc["local_layer_ids"])
    lc = tc.get("loop_config") or {}
    core = range(lc.get("core_start", 0), lc.get("core_end", 0))
    passes = lc.get("num_passes", 1)
    return sum((passes if il in core else 1) for il in range(tc["num_hidden_layers"]) if il not in local)


class Inkling2LoopSglConfig(PretrainedConfig):
    model_type = "inkling2_loop"

    def __init__(self, text_config=None, vision_config=None, mtp_config=None, **kw):
        kw.pop("quantization_config", None)
        kw.setdefault("architectures", ["Inkling2LoopForConditionalGeneration"])
        super().__init__(**kw)
        tc = dict(text_config or {})
        self.inkling_text = tc
        self.inkling_vision = vision_config
        self.inkling_mtp = mtp_config
        if tc:
            self.hidden_size = tc["hidden_size"]
            self.num_attention_heads = tc["num_attention_heads"]
            self.num_key_value_heads = tc["num_key_value_heads"]
            self.head_dim = tc["head_dim"]
            self.num_hidden_layers = n_global_visits(tc)
            self.vocab_size = tc.get("unpadded_vocab_size", tc["vocab_size"])
            self.max_position_embeddings = tc.get("model_max_length", 1048576)
            self.final_logit_softcapping = tc.get("final_logit_softcapping")
            self.rms_norm_eps = tc.get("rms_norm_eps", 1e-6)
            self.intermediate_size = tc.get("intermediate_size")
            self.tie_word_embeddings = False


def register() -> None:
    from transformers import AutoConfig
    try:
        AutoConfig.register("inkling2_loop", Inkling2LoopSglConfig)
    except ValueError:
        pass
    try:
        from sglang.srt.utils.hf_transformers.common import _CONFIG_REGISTRY
        _CONFIG_REGISTRY["inkling2_loop"] = Inkling2LoopSglConfig
    except Exception:  # pragma: no cover
        pass
    from sglang.srt.models.registry import ModelRegistry
    ModelRegistry.register("exl3xpu.inkling2.sgl_models", overwrite=True)
    import os
    if os.environ.get("INKLING2_VISION", "1") == "1":
        from .sgl_model import Inkling2LoopForConditionalGeneration
        from .sgl_mm import register as register_mm
        register_mm(Inkling2LoopForConditionalGeneration)
