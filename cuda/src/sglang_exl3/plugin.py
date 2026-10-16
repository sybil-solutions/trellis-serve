"""SGLang general plugin (`sglang.srt.plugins` entry point): registers the `exl3` quantization method and installs
the model-side shims EXL3 checkpoints need (MTP fc/embedding sharing, EXL3 DFlash drafts, optional host embedding).
Runs in the launcher, the engine and every scheduler subprocess."""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)
_done = False


def activate() -> None:
    global _done
    if _done:
        return
    _done = True
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.arg_groups.choices import QUANTIZATION_CHOICES, add_quantization_method_choices
    from .sglang_glue.config import Exl3Config
    QUANTIZATION_METHODS["exl3"] = Exl3Config
    if "exl3" not in QUANTIZATION_CHOICES:
        add_quantization_method_choices(["exl3"])
    _patch_mtp_fc()
    _patch_qwen4_mtp()
    _patch_dflash_exl3_draft()
    if os.environ.get("SGLANG_EXL3_EMBED_HOST", "0") == "1":
        _patch_host_embedding()
    from .sglang_glue import qsa_sm86
    qsa_sm86.install()                       # QSA decode attention without FA2/FA4 on sm_86
    _patch_decode_token_ids_logprobs()
    from .sglang_glue import qsa_fp8e4m3
    qsa_fp8e4m3.install()                    # e4m3 KV pool readable by the QSA kernels on sm_86
    from .sglang_glue import qsa_qkv
    qsa_qkv.install()                        # SGLANG_EXL3_KV_BITS=4|3: exllamav3-quantized QSA KV (off by default)
    if os.environ.get("SGLANG_EXL3_NGRAM_HOST", "1") == "1":
        # Qwen3.8-Flash-Next EXL3 n-gram table: pinned host memory + zero-copy gather/decode (offload/ngram_host.py);
        # a no-op for checkpoints without ngram_embedding.safetensors
        try:
            from .offload import ngram_host
            ngram_host.install()
        except ImportError as e:  # pragma: no cover - SGLang without qwen4_exp
            logger.info("sglang-exl3: n-gram host table shim not installed (%s)", e)
    if os.environ.get("SGLANG_EXL3_VISION", "1") == "1":
        # EXL3-quantized ViT linears decoded into SGLang's dense tower at load + PLE image-id fix (vision_exl3.py)
        from .sglang_glue import vision_exl3
        vision_exl3.install()
    if os.environ.get("SGLANG_EXL3_VIT_SDPA", "0") == "1":
        from .sglang_glue import vit_attn
        vit_attn.install()
        logger.info("sglang-exl3: vision attention 'triton_attn' served by per-segment SDPA (flash/efficient)")
    if os.environ.get("SGLANG_EXL3_MM_FAST_CPU", "0") == "1":
        from .sglang_glue import vit_attn
        vit_attn.install_fast_processor_cpu()
    if int(os.environ.get("SGLANG_EXL3_VIT_MLP_CHUNK", "0")) > 0:
        from .sglang_glue import vit_attn
        vit_attn.install_mlp_chunk(int(os.environ["SGLANG_EXL3_VIT_MLP_CHUNK"]))
        logger.info("sglang-exl3: vision MLP row-chunked at %s rows", os.environ["SGLANG_EXL3_VIT_MLP_CHUNK"])
    if os.environ.get("SGLANG_EXL3_HC_INT8", "0") not in ("", "0", "off"):
        from .sglang_glue import hc_int8_patch
        hc_int8_patch.install()          # int8 hyper-connection mix weights (K06)
    if float(os.environ.get("SGLANG_EXL3_OFFLOAD_ELASTIC_GB", "0") or 0) > 0:
        _patch_offload_elastic()
    logger.info("sglang-exl3: registered quantization method 'exl3'")


def _patch_offload_elastic() -> None:
    """Elastic expert cache: before every extend forward the flex chunks are handed back to torch, before every decode
    forward they are re-taken (sglang_glue/offload_moe_method.elastic_*_all). Decode runs as graph replays, so this is
    the only host hook."""
    try:
        from sglang.srt.model_executor.model_runner import ModelRunner
    except Exception:  # pragma: no cover
        return
    if getattr(ModelRunner, "_exl3_elastic_patch", False):
        return
    orig = ModelRunner.forward

    def forward(self, forward_batch, *args, **kwargs):
        try:
            mode = forward_batch.forward_mode
            from .sglang_glue import offload_moe_method as omm
            if mode.is_decode():
                omm.elastic_exit_all()          # graph replay: the flex chunks are cache again
            elif not mode.is_idle():
                # eager extend: hand back as much flex memory as its transients need (sized by its token count)
                ids = getattr(forward_batch, "input_ids", None)
                omm.elastic_enter_all(int(ids.shape[0]) if ids is not None else None)
        except AttributeError:
            pass
        return orig(self, forward_batch, *args, **kwargs)

    ModelRunner.forward = forward
    ModelRunner._exl3_elastic_patch = True
    logger.info("sglang-exl3: elastic expert cache hook installed (ModelRunner.forward)")


def _patch_decode_token_ids_logprobs() -> None:
    """SGLang 0.5.20 bug: a decode step of a request with `token_ids_logprob` crashes the scheduler
    (`_normalize_decode_outputs` calls .tolist() on next_token_token_ids_logprobs_val entries that are already lists).
    Seen with the reference-panel scorer; tolerate both forms."""
    try:
        from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor as P
    except Exception:  # pragma: no cover
        return
    if getattr(P, "_exl3_tolist_patch", False):
        return
    orig = P._normalize_decode_outputs

    def _normalize_decode_outputs(self, *, batch, result, logits_output, next_token_ids):
        vals = getattr(logits_output, "next_token_token_ids_logprobs_val", None)
        if vals and batch.return_logprob:
            logits_output.next_token_token_ids_logprobs_val = [
                v if (v is None or torch.is_tensor(v)) else torch.tensor(v, dtype=torch.float32) for v in vals]
        return orig(self, batch=batch, result=result, logits_output=logits_output, next_token_ids=next_token_ids)
    P._normalize_decode_outputs, P._exl3_tolist_patch = _normalize_decode_outputs, True


HOST_EMBEDS: list = []


def _patch_host_embedding() -> None:
    """SGLANG_EXL3_EMBED_HOST=1: keep the target's bf16 token embedding (2.5 GB on Qwen3.8-27B) in pinned host memory and
    gather rows over PCIe (SGLang's own pinned-host embedding from qwen4_exp, Triton gather, CUDA-graph safe). Decode reads
    a few 10 KB rows per step; the freed VRAM goes to the KV pool (~80k fp8 tokens on the 27B). The MTP draft shares it."""
    from sglang.srt.models import qwen3_5 as q
    from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    cls = q.Qwen3_5ForCausalLM
    if getattr(cls, "_exl3_host_embed", False):
        return
    orig = cls._build_embed_tokens

    def _build_embed_tokens(self, config):
        emb = orig(self, config)
        if not isinstance(emb, VocabParallelEmbedding):
            return emb
        if HOST_EMBEDS:              # the MTP draft (built second): reuse the target's host table later
            return emb
        emb.weight_scale = None
        host = Qwen4ExpPinnedHostEmbedding(emb, backend="pinned")
        HOST_EMBEDS.append(host)
        logger.info("sglang-exl3: token embedding %s kept in pinned host memory (%.2f GB)",
                    tuple(host.weight.shape), host.weight.numel() * host.weight.element_size() / 2**30)
        return host

    cls._build_embed_tokens, cls._exl3_host_embed = _build_embed_tokens, True

    # Qwen4-Exp (Qwen3.8-Flash-Next) overrides _build_embed_tokens: same treatment (1.27 GB bf16 table -> host)
    try:
        from sglang.srt.models import qwen4_exp as q4
    except Exception:  # pragma: no cover
        return
    cls4 = q4.Qwen4ExpModel
    if "_exl3_host_embed" in cls4.__dict__:        # (the flag set on Qwen3_5ForCausalLM above is inherited)
        return
    orig4 = cls4._build_embed_tokens

    def _build_embed_tokens4(self, config):
        if HOST_EMBEDS:                  # the MTP draft (built second): share the target's host table, allocate nothing
            return HOST_EMBEDS[0]
        emb = orig4(self, config)
        if not isinstance(emb, VocabParallelEmbedding):
            return emb
        emb.weight_scale = None
        host = Qwen4ExpPinnedHostEmbedding(emb, backend="pinned")
        HOST_EMBEDS.append(host)
        logger.info("sglang-exl3: qwen4_exp token embedding %s kept in pinned host memory (%.2f GB)",
                    tuple(host.weight.shape), host.weight.numel() * host.weight.element_size() / 2**30)
        return host

    cls4._build_embed_tokens, cls4._exl3_host_embed = _build_embed_tokens4, True


def _patch_mtp_fc() -> None:
    """The Qwen3.5 MTP draft builds `self.fc = nn.Linear(2h, h)`; EXL3 checkpoints quantize `mtp.fc`. Replace it
    with an EXL3 dense module after construction and finalize it after loading (nn.Linear has no quant hook)."""
    try:
        from sglang.srt.models import qwen3_5_mtp as m
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-exl3: MTP shim not installed (%s)", e)
        return
    cls = m.Qwen3_5ForCausalLMMTP
    if getattr(cls, "_exl3_patched", False):
        return
    orig_init, orig_load = cls.__init__, cls.load_weights

    def __init__(self, config, quant_config=None, prefix="", *a, **k):
        orig_init(self, config, quant_config, prefix, *a, **k)
        from .sglang_glue.config import Exl3Config
        from .sglang_glue.linear import Exl3Dense
        qc = getattr(self, "quant_config", quant_config)
        if isinstance(qc, Exl3Config) and os.environ.get("SGLANG_EXL3_DRAFT_SHARE_EMBED", "1") == "1":
            # The draft receives the target's embed_tokens via set_embed_and_head (init_lm_head); on a 24 GB card the
            # draft's own 2.5 GB bf16 copy would otherwise sit in memory when the KV pool is sized. Keep a 0-row
            # placeholder until the target's tensor is shared.
            emb = self.model.embed_tokens
            w = emb.weight
            emb.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
            for name, val in vars(w).items():
                if name != "data" and not hasattr(emb.weight, name):
                    setattr(emb.weight, name, val)
            self._exl3_skip_embed = True
            del w
            torch.cuda.empty_cache()
        if isinstance(qc, Exl3Config):
            info = qc.lookup("mtp.fc")
            if info is not None:
                self.fc = Exl3Dense(self.fc.in_features, self.fc.out_features, info, "mtp.fc",
                                    params_dtype=self.fc.weight.dtype)
                logger.info("sglang-exl3: MTP fc replaced by an EXL3 linear (%s)", info[:4])

    def load_weights(self, weights, *a, **k):
        if getattr(self, "_exl3_skip_embed", False):
            weights = ((n, w) for n, w in weights if not n.endswith("embed_tokens.weight"))
        out = orig_load(self, weights, *a, **k)
        fc = getattr(self, "fc", None)
        if fc is not None and hasattr(fc, "process_weights_after_loading") and hasattr(fc, "exl3_shards"):
            fc.process_weights_after_loading()
        return out

    orig_set = cls.set_embed_and_head

    def set_embed_and_head(self, embed, head):
        if embed is not None and not embed.is_cuda and HOST_EMBEDS:
            # target embedding lives in host memory: share the target's pinned-embedding module with the draft
            self.model.embed_tokens = HOST_EMBEDS[0]
            embed = None
        orig_set(self, embed, head)
        if head is not None and head.dim() == 2 and head.shape[1] == 0:
            _install_hot_head(self)

    cls.__init__, cls.load_weights, cls.set_embed_and_head, cls._exl3_patched = __init__, load_weights, set_embed_and_head, True


def _patch_qwen4_mtp() -> None:
    """Qwen4-Exp (Qwen3.8-Flash-Next) MTP draft with an EXL3 checkpoint:
    - `fc_embedding` / `fc_hidden` are EXL3 in the checkpoint but plain nn.Linear in SGLang -> Exl3Dense;
    - the draft's final hyper-connection mixer ships in `mtp_hyper_connection_mixer_patch.safetensors` (not in the
      index) -> appended to the weight stream if the loader did not deliver it;
    - the draft shares the target's EXL3 lm_head MODULE (the target's `.weight` is a zero-width placeholder), full
      vocabulary, no hot-token map needed."""
    try:
        from sglang.srt.models import qwen4_exp_mtp as m
    except Exception as e:  # pragma: no cover
        logger.info("sglang-exl3: qwen4_exp MTP shim not installed (%s)", e)
        return
    cls = m.Qwen4ExpForCausalLMMTP
    if "_exl3_patched" in cls.__dict__:
        return
    orig_init, orig_load, orig_set = cls.__init__, cls.load_weights, cls.set_embed_and_head

    def __init__(self, config, quant_config=None, prefix=""):
        orig_init(self, config, quant_config, prefix)
        from .sglang_glue.config import Exl3Config
        from .sglang_glue.linear import Exl3Dense
        qc = getattr(self, "quant_config", quant_config)
        self._exl3_dense = []
        if isinstance(qc, Exl3Config):
            self._exl3_model_path = qc.model_path
            for name in ("fc_embedding", "fc_hidden", "fc"):
                lin = getattr(self, name, None)
                info = qc.lookup(f"mtp.{name}")
                if isinstance(lin, torch.nn.Linear) and info is not None:
                    setattr(self, name, Exl3Dense(lin.in_features, lin.out_features, info, f"mtp.{name}",
                                                  params_dtype=lin.weight.dtype))
                    self._exl3_dense.append(name)
            if self._exl3_dense:
                logger.info("sglang-exl3: qwen4_exp MTP %s replaced by EXL3 linears", self._exl3_dense)

    def load_weights(self, weights, *a, **k):
        seen = set()

        def stream():
            for n, w in weights:
                seen.add(n)
                yield n, w
            path = getattr(self, "_exl3_model_path", None)
            fn = os.path.join(path, "mtp_hyper_connection_mixer_patch.safetensors") if path else None
            if fn and os.path.exists(fn):
                from safetensors import safe_open
                with safe_open(fn, "pt", device="cpu") as f:
                    extra = [n for n in f.keys() if n not in seen]
                    for n in extra:
                        yield n, f.get_tensor(n)
                if extra:
                    logger.info("sglang-exl3: qwen4_exp MTP mixer patch: %d tensors from %s", len(extra), fn)
        out = orig_load(self, stream(), *a, **k)
        for name in getattr(self, "_exl3_dense", []):
            getattr(self, name).process_weights_after_loading()
        return out

    def set_embed_and_head(self, embed, head):
        from .sglang_glue.linear import TARGET_LM_HEADS
        if head is not None and head.dim() == 2 and head.shape[1] == 0 and TARGET_LM_HEADS:
            self.lm_head = TARGET_LM_HEADS[-1]           # share the target's EXL3 head module
            head = None
            logger.info("sglang-exl3: qwen4_exp MTP draft shares the target EXL3 lm_head")
        if embed is not None and not embed.is_cuda and HOST_EMBEDS:
            self.model.embed_tokens = HOST_EMBEDS[0]
            embed = None
        return orig_set(self, embed, head)

    cls.__init__, cls.load_weights, cls.set_embed_and_head, cls._exl3_patched = __init__, load_weights, set_embed_and_head, True


def _patch_dflash_exl3_draft() -> None:
    """EXL3-quantized DFlash drafts (`--speculative-draft-model-quantization exl3`, e.g. an exllamav3 quant of
    z-lab/Qwen3.8-27B-DFlash2). SGLang builds the DFlash decoder layers with an empty prefix, so every draft linear
    would look up the same checkpoint key, fall back to unquantized, and the EXL3 tensors would be skipped by
    load_weights (a silently random draft). Give the layers their checkpoint prefixes, serve the quantized `fc`
    (a plain nn.Linear upstream) as an EXL3 linear, and fail loudly if any EXL3 layer was left without weights."""
    try:
        from sglang.srt.models import dflash as d
    except Exception as e:  # pragma: no cover
        logger.warning("sglang-exl3: DFlash EXL3 draft shim not installed (%s)", e)
        return
    layer_cls, model_cls = d.DFlashDecoderLayer, d.DFlashDraftModel
    if getattr(model_cls, "_exl3_patched", False):
        return
    orig_layer_init, orig_init, orig_load = layer_cls.__init__, model_cls.__init__, model_cls.load_weights

    def _is_exl3(qc):
        from .sglang_glue.config import Exl3Config
        return isinstance(qc, Exl3Config)

    def layer_init(self, config, layer_id, attention_conv=None, mlp_conv=None, quant_config=None, prefix=""):
        if not prefix and _is_exl3(quant_config):
            prefix = f"layers.{layer_id}"
        orig_layer_init(self, config, layer_id, attention_conv=attention_conv, mlp_conv=mlp_conv,
                        quant_config=quant_config, prefix=prefix)

    def __init__(self, config, quant_config=None, prefix=""):
        orig_init(self, config, quant_config=quant_config, prefix=prefix)
        if _is_exl3(quant_config) and isinstance(self.fc, torch.nn.Linear):
            info = quant_config.lookup("fc")
            if info is not None:
                from .sglang_glue.linear import Exl3Dense
                fin, fout = self.fc.in_features, self.fc.out_features
                self.fc = Exl3Dense(fin, fout, info, "fc", params_dtype=self.fc.weight.dtype)
                self.fc.in_features, self.fc.out_features = fin, fout
                logger.info("sglang-exl3: DFlash fc replaced by an EXL3 linear (%s)", info[:4])

    def load_weights(self, weights, *a, **k):
        out = orig_load(self, weights, *a, **k)
        empty = [n for n, m in self.named_modules() if hasattr(m, "exl3_shards") and not m.exl3_shards.get("trellis")]
        if empty:
            raise RuntimeError(f"sglang-exl3: EXL3 DFlash draft layers received no weights: {empty[:6]}")
        from .sglang_glue.linear import Exl3Dense
        if isinstance(self.fc, Exl3Dense):
            self.fc.process_weights_after_loading()
        return out

    layer_cls.__init__, model_cls.__init__, model_cls.load_weights = layer_init, __init__, load_weights
    model_cls._exl3_patched = True


@torch.no_grad()
def _hot_head_weight(target, hot: torch.Tensor) -> torch.Tensor:
    """Dense bf16 rows of the EXL3 target head for the `hot` token ids: identity blocks pushed through the quantized
    linear, i.e. the exact effective weights the target head applies."""
    k = target.exl3_suh_0.numel()
    w = torch.empty((hot.numel(), k), dtype=torch.bfloat16, device=hot.device)
    step = 256
    for i in range(0, k, step):
        eye = torch.zeros((min(step, k - i), k), dtype=torch.bfloat16, device=hot.device)
        eye[torch.arange(eye.shape[0]), torch.arange(i, i + eye.shape[0])] = 1
        y = target.quant_method.apply(target, eye)          # (block, vocab): rows of the effective weight
        w[:, i:i + eye.shape[0]] = y[:, hot].t()
        del y, eye
    torch.cuda.empty_cache()
    return w


@torch.no_grad()
def _install_hot_head(draft) -> None:
    """--speculative-token-map with an EXL3 target: SGLang sliced the target's zero-width lm_head placeholder, so the
    draft has no weights. Build a dense bf16 head for the hot tokens from the target's EXL3 head and serve it
    unquantized. The target keeps verifying with the full quantized head, so outputs are unchanged; only draft
    acceptance can move."""
    from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
    from sglang.srt.runtime_context import get_spec
    from sglang.srt.speculative.spec_utils import load_token_map
    from .sglang_glue.linear import TARGET_LM_HEADS
    if not TARGET_LM_HEADS:
        raise RuntimeError("sglang-exl3: token map requested but no quantized target lm_head was registered")
    target = TARGET_LM_HEADS[-1]
    hot = load_token_map(get_spec().speculative_token_map).to(target.exl3_svh_0.device)
    w = _hot_head_weight(target, hot)
    lm = draft.lm_head
    lm.quant_method = UnquantizedEmbeddingMethod()
    lm.weight = torch.nn.Parameter(w, requires_grad=False)
    logger.info("sglang-exl3: draft hot-token head %s built from the EXL3 target head (%.0f MB)", tuple(w.shape),
                w.numel() * 2 / 2**20)
