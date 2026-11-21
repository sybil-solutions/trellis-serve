"""Inkling-2-Mini image input for SGLang 0.5.20 (XPU).

The HF chat template renders each image as <|content_image|><|unused_200054|>; the processor expands the single
<|unused_200054|> placeholder into one token per 40x40 patch and attaches the patches (vendored upstream
Inkling2ImageProcessor). The model's vision tower (vendored upstream Inkling2VisionEncoder) turns them into text-width
embeddings that replace the placeholder rows (general_mm_embed_routine).
"""
from __future__ import annotations

import base64
from collections.abc import Mapping

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor

from .vendor.image_processing import Inkling2ImageProcessor

IMAGE_TOKEN_ID = 200054


class Inkling2HFProcessor:
    """Stand-in for the HF processor object SGLang loads with AutoProcessor (Inkling2Processor is not in transformers)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = Inkling2ImageProcessor()

    def __call__(self, *a, **k):  # pragma: no cover - SGLang never calls it for this processor
        raise NotImplementedError


def _resolve(item):
    url = item if isinstance(item, str) else item.get("url") if isinstance(item, Mapping) else getattr(item, "url", None)
    if not isinstance(url, str):
        return item
    if url.startswith("data:"):
        header, _, payload = url.partition(",")
        return base64.b64decode(payload) if ";base64" in header else payload.encode()
    if url.startswith(("http://", "https://")):
        from sglang.srt.utils.common import download_remote_media
        return download_remote_media(url, timeout=30)
    return url


class Inkling2XPUMultimodalProcessor(BaseMultimodalProcessor):
    models: list = []          # filled by register()

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.image_processor = getattr(_processor, "image_processor", None) or Inkling2ImageProcessor()
        self.IMAGE_TOKEN_ID = IMAGE_TOKEN_ID

    def assemble(self, input_ids: list[int], image_data=None) -> MultimodalProcessorOutput:
        images = list(image_data or [])
        count = input_ids.count(IMAGE_TOKEN_ID)
        if count != len(images):
            raise ValueError(f"Inkling2: {count} image placeholder(s) but {len(images)} image(s)")
        feats = self.image_processor.preprocess(images) if images else None
        out, items, off = [], [], 0
        for tok in input_ids:
            if tok != IMAGE_TOKEN_ID:
                out.append(tok)
                continue
            i = len(items)
            n = feats["num_tokens"][i]
            start = len(out)
            out.extend([IMAGE_TOKEN_ID] * n)
            items.append(MultimodalDataItem(
                modality=Modality.IMAGE,
                feature=feats["vision_patches_bthwc"][off:off + n],
                offsets=[(start, len(out) - 1)],
                model_specific_data={"vision_grid_thw": feats["vision_grid_thw"][i:i + 1]},
            ))
            off += n
        return MultimodalProcessorOutput(input_ids=out, mm_items=items, im_token_id=IMAGE_TOKEN_ID)

    async def process_mm_data_async(self, image_data=None, audio_data=None, input_text="", request_obj=None, *args, **kwargs):
        if audio_data:
            raise ValueError("Inkling2 supports text and images only")
        input_ids = getattr(request_obj, "input_ids", None)
        if input_ids is None:
            input_ids = self._tokenizer.encode(input_text, add_special_tokens=False) if isinstance(input_text, str) else input_text
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.flatten().tolist()
        images = [_resolve(it) for it in (image_data or [])]
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(self.io_executor, self.assemble, list(input_ids), images)


def register(model_cls) -> None:
    """Make SGLang 0.5.20 treat Inkling2LoopForConditionalGeneration as multimodal with this processor."""
    import sglang.srt.configs.model_config as MC
    if "Inkling2LoopForConditionalGeneration" not in MC.multimodal_model_archs:
        MC.multimodal_model_archs.append("Inkling2LoopForConditionalGeneration")
    from sglang.srt.managers import multimodal_processor as MP
    Inkling2XPUMultimodalProcessor.models = [model_cls]
    MP.PROCESSOR_MAPPING[model_cls] = Inkling2XPUMultimodalProcessor
    # AutoProcessor cannot build Inkling2Processor: hand SGLang the stand-in for this checkpoint type
    import sglang.srt.utils.hf_transformers.processor as HP
    if not getattr(HP.get_processor, "_inkling2", False):
        orig = HP.get_processor

        def get_processor(tokenizer_name, *args, **kwargs):
            import json
            import os
            try:
                with open(os.path.join(tokenizer_name, "config.json")) as f:
                    if json.load(f).get("model_type") == "inkling2_loop":
                        from .sgl_tokenizer import load_tokenizer
                        return Inkling2HFProcessor(load_tokenizer(tokenizer_name))
            except (OSError, ValueError):
                pass
            return orig(tokenizer_name, *args, **kwargs)

        get_processor._inkling2 = True
        HP.get_processor = get_processor
        for mod in ("sglang.srt.utils.hf_transformers", "sglang.srt.managers.tokenizer_manager",
                    "sglang.srt.managers.multimodal_processor", "sglang.srt.managers.scheduler",
                    "sglang.srt.managers.tp_worker"):
            try:
                import importlib
                m = importlib.import_module(mod)
                if getattr(m, "get_processor", None) is orig:
                    m.get_processor = get_processor
            except Exception:
                pass
