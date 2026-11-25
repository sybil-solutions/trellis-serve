#!/usr/bin/env python3
"""CPU unit test for EXL3 SC mixed-bits routing (DESIGN §5.2.0).

Exercises Exl3Config._bits_for and Exl3LinearMethod.create_weights / _load_trellis
for Muse-style [4,6,6] and [3,5,5] without GPU.

Requires torch. vLLM is stubbed so the plugin can import offline.
Skip (exit 0) if torch is missing — document in CODER-REPORT.md.
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:
    print("SKIP: torch not installed — cannot run create_weights/_load_trellis shape tests")
    print("Install CPU torch in a venv, then: python3 xpu/tests/test_bits_for_mixed.py")
    sys.exit(0)


def _stub_vllm():
    """Minimal vLLM surface so exl3xpu.vllm_plugin imports without the real package."""
    if "vllm" in sys.modules:
        return

    def _pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = m
        return m

    vllm = _pkg("vllm")
    logger_mod = _pkg("vllm.logger")
    logger_mod.init_logger = lambda name: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, error=lambda *a, **k: None
    )

    me = _pkg("vllm.model_executor")
    layers = _pkg("vllm.model_executor.layers")
    linear = _pkg("vllm.model_executor.layers.linear")
    quant = _pkg("vllm.model_executor.layers.quantization")
    base_cfg = _pkg("vllm.model_executor.layers.quantization.base_config")
    vocab = _pkg("vllm.model_executor.layers.vocab_parallel_embedding")

    class LinearBase(nn.Module):
        pass

    class LinearMethodBase:
        pass

    class UnquantizedLinearMethod(LinearMethodBase):
        pass

    class QuantizationConfig:
        def __init__(self):
            pass

    class ParallelLMHead(nn.Module):
        pass

    def register_quantization_config(name):
        def deco(cls):
            return cls
        return deco

    linear.LinearBase = LinearBase
    linear.LinearMethodBase = LinearMethodBase
    linear.UnquantizedLinearMethod = UnquantizedLinearMethod
    quant.register_quantization_config = register_quantization_config
    base_cfg.QuantizationConfig = QuantizationConfig
    vocab.ParallelLMHead = ParallelLMHead

    # Keep attribute chains importable
    vllm.logger = logger_mod
    vllm.model_executor = me
    me.layers = layers
    layers.linear = linear
    layers.quantization = quant
    layers.vocab_parallel_embedding = vocab
    quant.base_config = base_cfg


_stub_vllm()

PLUGIN_ROOT = Path(__file__).resolve().parents[1]  # xpu/ — package root for exl3xpu
sys.path.insert(0, str(PLUGIN_ROOT))

from exl3xpu.vllm_plugin import Exl3Config, Exl3LinearMethod  # noqa: E402


def _with_shard(data: torch.Tensor, shard_id) -> torch.Tensor:
    """Attach vLLM-style shard_id on the tensor object (real loaders do this)."""
    data.shard_id = shard_id  # type: ignore[attr-defined]
    return data


class TestBitsForMixed(unittest.TestCase):
    def _cfg(self, storage: dict) -> Exl3Config:
        return Exl3Config(bits=3.0, head_bits=6, codebook="mul1", storage=storage)

    def test_bits_for_uniform_and_mixed(self):
        cfg = self._cfg({
            "layers.0.self_attn.q_proj": 4,
            "layers.0.self_attn.k_proj": 6,
            "layers.0.self_attn.v_proj": 6,
            "layers.0.mlp.gate_proj": 3,
            "layers.0.mlp.up_proj": 3,
            "layers.1.self_attn.q_proj": 3,
            "layers.1.self_attn.k_proj": 5,
            "layers.1.self_attn.v_proj": 5,
        })
        self.assertEqual(
            cfg._bits_for("model.language_model.layers.0.self_attn.qkv_proj"),
            [4, 6, 6],
        )
        self.assertEqual(
            cfg._bits_for("model.language_model.layers.1.self_attn.qkv_proj"),
            [3, 5, 5],
        )
        self.assertEqual(
            cfg._bits_for("model.language_model.layers.0.mlp.gate_up_proj"),
            3,
        )
        # Partial EXL3 inside a fused leaf must error (not silently drop members).
        partial = self._cfg({"layers.0.self_attn.q_proj": 4})
        with self.assertRaises(ValueError):
            partial._bits_for("layers.0.self_attn.qkv_proj")

    def test_create_and_load_mixed(self):
        for Ks, label in (([4, 6, 6], "sc4"), ([3, 5, 5], "sc3")):
            with self.subTest(label=label, Ks=Ks):
                self._check_mixed_qkv(Ks)

    def _check_mixed_qkv(self, Ks: list[int]):
        # Muse-ish L0 dims: k=6656 (=416*16), q/k/v out = 4096 / 256 / 256 (multiples of 128)
        k = 6656
        sizes = [4096, 256, 256]
        layer = nn.Module()
        method = Exl3LinearMethod(
            Exl3Config(bits=3.0, head_bits=6, codebook="mul1"),
            Ks,
        )
        self.assertTrue(method.mixed)
        self.assertIsNone(method.K)
        self.assertEqual(method.Ks, Ks)

        method.create_weights(
            layer,
            input_size_per_partition=k,
            output_partition_sizes=sizes,
            input_size=k,
            output_size=sum(sizes),
            params_dtype=torch.float16,
        )
        self.assertTrue(layer.exl3_mixed)
        self.assertEqual(layer.exl3_Ks, Ks)
        self.assertFalse(hasattr(layer, "exl3_K"))
        self.assertEqual(tuple(layer.trellis.shape), (0,))

        expected_last = [16 * Ki for Ki in Ks]
        for i, Ki in enumerate(Ks):
            t = layer.get_parameter(f"exl3_trellis_{i}")
            want = (k // 16, sizes[i] // 16, 16 * Ki)
            self.assertEqual(tuple(t.shape), want, f"shard {i} K={Ki}")
            self.assertEqual(t.shape[-1], expected_last[i])

        # Per-shard trellis load via sentinel weight_loader (as packed modules do)
        loader = layer.trellis.weight_loader
        for i, Ki in enumerate(Ks):
            w = torch.zeros((k // 16, sizes[i] // 16, 16 * Ki), dtype=torch.int16)
            w.fill_(i + 1)
            loader(layer.trellis, _with_shard(w, i), sid=i)
            dest = layer.get_parameter(f"exl3_trellis_{i}")
            self.assertTrue(torch.equal(dest.data, w))
            self.assertIn((i,), layer.exl3_groups)

        # Wrong last-dim must assert (bits mismatch)
        bad = torch.zeros((k // 16, sizes[0] // 16, 16 * (Ks[0] + 1)), dtype=torch.int16)
        with self.assertRaises(AssertionError):
            loader(layer.trellis, _with_shard(bad, 0), sid=0)

    def test_uniform_create_unchanged(self):
        k = 6656
        sizes = [4096, 256, 256]
        layer = nn.Module()
        method = Exl3LinearMethod(
            Exl3Config(bits=3.0, head_bits=6, codebook="mul1"),
            3,
        )
        self.assertFalse(method.mixed)
        method.create_weights(
            layer, k, sizes, k, sum(sizes), torch.float16,
        )
        self.assertFalse(layer.exl3_mixed)
        self.assertEqual(layer.exl3_K, 3)
        self.assertEqual(tuple(layer.trellis.shape), (k // 16, sum(sizes) // 16, 48))


if __name__ == "__main__":
    unittest.main()
