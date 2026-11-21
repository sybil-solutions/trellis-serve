"""Inkling-2-Mini (inkling2_loop) reference forward in plain PyTorch.

Correctness oracle for the SGLang-XPU port: one sequence, dense attention, experts dequantized from the
checkpoint's native NVFP4 (layers 0-19) / MXFP8 (layers 20-23) on use. Semantics follow llama.cpp
src/models/inkling2.cpp (Unsloth b12588-mix, ppl 7.65) and sglang PR #51926.

Layer visits: prelude 0-5, core 6-17 x 3 passes, coda 18-23 -> 48 visits. Every visit has its own
attention KV and short-conv history.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def nvfp4_dequant(w: torch.Tensor, scale: torch.Tensor, alpha: torch.Tensor | float, dtype=torch.bfloat16) -> torch.Tensor:
    """w: uint8 [..., N, K/2] (low nibble = even element), scale: fp8e4m3 [..., N, K/16], alpha: per-expert global."""
    lut = E2M1.to(w.device, torch.float32)
    lo = lut[(w & 0xF).long()]
    hi = lut[(w >> 4).long()]
    v = torch.stack((lo, hi), dim=-1).flatten(-2)                       # [..., N, K]
    s = scale.to(torch.float32).repeat_interleave(16, dim=-1)
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.to(torch.float32).view(*alpha.shape, *([1] * (v.dim() - alpha.dim())))
    return (v * s * alpha).to(dtype)


def mxfp8_dequant(w: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    """w: fp8e4m3 [..., N, K], scale: uint8 e8m0 [..., N, K/32]."""
    s = torch.exp2(scale.to(torch.float32) - 127.0).repeat_interleave(32, dim=-1)
    return (w.to(torch.float32) * s).to(dtype)


def rms(x: torch.Tensor, w: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return y * w.float() if w is not None else y


def l2_clip(x: torch.Tensor, scale: torch.Tensor | None, thr: float) -> torch.Tensor:
    if scale is not None:
        x = x * scale.float()
    n = x.norm(dim=-1, keepdim=True) + 1e-6
    return x * (thr / n).clamp(max=1.0)


def mix(blocks: list[torch.Tensor], cur: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Attention over depth. blocks/cur: [T, D]; queries: [Q, D] -> [Q, T, D] (values RMS-normalized)."""
    states = rms(torch.stack([*blocks, cur], dim=1))                    # [T, S, D] fp32
    scores = torch.einsum("tsd,qd->tqs", states, queries.float()) / math.sqrt(states.shape[-1])
    p = scores.softmax(-1)
    return torch.einsum("tqs,tsd->qtd", p, states)


@dataclass
class VisitCache:
    k: torch.Tensor | None = None       # [L, Hkv, d]
    v: torch.Tensor | None = None
    pos: torch.Tensor | None = None     # [L]
    conv: dict = field(default_factory=dict)   # name -> [3, C] last 3 conv inputs (oldest first)


class Inkling2Ref:
    def __init__(self, model_dir: str, device="xpu", dtype=torch.bfloat16):
        self.dev, self.dtype = torch.device(device), dtype
        cfg = json.loads((Path(model_dir) / "config.json").read_text())["text_config"]
        self.cfg = cfg
        self.D, self.H, self.Hkv, self.hd = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]
        self.local = set(cfg["local_layer_ids"])
        lc = cfg["loop_config"]
        self.core = range(lc["core_start"], lc["core_end"])
        self.passes = lc["num_passes"]
        self.n_layers = cfg["num_hidden_layers"]
        self.window, self.rel_extent, self.d_rel = cfg["sliding_window_size"], cfg["rel_extent"], cfg["d_rel"]
        self.block = cfg["n_layers_per_attn_res_block"]
        self.w: dict[str, torch.Tensor] = {}
        with safe_open(str(Path(model_dir) / "model-00001-of-00001.safetensors"), "pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.visual"):
                    continue
                t = f.get_tensor(k)
                self.w[k.removeprefix("model.llm.")] = t.to(self.dev)
        self.visits = [(il, 0) for il in range(self.core.start)]
        for p in range(self.passes):
            self.visits += [(il, p) for il in self.core]
        self.visits += [(il, 0) for il in range(self.core.stop, self.n_layers)]
        self.reset()

    def reset(self):
        self.caches = [VisitCache() for _ in self.visits]
        self.n_past = 0

    # ---- pieces ----------------------------------------------------------------------------------
    def lin(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.to(self.dtype), self.w[name].to(self.dtype)).float()

    def convolve(self, cache: VisitCache, name: str, x: torch.Tensor, weight: torch.Tensor, dynamic: torch.Tensor | None):
        """x: [T, C] fp32; weight [C, 1, 4]; dynamic [T, 4*heads] (lag-major per head? see below) or None."""
        T, C = x.shape
        n = weight.shape[-1]
        past = cache.conv.get(name)
        if past is None:
            past = x.new_zeros(n - 1, C)
        xx = torch.cat([past, x], 0)                                     # [T+3, C]
        cache.conv[name] = xx[-(n - 1):].clone()
        w = weight.float().squeeze(1)                                    # [C, 4]
        win = xx.unfold(0, n, 1)                                         # [T, C, 4] (oldest..current)
        out = (win * w).sum(-1)
        if dynamic is not None:
            heads = dynamic.shape[-1] // n
            tap = dynamic.view(T, heads, n)                              # [T, heads, lag]  (llama.cpp: ne0 = n_conv)
            wv = win.view(T, heads, C // heads, n)                       # window index j = n-1-lag
            for lag in range(n):
                out = out + (wv[..., n - 1 - lag] * tap[..., lag:lag + 1]).reshape(T, C)
        return (x + out) / math.sqrt((2 if dynamic is not None else 1) * n + 1)

    def attention(self, il: int, cache: VisitCache, inq, ink, inv, pos: torch.Tensor):
        p = f"layers.{il}.attn."
        T = inq.shape[0]
        is_local = il in self.local
        q = self.lin(p + "wq_du.weight", inq).view(T, self.H, self.hd)
        if is_local:
            kw, vw, kdyn, vdyn = self.w[p + "k_sconv.weight"], self.w[p + "v_sconv.weight"], None, None
        else:  # static_w [4, C] is lag-major (row 0 = current token); weight_proj gives per-head taps (head*4 + lag)
            kw = self.w[p + "k_sconv.static_w"].flip(0).T.unsqueeze(1)
            vw = self.w[p + "v_sconv.static_w"].flip(0).T.unsqueeze(1)
            kdyn = self.lin(p + "k_sconv.weight_proj", ink)
            vdyn = self.lin(p + "v_sconv.weight_proj", inv)
        k = self.convolve(cache, "k", self.lin(p + "wk_dv.weight", ink), kw, kdyn)
        v = self.convolve(cache, "v", self.lin(p + "wv_dv.weight", inv), vw, vdyn)
        r = self.lin(p + "wr_du.weight", inq).view(T, self.H, self.d_rel)
        thr = self.cfg["qk_l2_clip_threshold"]
        q = l2_clip(q, self.w[p + "q_l2_clip.scale"], thr)
        k = l2_clip(k.view(T, self.Hkv, self.hd), None, thr)
        v = rms(v.view(T, self.Hkv, self.hd))
        r = l2_clip(r, self.w[p + "rel_l2_clip.scale"], self.cfg["rel_l2_clip_threshold"])
        if not is_local:
            tau = 1 + self.cfg["log_scaling_alpha"] * torch.log(torch.clamp((pos.float() + 1) / self.cfg["log_scaling_n_floor"], min=1.0))
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]
        sink = self.lin(p + "attn_sink_proj.weight", inq)                # [T, H]
        # cache
        cache.k = k if cache.k is None else torch.cat([cache.k, k])
        cache.v = v if cache.v is None else torch.cat([cache.v, v])
        cache.pos = pos if cache.pos is None else torch.cat([cache.pos, pos])
        K, V, kp = cache.k, cache.v, cache.pos
        if is_local:                                                     # keep the window only
            keep = kp > kp[-1] - self.window - 64
            cache.k, cache.v, cache.pos = K[keep], V[keep], kp[keep]
        dist = pos[:, None] - kp[None, :]                                # [T, L]
        valid = dist >= 0
        if is_local:
            valid &= dist < self.window
        proj = self.w[p + "rel_logits_proj.proj"].float()                # [d_rel, extent]
        table = torch.einsum("thr,re->the", r, proj)                     # [T, H, extent]
        table = torch.cat([table, table.new_zeros(T, self.H, 1)], -1)
        idx = torch.where(valid & (dist < self.rel_extent), dist, self.rel_extent).clamp(min=0)
        bias = torch.gather(table, 2, idx[:, None, :].expand(T, self.H, -1))   # [T, H, L]
        g = self.H // self.Hkv
        Kh = K.float().repeat_interleave(g, dim=1)                       # [L, H, d]
        Vh = V.float().repeat_interleave(g, dim=1)
        s = torch.einsum("thd,lhd->thl", q, Kh) / math.sqrt(self.hd) + bias
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        s = torch.cat([s, sink[:, :, None]], -1)                         # sink logit, no value
        pr = s.softmax(-1)[..., :-1]
        o = torch.einsum("thl,lhd->thd", pr, Vh).reshape(T, self.H * self.hd)
        o = o * torch.sigmoid(self.lin(p + "gate_attn.weight", inq))
        o = self.lin(p + "wo_ud.weight", o)
        return self.convolve(cache, "attn_out", o, self.w[f"layers.{il}.attn_sconv.weight"], None)

    def experts(self, il: int, x: torch.Tensor) -> torch.Tensor:
        p = f"layers.{il}.mlp."
        T = x.shape[0]
        logits = F.linear(x.float(), self.w[p + "gate.weight"].float())  # [T, E] router in fp32
        probs = torch.sigmoid(logits)
        sel = probs + self.w[p + "gate.bias"].float()
        topi = sel.topk(self.cfg["num_experts_per_tok"], -1).indices
        topw = probs.gather(1, topi)
        topw = topw / topw.sum(-1, keepdim=True) * self.cfg["route_scale"]
        out = torch.zeros(T, self.D, device=x.device, dtype=torch.float32)
        e = p + "experts."
        nv = (e + "w13_weight_amax") in self.w
        for ex in topi.unique().tolist():
            rows, slot = (topi == ex).nonzero(as_tuple=True)
            if nv:
                a13 = self.w[e + "w13_weight_amax"][ex] / self.w[e + "w13_weight_outer_scale_denominator"][0]
                a2 = self.w[e + "w2_weight_amax"][ex] / self.w[e + "w2_weight_outer_scale_denominator"][0]
                w13 = nvfp4_dequant(self.w[e + "w13_weight"][ex], self.w[e + "w13_weight_scale"][ex], a13)
                w2 = nvfp4_dequant(self.w[e + "w2_weight"][ex], self.w[e + "w2_weight_scale"][ex], a2)
            else:
                w13 = mxfp8_dequant(self.w[e + "w13_weight"][ex], self.w[e + "w13_weight_scale"][ex])
                w2 = mxfp8_dequant(self.w[e + "w2_weight"][ex], self.w[e + "w2_weight_scale"][ex])
            h = F.linear(x[rows].to(self.dtype), w13).float()
            gte, up = h[:, 0::2], h[:, 1::2]          # checkpoint interleaves gate/up rows
            gc, uc = self.cfg["mlp_gate_softcap"], self.cfg["mlp_up_softcap"]
            act = gc * torch.tanh(gte / gc) * torch.sigmoid(gte) * (uc * torch.tanh(up / uc))
            y = F.linear(act.to(self.dtype), w2).float()
            out.index_add_(0, rows, y * topw[rows, slot, None])
        return out

    def control(self, il: int, module: str, cur, msg):
        p = f"layers.{il}.controllers.{module}."
        h = torch.cat([rms(cur), rms(msg)], -1)
        h = self.lin(p + "down_weight", h)
        gain = torch.tanh(self.lin(p + "gain_weight", h))
        forget = torch.tanh(self.lin(p + "forget_weight", h))
        return (1 - forget) * cur + gain * msg

    # ---- forward -----------------------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [T] continuing the cached sequence. Returns fp32 logits [T, vocab_unpadded]."""
        T = ids.shape[0]
        pos = torch.arange(self.n_past, self.n_past + T, device=self.dev)
        cur = rms(F.embedding(ids, self.w["embed.weight"]), self.w["embed_norm.weight"])
        blocks: list[torch.Tensor] = []
        prelude = None
        bank, bank_last = None, None
        q_loop = self.w["middle_cycle_cross_pass_queries"]
        kv_loop = self.w["middle_cycle_cross_pass_kv_queries"]
        for vi, (il, pss) in enumerate(self.visits):
            cache = self.caches[vi]
            is_local = il in self.local
            msgs = None
            if il in self.core:
                if il == self.core.start:
                    if pss == 0:
                        prelude = (list(blocks), cur)
                    else:
                        bank, bank_last = blocks[len(prelude[0]) + 1:], cur
                        blocks, cur = list(prelude[0]), prelude[1]
                if pss > 0:
                    row = 2 * (il - self.core.start)
                    m_attn = mix(bank, bank_last, q_loop[row:row + 2])   # (attn, mlp)
                    m_kv = mix(bank, bank_last, kv_loop[row:row + 2])    # (k, v)
                    msgs = (m_attn[0], m_kv[0], m_kv[1], m_attn[1])
            inp_sa = cur
            a = f"layers.{il}."
            if il > 0:
                qn = self.w[f"attn_res.attn_queries.{il - 1}"]
                mixed = mix(blocks, cur, qn.view(-1, self.D))
                inputs = [mixed[0 if is_local else j] for j in range(3)]
            else:
                inputs = [cur, cur, cur]
            if msgs is not None:
                inputs = [self.control(il, m, inputs[j], msgs[j]) for j, m in enumerate(("attn", "k", "v"))]
            if is_local:
                inputs = [rms(x, self.w[a + "attn_norm.weight"]) for x in inputs]
            else:
                inputs = [rms(inputs[0], self.w[a + "q_input_norm.weight"]), rms(inputs[1], self.w[a + "k_input_norm.weight"]),
                          rms(inputs[2], self.w[a + "v_input_norm.weight"])]
            att = self.attention(il, cache, *inputs, pos)
            if il % self.block == 0:
                blocks.append(inp_sa)
                cur = att
            else:
                cur = inp_sa + att
            ffn_inp = cur
            x = mix(blocks, cur, self.w[f"attn_res.mlp_queries.{il}"].view(1, -1))[0]
            if msgs is not None:
                x = self.control(il, "mlp", x, msgs[3])
            x = rms(x, self.w[a + "mlp_norm.weight"])
            y = self.experts(il, x)
            y = self.convolve(cache, "mlp_out", y, self.w[a + "mlp_sconv.weight"], None)
            cur = ffn_inp + y
        h = mix(blocks, cur, self.w["attn_res.output_query"].view(1, -1))[0]
        h = rms(h, self.w["norm.weight"])
        logits = F.linear(h.to(self.dtype), self.w["unembed.weight"]).float()
        cap = self.cfg["final_logit_softcapping"]
        logits = cap * torch.tanh(logits / cap)
        self.n_past += T
        return logits[:, : self.cfg["unpadded_vocab_size"]]
