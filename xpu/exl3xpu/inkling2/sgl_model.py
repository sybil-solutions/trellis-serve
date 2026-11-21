"""Inkling-2-Mini (Inkling2LoopForConditionalGeneration) for SGLang 0.5.20 on Intel XPU.

Registered through SGLANG_EXTERNAL_MODEL_PACKAGE=exl3xpu.inkling2.sgl_models (see sgl_models/__init__.py).

Caches
- Global attention (layers 5, 11, 17, 23 -> 8 visits with the loop): SGLang's token KV pool, one pool layer per visit
  (the config facade reports 8 layers). Paged through req_to_token, fp8 or bf16.
- SWA attention (40 visits, window 1024): model-owned ring per request slot, R = 1024 + 64 rows, keyed by position.
- Short convolutions (48 visits x [k 1024 | v 1024 | attn_out 2304 | mlp_out 2304] channels, kernel 4): model-owned
  ring per slot, RC = 16 rows keyed by position. Rings make speculative verify/rollback free: entries are validated
  by their stored position, never by a write cursor.
Radix (prefix) caching must be disabled: the rings are per request.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from .attn_triton import attention_ragged, attention_decode_sycl as attention_decode, conv_decode, conv_decode1, attention_prefill_esimd
from .dense import Q8Linear
from .moe_triton import Inkling2ExpertsSYCL
from .ref import rms
from .fused import rms_norm, mix as fmix, route as froute, route_gemv

logger = logging.getLogger(__name__)

RING = 1024 + 64
_DEBUG = os.environ.get("INKLING2_DEBUG") == "1"
_STATIC = os.environ.get("INKLING2_STATIC", "1") == "1"
_FUSED_CONV = os.environ.get("INKLING2_FUSED_CONV", "1") == "1"
PREFILL_QTW = int(os.environ.get("INKLING2_PREFILL_QTW", "4"))
NSPLIT_G = int(os.environ.get("INKLING2_NSPLIT", "64"))
CONV_RING = 16
N_CONV = 4


def _l2_clip(x, scale, thr):
    if scale is not None:
        x = x * scale.float()
    n = x.norm(dim=-1, keepdim=True) + 1e-6
    return x * (thr / n).clamp(max=1.0)


def _mix(blocks, cur, q_scaled):
    """depth attention: blocks/cur [T, D]; q_scaled [Q, D] (already * D^-0.5) -> [Q, T, D] fp32."""
    states = rms(torch.stack([*blocks, cur], 1))
    p = torch.einsum("tsd,qd->tqs", states, q_scaled).softmax(-1)
    return torch.einsum("tqs,tsd->qtd", p, states)


class _LMHeadMethod:
    def __init__(self, q8: Q8Linear):
        self.q8 = q8

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, layer, x, bias=None):
        return self.q8(x)


class _LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant_method = None
        self.weight = None


class Inkling2LoopForConditionalGeneration(nn.Module):
    def __init__(self, config, quant_config=None, prefix: str = ""):
        super().__init__()
        self.config = config
        tc = config.inkling_text
        self.tc = tc
        self.D, self.H, self.Hkv, self.hd = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        self.n_layers = tc["num_hidden_layers"]
        self.local = set(tc["local_layer_ids"])
        lc = tc["loop_config"]
        self.core = range(lc["core_start"], lc["core_end"])
        self.passes = lc["num_passes"]
        self.window, self.extent, self.d_rel = tc["sliding_window_size"], tc["rel_extent"], tc["d_rel"]
        self.block = tc["n_layers_per_attn_res_block"]
        self.vocab = tc["unpadded_vocab_size"]
        self.visits = [(il, 0) for il in range(self.core.start)]
        for p in range(self.passes):
            self.visits += [(il, p) for il in self.core]
        self.visits += [(il, 0) for il in range(self.core.stop, self.n_layers)]
        gi = si = 0
        self.visit_slot = []
        for il, _ in self.visits:
            if il in self.local:
                self.visit_slot.append(si); si += 1
            else:
                self.visit_slot.append(gi); gi += 1
        self.n_global, self.n_swa = gi, si
        self.conv_ch = {"k": (0, self.Hkv * self.hd), "v": (self.Hkv * self.hd, self.Hkv * self.hd),
                        "attn_out": (2 * self.Hkv * self.hd, self.D), "mlp_out": (2 * self.Hkv * self.hd + self.D, self.D)}
        self.n_conv_ch = 2 * self.Hkv * self.hd + 2 * self.D
        self.w: dict[str, torch.Tensor] = {}
        self.lin: dict[str, Q8Linear] = {}
        self.experts: dict[int, Inkling2ExpertsSYCL] = {}
        self.lm_head = _LMHead()
        from sglang.srt.layers.logits_processor import LogitsProcessor
        self.logits_processor = LogitsProcessor(config)
        n_slots = os.environ.get("INKLING2_SLOTS")
        if n_slots is None:
            try:
                from sglang.srt.runtime_context import get_server_args
                mrr = get_server_args().max_running_requests
                n_slots = (int(mrr) + 2) if mrr else 8
            except Exception:
                n_slots = 8
        self.n_slots = int(n_slots)
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)   # gives the module a device
        self.visual = None
        if config.inkling_vision and os.environ.get("INKLING2_VISION", "1") == "1":
            from .vendor.inkling2_vision import Inkling2VisionEncoder
            self.visual = Inkling2VisionEncoder(dict(config.inkling_vision), self.D)
        self.start_layer, self.end_layer = 0, self.n_global

    # ------------------------------------------------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        dev = torch.device("xpu", torch.xpu.current_device())
        raw = {}
        vis_params = dict(self.visual.named_parameters()) if self.visual is not None else {}
        for name, t in weights:
            if name.startswith("model.visual."):
                vn = name.removeprefix("model.visual.")
                if vn in vis_params:
                    vis_params[vn].data = t.to(dev, vis_params[vn].dtype, copy=True)
                continue
            if not name.startswith("model.llm."):
                continue
            raw[name.removeprefix("model.llm.")] = t.to(dev, copy=True)   # never keep loader staging/mmap buffers
        D = self.D
        dense_names = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "wo_ud", "attn_sink_proj")
        for k, t in raw.items():
            if ".mlp.experts." in k:
                continue
            if any(f".attn.{n}.weight" in k for n in dense_names) or "sconv.weight_proj" in k or ".controllers." in k:
                self.lin[k] = Q8Linear(t.to(dev, torch.bfloat16))
                continue
            if k in ("embed.weight", "unembed.weight"):
                t = t[: self.vocab]
            self.w[k] = t.to(dev)
        for il in range(self.n_layers):
            e = f"layers.{il}.mlp.experts."
            if e + "w13_weight_amax" in raw:
                a13 = (raw[e + "w13_weight_amax"] / raw[e + "w13_weight_outer_scale_denominator"]).to(dev)
                a2 = (raw[e + "w2_weight_amax"] / raw[e + "w2_weight_outer_scale_denominator"]).to(dev)
            else:
                a13 = a2 = None
            self.experts[il] = Inkling2ExpertsSYCL(raw[e + "w13_weight"].to(dev), raw[e + "w13_weight_scale"].to(dev),
                                                   raw[e + "w2_weight"].to(dev), raw[e + "w2_weight_scale"].to(dev), a13, a2,
                                                   self.tc["mlp_gate_softcap"], self.tc["mlp_up_softcap"])
        del raw
        if self.visual is not None:
            self.visual.to(dev)
            self.visual.eval()
        # global-layer static conv taps: static_w [4, C] lag-major -> [C, 4] oldest..current
        for il in range(self.n_layers):
            for kv in "kv":
                n = f"layers.{il}.attn.{kv}_sconv.static_w"
                if n in self.w:
                    self.w[f"layers.{il}.attn.{kv}_sconv.weight"] = self.w.pop(n).flip(0).T.contiguous().unsqueeze(1)
        # fused projections sharing an input (one GEMV instead of 3-4 small ones)
        def cat_q8(names):
            ws = [self.lin.pop(n).w for n in names]
            return Q8Linear(torch.cat(ws, 0)), [w.shape[0] for w in ws]
        self.fused = {}
        for il in range(self.n_layers):
            a = f"layers.{il}.attn."
            self.fused[(il, "qrgs")] = cat_q8([a + "wq_du.weight", a + "wr_du.weight", a + "gate_attn.weight", a + "attn_sink_proj.weight"])
            if il in self.local:
                self.fused[(il, "k")] = cat_q8([a + "wk_dv.weight"])
                self.fused[(il, "v")] = cat_q8([a + "wv_dv.weight"])
            else:
                self.fused[(il, "k")] = cat_q8([a + "wk_dv.weight", a + "k_sconv.weight_proj"])
                self.fused[(il, "v")] = cat_q8([a + "wv_dv.weight", a + "v_sconv.weight_proj"])
            if il in self.core:
                for mod in ("attn", "k", "v", "mlp"):
                    c = f"layers.{il}.controllers.{mod}."
                    self.fused[(il, "gf" + mod)] = cat_q8([c + "gain_weight", c + "forget_weight"])
        self.f32 = {}
        for k in list(self.w):
            if k.endswith(("sconv.weight", "_sconv.weight")) or k.endswith(".attn_sconv.weight") or k.endswith(".mlp_sconv.weight"):
                self.f32[k] = self.w[k].float().reshape(self.w[k].shape[0], -1).contiguous()     # [C, 4] oldest..current
            elif k.endswith(("l2_clip.scale", "rel_logits_proj.proj", "mlp.gate.weight", "mlp.gate.bias")):
                self.f32[k] = self.w[k].float().contiguous()
        # scaled depth-attention queries
        sc = D ** -0.5
        for k in list(self.w):
            if k.startswith("attn_res.") or k.startswith("middle_cycle_cross_pass"):
                self.w[k] = self.w[k].float() * sc
        self.n_block_slots = 12
        self._idx_cache = {}
        self._qkv_norm = {il: torch.stack([self.w[f"layers.{il}.{n}_input_norm.weight"].float() for n in "qkv"]).contiguous()
                          for il in range(self.n_layers) if il not in self.local}
        self.lm_q8 = Q8Linear(self.w.pop("unembed.weight").to(torch.bfloat16), keep_bf16=False)
        # SGLang only reads shape/dtype of lm_head.weight; the logits come from the Q8 GEMV (no bf16 copy kept)
        self.lm_head.weight = torch.zeros(1, 1, device=dev, dtype=torch.bfloat16).expand(self.vocab, self.D)
        self.lm_head.quant_method = _LMHeadMethod(self.lm_q8)
        # model-owned state rings
        S = self.n_slots
        self.swa_k = torch.zeros(self.n_swa, S * RING, self.Hkv, self.hd, device=dev, dtype=torch.bfloat16)
        self.swa_v = torch.zeros_like(self.swa_k)
        self.swa_pos = torch.full((S, RING), -1, device=dev, dtype=torch.int32)
        self.conv = torch.zeros(len(self.visits), S, CONV_RING, self.n_conv_ch, device=dev, dtype=torch.float32)
        self.conv_pos = torch.full((S, CONV_RING), -1, device=dev, dtype=torch.int64)
        torch.xpu.empty_cache()
        logger.info("inkling2: loaded; %d global / %d SWA visits, %d slots", self.n_global, self.n_swa, S)

    # ---- multimodal ------------------------------------------------------------------------------------------------
    def pad_input_ids(self, input_ids, mm_inputs):
        from sglang.srt.managers.mm_utils import MultiModalityDataPaddingPatternMultimodalTokens
        return MultiModalityDataPaddingPatternMultimodalTokens().pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def get_image_feature(self, items):
        dev = self.swa_pos.device
        patches = torch.cat([it.feature for it in items]).to(dev, torch.bfloat16)
        grids = torch.cat([it.model_specific_data["vision_grid_thw"] for it in items]).cpu()
        out = self.visual(patches, grids)
        if _DEBUG:
            logger.info("inkling2 vision: patches %s grids %s -> %s (mean |x| %.3f)", tuple(patches.shape), grids.tolist(),
                        tuple(out.shape), float(out.float().abs().mean()))
        return out

    def get_input_embeddings(self):
        w, nw = self.w["embed.weight"], self.w["embed_norm.weight"]

        def embed(ids):
            return rms_norm(F.embedding(ids.clamp(0, w.shape[0] - 1), w), nw)
        embed.weight = w
        embed.num_embeddings = w.shape[0]
        return embed

    # ------------------------------------------------------------------------------------------------------------
    def L(self, name, x):
        return self.lin[name](x)

    def _batch(self, fb):
        """Per-request layout of this forward: list of (slot, start_row, n_rows, first_pos)."""
        mode = fb.forward_mode
        slots = fb.req_pool_indices.tolist()
        if mode.is_extend() and not mode.is_target_verify():
            ns = fb.extend_seq_lens_cpu if fb.extend_seq_lens_cpu is not None else fb.extend_seq_lens.tolist()
            pre = fb.extend_prefix_lens_cpu if fb.extend_prefix_lens_cpu is not None else fb.extend_prefix_lens.tolist()
        elif mode.is_target_verify():
            n = fb.spec_info.draft_token_num
            ns = [n] * len(slots)
            pre = fb.seq_lens_cpu.tolist() if fb.seq_lens_cpu is not None else fb.seq_lens.tolist()
        else:
            ns = [1] * len(slots)
            pre = [s - 1 for s in (fb.seq_lens_cpu.tolist() if fb.seq_lens_cpu is not None else fb.seq_lens.tolist())]
        reqs, r = [], 0
        for s, n, p in zip(slots, ns, pre):
            assert s < self.n_slots, f"req slot {s} >= INKLING2_SLOTS {self.n_slots}"
            reqs.append((s, r, int(n), int(p)))
            r += int(n)
        return reqs

    def _conv_static(self, vi, name, x, weight, dynamic):
        """decode/verify conv, graph-capturable: write this step's inputs into the position ring, then read windows."""
        c0, C = self.conv_ch[name]
        if _FUSED_CONV:
            return conv_decode1(x.contiguous(), weight, dynamic.contiguous() if dynamic is not None else None,
                                self.conv[vi], self.conv_pos, self._st_slot, self._st_pos, c0, self._st_nq, N_CONV)
        T = x.shape[0]
        ring = self.conv[vi]
        sl, pos = self._st_slot, self._st_pos
        ring[sl, pos.remainder(CONV_RING), c0:c0 + C] = x.float()
        cols = []
        for j in range(N_CONV):                                  # oldest .. current
            pj = pos - (N_CONV - 1 - j)
            idx = pj.remainder(CONV_RING)
            ok = (self.conv_pos[sl, idx] == pj) & (pj >= 0)
            cols.append(ring[sl, idx, c0:c0 + C] * ok[:, None])
        win = torch.stack(cols, -1)                              # [T, C, 4]
        o = (win * weight.float().squeeze(1)).sum(-1)
        if dynamic is not None:
            heads = dynamic.shape[-1] // N_CONV
            tap = dynamic.view(T, heads, N_CONV)
            o = o + (win.reshape(T, heads, C // heads, N_CONV) * tap.flip(-1)[:, :, None, :]).sum(-1).reshape(T, C)
        return (x.float() + o) / math.sqrt((2 if dynamic is not None else 1) * N_CONV + 1)

    def _conv(self, vi, name, x, weight, dynamic, reqs, pos):
        if self._static:
            return self._conv_static(vi, name, x, weight, dynamic)
        """causal short conv (+ per-head dynamic taps) with per-slot position-keyed history ring."""
        c0, C = self.conv_ch[name]
        T = x.shape[0]
        ring = self.conv[vi]                                   # [S, RC, NCH]
        hist = []
        for (s, r0, n, p0) in reqs:
            # previous 3 inputs: from the ring if their stored position matches, else zero
            want = torch.arange(p0 - (N_CONV - 1), p0, device=x.device)
            idx = want.remainder(CONV_RING)
            ok = (self.conv_pos[s, idx] == want) & (want >= 0)
            past = ring[s, idx, c0:c0 + C] * ok[:, None]
            hist.append(torch.cat([past, x[r0:r0 + n].float()]))
        out = []
        w = weight.float().squeeze(1)                            # [C, 4]
        for (s, r0, n, p0), xx in zip(reqs, hist):
            win = xx.unfold(0, N_CONV, 1)                        # [n, C, 4]
            o = (win * w).sum(-1)
            if dynamic is not None:
                heads = dynamic.shape[-1] // N_CONV
                tap = dynamic[r0:r0 + n].view(n, heads, N_CONV)
                wv = win.reshape(n, heads, C // heads, N_CONV)
                o = o + (wv * tap.flip(-1)[:, :, None, :]).sum(-1).reshape(n, C)
            out.append(o)
            # store the last min(n, RC) inputs
            m = min(n, CONV_RING)
            pp = torch.arange(p0 + n - m, p0 + n, device=x.device)
            ring[s, pp.remainder(CONV_RING), c0:c0 + C] = x[r0 + n - m:r0 + n].float()
        o = torch.cat(out)
        return (x.float() + o) / math.sqrt((2 if dynamic is not None else 1) * N_CONV + 1)

    def _attention(self, il, vi, inq, ink, inv, pos, reqs, fb, glob):
        p = f"layers.{il}.attn."
        T = inq.shape[0]
        is_local = il in self.local
        tc = self.tc
        lq, sq = self.fused[(il, "qrgs")]
        q, r, gate, sink = lq(inq).split(sq, -1)
        q = q.reshape(T, self.H, self.hd)
        lk, sk_ = self.fused[(il, "k")]
        lv, sv_ = self.fused[(il, "v")]
        kk, vv = lk(ink), lv(inv)
        kdyn = vdyn = None
        if not is_local:
            kk, kdyn = kk.split(sk_, -1)
            vv, vdyn = vv.split(sv_, -1)
        k = self._conv(vi, "k", kk, self.f32[p + "k_sconv.weight"], kdyn, reqs, pos)
        v = self._conv(vi, "v", vv, self.f32[p + "v_sconv.weight"], vdyn, reqs, pos)
        r = r.reshape(T, self.H, self.d_rel)
        q = _l2_clip(q, self.f32[p + "q_l2_clip.scale"], tc["qk_l2_clip_threshold"])
        k = _l2_clip(k.view(T, self.Hkv, self.hd), None, tc["qk_l2_clip_threshold"])
        v = rms_norm(v.view(T, self.Hkv, self.hd))
        r = _l2_clip(r, self.f32[p + "rel_l2_clip.scale"], tc["rel_l2_clip_threshold"])
        if not is_local:
            tau = 1 + tc["log_scaling_alpha"] * torch.log(torch.clamp((pos.float() + 1) / tc["log_scaling_n_floor"], min=1.0))
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]
        sink = sink.contiguous()
        proj = self.f32[p + "rel_logits_proj.proj"]
        if self._static:
            sl = self._st_slot
            if is_local:
                si = self.visit_slot[vi]
                kb, vb = self.swa_k[si], self.swa_v[si]
                rows = sl * RING + pos.remainder(RING)
                kb[rows] = k.to(torch.bfloat16); vb[rows] = v.to(torch.bfloat16)
                o = attention_decode(q, kb, vb, self._st_req_slots, pos.int(), r, proj, sink, n_q=self._st_nq, paged=False,
                                     ring_pos=self.swa_pos, ring=RING, window=self.window, nsplit=8)
            else:
                gi = self.visit_slot[vi]
                pool = self._kv_pool
                pool.set_kv_buffer(None, fb.out_cache_loc, k.to(torch.bfloat16), v.to(torch.bfloat16), layer_id_override=gi)
                kb, vb = pool.get_key_buffer(gi), pool.get_value_buffer(gi)
                o = attention_decode(q, kb, vb, self._st_req_slots, pos.int(), r, proj, sink, n_q=self._st_nq, paged=True,
                                     r2t=self._r2t, kv_len=self._st_kvlen, nsplit=NSPLIT_G)
            o = o.reshape(T, self.H * self.hd) * torch.sigmoid(gate)
            o = self.L(p + "wo_ud.weight", o)
            return self._conv(vi, "attn_out", o, self.f32[f"layers.{il}.attn_sconv.weight"], None, reqs, pos)
        table = torch.einsum("thr,re->the", r, proj).to(torch.bfloat16)
        dev = q.device
        outs = []
        if is_local:
            si = self.visit_slot[vi]
            kb, vb = self.swa_k[si], self.swa_v[si]
            for (s, r0, n, p0) in reqs:
                lo = max(0, p0 - (self.window - 1))
                rows = s * RING + torch.arange(lo, p0, device=dev).remainder(RING)
                ks = torch.cat([kb[rows], k[r0:r0 + n].to(torch.bfloat16)])
                vs = torch.cat([vb[rows], v[r0:r0 + n].to(torch.bfloat16)])
                outs.append(attention_prefill_esimd(q[r0:r0 + n], ks, vs, p0, lo, table[r0:r0 + n], sink[r0:r0 + n],
                                                    window=self.window, qtw=PREFILL_QTW))
            kb[self._ring_rows] = k[self._ring_src].to(torch.bfloat16)
            vb[self._ring_rows] = v[self._ring_src].to(torch.bfloat16)
        else:
            gi = self.visit_slot[vi]
            pool = self._kv_pool
            pool.set_kv_buffer(None, fb.out_cache_loc, k.to(torch.bfloat16), v.to(torch.bfloat16), layer_id_override=gi)
            kp, vp = pool.get_key_buffer(gi), pool.get_value_buffer(gi)
            for (s, r0, n, p0) in reqs:
                loc = self._r2t_ext[s, :p0 + n].long()
                outs.append(attention_prefill_esimd(q[r0:r0 + n], kp[loc], vp[loc], p0, 0, table[r0:r0 + n], sink[r0:r0 + n],
                                                    window=0, qtw=PREFILL_QTW))
        o = torch.cat(outs) if len(outs) > 1 else outs[0]
        o = o.reshape(T, self.H * self.hd) * torch.sigmoid(gate)
        o = self.L(p + "wo_ud.weight", o)
        return self._conv(vi, "attn_out", o, self.f32[f"layers.{il}.attn_sconv.weight"], None, reqs, pos)

    def _control(self, il, module, cur, msg):
        p = f"layers.{il}.controllers.{module}."
        h = torch.cat([rms_norm(cur), rms_norm(msg)], -1)
        h = self.L(p + "down_weight", h)
        lgf, sgf = self.fused[(il, "gf" + module)]
        gain, forget = torch.tanh(lgf(h)).split(sgf, -1)
        return (1 - forget) * cur + gain * msg

    def _moe(self, il, x):
        p = f"layers.{il}.mlp."
        if x.shape[0] <= 8:
            topi, topw = route_gemv(x, self.f32[p + "gate.weight"], self.f32[p + "gate.bias"], self.tc["num_experts_per_tok"],
                                    self.tc["route_scale"])
        else:
            logits = F.linear(x.float(), self.f32[p + "gate.weight"])
            topi, topw = froute(logits, self.f32[p + "gate.bias"], self.tc["num_experts_per_tok"], self.tc["route_scale"])
        return self.experts[il](x, topi, topw)

    def _prepare(self, fb, reqs, pos):
        dev = pos.device
        # new requests: invalidate their rings
        for (s, r0, n, p0) in reqs:
            if p0 == 0:
                self.swa_pos[s].fill_(-1)
                self.conv_pos[s].fill_(-1)
        # global: paged keys 0 .. p0+n-1 of each request
        from sglang.srt.model_executor.forward_context import get_attn_backend
        be = get_attn_backend()
        self._kv_pool = be.token_to_kv_pool
        r2t = be.req_to_token
        self._r2t_ext = r2t
        locs, starts, kp = [], [0], []
        if True:
            self._g_loc = self._g_kpos = self._g_start = None
            self._g_max = 0
            locs = None

        # SWA ring rows written by this forward (last min(n, RING) tokens of each request)
        rows, src = [], []
        for (s, r0, n, p0) in reqs:
            m = min(n, RING)
            pp = torch.arange(p0 + n - m, p0 + n, device=dev)
            rows.append(s * RING + pp.remainder(RING)); src.append(torch.arange(r0 + n - m, r0 + n, device=dev))
        self._ring_rows, self._ring_src = torch.cat(rows), torch.cat(src)
        small = all(n <= 64 for (_, _, n, _) in reqs)
        if small:
            # decode/verify: write positions first, attend over the ring
            for (s, r0, n, p0) in reqs:
                pp = torch.arange(p0, p0 + n, device=dev)
                self.swa_pos[s, pp.remainder(RING)] = pp.int()
            self._swa_scratch = None
            self._ring_loc = torch.cat([torch.arange(s * RING, (s + 1) * RING, device=dev, dtype=torch.int32) for (s, _, _, _) in reqs])
            self._ring_kpos = torch.cat([self.swa_pos[s] for (s, _, _, _) in reqs])
            self._ring_start = torch.arange(0, (len(reqs) + 1) * RING, RING, device=dev, dtype=torch.int32)
        else:
            # extend: scratch = [ring of req0 | chunk of req0 | ring of req1 | ...]
            tot = sum(RING + n for (_, _, n, _) in reqs)
            loc, kpos, start, wr = [], [], [0], []
            off = 0
            for (s, r0, n, p0) in reqs:
                loc.append(torch.arange(off, off + RING + n, device=dev, dtype=torch.int32))
                kpos.append(torch.cat([self.swa_pos[s], torch.arange(p0, p0 + n, device=dev, dtype=torch.int32)]))
                wr.append(torch.arange(off + RING, off + RING + n, device=dev))
                off += RING + n
                start.append(off)
            ring_idx = [torch.arange(s * RING, (s + 1) * RING, device=dev) for (s, _, _, _) in reqs]
            self._ext = [torch.cat(loc), torch.cat(kpos), torch.tensor(start, device=dev, dtype=torch.int32),
                         torch.cat(wr), tot, ring_idx]
            self._swa_scratch = "extend"
        # conv ring positions are written per visit by _conv (values); positions updated once here (same for all)
        for (s, r0, n, p0) in reqs:
            m = min(n, CONV_RING)
            pp = torch.arange(p0 + n - m, p0 + n, device=dev)
            self._conv_pos_update = getattr(self, "_conv_pos_update", [])
            self._conv_pos_update.append((s, pp))

    def _swa_extend_buffers(self, si):
        """fill the scratch for SWA visit si: ring contents of each request (chunk rows filled by _attention)."""
        loc, kpos, start, wr, tot, ring_idx = self._ext
        sk = torch.empty(tot, self.Hkv, self.hd, device=loc.device, dtype=torch.bfloat16)
        sv = torch.empty_like(sk)
        off = 0
        for ridx, (s, r0, n, p0) in zip(ring_idx, self._reqs):
            sk[off:off + RING] = self.swa_k[si][ridx]
            sv[off:off + RING] = self.swa_v[si][ridx]
            off += RING + n
        return sk, sv, loc, kpos, start, wr

    @torch.no_grad()
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, _mm_done=False, **kwargs):
        fb = forward_batch
        if fb.forward_mode.is_idle():
            h = torch.zeros(0, self.D, device=input_ids.device, dtype=torch.bfloat16)
            return self.logits_processor(input_ids, h, self.lm_head, fb)
        mode = fb.forward_mode
        self._static = _STATIC and (mode.is_decode() or mode.is_target_verify())
        if self._static:
            from sglang.srt.model_executor.forward_context import get_attn_backend
            be = get_attn_backend()
            self._kv_pool, self._r2t = be.token_to_kv_pool, be.req_to_token
            nq = fb.spec_info.draft_token_num if mode.is_target_verify() else 1
            self._st_nq = nq
            self._st_req_slots = fb.req_pool_indices.to(torch.int32)
            self._st_slot = fb.req_pool_indices.long().repeat_interleave(nq) if nq > 1 else fb.req_pool_indices.long()
            self._st_pos = positions.long()
            self._st_kvlen = (fb.seq_lens + (nq if mode.is_target_verify() else 0)).to(torch.int32)
            self.swa_pos[self._st_slot, self._st_pos.remainder(RING)] = self._st_pos.int()
            self.conv_pos[self._st_slot, self._st_pos.remainder(CONV_RING)] = self._st_pos
            return self._forward_body(input_ids, positions, fb, None)
        if _DEBUG and not mode.is_decode():
            logger.info("inkling2 forward mode=%s mm=%s _mm_done=%s embeds=%s", mode, getattr(fb, "contains_mm_inputs", lambda: None)(),
                        _mm_done, None if input_embeds is None else tuple(input_embeds.shape))
        if (not _mm_done and self.visual is not None and not mode.is_decode() and not mode.is_target_verify()
                and getattr(fb, "contains_mm_inputs", None) and fb.contains_mm_inputs()):
            from sglang.srt.managers.mm_utils import general_mm_embed_routine
            from sglang.srt.managers.schedule_batch import Modality
            outer = self

            class _LM:
                def get_input_embeddings(self):
                    return outer.get_input_embeddings()

                def __call__(self, input_ids=None, forward_batch=None, input_embeds=None, **kw):
                    if _DEBUG:
                        mags = input_embeds.float().abs().mean(-1)
                        logger.info("inkling2 mm embeds: rows %d, low-magnitude (vision-like) rows %d, dtype %s, kw %s",
                                    mags.numel(), int((mags < 0.3).sum()), input_embeds.dtype, list(kw))
                    return outer.forward(input_ids_raw, positions, forward_batch, input_embeds=input_embeds, _mm_done=True)
            input_ids_raw = input_ids
            return general_mm_embed_routine(input_ids=input_ids, forward_batch=fb, language_model=_LM(),
                                            data_embedding_funcs={Modality.IMAGE: self.get_image_feature})
        reqs = self._batch(fb)
        self._reqs = reqs
        if _DEBUG and reqs is not None:
            logger.info("inkling2 fwd mode=%s reqs=%s ids[min,max]=%s pos=%s..%s loc=%s..%s", fb.forward_mode, reqs,
                        (int(input_ids.min()), int(input_ids.max())), int(positions.min()), int(positions.max()),
                        int(fb.out_cache_loc.min()), int(fb.out_cache_loc.max()))
        pos = positions.long()
        self._conv_pos_update = []
        self._prepare(fb, reqs, pos)
        return self._forward_body(input_ids, positions, fb, reqs, input_embeds)

    def _idx(self, slots):
        key = tuple(slots)
        t = self._idx_cache.get(key)
        if t is None:
            t = torch.tensor(key if key else [0], dtype=torch.int32, device=self.swa_pos.device)[: len(key)]
            self._idx_cache[key] = t
        return t

    def _forward_body(self, input_ids, positions, fb, reqs, input_embeds=None):
        pos = positions.long()
        T = input_ids.shape[0]
        extend_scratch = None if self._static else self._swa_scratch
        if input_embeds is not None:
            cur = input_embeds.float()   # text rows embedded + normed (get_input_embeddings); image rows = raw vision output
        else:
            cur = rms_norm(F.embedding(input_ids, self.w["embed.weight"]), self.w["embed_norm.weight"])
        # block store: every pushed block gets its own slot for this forward (banks of earlier passes stay valid)
        bbuf = torch.empty(self.n_block_slots, T, self.D, device=cur.device, dtype=torch.float32)
        nslot = 0
        blocks: list[int] = []
        prelude = bank = bank_last = None
        qloop, kvloop = self.w["middle_cycle_cross_pass_queries"], self.w["middle_cycle_cross_pass_kv_queries"]
        for vi, (il, pss) in enumerate(self.visits):
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
                    bidx = self._idx(bank)
                    m_attn = fmix(bbuf, bidx, bank_last, qloop[row:row + 2])
                    m_kv = fmix(bbuf, bidx, bank_last, kvloop[row:row + 2])
                    msgs = (m_attn[0], m_kv[0], m_kv[1], m_attn[1])
            inp_sa = cur
            a = f"layers.{il}."
            nw = self.w[a + "attn_norm.weight"].float().view(1, -1) if is_local else self._qkv_norm[il]
            if msgs is None:
                if il > 0:
                    mixed = fmix(bbuf, self._idx(blocks), cur, self.w[f"attn_res.attn_queries.{il - 1}"].view(-1, self.D), nw)
                    inputs = [mixed[0]] * 3 if is_local else [mixed[0], mixed[1], mixed[2]]
                else:
                    x0 = rms_norm(cur, self.w[a + "attn_norm.weight"])
                    inputs = [x0, x0, x0]
            else:
                mixed = fmix(bbuf, self._idx(blocks), cur, self.w[f"attn_res.attn_queries.{il - 1}"].view(-1, self.D))
                raw = [mixed[0 if is_local else j] for j in range(3)]
                raw = [self._control(il, m, raw[j], msgs[j]) for j, m in enumerate(("attn", "k", "v"))]
                inputs = [rms_norm(raw[j], nw[0 if is_local else j]) for j in range(3)]
            if _DEBUG:
                torch.xpu.synchronize(); logger.info("inkling2 visit %d layer %d pre-attn ok", vi, il)
            att = self._attention(il, vi, *inputs, pos, reqs, fb, None)
            if _DEBUG:
                torch.xpu.synchronize(); logger.info("inkling2 visit %d attn ok", vi)
            if il % self.block == 0:
                bbuf[nslot].copy_(inp_sa); blocks.append(nslot); nslot += 1
                cur = att
            else:
                cur = inp_sa + att
            ffn_inp = cur
            mq = self.w[f"attn_res.mlp_queries.{il}"].view(1, -1)
            mn = self.w[a + "mlp_norm.weight"].float().view(1, -1)
            if msgs is None:
                x = fmix(bbuf, self._idx(blocks), cur, mq, mn)[0]
            else:
                x = fmix(bbuf, self._idx(blocks), cur, mq)[0]
                x = rms_norm(self._control(il, "mlp", x, msgs[3]), mn[0])
            y = self._conv(vi, "mlp_out", self._moe(il, x), self.f32[a + "mlp_sconv.weight"], None, reqs, pos)
            cur = ffn_inp + y
        if not self._static:
            for (s, pp) in self._conv_pos_update:
                self.conv_pos[s, pp.remainder(CONV_RING)] = pp
        if extend_scratch is not None:
            for (s, r0, n, p0) in reqs:
                m = min(n, RING)
                pp = torch.arange(p0 + n - m, p0 + n, device=pos.device)
                self.swa_pos[s, pp.remainder(RING)] = pp.int()
        h = fmix(bbuf, self._idx(blocks), cur, self.w["attn_res.output_query"].view(1, -1),
                 self.w["norm.weight"].float().view(1, -1))[0].to(torch.bfloat16)
        return self.logits_processor(input_ids, h, self.lm_head, fb)


EntryClass = Inkling2LoopForConditionalGeneration
