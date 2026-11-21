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

from .attn_triton import attention_ragged
from .dense import Q8Linear
from .moe_triton import Inkling2ExpertsSYCL
from .ref import rms

logger = logging.getLogger(__name__)

RING = 1024 + 64
_DEBUG = os.environ.get("INKLING2_DEBUG") == "1"
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
        return self.q8(x).to(torch.bfloat16) if x.shape[0] > self.q8.max_gemv_rows else self.q8(x)


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
        self.n_slots = int(os.environ.get("INKLING2_SLOTS", "8"))
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)   # gives the module a device
        self.start_layer, self.end_layer = 0, self.n_global

    # ------------------------------------------------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        dev = torch.device("xpu", torch.xpu.current_device())
        raw = {}
        for name, t in weights:
            if not name.startswith("model.llm."):
                continue   # vision tower: TODO (phase 2)
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
        # global-layer static conv taps: static_w [4, C] lag-major -> [C, 4] oldest..current
        for il in range(self.n_layers):
            for kv in "kv":
                n = f"layers.{il}.attn.{kv}_sconv.static_w"
                if n in self.w:
                    self.w[f"layers.{il}.attn.{kv}_sconv.weight"] = self.w.pop(n).flip(0).T.contiguous().unsqueeze(1)
        # scaled depth-attention queries
        sc = D ** -0.5
        for k in list(self.w):
            if k.startswith("attn_res.") or k.startswith("middle_cycle_cross_pass"):
                self.w[k] = self.w[k].float() * sc
        self.lm_q8 = Q8Linear(self.w["unembed.weight"].to(torch.bfloat16))
        self.lm_head.weight = self.w["unembed.weight"]
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

    def _conv(self, vi, name, x, weight, dynamic, reqs, pos):
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
        q = self.L(p + "wq_du.weight", inq).view(T, self.H, self.hd)
        kdyn = vdyn = None
        if not is_local:
            kdyn = self.L(p + "k_sconv.weight_proj", ink)
            vdyn = self.L(p + "v_sconv.weight_proj", inv)
        k = self._conv(vi, "k", self.L(p + "wk_dv.weight", ink), self.w[p + "k_sconv.weight"], kdyn, reqs, pos)
        v = self._conv(vi, "v", self.L(p + "wv_dv.weight", inv), self.w[p + "v_sconv.weight"], vdyn, reqs, pos)
        r = self.L(p + "wr_du.weight", inq).view(T, self.H, self.d_rel)
        q = _l2_clip(q, self.w[p + "q_l2_clip.scale"], tc["qk_l2_clip_threshold"])
        k = _l2_clip(k.view(T, self.Hkv, self.hd), None, tc["qk_l2_clip_threshold"])
        v = rms(v.view(T, self.Hkv, self.hd))
        r = _l2_clip(r, self.w[p + "rel_l2_clip.scale"], tc["rel_l2_clip_threshold"])
        if not is_local:
            tau = 1 + tc["log_scaling_alpha"] * torch.log(torch.clamp((pos.float() + 1) / tc["log_scaling_n_floor"], min=1.0))
            q = q * tau[:, None, None]
            r = r * tau[:, None, None]
        sink = self.L(p + "attn_sink_proj.weight", inq)
        table = torch.einsum("thr,re->the", r, self.w[p + "rel_logits_proj.proj"].float()).to(torch.bfloat16)
        q_counts = [n for (_, _, n, _) in reqs]
        if is_local:
            si = self.visit_slot[vi]
            kb, vb = self.swa_k[si], self.swa_v[si]
            if self._swa_scratch is None:      # decode/verify: tokens already in the ring
                kb[self._ring_rows] = k.to(torch.bfloat16)
                vb[self._ring_rows] = v.to(torch.bfloat16)
                o = attention_ragged(q, kb, vb, self._ring_loc, self._ring_kpos, self._ring_start, pos.int(), q_counts,
                                     table, sink, window=self.window)
            else:                               # extend: ring (prefix) + this chunk in a scratch buffer
                sk, sv, sloc, skpos, sstart, wr = self._swa_scratch
                sk[wr] = k.to(torch.bfloat16); sv[wr] = v.to(torch.bfloat16)
                o = attention_ragged(q, sk, sv, sloc, skpos, sstart, pos.int(), q_counts, table, sink, window=self.window)
                kb[self._ring_rows] = k[self._ring_src].to(torch.bfloat16)
                vb[self._ring_rows] = v[self._ring_src].to(torch.bfloat16)
        else:
            gi = self.visit_slot[vi]
            pool = self._kv_pool
            pool.set_kv_buffer(None, fb.out_cache_loc, k.to(torch.bfloat16), v.to(torch.bfloat16), layer_id_override=gi)
            kb, vb = pool.get_key_buffer(gi), pool.get_value_buffer(gi)
            o = attention_ragged(q, kb, vb, self._g_loc, self._g_kpos, self._g_start, pos.int(), q_counts, table, sink,
                                 window=0, max_kv=self._g_max)
        o = o.reshape(T, self.H * self.hd)
        o = o * torch.sigmoid(self.L(p + "gate_attn.weight", inq))
        o = self.L(p + "wo_ud.weight", o)
        return self._conv(vi, "attn_out", o, self.w[f"layers.{il}.attn_sconv.weight"], None, reqs, pos)

    def _control(self, il, module, cur, msg):
        p = f"layers.{il}.controllers.{module}."
        h = torch.cat([rms(cur), rms(msg)], -1)
        h = self.L(p + "down_weight", h)
        return (1 - torch.tanh(self.L(p + "forget_weight", h))) * cur + torch.tanh(self.L(p + "gain_weight", h)) * msg

    def _moe(self, il, x):
        p = f"layers.{il}.mlp."
        probs = torch.sigmoid(F.linear(x.float(), self.w[p + "gate.weight"].float()))
        topi = (probs + self.w[p + "gate.bias"].float()).topk(self.tc["num_experts_per_tok"], -1).indices
        topw = probs.gather(1, topi)
        topw = topw / topw.sum(-1, keepdim=True) * self.tc["route_scale"]
        return self.experts[il](x, topi.to(torch.int32), topw)

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
        locs, starts, kp = [], [0], []
        for (s, r0, n, p0) in reqs:
            L = p0 + n
            locs.append(r2t[s, :L].to(torch.int32)); kp.append(torch.arange(L, device=dev, dtype=torch.int32))
            starts.append(starts[-1] + L)
        self._g_loc, self._g_kpos = torch.cat(locs), torch.cat(kp)
        self._g_start = torch.tensor(starts, device=dev, dtype=torch.int32)
        self._g_max = max(p0 + n for (_, _, n, p0) in reqs)
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
    def forward(self, input_ids, positions, forward_batch, input_embeds=None, **kwargs):
        fb = forward_batch
        if fb.forward_mode.is_idle():
            h = torch.zeros(0, self.D, device=input_ids.device, dtype=torch.bfloat16)
            return self.logits_processor(input_ids, h, self.lm_head, fb)
        reqs = self._batch(fb)
        self._reqs = reqs
        if _DEBUG:
            logger.info("inkling2 fwd mode=%s reqs=%s ids[min,max]=%s pos=%s..%s loc=%s..%s", fb.forward_mode, reqs,
                        (int(input_ids.min()), int(input_ids.max())), int(positions.min()), int(positions.max()),
                        int(fb.out_cache_loc.min()), int(fb.out_cache_loc.max()))
        pos = positions.long()
        self._conv_pos_update = []
        self._prepare(fb, reqs, pos)
        extend_scratch = self._swa_scratch
        cur = rms(F.embedding(input_ids, self.w["embed.weight"]), self.w["embed_norm.weight"])
        blocks: list[torch.Tensor] = []
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
                    m_attn = _mix(bank, bank_last, qloop[row:row + 2])
                    m_kv = _mix(bank, bank_last, kvloop[row:row + 2])
                    msgs = (m_attn[0], m_kv[0], m_kv[1], m_attn[1])
            inp_sa = cur
            a = f"layers.{il}."
            if il > 0:
                mixed = _mix(blocks, cur, self.w[f"attn_res.attn_queries.{il - 1}"].view(-1, self.D))
                inputs = [mixed[0 if is_local else j] for j in range(3)]
            else:
                inputs = [cur, cur, cur]
            if msgs is not None:
                inputs = [self._control(il, m, inputs[j], msgs[j]) for j, m in enumerate(("attn", "k", "v"))]
            if is_local:
                inputs = [rms(x, self.w[a + "attn_norm.weight"]) for x in inputs]
            else:
                inputs = [rms(inputs[0], self.w[a + "q_input_norm.weight"]), rms(inputs[1], self.w[a + "k_input_norm.weight"]),
                          rms(inputs[2], self.w[a + "v_input_norm.weight"])]
            if is_local and extend_scratch is not None:
                self._swa_scratch = self._swa_extend_buffers(self.visit_slot[vi])
            if _DEBUG:
                torch.xpu.synchronize(); logger.info("inkling2 visit %d layer %d pre-attn ok", vi, il)
            att = self._attention(il, vi, *inputs, pos, reqs, fb, None)
            if _DEBUG:
                torch.xpu.synchronize(); logger.info("inkling2 visit %d attn ok", vi)
            if il % self.block == 0:
                blocks.append(inp_sa); cur = att
            else:
                cur = inp_sa + att
            ffn_inp = cur
            x = _mix(blocks, cur, self.w[f"attn_res.mlp_queries.{il}"].view(1, -1))[0]
            if msgs is not None:
                x = self._control(il, "mlp", x, msgs[3])
            x = rms(x, self.w[a + "mlp_norm.weight"])
            y = self._conv(vi, "mlp_out", self._moe(il, x), self.w[a + "mlp_sconv.weight"], None, reqs, pos)
            cur = ffn_inp + y
        for (s, pp) in self._conv_pos_update:
            self.conv_pos[s, pp.remainder(CONV_RING)] = pp
        if extend_scratch is not None:
            for (s, r0, n, p0) in reqs:
                m = min(n, RING)
                pp = torch.arange(p0 + n - m, p0 + n, device=pos.device)
                self.swa_pos[s, pp.remainder(RING)] = pp.int()
        h = _mix(blocks, cur, self.w["attn_res.output_query"].view(1, -1))[0]
        h = rms(h, self.w["norm.weight"]).to(torch.bfloat16)
        return self.logits_processor(input_ids, h, self.lm_head, fb)


EntryClass = Inkling2LoopForConditionalGeneration
