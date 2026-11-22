"""Inkling-2-Mini DSpark block drafter on XPU (mtp.safetensors: 3 Qwen-style layers + recurrent Markov head, block 8).

Semantics follow llama.cpp src/models/dspark.cpp graph_inkling2 (Unsloth b12588) and sglang PR #51926 inkling2_mtp.py:
  context (per target position t, features F_t = target hidden at visits [5, 23, 41] and the final normed hidden):
      ctx_l = RMSNorm(sum_j softmax(fuse[l])_j F_t[j]) * dflare_norm[l]        (l = 0..2)
      K_ctx = RoPE(RMSNorm(k_proj_ctx(ctx_l)) * k_norm_ctx), V_ctx = v_proj_ctx(ctx_l)   -> draft KV at position t
      f_rnn = RMSNorm(sum_j softmax(fuse[3])_j F_t[j]) * markov_head.dflare_norm   (of the last committed position)
  block (anchor token a at position p, positions p..p+B-1): x0 = embed_norm(embed(a)), x_i = mask_embed (i>0)
      3 layers: h = x + o_proj(attn(RoPE(q_norm(q)), RoPE(k_norm(k)), v)); x = h + down(silu(gate)*up)  (pre-norm)
      attention: every block row sees all block keys and the context keys at positions < p (SWA layers: window 1024)
      base_i = softcap? no: base logits = LM_head(final_norm(x_i))
  markov (i = 1..B-1, greedy): joint = [state, w1[prev], x_i, f_rnn] -> rnn_proj -> (gate, cand, out)
      state = cand + sigmoid(gate)*(state - cand); bias = LM_head(hidden_out(tanh(out)))  (folded: W_fold latent + b_fold)
      logits_i = 30 tanh((base_i + bias)/30); prev = argmax(logits_i)
All draft state is keyed by position (rings), so rejected tokens need no rollback: the target forward writes context K/V
and features for every row it processes; positions that were rejected are overwritten later.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from .dense import Q8Linear
from .fused import rms_norm

SWA_RING = 1024 + 64
_REF = os.environ.get("INKLING2_DRAFT_REF", "0") == "1"
FEAT_RING = 32


def _rope_neox(x: torch.Tensor, pos: torch.Tensor, theta: float) -> torch.Tensor:
    """x [T, H, D] rotate_half RoPE at positions pos [T]."""
    D = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = pos.float()[:, None] * inv[None, :]                      # [T, D/2]
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
    x1, x2 = x[..., : D // 2].float(), x[..., D // 2:].float()
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class Inkling2DSpark:
    def __init__(self, mtp_path: str, mtp_cfg: dict, target, n_slots: int, dev):
        from safetensors import safe_open
        self.cfg = mtp_cfg
        self.dev = dev
        self.t = target                                             # Inkling2LoopForConditionalGeneration
        self.D = mtp_cfg["hidden_size"]
        self.H, self.Hkv, self.hd = mtp_cfg["num_attention_heads"], mtp_cfg["num_key_value_heads"], mtp_cfg["head_dim"]
        self.L = mtp_cfg["num_hidden_layers"]
        self.B = mtp_cfg["block_size"]
        self.rank = mtp_cfg["markov_rank"]
        self.theta = float(mtp_cfg.get("rope_theta", 1e6))
        self.eps = float(mtp_cfg.get("rms_norm_eps", 1e-6))
        self.local = set(mtp_cfg.get("local_layer_ids", []))
        self.window = int(mtp_cfg.get("sliding_window_size", 1024))
        self.gctx = int(os.environ.get("INKLING2_DRAFT_CTX", "16384"))   # context window kept for the global draft layer
        self.n_feat = len(mtp_cfg["target_layer_ids"])
        w = {}
        with safe_open(mtp_path, "pt", device="cpu") as f:
            for k in f.keys():
                w[k] = f.get_tensor(k).to(dev, copy=True)
        self.w = w
        self.lin = {}
        for l in range(self.L):
            a, m = f"layers.{l}.self_attn.", f"layers.{l}.mlp."
            self.lin[(l, "qkv")] = Q8Linear(torch.cat([w[a + "q_proj.weight"], w[a + "k_proj.weight"], w[a + "v_proj.weight"]]), keep_bf16=False)
            self.lin[(l, "ctx")] = Q8Linear(torch.cat([w[a + "k_proj_ctx.weight"], w[a + "v_proj_ctx.weight"]]))
            self.lin[(l, "o")] = Q8Linear(w[a + "o_proj.weight"], keep_bf16=False)
            self.lin[(l, "gu")] = Q8Linear(torch.cat([w[m + "gate_proj.weight"], w[m + "up_proj.weight"]]), keep_bf16=False)
            self.lin[(l, "down")] = Q8Linear(w[m + "down_proj.weight"], keep_bf16=False)
        if os.environ.get("INKLING2_DRAFT_BF16", "0") == "1":     # reference: bf16 drafter linears (F.linear)
            for (l, nm) in list(self.lin):
                a, m = f"layers.{l}.self_attn.", f"layers.{l}.mlp."
                src = {"qkv": [a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"], "ctx": [a + "k_proj_ctx.weight", a + "v_proj_ctx.weight"],
                       "o": [a + "o_proj.weight"], "gu": [m + "gate_proj.weight", m + "up_proj.weight"], "down": [m + "down_proj.weight"]}[nm]
                q = Q8Linear(torch.cat([w[k] for k in src]), keep_bf16=True)
                q.max_gemv_rows = 0
                self.lin[(l, nm)] = q
        self.fuse_alpha = torch.softmax(w["fuse_weight"].float(), -1)           # [L+1, n_feat]
        self.rnn = Q8Linear(w["markov_head.rnn_proj.weight"], keep_bf16=False)   # [3r, 2r + 2D]
        self.rnn_b = w["markov_head.rnn_proj.bias"].float()
        # fold hidden_out into the (unpadded) target LM head: bias = latent @ W_fold^T + b_fold
        unemb = target.lm_q8                                           # Q8 [V, D]
        V = unemb.N
        Wout = w["markov_head.hidden_out.weight"].float()             # [D, r]
        bout = w["markov_head.hidden_out.bias"].float()               # [D]
        from .fused import q8_dequant
        fold, fold_b = [], []
        for s in range(0, V, 16384):
            head = q8_dequant(unemb.q[:, s:s + 16384].contiguous(), unemb.s[:, s:s + 16384].contiguous()).float()
            fold.append((head @ Wout).to(torch.bfloat16)); fold_b.append(head @ bout)
        self.fold_full = Q8Linear(torch.cat(fold), keep_bf16=False)    # [V, r]
        self.fold_b_full = torch.cat(fold_b)
        self.V = V
        self.set_draft_vocab(int(os.environ.get("INKLING2_DRAFT_VOCAB", "0")) or V)
        for k in [k for k in w if k.endswith("_proj.weight") or k.startswith("markov_head.hidden_out.")]:
            del w[k]                                                    # quantized copies above; keep norms / embeddings
        del fold, head
        torch.xpu.empty_cache()
        # state rings
        S = n_slots
        self.ring = {}
        for l in range(self.L):
            R = SWA_RING if l in self.local else self.gctx + 64
            self.ring[l] = (torch.zeros(S, R, self.Hkv, self.hd, device=dev, dtype=torch.bfloat16),
                            torch.zeros(S, R, self.Hkv, self.hd, device=dev, dtype=torch.bfloat16),
                            torch.full((S, R), -1, device=dev, dtype=torch.int32), R)
        self.feat = torch.zeros(S, FEAT_RING, self.n_feat, self.D, device=dev, dtype=torch.bfloat16)
        self.feat_pos = torch.full((S, FEAT_RING), -1, device=dev, dtype=torch.int64)
        self.slots = torch.arange(S, device=dev)
        self.raw_anchor, self.markov = False, True

    def set_draft_vocab(self, k: int):
        """drafts only from token ids < k (BPE ids are roughly frequency-ordered): the draft LM head GEMV reads k/V of
        the head. k >= V: full vocabulary (shares the target's INT8 head)."""
        from .fused import q8_dequant
        if k >= self.V:
            self.Vd, self.head, self.fold, self.fold_b = self.V, self.t.lm_q8, self.fold_full, self.fold_b_full
            return
        u = self.t.lm_q8
        self.Vd = k
        self.head = Q8Linear(q8_dequant(u.q[:, :k].contiguous(), u.s[:, :k].contiguous()), keep_bf16=False)
        f = self.fold_full
        self.fold = Q8Linear(q8_dequant(f.q[:, :k].contiguous(), f.s[:, :k].contiguous()), keep_bf16=False)
        self.fold_b = self.fold_b_full[:k].contiguous()

    # ---- context ingestion (called from the target forward for every processed row) -----------------------
    def _mix(self, feats: torch.Tensor, row: int, norm_w: torch.Tensor) -> torch.Tensor:
        x = (feats.float() * self.fuse_alpha[row][None, :, None]).sum(1)
        return rms_norm(x, norm_w, self.eps)

    @torch.no_grad()
    def ingest(self, slot_rows: torch.Tensor, pos: torch.Tensor, feats: torch.Tensor, keep_last: int | None = None):
        """slot_rows/pos [T] (long), feats [T, n_feat, D]. Writes draft context K/V and the feature ring (rollback-free)."""
        if keep_last is not None and feats.shape[0] > keep_last:
            slot_rows, pos, feats = slot_rows[-keep_last:], pos[-keep_last:], feats[-keep_last:]
        pos = pos.long()
        fi = pos.remainder(FEAT_RING)
        self.feat[slot_rows, fi] = feats.to(torch.bfloat16)
        self.feat_pos[slot_rows, fi] = pos
        for l in range(self.L):
            self._ingest_layer(l, slot_rows, pos, feats)

    @torch.no_grad()
    def ingest_reqs(self, reqs, pos: torch.Tensor, feats: torch.Tensor):
        """extend path: per request, only the rows each ring can hold (no duplicate ring writes within one call)."""
        for (s, r0, n, _p0) in reqs:
            if n <= 0:
                continue
            p, f = pos[r0:r0 + n].long(), feats[r0:r0 + n]
            rows = torch.full((n,), s, device=f.device, dtype=torch.long)
            k = max(0, n - FEAT_RING)
            self.feat[rows[k:], p[k:].remainder(FEAT_RING)] = f[k:]
            self.feat_pos[rows[k:], p[k:].remainder(FEAT_RING)] = p[k:]
            for l in range(self.L):
                k = max(0, n - min(self.ring[l][3], SWA_RING if l in self.local else self.gctx))
                self._ingest_layer(l, rows[k:], p[k:], f[k:])

    def _ingest_layer(self, l, slot_rows, pos, feats):
        T = feats.shape[0]
        kvsz = self.Hkv * self.hd
        a = f"layers.{l}."
        ctx = self._mix(feats, l, self.w[a + "dflare_norm.weight"])
        kv = self.lin[(l, "ctx")](ctx)
        k = rms_norm(kv[:, :kvsz].reshape(T, self.Hkv, self.hd), self.w[a + "self_attn.k_norm_ctx.weight"], self.eps)
        k = _rope_neox(k, pos, self.theta)
        v = kv[:, kvsz:].reshape(T, self.Hkv, self.hd)
        kr, vr, pr, R = self.ring[l]
        ri = pos.remainder(R)
        kr[slot_rows, ri] = k.to(torch.bfloat16)
        vr[slot_rows, ri] = v.to(torch.bfloat16)
        pr[slot_rows, ri] = pos.int()

    def _ref_attn(self, q, kr, vr, pr, slots, anchor_pos, win):
        """plain torch: every block row sees keys with p+B-1-win < kpos <= p+B-1 (bf16 math in fp32)."""
        B = self.B
        outs = []
        for b in range(slots.shape[0]):
            s = int(slots[b]); qp = int(anchor_pos[b]) + B - 1
            kp = pr[s].long()
            ok = (kp >= 0) & (kp <= qp) & (qp - kp < win)
            K = kr[s][ok].float(); V = vr[s][ok].float()                     # [L, Hkv, D]
            qq = q[b * B:(b + 1) * B].float().view(B, self.Hkv, 3, self.hd)
            sc = torch.einsum("thgd,lhd->thgl", qq, K) / math.sqrt(self.hd)
            o = torch.einsum("thgl,lhd->thgd", sc.softmax(-1), V)
            outs.append(o.reshape(B, self.H, self.hd))
        return torch.cat(outs)

    # ---- block proposal --------------------------------------------------------------------------------------
    @torch.no_grad()
    def propose(self, slots: torch.Tensor, anchor: torch.Tensor, anchor_pos: torch.Tensor, n_draft: int):
        """slots/anchor/anchor_pos [bs] (long). Returns greedy drafts [bs, n_draft] (int64) for positions p+1..p+n."""
        from .attn_triton import attention_decode_sycl
        bs = slots.shape[0]
        B = self.B
        n = min(n_draft, B - 1)
        t = self.t
        emb = F.embedding(anchor, t.w["embed.weight"]).float()
        if not self.raw_anchor:
            emb = rms_norm(emb, t.w["embed_norm.weight"])                                         # [bs, D]
        x = self.w["mask_embed"].float()[None, None, :].expand(bs, B, self.D).clone()
        x[:, 0] = emb
        x = x.reshape(bs * B, self.D)
        pos = (anchor_pos[:, None] + torch.arange(B, device=self.dev)[None, :]).reshape(-1)     # [bs*B]
        slot_rows = slots.repeat_interleave(B)
        qsz, kvsz = self.H * self.hd, self.Hkv * self.hd
        # block rows all use the last block position as their "query position": block-uniform visibility
        qpos_uniform = (anchor_pos + B - 1).repeat_interleave(B).int()
        zr = torch.zeros(bs * B, self.H, 16, device=self.dev)
        zproj = torch.zeros(16, 1, device=self.dev)
        nosink = torch.full((bs * B, self.H), -1e30, device=self.dev)
        for l in range(self.L):
            a = f"layers.{l}."
            h = rms_norm(x, self.w[a + "input_layernorm.weight"], self.eps)
            qkv = self.lin[(l, "qkv")](h)
            q = rms_norm(qkv[:, :qsz].reshape(-1, self.H, self.hd), self.w[a + "self_attn.q_norm.weight"], self.eps)
            k = rms_norm(qkv[:, qsz:qsz + kvsz].reshape(-1, self.Hkv, self.hd), self.w[a + "self_attn.k_norm.weight"], self.eps)
            v = qkv[:, qsz + kvsz:].reshape(-1, self.Hkv, self.hd)
            q = _rope_neox(q, pos, self.theta)
            k = _rope_neox(k, pos, self.theta)
            kr, vr, pr, R = self.ring[l]
            ri = pos.remainder(R)
            kr[slot_rows, ri] = k.to(torch.bfloat16)                   # block K/V live in the ring until overwritten
            vr[slot_rows, ri] = v.to(torch.bfloat16)
            pr[slot_rows, ri] = pos.int()
            # block-uniform: the whole block uses the anchor's row (keys >= p - window, SWA) -> qpos p+B-1, window + B
            win = self.window + B if l in self.local else self.gctx
            if _REF:
                o = self._ref_attn(q, kr, vr, pr, slots, anchor_pos, win)
            else:
                o = attention_decode_sycl(q.to(torch.bfloat16), kr.view(-1, self.Hkv, self.hd), vr.view(-1, self.Hkv, self.hd),
                                          slots.int(), qpos_uniform, zr, zproj, nosink, n_q=B, paged=False,
                                          ring_pos=pr, ring=R, window=win, nsplit=8,
                                          kv_len=(anchor_pos + B).clamp(max=R).int())   # scan only the filled ring
            x = x + self.lin[(l, "o")](o.reshape(-1, qsz))
            h = rms_norm(x, self.w[a + "post_attention_layernorm.weight"], self.eps)
            gu = self.lin[(l, "gu")](h)
            I = gu.shape[-1] // 2
            x = x + self.lin[(l, "down")](F.silu(gu[:, :I]) * gu[:, I:])
        xf = rms_norm(x, self.w["norm.weight"], self.eps).reshape(bs, B, self.D)
        base = self.head(xf[:, 1:n + 1].reshape(-1, self.D).contiguous()).reshape(bs, n, self.Vd)
        # f_rnn from the features of position p-1 (the last committed target position)
        fp = anchor_pos - 1
        fi = fp.remainder(FEAT_RING)
        fe = self.feat[slots, fi]                                       # [bs, n_feat, D]
        f_rnn = self._mix(fe, self.L, self.w["markov_head.dflare_norm.weight"])
        state = torch.zeros(bs, self.rank, device=self.dev)
        prev = anchor
        out = []
        for i in range(1, n + 1):
            joint = torch.cat([state, self.w["markov_head.markov_w1.weight"][prev].float(), xf[:, i].float(), f_rnn], -1)
            pr_ = self.rnn(joint) + self.rnn_b
            r = self.rank
            gate, cand, o_ = torch.sigmoid(pr_[:, :r]), torch.tanh(pr_[:, r:2 * r]), torch.tanh(pr_[:, 2 * r:])
            state = cand + gate * (state - cand)
            logits = base[:, i - 1] + (self.fold(o_) + self.fold_b if self.markov else 0.0)
            logits = 30.0 * torch.tanh(logits / 30.0)
            prev = logits.argmax(-1)
            out.append(prev)
        return torch.stack(out, 1)
