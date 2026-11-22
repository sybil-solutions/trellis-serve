"""Inkling-2-Mini DSpark speculative decoding on SGLang 0.5.20 XPU.

SGLang's own DSpark worker is CUDA/NPU-only and its RNN head differs from Inkling 2's, so this plugs the model-owned
drafter (dspark.py) into the NGRAM worker, which already runs a chain/tree TARGET_VERIFY step, accept, KV commit and
the overlap bookkeeping:

  --speculative-algorithm NGRAM --speculative-num-draft-tokens N --speculative-ngram-max-bfs-breadth 1   (+ INKLING2_DSPARK=1)

Replaced: the CPU n-gram corpus (drafts come from the GPU drafter: anchor = last committed token at position seq_len,
drafts for seq_len+1..), the CUDA-only tree-index kernel (chain links are static), the CUDA/CPU device check, and the
verify sampler. On XPU, SGLang's eagle_sample always verifies against the target argmax; here each verify row is
sampled from the target distribution (temperature / top-k / top-p / min-p) and drafts are accepted while they equal
the sample. With deterministic (argmax) drafts this is exactly speculative sampling: P(accept d) = p(d), and a rejected
row's sample is distributed as p restricted to tokens != d.
"""
from __future__ import annotations

import logging
import os
import sys
import types

import torch

logger = logging.getLogger(__name__)
_done = False
_DRAFT_GRAPH = os.environ.get("INKLING2_DRAFT_GRAPH", "1") == "1"
_PROF = os.environ.get("INKLING2_SPEC_PROF", "0") == "1"
_TPROF = os.environ.get("INKLING2_SPEC_TRACE", "0") == "1"
_T = {}


def _tick(name):
    if _PROF:
        import time
        torch.xpu.synchronize()
        now = time.perf_counter()
        last = _T.get("_last")
        if last is not None and name != "start":
            _T[name] = _T.get(name, 0.0) + now - last
        _T["_last"] = now


def _stub_cuda_ops():
    def _unavailable(*a, **k):
        raise RuntimeError("reconstruct_indices_from_tree_mask is CUDA-only (unused by the Inkling2 chain drafter)")
    try:
        import sgl_kernel.speculative as sks  # noqa
        if not hasattr(sks, "reconstruct_indices_from_tree_mask"):
            sks.reconstruct_indices_from_tree_mask = _unavailable
    except Exception:
        mod = types.ModuleType("sgl_kernel.speculative")
        mod.reconstruct_indices_from_tree_mask = _unavailable
        sys.modules["sgl_kernel.speculative"] = mod


def _patch_ngram_args():
    from sglang.srt.arg_groups import speculative_hook as H
    orig = H._handle_ngram
    if getattr(orig, "_inkling2", False):
        return

    def _handle_ngram(server_args):
        cfg = H.resolving_view(server_args)
        if cfg.device != "xpu":
            return orig(server_args)
        if cfg.max_running_requests is None:
            H.declare_resolution(server_args, "_handle_ngram", max_running_requests=4)
        H.declare_resolution(server_args, "_handle_ngram", enable_mixed_chunk=False)
        H.declare_resolution(server_args, "_handle_ngram", speculative_eagle_topk=1)
        if cfg.speculative_num_draft_tokens is None:
            H.declare_resolution(server_args, "_handle_ngram", speculative_num_draft_tokens=4)
        if cfg.speculative_num_steps is None:
            H.declare_resolution(server_args, "_handle_ngram",
                                 speculative_num_steps=H.resolving_view(server_args).speculative_num_draft_tokens)
        logger.info("inkling2: NGRAM slot runs the DSpark drafter on XPU (%s draft tokens)",
                    H.resolving_view(server_args).speculative_num_draft_tokens)

    _handle_ngram._inkling2 = True
    H._handle_ngram = _handle_ngram


class _NullCorpus:
    def synchronize(self): pass
    def reset(self): pass
    def batch_put(self, *a, **k): pass
    def erase_match_state(self, *a, **k): pass


def _expand(t, n):
    return torch.repeat_interleave(t, n, dim=0)


def inkling_sample(verify_input, batch, logits_output, grammar_mask=None, uno_target_max_top_k=None):
    """eagle_sample for chain drafts: target rows sampled (or argmax), accept while sample == draft."""
    from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
    device = batch.device
    if batch.forward_mode.is_idle():
        e = torch.empty(0, dtype=torch.int32, device=device)
        return e, e, e
    bs = len(batch.seq_lens)
    n = verify_input.draft_token_num
    si = batch.sampling_info
    logits = logits_output.next_token_logits
    if si.acc_additive_penalties is not None:
        logits.add_(_expand(si.acc_additive_penalties, n))
    if si.acc_scaling_penalties is not None:
        from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
        apply_scaling_penalties(logits, _expand(si.acc_scaling_penalties, n))
    if si.logit_bias is not None:
        logits.add_(_expand(si.logit_bias, n))
    if grammar_mask is not None:
        grammar_mask.apply(logits)
    _tick("verify")
    if si.is_all_greedy:
        target = logits.argmax(-1)
    else:
        target = _sample_rows(logits.float(), si, n)
    target = target.reshape(bs, n)
    m = _MODEL[0]
    if m is not None and _in_graph(m):
        # drafts were written into the verify input buffer by the drafter inside the verify graph (or, for an eager
        # step, into the batch's own input ids)
        if m.drafts_eager:
            src = m.spec_drafts
        else:
            src = m.graph_drafts[min(b for b in m.graph_drafts if b >= bs)]
        verify_input.draft_token.view(bs, n).copy_(src[:bs].to(verify_input.draft_token.dtype))
    candidates = verify_input.draft_token.reshape(bs, n)
    predict = torch.zeros(bs * n, dtype=torch.int32, device=device)
    accept_index = torch.full((bs, n), -1, dtype=torch.int32, device=device)
    num_correct = torch.empty((bs,), dtype=torch.int32, device=device)
    predict, accept_index, num_correct = verify_tree_greedy_func(
        predicts=predict, accept_index=accept_index, accept_token_num=num_correct, candidates=candidates,
        retrieve_index=verify_input.retrieve_index, retrieve_next_token=verify_input.retrieve_next_token,
        retrieve_next_sibling=verify_input.retrieve_next_sibling, target_predict=target, topk=-1)
    return predict, num_correct + 1, accept_index


def _mass_threshold(p, target, iters=18):
    """largest t with sum(p[p >= t]) >= target (per row), by bisection: top-p without a 200K-wide sort."""
    lo = torch.zeros_like(target)
    hi = p.amax(-1, keepdim=True)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = (p * (p >= mid)).sum(-1, keepdim=True) >= target
        lo = torch.where(ok, mid, lo)
        hi = torch.where(ok, hi, mid)
    return lo


def _sample_rows(logits, si, n):
    """one sample per row from softmax(logits / T) with per-request top-k, then top-p (renormalised), then min-p;
    rows of a request share its parameters. Greedy requests (top_k == 1) take the argmax."""
    temps = _expand(si.temperatures.view(-1, 1).float(), n)
    probs = torch.softmax(logits / temps.clamp(min=1e-5), -1)
    zero = torch.zeros((), device=probs.device)
    if getattr(si, "need_top_k_sampling", False):
        ks = _expand(si.top_ks.view(-1, 1), n).clamp(min=1, max=probs.shape[-1])
        kmax = int(ks.max())
        if kmax < probs.shape[-1]:
            top = probs.topk(kmax, -1).values
            probs = torch.where(probs >= top.gather(-1, ks - 1), probs, zero)
    if getattr(si, "need_top_p_sampling", False):
        ps = _expand(si.top_ps.view(-1, 1).float(), n)
        t = _mass_threshold(probs, ps * probs.sum(-1, keepdim=True))
        probs = torch.where(probs >= t, probs, zero)
    if getattr(si, "need_min_p_sampling", False):
        mp = _expand(si.min_ps.view(-1, 1).float(), n)
        probs = torch.where(probs >= mp * probs.amax(-1, keepdim=True), probs, zero)
    out = (probs / torch.empty_like(probs).exponential_()).argmax(-1)     # exponential race = categorical sample
    greedy = _expand(si.top_ks.view(-1), n) <= 1
    return torch.where(greedy, logits.argmax(-1), out)


_MODEL = [None]


def _in_graph(model):
    from . import sgl_model as SM
    return SM._DRAFT_IN_GRAPH and model.drafter is not None and (model.drafts_eager or bool(model.graph_drafts))


def _worker_cls():
    if "sglang.srt.speculative.cpp_ngram.ngram_corpus" not in sys.modules:
        try:
            import sglang.srt.speculative.cpp_ngram.ngram_corpus  # noqa: F401
        except Exception:   # tvm_ffi-built C++ corpus is not in the XPU image; the drafter replaces it
            m = types.ModuleType("sglang.srt.speculative.cpp_ngram.ngram_corpus")
            m.NgramCorpus = _NullCorpus
            sys.modules[m.__name__] = m
    from sglang.srt.speculative import ngram_worker as NW
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.speculative.spec_utils import prepare_mamba_track_for_verify
    from sglang.kernels.ops.speculative.cache_locs import assign_extend_cache_locs_func

    NW.eagle_sample = inkling_sample

    class InklingDSparkWorker(NW.NGRAMWorker):
        def __init__(self, server_args, gpu_id, ps, nccl_port, target_worker):
            NW.BaseSpecWorker.__init__(self)
            from sglang.srt.runtime_context import get_device, get_schedule, get_spec
            self.server_args = server_args
            self.enable_overlap = not get_schedule().disable_overlap_schedule
            self._target_worker = target_worker
            self.model_runner = target_worker.model_runner
            self.tp_rank = ps.tp_rank
            self.page_size = get_schedule().page_size
            self.draft_token_num = get_spec().speculative_num_draft_tokens
            self.speculative_num_draft_tokens = self.draft_token_num
            self.max_trie_depth = 1
            self.topk = 1
            self.speculative_num_steps = get_spec().speculative_num_steps
            self.device = get_device().device
            self.adaptive_controller = None
            self._prev_decode_rids = set()
            self.grammar_tree_host = None
            self.ngram_corpus = _NullCorpus()
            self._links = {}
            self._pgraphs = {}
            self.n_steps = 0
            self.n_accepted = 0
            logger.info("inkling2: DSpark worker, %d tokens per verify, overlap=%s", self.draft_token_num, self.enable_overlap)

        def clear_cache_pool(self):
            pass

        def _propose(self, dr, bs, slots, root, seq, nd):
            """drafter block forward, one XPU graph per batch size (static input buffers; the drafter's state rings
            are written in place by the replay)."""
            if not _DRAFT_GRAPH:
                return dr.propose(slots, root, seq, nd)
            g = self._pgraphs.get(bs)
            if g is None:
                bufs = (slots.clone(), root.clone(), seq.clone())
                st = torch.xpu.Stream()
                st.wait_stream(torch.xpu.current_stream())
                with torch.xpu.stream(st):
                    for _ in range(2):
                        dr.propose(*bufs, nd)      # warm-up: same inputs as the real step, so its ring writes are valid
                torch.xpu.current_stream().wait_stream(st)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph):
                    out = dr.propose(*bufs, nd)
                g = self._pgraphs[bs] = (graph, bufs, out)
                logger.info("inkling2: captured drafter graph for batch %d", bs)
            graph, bufs, out = g
            bufs[0].copy_(slots); bufs[1].copy_(root); bufs[2].copy_(seq)
            graph.replay()
            return out.clone()

        def _update_ngram_corpus(self, batch):
            pass

        def _chain(self, bs):
            t = self._links.get(bs)
            if t is None:
                n, dev = self.draft_token_num, self.device
                ri = torch.arange(bs * n, device=dev, dtype=torch.int64).view(bs, n)
                nt = torch.arange(1, n + 1, device=dev, dtype=torch.int64).repeat(bs, 1)
                nt[:, -1] = -1
                ns = torch.full((bs, n), -1, device=dev, dtype=torch.int64)
                tri = torch.tril(torch.ones(n, n, dtype=torch.bool, device=dev))
                t = self._links[bs] = (ri, nt, ns, tri, torch.arange(n, device=dev))
            return t

        @torch.no_grad()
        def _prepare_for_speculative_decoding(self, batch):
            if not batch.forward_mode.is_decode():
                return
            bs = len(batch.reqs)
            n, dev = self.draft_token_num, self.device
            model = self.model_runner.model
            ri, nt, ns, tri, ar = self._chain(bs)
            spec = batch.spec_info
            if (self.enable_overlap and not batch.grammar_needs_sync() and spec is not None
                    and getattr(spec, "accept_tokens", None) is not None):
                at = spec.accept_tokens.view(bs, -1)
                root = at[torch.arange(bs, device=at.device), spec.accept_lens.long() - 1].to(dev).long()
            else:
                root = torch.tensor([r.output_ids[-1] if len(r.output_ids) else r.origin_input_ids[-1] for r in batch.reqs],
                                    device=dev, dtype=torch.long)
            seq = batch.seq_lens.long()
            _tick("start")
            _MODEL[0] = model
            model.drafts_eager = False
            from . import sgl_model as SM
            if SM._DRAFT_IN_GRAPH:
                drafts = torch.zeros(bs, n - 1, device=dev, dtype=torch.long)   # filled inside the verify graph
            else:
                drafts = self._propose(model.drafter, bs, batch.req_pool_indices.long(), root, seq, n - 1)
            _tick("propose")
            draft_tokens = torch.cat([root[:, None], drafts.long()], 1).reshape(-1)
            positions = (seq[:, None] + ar[None, :]).reshape(-1)
            # full mask (prefix ones + causal chain), as the NGRAM worker builds it; the Inkling2 model itself
            # attends causally by position and ignores it
            sl = batch.seq_lens_cpu.tolist()
            mask = torch.cat([torch.cat([torch.ones(n, int(sl[i]), dtype=torch.bool, device=dev), tri], 1).reshape(-1)
                              for i in range(bs)])
            batch.forward_mode = ForwardMode.TARGET_VERIFY
            batch.input_ids = draft_tokens
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices, req_to_token=batch.req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens, end_offset=batch.seq_lens + n, batch_size=bs, draft_token_num=n,
                device=dev)
            prepare_mamba_track_for_verify(batch)
            batch.spec_info = NgramVerifyInput(
                draft_token=draft_tokens, custom_mask=mask, positions=positions, retrieve_index=ri,
                retrieve_next_token=nt, retrieve_next_sibling=ns, draft_token_num=n)
            _tick("mask_locs")

        def forward_batch_generation(self, batch, *a, **k):
            if _TPROF and batch.forward_mode.is_decode():
                self._tp_n = getattr(self, "_tp_n", 0) + 1
                if self._tp_n == 150:
                    from torch.profiler import profile, ProfilerActivity
                    self._tp = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
                    self._tp.__enter__()
                elif self._tp_n == 170:
                    torch.xpu.synchronize()
                    self._tp.__exit__(None, None, None)
                    ka = self._tp.key_averages()
                    cpu = sorted(ka, key=lambda e: -e.cpu_time_total)[:45]
                    logger.info("inkling2 spec trace (20 steps), top CPU ops:\n%s", "\n".join(
                        f"{e.count:6d}x cpu {e.cpu_time_total / 20e3:8.2f} ms/step self {e.self_cpu_time_total / 20e3:7.2f} dev {e.self_device_time_total / 20e3:7.2f}  {e.key[:110]}" for e in cpu))
                    self._tp.export_chrome_trace("/hc-cache/spec-trace.json")
            r = super().forward_batch_generation(batch, *a, **k)
            if _PROF and "propose" in _T:
                _tick("after_sample")
                self.n_steps += 1
                if self.n_steps % 100 == 0:
                    logger.info("inkling2 spec step ms (avg of 100): %s", {kk: round(v * 10, 2) for kk, v in _T.items() if kk != "_last"})
                    for kk in list(_T):
                        if kk != "_last":
                            _T[kk] = 0.0
            return r

    return InklingDSparkWorker


def register():
    """Called from the exl3xpu SGLang plugin when INKLING2_DSPARK=1 (launcher, tokenizer and scheduler processes)."""
    global _done
    if _done:
        return
    _done = True
    _stub_cuda_ops()
    try:
        _patch_ngram_args()
    except Exception as e:  # pragma: no cover
        logger.warning("inkling2: could not patch NGRAM server-arg handling: %s", e)
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    orig = SpeculativeAlgorithm.create_worker

    def create_worker(self, server_args):
        if self.is_ngram():
            return _worker_cls()
        return orig(self, server_args)

    SpeculativeAlgorithm.create_worker = create_worker
    logger.info("inkling2: DSpark speculative worker registered (NGRAM slot)")
