"""Marlin MoE backend: grouped EXL3 expert kernels (csrc/exl3_marlin_moe.cu, module `trellis_exl3_moe_kernels`).

All experts of a projection are stacked `[E, ...]` in the lane-major layout of the dense Marlin kernel (K = 4: a pure
permutation of the stream words), gate and up concatenated along n. One MoE layer is five launches + the router-side
`moe_align_block_size` (done by the caller: this layer may not import vLLM):

    had_in (gather, per-expert suh, 2 slabs) -> grouped GEMM gate+up (output transform inside the launch)
    -> glu_had_in (fp32 silu * up -> fp16 -> per-expert suh) -> grouped GEMM down -> fp32 router-weighted combine

Decoded expert weights are bit-identical to ExLlamaV3's; accumulation order differs (parity/marlin_moe_parity.py).
Every (token, expert) slot has its own input transform because every expert has its own suh (verified on
Qwen3.6-35B-A3B: suh / svh differ between experts, and gate suh != up suh inside an expert; tools/moe_inspect.py).
Shapes are static for a fixed (tokens, top_k, moe block size), so a captured CUDA graph replays correctly; the number
of valid moe blocks is read on the device.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field

import torch

from .base import BackendStatus

_mod = None
BLOCK_SIZES = (8, 16, 32, 48, 64)
CB_MCG_LUT = 13             # csrc: MCG through a 128 KB shared-memory table (decode block families 8 / 16 rows only)
LUT_MCG = os.environ.get("TRELLIS_EXL3_MOE_LUT", "0") == "1"


def effective_codebook(codebook: int, block: int) -> int:
    """cb 13 (table decode) for MCG experts at decode block sizes when TRELLIS_EXL3_MOE_LUT=1; identical values."""
    return CB_MCG_LUT if (LUT_MCG and codebook == 1 and block <= 16) else codebook


def _load():
    global _mod
    if _mod is None:
        _mod = importlib.import_module("trellis_exl3_moe_kernels")
    return _mod


def probe() -> BackendStatus:
    try:
        _load()
    except ImportError as e:
        return BackendStatus(False, detail=str(e))
    return BackendStatus(True)


def moe_block_size(tokens: int, top_k: int, num_experts: int) -> int:
    """Rows per moe block: vLLM fused_marlin_moe's data-independent ladder (depends on shapes only: graph-safe)."""
    for block in BLOCK_SIZES:
        if tokens * top_k / num_experts / block < 0.9:
            break
    return block


def sorted_ids_capacity(tokens: int, top_k: int, num_experts: int, block: int) -> int:
    """Length of moe_align_block_size's sorted_token_ids for these shapes (its own upper bound)."""
    return tokens * top_k + num_experts * (block - 1)


def stack_repacked(trellis: list[torch.Tensor]) -> torch.Tensor:
    """E x int16 [k/16, n/16, 16K] -> the grouped kernel's stack, a pure permutation / view of the stored words:
    K = 4: int32 [E, k/16, n/64, 32, 4], per expert exactly `trellis_exl3_kernels.repack_trellis`
           (word l of tile (i, 4g + j) -> [i, g, l, j]);
    K = 3: int32 [E, k/16, n/64, 4, 24], the tiles exactly as stored (24 words each); the kernel stages them
           byte-exact and every lane gathers its two words (csrc/exl3_decode.cuh, dq8_regs_3bits);
    K = 2: int32 [E, k/16, n/64, 4, 16], likewise (16 words per tile, dq8_regs_2bits)."""
    t = torch.stack([w.contiguous() for w in trellis])
    e, kt, nt, words = t.shape
    if t.dtype != torch.int16 or words not in (32, 48, 64) or kt % 8 or nt % 8:
        raise ValueError(f"experts must be K=2, 3 or 4 trellises with k, n multiples of 128, got {tuple(t.shape)} {t.dtype}")
    if words in (32, 48):
        return t.view(torch.int32).view(e, kt, nt // 4, 4, words // 2).contiguous()
    return t.view(torch.int32).view(e, kt, nt // 4, 4, 32).permute(0, 1, 2, 4, 3).contiguous()


def bits_of_stack(stack: torch.Tensor) -> int:
    return {24: 3, 16: 2}.get(stack.shape[-1], 4)


@dataclass
class ExpertPack:
    w13: torch.Tensor      # int32 [E, H/16, 2I/64, 32, 4]   gate | up along n
    w2: torch.Tensor       # int32 [E, I/16, H/64, 32, 4]
    suh13: torch.Tensor    # fp16 [E, 2, H]
    svh13: torch.Tensor    # fp16 [E, 2I]
    suh2: torch.Tensor     # fp16 [E, I]
    svh2: torch.Tensor     # fp16 [E, H]
    codebook: int
    hidden: int
    inter: int
    num_experts: int
    bits: int = 4          # EXL3 K of these experts (uniform inside a pack; mixed-K layers use one pack per K)

    def tensors(self) -> list[torch.Tensor]:
        return [self.w13, self.w2, self.suh13, self.svh13, self.suh2, self.svh2]


def supports(gate: list[torch.Tensor], up: list[torch.Tensor], down: list[torch.Tensor], codebook: int) -> tuple[bool, str]:
    shapes = {tuple(t.shape) for t in gate} | {tuple(t.shape) for t in up}
    if len(shapes) != 1 or len({tuple(t.shape) for t in down}) != 1:
        return False, "experts differ in shape / bitrate"
    (kt, nt, words), (kt2, nt2, words2) = shapes.pop(), tuple(down[0].shape)
    if words not in (32, 48, 64) or words2 not in (32, 48, 64):
        return False, "only K = 2, 3 and 4 experts"
    if kt % 8 or nt % 8 or kt2 != nt or nt2 != kt:
        return False, "hidden / intermediate not multiples of 128, or down does not mirror gate/up"
    if codebook not in (1, 2):
        return False, "codebook must be mcg or mul1"
    return True, ""


def prepare(gate, up, down, codebook: int) -> ExpertPack:
    """gate / up / down = (trellis list, suh list, svh list) indexed by expert. Pure re-layout, no value changes.
    The per-expert tensors may live on the CPU: the stacks are then built there and moved once, so the GPU holds
    only the pack (the peak is otherwise two copies of the experts)."""
    mod = _load()
    dev = gate[0][0].device
    if dev.type != "cuda":
        dev = torch.device("cuda", torch.cuda.current_device())
    mod.moe_init_device(dev.index)                 # per-device state must exist before any CUDA-graph capture
    w13 = torch.cat([stack_repacked(gate[0]), stack_repacked(up[0])], dim=2).contiguous().to(dev)
    w2 = stack_repacked(down[0]).to(dev)
    suh13 = torch.stack([torch.stack(gate[1]), torch.stack(up[1])], dim=1).contiguous().to(dev)
    svh13 = torch.cat([torch.stack(gate[2]), torch.stack(up[2])], dim=1).contiguous().to(dev)
    suh2, svh2 = torch.stack(down[1]).contiguous().to(dev), torch.stack(down[2]).contiguous().to(dev)
    hidden, inter = w13.shape[1] * 16, w2.shape[1] * 16
    return ExpertPack(w13, w2, suh13, svh13, suh2, svh2, codebook, hidden, inter, w13.shape[0], bits_of_stack(w13))


@dataclass
class MixedPack:
    """Mixed K per expert (e.g. 192 experts at K=3 + 64 at K=4 per layer): one ExpertPack per K class in LOCAL expert
    order, plus the union input transforms in GLOBAL order and, per class, vLLM's expert_map (global id -> local id,
    -1 elsewhere). Routing: `moe_align_block_size(topk_ids, block_c, num_experts, expert_map_c, ignore_invalid_experts=True)`
    per class gives blocks for that class's slots only, with local expert ids; every slot belongs to exactly one class,
    so the class GEMM launches together fill the shared gate/up and down slabs. One input transform, one GLU, one
    combine, no remapped ids, no dummy experts: 2 + 2 launches + the transforms per layer."""
    packs: list[ExpertPack]
    expert_maps: list[torch.Tensor]     # int32 [E] per class: global id -> local id, -1 elsewhere
    suh13: torch.Tensor                 # fp16 [E, 2, H], global order
    suh2: torch.Tensor                  # fp16 [E, I], global order
    num_experts: int
    hidden: int
    inter: int
    codebook: int
    expert_ids: list[torch.Tensor] = field(default_factory=list)    # int32 [E_c] per class: local id -> global id

    def __post_init__(self):
        if not self.expert_ids:
            self.expert_ids = [(m >= 0).nonzero().flatten().to(torch.int32).contiguous() for m in self.expert_maps]

    @property
    def bits(self) -> list[int]:
        return [p.bits for p in self.packs]

    def block_sizes(self, tokens: int, top_k: int) -> list[int]:
        return [moe_block_size(tokens, top_k, p.num_experts) for p in self.packs]


def prepare_mixed(gate, up, down, codebook: int) -> MixedPack:
    """Like `prepare`, for experts of mixed K (any device for the per-expert tensors; the GPU holds only the packs)."""
    words = [w.shape[2] for w in gate[0]]
    classes = sorted(set(words))
    e = len(words)
    packs, maps = [], []
    for wd in classes:
        ids = [i for i in range(e) if words[i] == wd]
        sub = lambda t: tuple([t[j][i] for i in ids] for j in range(3))
        packs.append(prepare(sub(gate), sub(up), sub(down), codebook))
        m = torch.full((e,), -1, dtype=torch.int32)
        m[ids] = torch.arange(len(ids), dtype=torch.int32)
        maps.append(m.to(packs[-1].w13.device))
    dev = packs[0].w13.device
    suh13 = torch.stack([torch.stack(gate[1]), torch.stack(up[1])], dim=1).contiguous().to(dev)
    suh2 = torch.stack(down[1]).contiguous().to(dev)
    return MixedPack(packs, maps, suh13, suh2, e, packs[0].hidden, packs[0].inter, codebook)


ALIGN_DECODE_MAX_SLOTS = 4096


def align_capacity(slots: int, num_experts_class: int, block: int) -> int:
    """vLLM's sorted_token_ids length for these shapes (moe_align_block_size), so the outputs are interchangeable."""
    cap = slots + num_experts_class * (block - 1)
    cap = (cap + block - 1) // block * block
    if slots < num_experts_class:
        cap = min(slots * block, cap)
    return cap


def align_decode(topk_ids: torch.Tensor, pack: MixedPack, tokens: int, top_k: int) -> list[tuple]:
    """Routings for run_mixed from our single-launch decode align (slots <= ALIGN_DECODE_MAX_SLOTS). Returns the
    same (sorted_ids, expert_ids, num_post, block) per class as vLLM's moe_align_block_size with the class's expert_map."""
    mod = _load()
    ids = topk_ids.reshape(-1).to(torch.int32).contiguous()
    slots = ids.numel()
    blocks = pack.block_sizes(tokens, top_k)
    sorted_l, eids_l, post_l = [], [], []
    for pk, block in zip(pack.packs, blocks):
        cap = align_capacity(slots, pk.num_experts, block)
        sorted_l.append(torch.empty((cap,), dtype=torch.int32, device=ids.device))
        eids_l.append(torch.empty(((cap + block - 1) // block,), dtype=torch.int32, device=ids.device))
        post_l.append(torch.empty((1,), dtype=torch.int32, device=ids.device))
    mod.moe_align_decode(ids, list(pack.expert_maps), [int(b) for b in blocks], pack.num_experts, sorted_l, eids_l, post_l)
    return [(s, e, p, b) for s, e, p, b in zip(sorted_l, eids_l, post_l, blocks)]


_OVERLAP = os.environ.get("TRELLIS_EXL3_MOE_OVERLAP", "1") == "1"   # mixed packs: the two K classes' GEMMs on two streams
_side_streams: dict[int, "torch.cuda.Stream"] = {}


def _side_stream(device: torch.device) -> "torch.cuda.Stream":
    idx = device.index if device.index is not None else torch.cuda.current_device()
    st = _side_streams.get(idx)
    if st is None:
        st = _side_streams[idx] = torch.cuda.Stream(device=idx)
    return st


def _class_gemms(mod, a, c, calls, device) -> None:
    """calls: per class (w, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb). Each class writes its own rows
    of c, so with two classes the second (smaller) one runs on a side stream with its own scratch set (fork / join by
    events: CUDA-graph capturable); the launcher hands each block only the shared memory it needs, so blocks of both
    launches co-reside on an SM. TRELLIS_EXL3_MOE_OVERLAP=0 keeps the serial order."""
    if len(calls) == 2 and _OVERLAP:
        cur, side = torch.cuda.current_stream(device), _side_stream(device)
        fork = torch.cuda.Event()
        fork.record(cur)
        side.wait_event(fork)
        w, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb = calls[1]
        with torch.cuda.stream(side):
            mod.moe_gemm(a, w, c, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb, -1, -1, 1)
        w, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb = calls[0]
        mod.moe_gemm(a, w, c, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb, -1, -1, 0)
        join = torch.cuda.Event()
        join.record(side)
        cur.wait_event(join)
        return
    for w, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb in calls:
        mod.moe_gemm(a, w, c, svh, sorted_ids, expert_ids, num_post, block, shard_end, cb)


def run_mixed(x, topk_weights, topk_ids, routings, pack: MixedPack, out: torch.Tensor | None = None) -> torch.Tensor:
    """routings: per class (sorted_ids, expert_ids, num_post_padded, block) from moe_align_block_size with that
    class's expert_map and ignore_invalid_experts=True (global topk_ids, local expert ids)."""
    mod = _load()
    tokens, hidden = x.shape
    top_k, inter = topk_ids.shape[1], pack.inter
    slots = tokens * top_k
    y = out if out is not None else torch.empty_like(x)
    if tokens == 0:
        return y
    f16 = dict(dtype=torch.float16, device=x.device)
    xh = torch.empty((2 * slots, hidden), **f16)
    mod.moe_had_in(x, pack.suh13, topk_ids, xh)
    gu = torch.empty((slots, 2 * inter), **f16)
    _class_gemms(mod, xh, gu, [(pk.w13, pk.svh13, s, e, p, b, inter, effective_codebook(pk.codebook, b))
                               for pk, (s, e, p, b) in zip(pack.packs, routings)], x.device)
    act, xd = torch.empty((slots, inter), **f16), torch.empty((slots, inter), **f16)
    mod.moe_glu_had_in(gu, pack.suh2, topk_ids, act, xd)
    yd = xh[:slots]
    _class_gemms(mod, xd, yd, [(pk.w2, pk.svh2, s, e, p, b, 0, effective_codebook(pk.codebook, b))
                               for pk, (s, e, p, b) in zip(pack.packs, routings)], x.device)
    mod.moe_combine(yd, topk_weights, topk_ids, pack.num_experts, y)
    return y


def run_tensors(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block: int, w13, w2, suh13, svh13,
                suh2, svh2, codebook: int, out: torch.Tensor | None = None, scratch: int = 0) -> torch.Tensor:
    """x fp16 | bf16 [T, H]; topk_weights float32 [T, top_k]; topk_ids int32 | int64 [T, top_k];
    sorted_ids / expert_ids / num_post_padded from moe_align_block_size(topk_ids, block, E). -> y [T, H] (x's dtype)."""
    mod = _load()
    tokens, hidden = x.shape
    top_k, inter = topk_ids.shape[1], suh2.shape[1]
    slots = tokens * top_k
    y = out if out is not None else torch.empty_like(x)
    if tokens == 0:
        return y
    f16 = dict(dtype=torch.float16, device=x.device)
    xh = torch.empty((2 * slots, hidden), **f16)
    mod.moe_had_in(x, suh13, topk_ids, xh)
    gu = torch.empty((slots, 2 * inter), **f16)
    cb = effective_codebook(codebook, block)
    mod.moe_gemm(xh, w13, gu, svh13, sorted_ids, expert_ids, num_post_padded, block, inter, cb, -1, -1, scratch)
    act, xd = torch.empty((slots, inter), **f16), torch.empty((slots, inter), **f16)
    mod.moe_glu_had_in(gu, suh2, topk_ids, act, xd)
    yd = xh[:slots]                                 # reuse: the gate slab is dead after the first GEMM
    mod.moe_gemm(xd, w2, yd, svh2, sorted_ids, expert_ids, num_post_padded, block, 0, cb, -1, -1, scratch)
    mod.moe_combine(yd, topk_weights, topk_ids, w13.shape[0], y)
    return y


def run(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block: int, pack: ExpertPack,
        out: torch.Tensor | None = None, scratch: int = 0) -> torch.Tensor:
    """scratch: 0 / 1 = which per-device lock + fp32-reduce scratch set the GEMMs use; two launches of this module that
    may run CONCURRENTLY (different streams) must use different sets."""
    return run_tensors(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block, *pack.tensors(),
                       pack.codebook, out, scratch)


# ---------------------------------------------------------------------------------------------------------------
# Building blocks for parity/marlin_moe_parity.py and tools/moe_microbench.py

def gemm_rotated(a, b, c, sorted_ids, expert_ids, num_post_padded, block: int, shard_end: int, codebook: int,
                 svh=None, thread_k: int = -1, thread_n: int = -1) -> torch.Tensor:
    """Grouped GEMM in the rotated basis (svh=None: no output transform). Identity rows of `a` read W_hat out."""
    _load().moe_gemm(a, b, c, svh, sorted_ids, expert_ids, num_post_padded, block, shard_end, codebook, thread_k, thread_n)
    return c


def set_gridstride_blocks(n: int) -> None:
    """Knob: geometry of the per-slot transform launches. 0 = one block per (row, 128-block) item; n > 0 = n grid-strided
    256-thread blocks, -1 = automatic (default). Same arithmetic per item, so results are bit-identical."""
    _load().moe_set_gridstride_blocks(int(n))


def set_large_block_bps(n: int) -> None:
    """Knob: cap of co-resident blocks per SM for moe blocks of more than 16 rows (default 1; 0 = Marlin MoE's choice)."""
    _load().moe_set_large_block_bps(int(n))


def set_blocks_per_sm(n: int) -> None:
    _load().moe_set_blocks_per_sm(int(n))


def set_grid_limit(n: int) -> None:
    """Knob: cap the grouped GEMM grid at n blocks (0 = sms x blocks per SM): longer k stripes per block."""
    _load().moe_set_grid_limit(int(n))
