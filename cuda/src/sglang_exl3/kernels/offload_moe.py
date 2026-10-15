"""Offload MoE: the grouped EXL3 Marlin MoE kernels with experts addressed through a per-expert POINTER TABLE.

An expert can live anywhere the GPU can dereference: a slot of a device cache arena, a device staging buffer (prefill
streaming), or pinned + mapped host memory (zero-copy over PCIe, UVA address). The runtime owns the table and rewrites
it in place; the kernels read it on the device, so a captured CUDA graph stays valid while experts move.

Expert RECORD (one contiguous blob per expert, identical in every tier, so moving an expert is one memcpy):

    field 0 w13    int32 [H/16, 2I/64, 4, 8K] (K=2, 3) | [H/16, 2I/64, 32, 4] (K=4)   gate | up along n (pack layout)
    field 1 w2     int32 [I/16, H/64, 4, 8K]  (K=2, 3) | [I/16, H/64, 32, 4]  (K=4)
    field 2 suh13  fp16  [2, H]     gate suh, up suh
    field 3 svh13  fp16  [2I]       gate svh | up svh
    field 4 suh2   fp16  [I]
    field 5 svh2   fp16  [H]

Every field starts on a 256-byte boundary (record_bytes is a multiple of 256). For Qwen3.8-Flash-Next (H 2560, I 640,
K=3): 1,228,800 + 614,400 + 10,240 + 2,560 + 1,280 + 5,120 = 1,862,400 B per expert (the checkpoint's 1,862,412 B
minus the three 4-byte mul1 markers), 953,548,800 B per layer. K=2 (2.05bpw): 819,200 + 409,600 + 19,200 = 1,248,000 B
per expert, 638,976,000 B per layer.

POINTER TABLE: int64 CUDA tensor [E, 6], row e = the six field addresses of LOGICAL expert e (router id). Build it from
per-expert record base addresses with `fill_table_` (device op, graph-capturable), e.g. from a slot map:
    base[e] = arena_base + slot[e] * record_bytes    if slot[e] >= 0   (GPU cache / staging slot)
            = host_base  + e       * record_bytes    otherwise         (zero-copy from the host bank)
Routing is unchanged: moe_align_block_size(topk_ids, block, E) with logical ids; the table does the mapping.

One MoE layer = moe_had_in_ptr -> moe_gemm_ptr (gate+up, svh in launch) -> moe_glu_had_in_ptr -> moe_gemm_ptr (down)
-> moe_combine: the same five launches and the same arithmetic as marlin_moe.run (bit-identical outputs).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import marlin_moe

FIELDS = ("w13", "w2", "suh13", "svh13", "suh2", "svh2")
F_W13, F_W2, F_SUH13, F_SVH13, F_SUH2, F_SVH2 = range(6)
ALIGN = 256


def _mod():
    return marlin_moe._load()


@dataclass(frozen=True)
class RecordLayout:
    hidden: int
    inter: int
    bits: int
    sizes: tuple
    offsets: tuple
    record_bytes: int

    def offsets_tensor(self, device) -> torch.Tensor:
        return torch.tensor(self.offsets, dtype=torch.int64, device=device)


def layout(hidden: int, inter: int, bits: int) -> RecordLayout:
    if bits not in (2, 3, 4) or hidden % 128 or inter % 128:
        raise ValueError("K must be 2, 3 or 4 and hidden / inter multiples of 128")
    w = lambda k, n: k * n * bits // 8
    sizes = (w(hidden, 2 * inter), w(inter, hidden), 2 * hidden * 2, 2 * inter * 2, inter * 2, hidden * 2)
    offs, o = [], 0
    for s in sizes:
        offs.append(o)
        o += (s + ALIGN - 1) // ALIGN * ALIGN
    return RecordLayout(hidden, inter, bits, sizes, tuple(offs), o)


def expert_record(gate, up, down, e: int, lay: RecordLayout, out: torch.Tensor | None = None) -> torch.Tensor:
    """gate / up / down = (trellis list, suh list, svh list) by expert (as marlin_moe.prepare). -> uint8 [record_bytes]
    on the tensors' device (or written into `out`, any device). Pure re-layout of expert e."""
    w13 = torch.cat([marlin_moe.stack_repacked([gate[0][e]]), marlin_moe.stack_repacked([up[0][e]])], dim=2)
    w2 = marlin_moe.stack_repacked([down[0][e]])
    if marlin_moe.bits_of_stack(w13) != lay.bits:
        raise ValueError("expert K does not match the layout")
    parts = (w13, w2, torch.stack([gate[1][e], up[1][e]]), torch.cat([gate[2][e], up[2][e]]), down[1][e], down[2][e])
    rec = torch.zeros((lay.record_bytes,), dtype=torch.uint8, device=w13.device) if out is None else out
    for p, off, size in zip(parts, lay.offsets, lay.sizes):
        b = p.contiguous().view(-1).view(torch.uint8)
        if b.numel() != size:
            raise ValueError(f"field size {b.numel()} != {size}")
        rec[off:off + size].copy_(b)
    return rec


def build_bank(gate, up, down, lay: RecordLayout, where: str = "host", device=None) -> torch.Tensor:
    """All experts of a layer as uint8 [E, record_bytes]: where='host' -> pinned + mapped host memory (exact-size
    cudaHostAlloc), 'device' -> a CUDA tensor (a cache arena / staging buffer holding every expert)."""
    e = len(gate[0])
    if where == "host":
        bank = _mod().host_alloc_mapped(e * lay.record_bytes).view(e, lay.record_bytes)
    else:
        bank = torch.empty((e, lay.record_bytes), dtype=torch.uint8, device=device or "cuda")
    for i in range(e):
        bank[i].copy_(expert_record(gate, up, down, i, lay))
    return bank


def base_address(t: torch.Tensor) -> int:
    """Kernel-visible address of a tensor: device pointer, or the UVA address of pinned + mapped host memory."""
    return t.data_ptr() if t.is_cuda else int(_mod().host_device_ptr(t))


def new_table(num_experts: int, device=None) -> torch.Tensor:
    return torch.zeros((num_experts, len(FIELDS)), dtype=torch.int64, device=device or "cuda")


def fill_table_(table: torch.Tensor, bases: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """table[e, f] = bases[e] + offsets[f], in place (device op, CUDA-graph capturable)."""
    torch.add(bases.unsqueeze(1), offsets.unsqueeze(0), out=table)
    return table


def slot_bases(slot_of_expert: torch.Tensor, arena_base: int, host_base: int, record_bytes: int) -> torch.Tensor:
    """Per-expert record base from a slot map (int32/int64 [E], -1 = not resident -> host bank row e)."""
    s = slot_of_expert.to(torch.int64)
    e = torch.arange(s.numel(), dtype=torch.int64, device=s.device)
    return torch.where(s >= 0, arena_base + s * record_bytes, host_base + e * record_bytes)


def fill_admit_(wb: torch.Tensor, dst_bases: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Admission table for `run(..., admit=wb)`: row e = expert e's record fields in its DESTINATION slot, or all 0
    (no admission) where dst_bases[e] == 0. In place, device op (graph-capturable)."""
    torch.where((dst_bases != 0).unsqueeze(1), dst_bases.unsqueeze(1) + offsets.unsqueeze(0),
                torch.zeros((), dtype=torch.int64, device=wb.device), out=wb)
    return wb


def run(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block: int, table: torch.Tensor,
        lay: RecordLayout, codebook: int, out: torch.Tensor | None = None, admit: torch.Tensor | None = None,
        scratch: int = 0) -> torch.Tensor:
    """x fp16 | bf16 [T, H]; topk_weights fp32 [T, top_k]; topk_ids int32 | int64 [T, top_k] (logical ids);
    sorted_ids / expert_ids / num_post_padded = moe_align_block_size(topk_ids, block, E); table int64 [E, 6].
    admit (optional, int64 [E, 6], see fill_admit_): fused admission - every routed expert with a nonzero admit row is
    also written to that destination record while it is computed (the B tiles from the GEMMs' shared-memory stages, the
    small fields by one copy launch), e.g. zero-copy misses into their victim cache slots. The caller re-points the
    table rows afterwards (stream order) and must not route an expert whose record is being overwritten."""
    mod = _mod()
    tokens, hidden = x.shape
    top_k, inter = topk_ids.shape[1], lay.inter
    slots = tokens * top_k
    y = out if out is not None else torch.empty_like(x)
    if tokens == 0:
        return y
    f16 = dict(dtype=torch.float16, device=x.device)
    xh = torch.empty((2 * slots, hidden), **f16)
    if admit is not None:   # small fields = the contiguous record tail (suh13 .. svh2)
        mod.moe_copy_fields(table, admit, topk_ids, F_SUH13, lay.record_bytes - lay.offsets[F_SUH13])
    mod.moe_had_in_ptr(x, table, F_SUH13, 2, topk_ids, xh)
    gu = torch.empty((slots, 2 * inter), **f16)
    cb = codebook
    mod.moe_gemm_ptr(xh, gu, table, F_W13, F_SVH13, lay.bits, sorted_ids, expert_ids, num_post_padded, block, inter, cb,
                     scratch=scratch, wb=admit)
    act, xd = torch.empty((slots, inter), **f16), torch.empty((slots, inter), **f16)
    mod.moe_glu_had_in_ptr(gu, table, F_SUH2, topk_ids, act, xd)
    yd = xh[:slots]
    mod.moe_gemm_ptr(xd, yd, table, F_W2, F_SVH2, lay.bits, sorted_ids, expert_ids, num_post_padded, block, 0, cb,
                     scratch=scratch, wb=admit)
    mod.moe_combine(yd, topk_weights, topk_ids, table.shape[0], y)
    return y


_ident_maps: dict = {}


def align_decode(topk_ids: torch.Tensor, block: int, num_experts: int):
    """Decode-sized moe_align_block_size in ONE launch (csrc moe_align_decode, identity expert map): same
    (sorted_ids, expert_ids, num_post_padded) contract as SGLang's align with ignore_invalid_expert=True (ids outside
    [0, E) dropped, padding = numel, padding blocks -1). Order of slots inside an expert's block may differ (rows are
    independent: results are bit-identical). topk_ids int32; slots <= 4096, E <= 1024. Graph-capturable."""
    key = (num_experts, topk_ids.device)
    m = _ident_maps.get(key)
    if m is None:
        m = _ident_maps[key] = torch.arange(num_experts, dtype=torch.int32, device=topk_ids.device)
    ids = topk_ids.reshape(-1)
    if ids.dtype != torch.int32:
        ids = ids.to(torch.int32)
    slots = ids.numel()
    cap = marlin_moe.align_capacity(slots, num_experts, block)
    sorted_ids = torch.empty((cap,), dtype=torch.int32, device=ids.device)
    eids = torch.empty(((cap + block - 1) // block,), dtype=torch.int32, device=ids.device)
    post = torch.empty((1,), dtype=torch.int32, device=ids.device)
    _mod().moe_align_decode(ids.contiguous(), [m], [int(block)], num_experts, [sorted_ids], [eids], [post])
    return sorted_ids, eids, post


class ExpertCache:
    """GPU expert cache shared by all MoE layers, managed on the device (csrc/exl3_offload_cache.cu).

    Owns: the slot arena (uint8 [slots, record_bytes]), per-layer pointer tables [L, E, 6] (initially every expert on
    its layer's host bank), admission tables, CLOCK state and per-layer hit/miss counters. Per layer and step:

        cache.step(layer, topk_ids)                 # before the MoE: hits pinned, misses get victim slots
        y = cache.run(layer, x, w, topk_ids, sorted_ids, expert_ids, npost, block)   # zero-copy misses + admission
        # (run = step + om.run(..., table=cache.tables[layer], admit=cache.admit[layer]) + commit)

    Everything is device-side with static shapes (CUDA-graph capturable). topk_ids must be int32 (layer-local ids,
    sentinel E = dropped slot). host_bases[l] = kernel-visible base address of layer l's host bank (record layout).
    Misses are served zero-copy from the host this step and written into their slot by the GEMM (fused admission);
    from the next step on they are hits. admit=False: misses are served zero-copy without admission (bypass)."""

    def __init__(self, num_layers: int, num_experts: int, lay: RecordLayout, host_bases, slots: int, device=None):
        dev = torch.device(device or "cuda")
        self.L, self.E, self.S, self.lay = num_layers, num_experts, slots, lay
        self.arena = torch.empty((max(slots, 1), lay.record_bytes), dtype=torch.uint8, device=dev)
        self.offs = lay.offsets_tensor(dev)
        self.host_bases = torch.tensor(list(host_bases), dtype=torch.int64, device=dev)
        e = torch.arange(num_experts, dtype=torch.int64, device=dev)
        rows = self.host_bases.view(-1, 1, 1) + (e * lay.record_bytes).view(1, -1, 1) + self.offs.view(1, 1, -1)
        self.tables = rows.contiguous()                                            # int64 [L, E, 6]
        self.admit = torch.zeros_like(self.tables)
        i32 = dict(dtype=torch.int32, device=dev)
        self.slot_of = torch.full((num_layers * num_experts,), -1, **i32)
        self.owner = torch.full((max(slots, 1),), -1, **i32)
        self.stamp = torch.zeros((max(slots, 1),), dtype=torch.int64, device=dev)
        self.ref = torch.zeros((max(slots, 1),), **i32)
        self.hand = torch.zeros((1,), **i32)
        self.clock = torch.zeros((1,), dtype=torch.int64, device=dev)
        self.stats = torch.zeros((num_layers, 2), dtype=torch.int64, device=dev)

    def step(self, layer: int, topk_ids: torch.Tensor, admit: bool = True) -> None:
        _mod().moe_cache_step(topk_ids.reshape(-1), layer, self.slot_of, self.owner, self.stamp, self.ref, self.hand,
                              self.clock, self.tables, self.admit, self.host_bases, self.arena.data_ptr(),
                              self.lay.record_bytes, self.offs, self.stats, admit and self.S > 0)

    def commit(self, layer: int, topk_ids: torch.Tensor) -> None:
        _mod().moe_cache_commit(topk_ids.reshape(-1), layer, self.tables, self.admit)

    def run(self, layer: int, x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block: int,
            codebook: int = 2, admit: bool = True, out=None) -> torch.Tensor:
        self.step(layer, topk_ids, admit)
        y = run(x, topk_weights, topk_ids, sorted_ids, expert_ids, num_post_padded, block, self.tables[layer], self.lay,
                codebook, out=out, admit=self.admit[layer] if admit else None)
        if admit:
            self.commit(layer, topk_ids)
        return y

    def set_layer_bank(self, layer: int, host_base: int) -> None:
        """(Re)point layer `layer` at its host bank (e.g. as layers finish loading). The layer must have no resident
        experts yet (fresh cache)."""
        e = torch.arange(self.E, dtype=torch.int64, device=self.tables.device)
        self.host_bases[layer] = host_base
        self.tables[layer].copy_(host_base + (e * self.lay.record_bytes).unsqueeze(1) + self.offs.unsqueeze(0))

    def preload(self, layer: int, experts) -> None:
        """Init-time warm fill (not graph-safe, host-side slot choice): copy `experts` of `layer` into free slots."""
        free = (self.owner < 0).nonzero().flatten()
        ex = torch.as_tensor(list(experts), dtype=torch.int64, device=self.tables.device)
        if ex.numel() > free.numel():
            raise ValueError("not enough free slots")
        sl = free[: ex.numel()].to(torch.int64)
        rec = self.lay.record_bytes
        _mod().copy_records(self.host_bases[layer] + ex * rec, self.arena.data_ptr() + sl * rec, rec)
        g = layer * self.E + ex
        self.slot_of[g] = sl.to(torch.int32)
        self.owner[sl] = g.to(torch.int32)
        rows = self.tables[layer].clone()
        rows[ex] = self.arena.data_ptr() + (sl * rec).unsqueeze(1) + self.offs.unsqueeze(0)
        self.tables[layer].copy_(rows)

    def prefill_plan(self, layer: int, staging_base: int, table_out: torch.Tensor, src_out: torch.Tensor,
                     dst_out: torch.Tensor) -> None:
        """Prefill of `layer` from a staging buffer (record layout, row e = expert e): table_out [E, 6] points resident
        experts at their slots and the others at staging row e; src_out / dst_out int64 [E] = the copy list for
        copy_records / copy_records_persistent (src 0 = resident, skipped): only the missing experts cross PCIe.
        Device ops only (graph-capturable)."""
        rec = self.lay.record_bytes
        e = torch.arange(self.E, dtype=torch.int64, device=self.tables.device)
        sl = self.slot_of[layer * self.E:(layer + 1) * self.E].to(torch.int64)
        stage = staging_base + e * rec
        res = sl >= 0
        fill_table_(table_out, torch.where(res, self.arena.data_ptr() + sl * rec, stage), self.offs)
        src_out.copy_(torch.where(res, torch.zeros_like(stage), self.host_bases[layer] + e * rec))
        dst_out.copy_(stage)

    def run_fused(self, layer: int, x, topk_weights, topk_ids, codebook: int = 2, out=None) -> torch.Tensor:
        """K07 decode path (T * top_k <= 1024 slots, block 8): ONE prologue launch (deferred commit of this layer's
        previous admissions + ids clean + align + cache step) then the five MoE launches with fused admission. Admitted
        experts are committed to the table by this layer's next prologue (so the admit rows stay set until then).
        Bit-identical to run() (same align order, same kernels)."""
        ids = topk_ids.reshape(-1)
        if ids.dtype != torch.int32:
            ids = ids.to(torch.int32)
        ids = ids.contiguous()
        n = ids.numel()
        dev = ids.device
        ids_out = torch.empty_like(ids)
        sorted_ids = torch.empty((n * 8,), dtype=torch.int32, device=dev)
        eids = torch.empty((n,), dtype=torch.int32, device=dev)
        post = torch.empty((1,), dtype=torch.int32, device=dev)
        _mod().moe_cache_decode_prologue(ids, layer, self.slot_of, self.owner, self.stamp, self.ref, self.hand, self.clock,
                                         self.tables, self.admit, self.host_bases, self.arena.data_ptr(),
                                         self.lay.record_bytes, self.offs, self.stats, self.S > 0,
                                         ids_out, sorted_ids, eids, post)
        return run(x, topk_weights, ids_out.view(topk_ids.shape), sorted_ids, eids, post, 8, self.tables[layer], self.lay,
                   codebook, out=out, admit=self.admit[layer] if self.S > 0 else None)

    def hit_rate(self) -> torch.Tensor:
        s = self.stats.double()
        return s[:, 0] / (s[:, 0] + s[:, 1]).clamp(min=1)
