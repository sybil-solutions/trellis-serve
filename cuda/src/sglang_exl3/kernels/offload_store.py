"""Pinned host expert store: every routed expert of every MoE layer in the offload RECORD layout (offload_moe), in ONE
page-aligned anonymous mapping, loaded straight from the checkpoint shards and pinned + mapped (UVA) per layer.

Load path (one copy of the experts is kept):
  * the shards' expert tensors are read with large sequential preadv calls (groups of adjacent tensors, <= group_bytes)
    by a reader thread into two reusable bounce buffers (double buffering);
  * the main thread re-lays each tensor out into its record field (K=3: gate/up trellis rows interleave into w13, down
    trellis / suh / svh are straight copies) with one threaded C++ scatter copy per group (host_scatter_copy, GIL
    released: per-tensor torch copies were Python-bound at ~2.2 GB/s), so the store never holds a second copy;
  * a layer whose 9 x E tensors are all in is cudaHostRegister'ed (Portable | Mapped) on a pin thread while later
    layers are still being read (pin-after-fill: registering first would fault and zero every page).
release() unregisters every layer BEFORE the mapping is unmapped (a stale UVA registration over a recycled VA range
corrupts later mappings: seen by the integration lane).
"""
from __future__ import annotations

import json
import mmap
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass, field

import torch

from . import offload_moe as om

_PAGE = 4096
_ROLES = {"gate": 0, "up": 1, "down": 2}
_KINDS = ("trellis", "suh", "svh")
_DT = {"I16": torch.int16, "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32}


def _shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


@dataclass
class LoadStats:
    seconds_total: float = 0.0
    seconds_read_wait: float = 0.0
    seconds_relayout: float = 0.0
    seconds_register: float = 0.0
    bytes_read: int = 0
    tensors: int = 0
    per_layer_register_s: list = field(default_factory=list)


class HostExpertStore:
    """layers: checkpoint layer ids to load (store index i = layers[i]); prefix: key prefix of those layers."""

    def __init__(self, model_dir: str, layers: list[int], num_experts: int, hidden: int, inter: int, bits: int = 3,
                 prefix: str = "model.language_model.layers.", register: bool = True, threads: int = 8, readers: int = 4):
        if bits not in (2, 3):
            raise NotImplementedError("HostExpertStore: fast relayout implemented for K=2 / K=3 experts only "
                                      "(byte-exact tiles; K=4 needs the lane-major repack)")
        self.model_dir, self.layers, self.E = model_dir, list(layers), num_experts
        self.lay = om.layout(hidden, inter, bits)
        self.L = len(self.layers)
        self.bank_bytes = num_experts * self.lay.record_bytes
        self.stride = (self.bank_bytes + _PAGE - 1) // _PAGE * _PAGE
        # private anonymous mapping (page aligned, lazily backed) with transparent huge pages: 2 MB first-touch faults
        # instead of 4 KB ones (a shared anonymous mapping is shmem: 4 KB pages, ~11 M zeroing faults for 46 GB)
        self._mm = mmap.mmap(-1, self.L * self.stride, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        if hasattr(mmap, "MADV_HUGEPAGE"):
            self._mm.madvise(mmap.MADV_HUGEPAGE)
        self.buf = torch.frombuffer(self._mm, dtype=torch.uint8)
        self.prefix, self.register, self.threads, self.readers = prefix, register, threads, readers
        self._registered = [False] * self.L
        self._key = re.compile(rf"^{re.escape(prefix)}(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.(trellis|suh|svh)$")
        self.stats = LoadStats()

    # ---- views
    def bank(self, i: int) -> torch.Tensor:
        """uint8 [E, record_bytes] of store layer i (CPU tensor over the pinned mapping)."""
        return self.buf[i * self.stride: i * self.stride + self.bank_bytes].view(self.E, self.lay.record_bytes)

    def base_address(self, i: int) -> int:
        return int(om._mod().host_device_ptr(self.bank(i))) if self._registered[i] else self.bank(i).data_ptr()

    # ---- load
    def _dst(self, i, e, role, kind, t_shape):
        """-> (destination view, shape the source must be viewed as) for one checkpoint tensor."""
        lay, rec = self.lay, self.bank(i)[e]
        H, I = lay.hidden, lay.inter
        o = lay.offsets
        if kind == "trellis":
            if role == "down":
                return rec[o[om.F_W2]:o[om.F_W2] + lay.sizes[om.F_W2]], None
            rows = H // 16
            w13 = rec[o[om.F_W13]:o[om.F_W13] + lay.sizes[om.F_W13]].view(rows, 2, -1)
            return w13[:, _ROLES[role], :], (rows, -1)
        if role == "down":
            f = om.F_SUH2 if kind == "suh" else om.F_SVH2
            return rec[o[f]:o[f] + lay.sizes[f]], None
        f = om.F_SUH13 if kind == "suh" else om.F_SVH13
        half = lay.sizes[f] // 2
        a = o[f] + _ROLES[role] * half
        return rec[a:a + half], None

    def _dst_entry(self, i, e, role, kind, nbytes):
        """-> (dst address, rows, row bytes, dst row stride) of one checkpoint tensor in its record field."""
        lay = self.lay
        o, H, I = lay.offsets, lay.hidden, lay.inter
        rec = self.buf.data_ptr() + i * self.stride + e * lay.record_bytes
        if kind == "trellis":
            if role == "down":
                f, size = om.F_W2, lay.sizes[om.F_W2]
                if nbytes != size:
                    raise ValueError(f"down trellis {nbytes} B vs {size}")
                return rec + o[f], 1, size, size
            rows = H // 16
            rb = lay.sizes[om.F_W13] // rows // 2
            if nbytes != rows * rb:
                raise ValueError(f"{role} trellis {nbytes} B vs {rows * rb}")
            return rec + o[om.F_W13] + _ROLES[role] * rb, rows, rb, 2 * rb
        if role == "down":
            f = om.F_SUH2 if kind == "suh" else om.F_SVH2
            size = lay.sizes[f]
        else:
            f = om.F_SUH13 if kind == "suh" else om.F_SVH13
            size = lay.sizes[f] // 2
        if nbytes != size:
            raise ValueError(f"{role}.{kind} {nbytes} B vs {size}")
        base = rec + o[f] + (0 if role == "down" else _ROLES[role] * size)
        return base, 1, size, size

    def load(self, group_bytes: int = 1 << 30, index_file: str = "model.safetensors.index.json") -> LoadStats:
        t0 = time.time()
        st = self.stats
        wm = json.load(open(os.path.join(self.model_dir, index_file)))["weight_map"]
        want = {}
        lmap = {l: i for i, l in enumerate(self.layers)}
        for k, fname in wm.items():
            m = self._key.match(k)
            if m and int(m.group(1)) in lmap and int(m.group(2)) < self.E:
                want.setdefault(fname, []).append((k, lmap[int(m.group(1))], int(m.group(2)), m.group(3), m.group(4)))
        expected = self.L * self.E * 9
        found = sum(len(v) for v in want.values())
        if found != expected:
            raise ValueError(f"HostExpertStore: {found} expert tensors in the index, expected {expected}")
        # groups of adjacent tensors per shard, in file order
        jobs = []
        for fname, items in sorted(want.items()):
            path = os.path.join(self.model_dir, fname)
            hdr, base = _shard_header(path)
            ents = []
            for k, i, e, role, kind in items:
                h = hdr[k]
                a, b = h["data_offsets"]
                ents.append((base + a, base + b, h["dtype"], tuple(h["shape"]), i, e, role, kind))
            ents.sort()
            g = []
            for ent in ents:
                if g and (ent[1] - g[0][0] > group_bytes or ent[0] - g[-1][1] > (1 << 20)):
                    jobs.append((path, g)); g = []
                g.append(ent)
            if g:
                jobs.append((path, g))
        cap = max(g[-1][1] - g[0][0] for _, g in jobs)
        nbuf = self.readers + 2
        bounce = [torch.empty(cap, dtype=torch.uint8) for _ in range(nbuf)]
        free_q, full_q, job_q = queue.Queue(), queue.Queue(), queue.Queue()
        for b in range(nbuf):
            free_q.put(b)
        for j in jobs:
            job_q.put(j)

        def reader():
            fds = {}
            try:
                while True:
                    try:
                        path, g = job_q.get_nowait()
                    except queue.Empty:
                        break
                    b = free_q.get()
                    fd = fds.get(path) or fds.setdefault(path, os.open(path, os.O_RDONLY))
                    a0, a1 = g[0][0], g[-1][1]
                    mv = memoryview(bounce[b].numpy())[: a1 - a0]
                    pos = 0
                    while pos < a1 - a0:                                      # 64 MB preadv calls (GIL released)
                        n = os.preadv(fd, [mv[pos: min(pos + (64 << 20), a1 - a0)]], a0 + pos)
                        if n <= 0:
                            raise IOError(f"short read {path} @ {a0 + pos}")
                        pos += n
                    full_q.put((b, g))
            except BaseException as ex:  # pragma: no cover
                full_q.put(ex)
            finally:
                for fd in fds.values():
                    os.close(fd)

        pin_q: queue.Queue = queue.Queue()

        def pinner():
            while True:
                i = pin_q.get()
                if i is None:
                    return
                ts = time.time()
                om._mod().host_register(self.bank(i))
                self._registered[i] = True
                st.per_layer_register_s.append(time.time() - ts)

        rts = [threading.Thread(target=reader, daemon=True) for _ in range(self.readers)]
        pt = threading.Thread(target=pinner, daemon=True)
        for r in rts:
            r.start()
        if self.register:
            pt.start()
        remaining = [self.E * 9] * self.L
        for _ in range(len(jobs)):
            tw = time.time()
            item = full_q.get()
            st.seconds_read_wait += time.time() - tw
            if isinstance(item, BaseException):
                raise item
            b, g = item
            tr = time.time()
            a0 = g[0][0]
            src_all = bounce[b]

            ent = [(s - a0, *self._dst_entry(i, e, role, kind, t - s)) for s, t, dt, shape, i, e, role, kind in g]
            om._mod().host_scatter_copy(src_all, torch.tensor(ent, dtype=torch.int64), self.threads)
            for s, t, dt, shape, i, e, role, kind in g:
                remaining[i] -= 1
                if remaining[i] == 0 and self.register:
                    pin_q.put(i)
                st.bytes_read += t - s
                st.tensors += 1
            st.seconds_relayout += time.time() - tr
            free_q.put(b)
        for r in rts:
            r.join()
        tp = time.time()
        pin_q.put(None)
        if self.register:
            pt.join()
        st.seconds_register = sum(st.per_layer_register_s)
        st.seconds_total = time.time() - t0
        st.register_tail_s = time.time() - tp
        del bounce
        if any(r != 0 for r in remaining):
            raise ValueError("HostExpertStore: incomplete layers after load")
        return st

    def release(self) -> None:
        for i in range(self.L):
            if self._registered[i]:
                om._mod().host_unregister(self.bank(i))
                self._registered[i] = False
        self.buf = None
        try:
            self._mm.close()
        except BufferError:  # a view is still alive somewhere: leave the (now unregistered) mapping to the GC
            pass

    def __del__(self):
        try:
            if any(self._registered):
                self.release()
        except Exception:
            pass
