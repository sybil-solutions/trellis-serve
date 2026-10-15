// Chunked-arena variant of exl3_offload_cache.cu (JIT-built, kernels/offload_moe.py): slot s lives at
// cbase[s / spc] + (s % spc) * rec, so arena chunks can be freed and re-allocated (elastic cache: the flex chunks are
// handed back to torch during prefill) without re-capturing the decode graphs; owner -2 = disabled slot (never a
// victim). Everything else is identical to exl3_offload_cache.cu.
// Device-side expert cache manager for the pointer-table MoE path (kernels/offload_moe.py ExpertCache).
//
// One global pool of S GPU slots shared by all layers (slot = one expert record). State, all on the device:
//   slot_of[L * E]  int32   slot of global expert g = layer * E + e, -1 = host-resident
//   owner[S]        int32   global expert in slot s, -1 = free
//   stamp[S]        int64   step clock of the last use of slot s (hit or admission)
//   ref[S]          int32   CLOCK reference bit
//   hand[1]         int32   CLOCK hand;  clock[1] int64 step clock (one tick per cache_step call)
//   tables[L, E, F] int64   the per-layer pointer tables the GEMMs read (row = record field addresses)
//   admit[L, E, F]  int64   admission rows consumed by the fused write-back GEMM (0 = none)
//   stats[L, 2]     int64   distinct routed experts that hit / missed, per layer
// cache_step(layer, topk_ids) runs BEFORE the layer's MoE: dedups the routed ids, stamps hits, picks a victim per miss
// with CLOCK (never a slot used in this step), evicts the victim's previous owner (its table row goes back to the host
// bank, possibly of another layer), and writes the miss's admission row (destination = the victim slot). The miss's
// table row keeps pointing at the host bank for this step: the GEMM reads it zero-copy and writes it into the slot.
// cache_commit(layer, topk_ids) runs AFTER the MoE: copies consumed admission rows into the table and clears them.
// Both are single-block kernels with static shapes: CUDA-graph capturable. Misses beyond what the pool can take in one
// step (all slots used in this step) are served zero-copy without admission.

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#define TRELLIS_CACHE_MAX_E 1024
#define TRELLIS_CACHE_MAX_F 8

__global__ __launch_bounds__(1024) void trellis_cache_step2_kernel(
    const int* __restrict__ ids, int nids, int layer, int E, int S, int F, int* __restrict__ slot_of,
    int* __restrict__ owner, int64_t* __restrict__ stamp, int* __restrict__ ref, int* __restrict__ hand,
    int64_t* __restrict__ clock, int64_t* __restrict__ tables, int64_t* __restrict__ admit,
    const int64_t* __restrict__ host_bases, const int64_t* __restrict__ cbase, int spc, int64_t rec, const int64_t* __restrict__ offs,
    int64_t* __restrict__ stats, int do_admit) {
  __shared__ int seen[TRELLIS_CACHE_MAX_E];
  __shared__ int miss_list[TRELLIS_CACHE_MAX_E];
  __shared__ int n_miss, n_hit;
  __shared__ int64_t c;
  __shared__ int64_t off_sh[TRELLIS_CACHE_MAX_F];
  const int t = threadIdx.x;
  for (int e = t; e < E; e += blockDim.x) seen[e] = 0;
  if (t < F) off_sh[t] = offs[t];
  if (t == 0) {
    n_miss = 0;
    n_hit = 0;
    c = ++clock[0];
  }
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {
    const int e = ids[s];
    if (e < 0 || e >= E) continue;  // dropped slot (sentinel E)
    if (atomicExch(&seen[e], 1) != 0) continue;
    const int g = layer * E + e;
    const int sl = slot_of[g];
    if (sl >= 0) {
      stamp[sl] = c;  // pinned for this step
      ref[sl] = 1;
      atomicAdd(&n_hit, 1);
    } else {
      miss_list[atomicAdd(&n_miss, 1)] = e;
    }
  }
  __syncthreads();
  if (t == 0) {
    stats[2 * layer + 0] += n_hit;
    stats[2 * layer + 1] += n_miss;
  }
  if (!do_admit || S == 0 || t >= 32) return;
  // CLOCK victim search by warp 0, 32 slots per step with the exact sequential semantics: lane j looks at slot
  // (h + j) % S; the first lane whose slot is free or (not used this step and ref == 0) is the victim; every slot the
  // hand passes before it (used this step: skipped; ref == 1: cleared) is treated as the sequential loop would. A
  // single-thread walk cost ~0.5 us per slot (dependent global loads) and dominated the layer at high miss rates.
  const int lane = t;
  int h = hand[0];
  for (int m = 0; m < n_miss; m++) {
    const int e = miss_list[m];
    int victim = -1;
    for (int scanned = 0; scanned < 2 * S + 32 && victim < 0; scanned += 32) {
      const int v = (h + lane) % S;
      const int o = owner[v];
      const bool pinned = o >= 0 && stamp[v] == c;
      const bool cand = o == -1 || (o >= 0 && !pinned && ref[v] == 0);
      const unsigned mask = __ballot_sync(0xffffffffu, cand);
      const int first = mask ? __ffs(mask) - 1 : 32;
      if (lane < first && o >= 0 && !pinned) ref[v] = 0;   // passed with a second chance
      __syncwarp();
      if (first < 32) {
        victim = (h + first) % S;
        h = (h + first + 1) % S;
      } else {
        h = (h + 32) % S;
      }
    }
    if (victim < 0) break;  // every slot is in use this step: remaining misses stay zero-copy only
    if (lane == 0) {
      const int o = owner[victim];
      if (o >= 0) {  // evict: the old owner goes back to its host bank row
        slot_of[o] = -1;
        const int ol = o / E, oe = o % E;
        int64_t* row = tables + (int64_t)o * F;
        const int64_t hb = host_bases[ol] + (int64_t)oe * rec;
        for (int f = 0; f < F; f++) row[f] = hb + off_sh[f];
        int64_t* orow = admit - (int64_t)layer * E * F + (int64_t)o * F;       // (admit = this layer's slice)
        for (int f = 0; f < F; f++) orow[f] = 0;
      }
      const int g = layer * E + e;
      owner[victim] = g;
      slot_of[g] = victim;
      stamp[victim] = c;
      ref[victim] = 1;
      int64_t* arow = admit + (int64_t)e * F;
      const int64_t db = cbase[victim / spc] + (int64_t)(victim % spc) * rec;
      for (int f = 0; f < F; f++) arow[f] = db + off_sh[f];
    }
    __syncwarp();
  }
  if (lane == 0) hand[0] = h;
}

__global__ __launch_bounds__(1024) void trellis_cache_commit2_kernel(const int* __restrict__ ids, int nids, int E, int F,
                                                                    int64_t* __restrict__ table,
                                                                    int64_t* __restrict__ admit) {
  __shared__ int seen[TRELLIS_CACHE_MAX_E];
  const int t = threadIdx.x;
  for (int e = t; e < E; e += blockDim.x) seen[e] = 0;
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {
    const int e = ids[s];
    if (e < 0 || e >= E) continue;
    if (atomicExch(&seen[e], 1) != 0) continue;
    int64_t* arow = admit + (int64_t)e * F;
    if (arow[0] == 0) continue;
    int64_t* trow = table + (int64_t)e * F;
    for (int f = 0; f < F; f++) {
      trow[f] = arow[f];
      arow[f] = 0;
    }
  }
}

static void check_i32(const at::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kInt && t.is_contiguous() && (n < 0 || t.numel() == n), name, ": int32 CUDA");
}
static void check_i64(const at::Tensor& t, int64_t n, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.dtype() == at::kLong && t.is_contiguous() && (n < 0 || t.numel() == n), name, ": int64 CUDA");
}

// tables / admit: int64 [L, E, F]; host_bases int64 [L]; offs int64 [F]; stats int64 [L, 2]
void moe_cache_step2(const at::Tensor& ids, int64_t layer, at::Tensor& slot_of, at::Tensor& owner, at::Tensor& stamp,
                    at::Tensor& ref, at::Tensor& hand, at::Tensor& clock, at::Tensor& tables, at::Tensor& admit,
                    const at::Tensor& host_bases, const at::Tensor& cbase, int64_t spc, int64_t rec, const at::Tensor& offs,
                    at::Tensor& stats, bool do_admit) {
  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  TORCH_CHECK(tables.dim() == 3 && admit.sizes() == tables.sizes());
  const int64_t L = tables.size(0), E = tables.size(1), F = tables.size(2), S = owner.numel();
  TORCH_CHECK(E <= TRELLIS_CACHE_MAX_E && F <= TRELLIS_CACHE_MAX_F && layer >= 0 && layer < L);
  check_i32(ids, -1, "ids"); check_i32(slot_of, L * E, "slot_of"); check_i32(owner, S, "owner");
  check_i64(stamp, S, "stamp"); check_i32(ref, S, "ref"); check_i32(hand, 1, "hand"); check_i64(clock, 1, "clock");
  check_i64(tables, -1, "tables"); check_i64(admit, -1, "admit"); check_i64(host_bases, L, "host_bases");
  check_i64(offs, F, "offs"); check_i64(stats, 2 * L, "stats");
  trellis_cache_step2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int>(), (int)ids.numel(), (int)layer, (int)E, (int)S, (int)F, slot_of.data_ptr<int>(),
      owner.data_ptr<int>(), stamp.data_ptr<int64_t>(), ref.data_ptr<int>(), hand.data_ptr<int>(),
      clock.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), admit[layer].data_ptr<int64_t>(),
      host_bases.data_ptr<int64_t>(), cbase.data_ptr<int64_t>(), (int)spc, rec, offs.data_ptr<int64_t>(), stats.data_ptr<int64_t>(),
      do_admit ? 1 : 0);
}

void moe_cache_commit2(const at::Tensor& ids, int64_t layer, at::Tensor& tables, at::Tensor& admit) {
  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  TORCH_CHECK(tables.dim() == 3 && admit.sizes() == tables.sizes());
  const int64_t E = tables.size(1), F = tables.size(2);
  check_i32(ids, -1, "ids"); check_i64(tables, -1, "tables"); check_i64(admit, -1, "admit");
  TORCH_CHECK(E <= TRELLIS_CACHE_MAX_E && layer >= 0 && layer < tables.size(0));
  trellis_cache_commit2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int>(), (int)ids.numel(), (int)E, (int)F, tables[layer].data_ptr<int64_t>(),
      admit[layer].data_ptr<int64_t>());
}

// ---------------------------------------------------------------------------------------------------------------
// K07: fused decode prologue (T * top_k <= 1024 slots, block 8): ONE launch per layer replaces
//   ids clean (negative / >= E -> E), moe_align_block_size (SGLang: 2 kernels), cache_step and cache_commit.
// Order inside: (1) deferred commit of THIS layer's admission rows from the previous step (rows of experts evicted
// since were cleared by the eviction, so a stale row can never land), (2) dedup + hits (stamp/ref) + per-expert counts,
// (3) align: experts in ascending id (as SGLang's align -> same moe blocks, same GEMM partition), padding = numel,
// (4) warp 0: CLOCK victims for the misses, evictions (table row -> host, pending admission row cleared), new
// admission rows (consumed by the GEMM write-back; committed by this layer's NEXT prologue).
#include <cub/cub.cuh>

__global__ __launch_bounds__(1024) void trellis_cache_decode_prologue2_kernel(
    const int* __restrict__ ids_in, int nids, int layer, int E, int S, int F, int* __restrict__ slot_of,
    int* __restrict__ owner, int64_t* __restrict__ stamp, int* __restrict__ ref, int* __restrict__ hand,
    int64_t* __restrict__ clock, int64_t* __restrict__ tables, int64_t* __restrict__ admit_all,
    const int64_t* __restrict__ host_bases, const int64_t* __restrict__ cbase, int spc, int64_t rec, const int64_t* __restrict__ offs,
    int64_t* __restrict__ stats, int do_admit, int* __restrict__ ids_out, int* __restrict__ sorted, int sorted_cap,
    int* __restrict__ eids, int eids_cap, int* __restrict__ post) {
  __shared__ int seen[TRELLIS_CACHE_MAX_E];
  __shared__ int cnt[TRELLIS_CACHE_MAX_E];
  __shared__ int start[TRELLIS_CACHE_MAX_E];
  __shared__ int miss_list[TRELLIS_CACHE_MAX_E];
  __shared__ int n_miss, n_hit, total_sh;
  __shared__ int64_t c;
  __shared__ int64_t off_sh[TRELLIS_CACHE_MAX_F];
  using Scan = cub::BlockScan<int, 1024>;
  __shared__ typename Scan::TempStorage scan_tmp;
  constexpr int kBlock = 8;
  const int t = threadIdx.x;
  int64_t* table = tables + (int64_t)layer * E * F;
  int64_t* admit = admit_all + (int64_t)layer * E * F;
  for (int e = t; e < E; e += blockDim.x) {
    seen[e] = 0;
    cnt[e] = 0;
    int64_t* arow = admit + (int64_t)e * F;           // (1) deferred commit
    if (arow[0] != 0) {
      int64_t* trow = table + (int64_t)e * F;
      for (int f = 0; f < F; f++) { trow[f] = arow[f]; arow[f] = 0; }
    }
  }
  if (t < F) off_sh[t] = offs[t];
  if (t == 0) { n_miss = 0; n_hit = 0; c = ++clock[0]; }
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {        // (2)
    int e = ids_in[s];
    if (e < 0 || e >= E) e = E;
    ids_out[s] = e;
    if (e == E) continue;
    atomicAdd(&cnt[e], 1);
    if (atomicExch(&seen[e], 1) != 0) continue;
    const int g = layer * E + e;
    const int sl = slot_of[g];
    if (sl >= 0) { stamp[sl] = c; ref[sl] = 1; atomicAdd(&n_hit, 1); }
    else miss_list[atomicAdd(&n_miss, 1)] = e;
  }
  __syncthreads();
  // (3) align: padded counts, exclusive scan in ascending expert order
  const int padded = (t < E) ? ((cnt[t] + kBlock - 1) / kBlock) * kBlock : 0;
  int excl, total;
  Scan(scan_tmp).ExclusiveSum(padded, excl, total);
  if (t < E) start[t] = excl;
  if (t == 0) { total_sh = total; post[0] = total; }
  __syncthreads();
  for (int i = t; i < sorted_cap; i += blockDim.x) sorted[i] = nids;          // padding = numel
  for (int i = t; i < eids_cap; i += blockDim.x) eids[i] = -1;
  __syncthreads();
  if (t < E && cnt[t] > 0)
    for (int i = start[t]; i < start[t] + padded; i += kBlock) eids[i / kBlock] = t;
  if (t < E) cnt[t] = 0;                                                      // reuse as fill counters
  __syncthreads();
  for (int s = t; s < nids; s += blockDim.x) {
    const int e = ids_out[s];
    if (e < E) sorted[start[e] + atomicAdd(&cnt[e], 1)] = s;
  }
  if (t == 0) { stats[2 * layer + 0] += n_hit; stats[2 * layer + 1] += n_miss; }
  if (!do_admit || S == 0 || t >= 32) return;
  // (4) warp 0: CLOCK victims (same policy as trellis_cache_step2_kernel)
  const int lane = t;
  int h = hand[0];
  for (int m = 0; m < n_miss; m++) {
    const int e = miss_list[m];
    int victim = -1;
    for (int scanned = 0; scanned < 2 * S + 32 && victim < 0; scanned += 32) {
      const int v = (h + lane) % S;
      const int o = owner[v];
      const bool pinned = o >= 0 && stamp[v] == c;
      const bool cand = o == -1 || (o >= 0 && !pinned && ref[v] == 0);
      const unsigned mask = __ballot_sync(0xffffffffu, cand);
      const int first = mask ? __ffs(mask) - 1 : 32;
      if (lane < first && o >= 0 && !pinned) ref[v] = 0;
      __syncwarp();
      if (first < 32) { victim = (h + first) % S; h = (h + first + 1) % S; }
      else h = (h + 32) % S;
    }
    if (victim < 0) break;
    if (lane == 0) {
      const int o = owner[victim];
      if (o >= 0) {
        slot_of[o] = -1;
        const int ol = o / E, oe = o % E;
        int64_t* row = tables + (int64_t)o * F;
        const int64_t hb = host_bases[ol] + (int64_t)oe * rec;
        for (int f = 0; f < F; f++) row[f] = hb + off_sh[f];
        int64_t* orow = admit_all + (int64_t)o * F;                            // pending admission of the evicted
        for (int f = 0; f < F; f++) orow[f] = 0;
      }
      const int g = layer * E + e;
      owner[victim] = g;
      slot_of[g] = victim;
      stamp[victim] = c;
      ref[victim] = 1;
      int64_t* arow = admit + (int64_t)e * F;
      const int64_t db = cbase[victim / spc] + (int64_t)(victim % spc) * rec;
      for (int f = 0; f < F; f++) arow[f] = db + off_sh[f];
    }
    __syncwarp();
  }
  if (lane == 0) hand[0] = h;
}

void moe_cache_decode_prologue2(const at::Tensor& ids, int64_t layer, at::Tensor& slot_of, at::Tensor& owner,
                               at::Tensor& stamp, at::Tensor& ref, at::Tensor& hand, at::Tensor& clock,
                               at::Tensor& tables, at::Tensor& admit, const at::Tensor& host_bases, const at::Tensor& cbase, int64_t spc,
                               int64_t rec, const at::Tensor& offs, at::Tensor& stats, bool do_admit,
                               at::Tensor& ids_out, at::Tensor& sorted, at::Tensor& eids, at::Tensor& post) {
  const at::cuda::OptionalCUDAGuard device_guard(ids.device());
  TORCH_CHECK(tables.dim() == 3 && admit.sizes() == tables.sizes());
  const int64_t L = tables.size(0), E = tables.size(1), F = tables.size(2), S = owner.numel();
  TORCH_CHECK(E <= TRELLIS_CACHE_MAX_E && E <= 1024 && F <= TRELLIS_CACHE_MAX_F && layer >= 0 && layer < L);
  check_i32(ids, -1, "ids"); check_i32(ids_out, ids.numel(), "ids_out"); check_i32(sorted, -1, "sorted");
  check_i32(eids, -1, "eids"); check_i32(post, 1, "post");
  check_i32(slot_of, L * E, "slot_of"); check_i32(owner, S, "owner");
  check_i64(stamp, S, "stamp"); check_i32(ref, S, "ref"); check_i32(hand, 1, "hand"); check_i64(clock, 1, "clock");
  check_i64(host_bases, L, "host_bases"); check_i64(offs, F, "offs"); check_i64(stats, 2 * L, "stats");
  TORCH_CHECK(ids.numel() <= 1024, "decode prologue: at most 1024 slots");
  TORCH_CHECK(sorted.numel() >= ids.numel() * 8 && eids.numel() * 8 >= sorted.numel(), "prologue buffers too small");
  trellis_cache_decode_prologue2_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
      ids.data_ptr<int>(), (int)ids.numel(), (int)layer, (int)E, (int)S, (int)F, slot_of.data_ptr<int>(),
      owner.data_ptr<int>(), stamp.data_ptr<int64_t>(), ref.data_ptr<int>(), hand.data_ptr<int>(),
      clock.data_ptr<int64_t>(), tables.data_ptr<int64_t>(), admit.data_ptr<int64_t>(), host_bases.data_ptr<int64_t>(),
      cbase.data_ptr<int64_t>(), (int)spc, rec, offs.data_ptr<int64_t>(), stats.data_ptr<int64_t>(), do_admit ? 1 : 0, ids_out.data_ptr<int>(),
      sorted.data_ptr<int>(), (int)sorted.numel(), eids.data_ptr<int>(), (int)eids.numel(), post.data_ptr<int>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_cache_step2", &moe_cache_step2);
  m.def("moe_cache_commit2", &moe_cache_commit2);
  m.def("moe_cache_decode_prologue2", &moe_cache_decode_prologue2);
}
