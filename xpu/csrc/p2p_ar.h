#pragma once
// Graph-capturable 2-GPU P2P AllReduce & AllGather over Level Zero IPC (Intel Xe2 / Arc Pro B70).
//
// Push-based 1-WG / 4-WG design:
//   1. Push input directly to PEER GDDR6 via 4x-unrolled 128-byte posted PCIe stores (L1UC_L3UC).
//   2. Single GPU-scope fence + WG barrier (+ arrival counter if n_wgs > 1) -> post peer_flag[slot].
//      (PCIe posted writes from the same device to the same peer are strictly ordered!)
//   3. Spin on local_flag[slot] >= next_seq (local memory only, zero PCIe polling traffic!).
//   4. Read received peer data from LOCAL GDDR6 (539 GB/s!) + local input -> store output.

#include <cstring>
#include <vector>
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <torch/all.h>
#include <c10/xpu/XPUStream.h>

namespace exl3p2p {

using namespace sycl::ext::intel::esimd;
using fp16 = sycl::half;

constexpr auto kUC = properties{cache_hint_L1<cache_hint::uncached>, cache_hint_L2<cache_hint::uncached>};
constexpr int kWgSize = 32;

struct State {
    bool initialized = false;
    int rank = 0;
    int64_t slot_bytes = 0;
    ze_context_handle_t l0_ctx = nullptr;
    void* local_base = nullptr;
    void* peer_base = nullptr;
};

static State& state_of(int dev_idx) {
    static State states[16];
    TORCH_CHECK(dev_idx >= 0 && dev_idx < 16, "p2p_ar: invalid dev_idx");
    return states[dev_idx];
}

struct PushAllReduceFp16 {
    const fp16* inp;
    fp16* out;
    const uint8_t* local_base;
    uint8_t* peer_base;
    uint32_t* local_ctrl;
    uint32_t* peer_ctrl;
    int64_t slot_bytes;
    int n_chunks;  // 64-fp16 (128-byte) chunks
    int n_32;      // total 32-fp16 (64-byte) chunks
    int n_wgs;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int lid = it.get_local_id(0);
        int gid = it.get_global_id(0);
        int gstride = it.get_global_range(0);

        uint32_t next_seq = block_load<uint32_t, 1>(local_ctrl + 32, kUC)[0] + 1u;
        uint32_t slot = next_seq & 1u;
        const fp16* lbuf = reinterpret_cast<const fp16*>(local_base + (size_t)slot * slot_bytes);
        fp16* pbuf = reinterpret_cast<fp16*>(peer_base + (size_t)slot * slot_bytes);

        // 1. Push input directly into PEER's GDDR6 slot (4x unrolled 128-byte posted PCIe writes)
        int c = gid;
        for (; c + 3 * gstride < n_chunks; c += 4 * gstride) {
            int i0 = c * 64, i1 = (c + gstride) * 64, i2 = (c + 2 * gstride) * 64, i3 = (c + 3 * gstride) * 64;
            simd<fp16, 64> v0 = block_load<fp16, 64>(inp + i0);
            simd<fp16, 64> v1 = block_load<fp16, 64>(inp + i1);
            simd<fp16, 64> v2 = block_load<fp16, 64>(inp + i2);
            simd<fp16, 64> v3 = block_load<fp16, 64>(inp + i3);
            block_store<fp16, 64>(pbuf + i0, v0, kUC);
            block_store<fp16, 64>(pbuf + i1, v1, kUC);
            block_store<fp16, 64>(pbuf + i2, v2, kUC);
            block_store<fp16, 64>(pbuf + i3, v3, kUC);
        }
        for (; c < n_chunks; c += gstride) {
            block_store<fp16, 64>(pbuf + c * 64, block_load<fp16, 64>(inp + c * 64), kUC);
        }
        if ((n_32 & 1) && gid == 0) {
            int tail = n_chunks * 64;
            block_store<fp16, 32>(pbuf + tail, block_load<fp16, 32>(inp + tail), kUC);
        }
        fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
        barrier();

        // 2. Signal peer_flag[slot] (posted PCIe write ordered behind pbuf writes!) and spin locally
        if (lid == 0) {
            uint32_t* pflag = peer_ctrl + slot * 16;
            simd<uint32_t, 1> flag_off(slot * 64u);
            if (n_wgs == 1) {
                block_store<uint32_t, 1>(local_ctrl + 32, simd<uint32_t, 1>(next_seq), kUC);
                block_store<uint32_t, 1>(pflag, simd<uint32_t, 1>(next_seq), kUC);
            } else {
                simd<uint32_t, 1> arr_off(192u + slot * 64u);
                simd<uint32_t, 1> old = atomic_update<atomic_op::inc, uint32_t, 1>(local_ctrl, arr_off);
                if (old[0] == (uint32_t)(n_wgs - 1)) {
                    atomic_update<atomic_op::store, uint32_t, 1>(local_ctrl, arr_off, simd<uint32_t, 1>(0u));
                    block_store<uint32_t, 1>(local_ctrl + 32, simd<uint32_t, 1>(next_seq), kUC);
                    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
                    block_store<uint32_t, 1>(pflag, simd<uint32_t, 1>(next_seq), kUC);
                }
            }
            while (true) {
                simd<uint32_t, 1> fv = atomic_update<atomic_op::load, uint32_t, 1>(local_ctrl, flag_off);
                if (fv[0] >= next_seq) break;
            }
        }
        barrier();

        // 3. Read peer's data from LOCAL GDDR6 (539 GB/s!) + local input -> output
        c = gid;
        for (; c + 3 * gstride < n_chunks; c += 4 * gstride) {
            int i0 = c * 64, i1 = (c + gstride) * 64, i2 = (c + 2 * gstride) * 64, i3 = (c + 3 * gstride) * 64;
            simd<fp16, 64> b0 = block_load<fp16, 64>(lbuf + i0, kUC);
            simd<fp16, 64> b1 = block_load<fp16, 64>(lbuf + i1, kUC);
            simd<fp16, 64> b2 = block_load<fp16, 64>(lbuf + i2, kUC);
            simd<fp16, 64> b3 = block_load<fp16, 64>(lbuf + i3, kUC);
            simd<fp16, 64> a0 = block_load<fp16, 64>(inp + i0);
            simd<fp16, 64> a1 = block_load<fp16, 64>(inp + i1);
            simd<fp16, 64> a2 = block_load<fp16, 64>(inp + i2);
            simd<fp16, 64> a3 = block_load<fp16, 64>(inp + i3);
            block_store<fp16, 64>(out + i0, convert<fp16>(convert<float>(a0) + convert<float>(b0)));
            block_store<fp16, 64>(out + i1, convert<fp16>(convert<float>(a1) + convert<float>(b1)));
            block_store<fp16, 64>(out + i2, convert<fp16>(convert<float>(a2) + convert<float>(b2)));
            block_store<fp16, 64>(out + i3, convert<fp16>(convert<float>(a3) + convert<float>(b3)));
        }
        for (; c < n_chunks; c += gstride) {
            int i0 = c * 64;
            simd<fp16, 64> b0 = block_load<fp16, 64>(lbuf + i0, kUC);
            simd<fp16, 64> a0 = block_load<fp16, 64>(inp + i0);
            block_store<fp16, 64>(out + i0, convert<fp16>(convert<float>(a0) + convert<float>(b0)));
        }
        if ((n_32 & 1) && gid == 0) {
            int tail = n_chunks * 64;
            simd<fp16, 32> b0 = block_load<fp16, 32>(lbuf + tail, kUC);
            simd<fp16, 32> a0 = block_load<fp16, 32>(inp + tail);
            block_store<fp16, 32>(out + tail, convert<fp16>(convert<float>(a0) + convert<float>(b0)));
        }
    }
};

struct PushAllGatherU32 {
    const uint32_t* inp;
    uint32_t* out;
    const uint8_t* local_base;
    uint8_t* peer_base;
    uint32_t* local_ctrl;
    uint32_t* peer_ctrl;
    int64_t slot_bytes;
    int rank;
    int M;
    int row_u32;
    int n_wgs;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int lid = it.get_local_id(0);
        int gid = it.get_global_id(0);
        int gstride = it.get_global_range(0);
        int total_u32 = M * row_u32;

        uint32_t next_seq = block_load<uint32_t, 1>(local_ctrl + 32, kUC)[0] + 1u;
        uint32_t slot = next_seq & 1u;
        const uint32_t* lbuf = reinterpret_cast<const uint32_t*>(local_base + (size_t)slot * slot_bytes);
        uint32_t* pbuf = reinterpret_cast<uint32_t*>(peer_base + (size_t)slot * slot_bytes);

        if ((total_u32 & 31) == 0) {
            int nc = total_u32 >> 5;
            for (int c = gid; c < nc; c += gstride) {
                block_store<uint32_t, 32>(pbuf + c * 32, block_load<uint32_t, 32>(inp + c * 32), kUC);
            }
        } else {
            for (int i = gid; i < total_u32; i += gstride) {
                block_store<uint32_t, 1>(pbuf + i, simd<uint32_t, 1>(inp[i]), kUC);
            }
        }
        fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
        barrier();

        if (lid == 0) {
            uint32_t* pflag = peer_ctrl + slot * 16;
            simd<uint32_t, 1> flag_off(slot * 64u);
            if (n_wgs == 1) {
                block_store<uint32_t, 1>(local_ctrl + 32, simd<uint32_t, 1>(next_seq), kUC);
                block_store<uint32_t, 1>(pflag, simd<uint32_t, 1>(next_seq), kUC);
            } else {
                simd<uint32_t, 1> arr_off(192u + slot * 64u);
                simd<uint32_t, 1> old = atomic_update<atomic_op::inc, uint32_t, 1>(local_ctrl, arr_off);
                if (old[0] == (uint32_t)(n_wgs - 1)) {
                    atomic_update<atomic_op::store, uint32_t, 1>(local_ctrl, arr_off, simd<uint32_t, 1>(0u));
                    block_store<uint32_t, 1>(local_ctrl + 32, simd<uint32_t, 1>(next_seq), kUC);
                    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpu>();
                    block_store<uint32_t, 1>(pflag, simd<uint32_t, 1>(next_seq), kUC);
                }
            }
            while (true) {
                simd<uint32_t, 1> fv = atomic_update<atomic_op::load, uint32_t, 1>(local_ctrl, flag_off);
                if (fv[0] >= next_seq) break;
            }
        }
        barrier();

        int my_off = rank * row_u32;
        int peer_off = (1 - rank) * row_u32;
        int out_stride = 2 * row_u32;
        if ((row_u32 & 31) == 0) {
            int chunks_per_row = row_u32 >> 5;
            int total_chunks = M * chunks_per_row;
            for (int idx = gid; idx < total_chunks; idx += gstride) {
                int m = idx / chunks_per_row;
                int c = (idx % chunks_per_row) * 32;
                simd<uint32_t, 32> pv = block_load<uint32_t, 32>(lbuf + m * row_u32 + c, kUC);
                simd<uint32_t, 32> lv = block_load<uint32_t, 32>(inp + m * row_u32 + c);
                block_store<uint32_t, 32>(out + m * out_stride + my_off + c, lv);
                block_store<uint32_t, 32>(out + m * out_stride + peer_off + c, pv);
            }
        } else {
            for (int idx = gid; idx < total_u32; idx += gstride) {
                int m = idx / row_u32;
                int j = idx % row_u32;
                out[m * out_stride + my_off + j] = inp[idx];
                out[m * out_stride + peer_off + j] = block_load<uint32_t, 1>(lbuf + idx, kUC)[0];
            }
        }
    }
};

struct PeerWarmupKernel {
    const uint32_t* peer_base;
    uint32_t* sink;
    int n_pages;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int gid = it.get_global_id(0);
        if (gid < n_pages) {
            simd<uint32_t, 1> v = block_load<uint32_t, 1>(peer_base + (size_t)gid * 1024, kUC);
            if (gid == 0) block_store<uint32_t, 1>(sink, v, kUC);
        }
    }
};

static std::vector<int64_t> p2p_init_local(int64_t dev_idx, int64_t slot_bytes) {
    TORCH_CHECK(slot_bytes > 0 && (slot_bytes % 4096) == 0, "p2p_init_local: slot_bytes must be 4K-aligned");
    auto& q = c10::xpu::getCurrentXPUStream(dev_idx).queue();
    auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto l0_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    auto& st = state_of((int)dev_idx);
    if (st.local_base) {
        zeMemFree(st.l0_ctx, st.local_base);
        st.local_base = nullptr;
    }
    size_t total = (size_t)slot_bytes * 2 + 4096;
    ze_device_mem_alloc_desc_t desc = {};
    desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    desc.flags = ZE_DEVICE_MEM_ALLOC_FLAG_BIAS_UNCACHED;
    void* ptr = nullptr;
    ze_result_t rc = zeMemAllocDevice(l0_ctx, &desc, total, 4096, l0_dev, &ptr);
    TORCH_CHECK(rc == ZE_RESULT_SUCCESS && ptr != nullptr, "zeMemAllocDevice failed: ", (int)rc);
    q.memset(ptr, 0, total).wait();

    ze_ipc_mem_handle_t handle = {};
    rc = zeMemGetIpcHandle(l0_ctx, ptr, &handle);
    TORCH_CHECK(rc == ZE_RESULT_SUCCESS, "zeMemGetIpcHandle failed: ", (int)rc);

    st.l0_ctx = l0_ctx;
    st.local_base = ptr;
    st.slot_bytes = slot_bytes;
    st.initialized = false;

    std::vector<int64_t> out(ZE_MAX_IPC_HANDLE_SIZE);
    for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE; ++i) {
        out[i] = (int64_t)(uint8_t)handle.data[i];
    }
    return out;
}

static bool p2p_open_peer(int64_t dev_idx, int64_t rank, std::vector<int64_t> peer_handle_bytes, int64_t peer_fd) {
    TORCH_CHECK(peer_handle_bytes.size() == ZE_MAX_IPC_HANDLE_SIZE, "bad IPC handle size");
    auto& q = c10::xpu::getCurrentXPUStream(dev_idx).queue();
    auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto l0_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    auto& st = state_of((int)dev_idx);
    TORCH_CHECK(st.local_base != nullptr, "call p2p_init_local first");

    ze_ipc_mem_handle_t h = {};
    for (size_t i = 0; i < ZE_MAX_IPC_HANDLE_SIZE; ++i) {
        h.data[i] = (char)(uint8_t)peer_handle_bytes[i];
    }
    if (peer_fd >= 0) {
        int32_t fd32 = (int32_t)peer_fd;
        std::memcpy(h.data, &fd32, sizeof(fd32));
    }
    void* peer_ptr = nullptr;
    ze_result_t rc = zeMemOpenIpcHandle(l0_ctx, l0_dev, h, ZE_IPC_MEMORY_FLAG_BIAS_UNCACHED, &peer_ptr);
    if (rc != ZE_RESULT_SUCCESS || peer_ptr == nullptr) {
        return false;
    }
    st.peer_base = peer_ptr;
    st.rank = (int)rank;
    st.initialized = true;

    int n_pages = (int)(((size_t)st.slot_bytes * 2 + 4096) / 4096);
    int global = ((n_pages + 31) / 32) * 32;
    uint32_t* sink = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(st.local_base) + 2 * st.slot_bytes + 320);
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(global, 32),
                         PeerWarmupKernel{reinterpret_cast<const uint32_t*>(st.peer_base), sink, n_pages});
    }).wait();
    return true;
}

static bool p2p_is_ready(int64_t dev_idx) {
    return state_of((int)dev_idx).initialized;
}

static at::Tensor p2p_all_reduce(at::Tensor input) {
    int dev_idx = input.device().index();
    auto& st = state_of(dev_idx);
    TORCH_CHECK(st.initialized, "p2p_all_reduce: not initialized");
    auto x = input.is_contiguous() ? input : input.contiguous();
    int64_t n = x.numel();
    int64_t n_bytes = n * x.element_size();
    TORCH_CHECK(x.scalar_type() == at::kHalf && (n % 32) == 0 && n_bytes <= st.slot_bytes,
                "p2p_all_reduce: requires fp16, numel%32==0, bytes<=slot_bytes");
    auto out = at::empty_like(x);
    if (n == 0) return out;

    auto& q = c10::xpu::getCurrentXPUStream(dev_idx).queue();
    uint8_t* lbase = reinterpret_cast<uint8_t*>(st.local_base);
    uint8_t* pbase = reinterpret_cast<uint8_t*>(st.peer_base);
    uint32_t* lctrl = reinterpret_cast<uint32_t*>(lbase + 2 * st.slot_bytes);
    uint32_t* pctrl = reinterpret_cast<uint32_t*>(pbase + 2 * st.slot_bytes);
    int n_32 = (int)(n / 32);
    int n_chunks = n_32 / 2;
    int n_wgs = (n_bytes <= 163840) ? 1 : 4;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * kWgSize, kWgSize),
                         PushAllReduceFp16{
                             reinterpret_cast<const fp16*>(x.data_ptr()),
                             reinterpret_cast<fp16*>(out.data_ptr()),
                             lbase, pbase, lctrl, pctrl, st.slot_bytes, n_chunks, n_32, n_wgs});
    });
    return out;
}

static at::Tensor p2p_all_gather_last(at::Tensor input) {
    int dev_idx = input.device().index();
    auto& st = state_of(dev_idx);
    TORCH_CHECK(st.initialized, "p2p_all_gather_last: not initialized");
    auto x = input.is_contiguous() ? input : input.contiguous();
    auto shape = x.sizes().vec();
    int64_t D = shape.back();
    int64_t M = x.numel() / D;
    int64_t row_bytes = D * x.element_size();
    int64_t n_bytes = M * row_bytes;
    TORCH_CHECK((row_bytes % 4) == 0 && n_bytes <= st.slot_bytes,
                "p2p_all_gather_last: row_bytes%4==0 and bytes<=slot_bytes required");
    shape.back() = D * 2;
    auto out = at::empty(shape, x.options());
    if (x.numel() == 0) return out;

    auto& q = c10::xpu::getCurrentXPUStream(dev_idx).queue();
    uint8_t* lbase = reinterpret_cast<uint8_t*>(st.local_base);
    uint8_t* pbase = reinterpret_cast<uint8_t*>(st.peer_base);
    uint32_t* lctrl = reinterpret_cast<uint32_t*>(lbase + 2 * st.slot_bytes);
    uint32_t* pctrl = reinterpret_cast<uint32_t*>(pbase + 2 * st.slot_bytes);
    int row_u32 = (int)(row_bytes / 4);
    int n_wgs = (n_bytes <= 163840 || (row_u32 & 31) != 0) ? 1 : 4;

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range<1>(n_wgs * kWgSize, kWgSize),
                         PushAllGatherU32{
                             reinterpret_cast<const uint32_t*>(x.data_ptr()),
                             reinterpret_cast<uint32_t*>(out.data_ptr()),
                             lbase, pbase, lctrl, pctrl, st.slot_bytes, st.rank, (int)M, row_u32, n_wgs});
    });
    return out;
}

}  // namespace exl3p2p
