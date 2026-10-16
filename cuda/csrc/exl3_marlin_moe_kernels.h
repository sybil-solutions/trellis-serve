// MoE grouped GEMM: kernel declaration, parameter list and instance table shared by the host code
// (exl3_marlin_moe.cu) and the generated instantiation units (setup.py writes one build/gen/moe_inst_mb<M>_cb<C>.cu
// per moe-block family and codebook). Same scheme as exl3_marlin_kernels.h / vLLM's Marlin MoE generate_kernels.py.
// Own namespace, so the dense kernels (trellis_exl3_marlin) and these never collide.
#pragma once

#ifndef MARLIN_NAMESPACE_NAME
#define MARLIN_NAMESPACE_NAME trellis_exl3_marlin_moe
#endif

#include "marlin.cuh"
#include "marlin_dtypes.cuh"
#include "core/scalar_type.hpp"

// Parameter order of vLLM's marlin_moe_wna16 kernel + the EXL3 tail (shard map, in-launch output transform).
#define TRELLIS_MOE_KERNEL_PARAMS                                                          \
  const int4 *__restrict__ A, const int4 *__restrict__ B, int4 *__restrict__ C,           \
      int4 *__restrict__ C_tmp, const int4 *__restrict__ b_bias_ptr,                      \
      const float *__restrict__ a_scales_ptr, const int4 *__restrict__ scales_ptr,        \
      const float *__restrict__ global_scale_ptr, const int4 *__restrict__ zp_ptr,        \
      const int *__restrict__ g_idx, const int32_t *__restrict__ sorted_token_ids_ptr,    \
      const int32_t *__restrict__ expert_ids_ptr,                                         \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,                             \
      const float *__restrict__ topk_weights_ptr, int top_k, bool mul_topk_weights,       \
      int num_groups, int prob_m, int prob_n, int prob_k, int *locks, bool has_bias,      \
      bool use_atomic_add, bool use_fp32_reduce, int a_shard_stride, int shard_end0,      \
      int shard_end1, int shard_end2, int out_flags,                                     \
      const int64_t *__restrict__ b_ptrs, const int64_t *__restrict__ s_ptrs, int ptr_stride, int ptr_rows, \
      const int64_t *__restrict__ wb_ptrs

namespace MARLIN_NAMESPACE_NAME {

#ifndef TRELLIS_MOE_KERNEL_DEFINED
template <const vllm::ScalarTypeId a_type_id, const vllm::ScalarTypeId b_type_id,
          const vllm::ScalarTypeId c_type_id, const vllm::ScalarTypeId s_type_id, const int threads,
          const int thread_m_blocks, const int thread_n_blocks, const int thread_k_blocks,
          const bool m_block_size_8, const int stages, const int group_blocks, const bool is_zp_float,
          const int exl3_cb>
__global__ void Marlin(TRELLIS_MOE_KERNEL_PARAMS);
#endif

using MoeFuncPtr = void (*)(TRELLIS_MOE_KERNEL_PARAMS);

static constexpr int kMoeStages = 4;

// EXL3 K = 3 experts: a 3-bit "weight type" that only selects the byte-exact 24-word tile staging in the template
// (vLLM's ScalarType has no 3-bit constant; the id is what the kernel template switches on).
static constexpr vllm::ScalarTypeId kExl3K3Id = vllm::ScalarType::uint(3, 0).id();
// EXL3 K = 2 experts: byte-exact 16-word tile staging (same scheme as K = 3).
static constexpr vllm::ScalarTypeId kExl3K2Id = vllm::ScalarType::uint(2, 0).id();

// Kernel type for (threads, thread_n, thread_k, mb, cb, bits); mb = moe block family: 0 = 8 rows per moe block
// (transposed MMA), m = 16m rows per moe block (1..4); bits = EXL3 K of the experts (2, 3: byte-exact tile staging,
// 4: Marlin 4-bit staging of the lane-major repack).
#define TRELLIS_MOE_KERNEL(threads, tn, tk, mb, cb, bits)                                                 \
  Marlin<vllm::kFloat16.id(), (bits == 3 ? kExl3K3Id : bits == 2 ? kExl3K2Id : vllm::kU4B8.id()), vllm::kFloat16.id(),          \
         vllm::kFloat16.id(), threads, (mb == 0 ? 1 : mb), tn / 16, tk / 16, (mb == 0), kMoeStages, -1,  \
         false, cb>

// X(threads, thread_n, thread_k): Marlin's thread configurations
#define TRELLIS_MOE_THREAD_CFGS(X, mb, cb, bits) \
  X(256, 128, 128, mb, cb, bits)                \
  X(256, 256, 64, mb, cb, bits)                 \
  X(128, 128, 64, mb, cb, bits)                 \
  X(128, 64, 128, mb, cb, bits)

}  // namespace MARLIN_NAMESPACE_NAME
