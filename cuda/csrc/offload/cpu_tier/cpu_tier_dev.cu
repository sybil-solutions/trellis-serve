// trellis-serve CPU expert tier, GPU side (ported from FreeToken-EXL3 ft_tier_cu.cu). Two kernels per offloaded MoE
// layer in decode, both on the compute stream, no host sync, static shapes (CUDA-graph capturable):
//  ct_split   (before the cache prologue): among the layer's GPU-cache misses pick the k for the CPU that minimise
//             max(GPU time, CPU time) under a linear cost model, publish the job to pinned host memory (x rows as
//             fp32, picks, then the seq flag) and hand the GPU path the routing with those picks dropped (sentinel E:
//             the prologue neither counts nor admits them, the GEMMs skip them, moe_combine ignores them)
//  ct_combine (after the GPU experts): wait for the worker's done flag, add its fp32 partial into the MoE output
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// policy p[]: 0 t_zc (us per zero-copy miss), 1 t_hit (us per GPU-resident expert), 2 cpu_a (us fixed per CPU job incl.
// handoff), 3 cpu_b (us per CPU expert), 4 cpu_tok (extra fraction per extra token of an expert), 5 max_cpu,
// 6 force_n (-1 = cost model), 7 publish empty jobs (1) or skip them (0)
// stats[]: 0 hits, 1 misses, 2 cpu picks, 3 calls, 4 wait ns, 5 waits, 6 timeouts, 7 jobs published
template <typename T>
__global__ void ct_split_k(const int* __restrict__ ids, const float* __restrict__ w, const T* __restrict__ x, int n, int bsz,
    int topk, int H, int E, const int* __restrict__ slotof, float* __restrict__ freq, const float* __restrict__ p,
    volatile long long* ctrl, float* hx, int* hpicks, long long* dseq, int li, int* ids_out, long long* dflag,
    long long* stats)
{
    __shared__ int cnt[1024];
    __shared__ unsigned char cpu[1024];
    __shared__ int s_miss[1024];
    __shared__ int npk, publish;
    __shared__ long long seq;
    const int t = threadIdx.x;
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; cpu[e] = 0; }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const int e = ids[s];
        if (e >= 0 && e < E && w[s] != 0.0f) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (t == 0)
    {
        // misses, coldest first (running routing frequency of this layer); decode has <= topk x bsz of them
        int nm = 0, nh = 0;
        int* miss = s_miss;
        for (int e = 0; e < E; ++e)
        {
            if (!cnt[e]) continue;
            freq[e] += 1.0f;
            if (slotof[e] >= 0) { ++nh; continue; }
            int b = nm++;
            while (b > 0 && freq[miss[b - 1]] > freq[e]) { miss[b] = miss[b - 1]; --b; }
            miss[b] = e;
        }
        int best = 0;
        const int force = (int) p[6], maxc = min((int) p[5], nm);
        if (force >= 0) best = min(force, nm);
        else
        {
            float bt = 1e30f, c = 0.0f;
            for (int k = 0; k <= maxc; ++k)
            {
                if (k) c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1));
                const float gpu = p[1] * nh + p[0] * (nm - k);
                const float tt = fmaxf(gpu, k ? p[2] + c : 0.0f);
                if (tt < bt - 1e-6f) { bt = tt; best = k; }
            }
        }
        for (int k = 0; k < best; ++k) cpu[miss[k]] = 1;
        publish = best > 0 || p[7] > 0.5f;
        stats[0] += nh; stats[1] += nm; stats[2] += best; stats[3] += 1;
        npk = 0;
        if (publish)
        {
            seq = dseq[0] + 1;
            dseq[0] = seq;
            for (int s = 0; s < n; ++s)
            {
                const int e = ids[s];
                if (e < 0 || e >= E || !cpu[e] || w[s] == 0.0f) continue;
                hpicks[npk * 3] = s / topk; hpicks[npk * 3 + 1] = e; hpicks[npk * 3 + 2] = __float_as_int(w[s]);
                ++npk;
            }
        }
    }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const int e = ids[s];
        const bool drop = e < 0 || e >= E || cpu[e];
        ids_out[s] = drop ? E : e;
    }
    if (publish)
    {
        for (int i = t; i < bsz * H; i += blockDim.x)
        {
            if constexpr (std::is_same<T, __nv_bfloat16>::value) hx[i] = __bfloat162float(x[i]);
            else hx[i] = __half2float(x[i]);
        }
        __threadfence_system();
        __syncthreads();
        if (t == 0)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = seq;
            __threadfence_system();
            dflag[0] = seq;
            stats[7] += 1;
        }
    }
    else if (t == 0) dflag[0] = 0;
}

template <typename T>
__global__ void ct_combine_k(T* out, int n, const long long* __restrict__ dflag, volatile long long* ctrl,
    const float* hout, long long* stats, long long timeout_ns)
{
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0;
        if (go)
        {
            unsigned long long t0, t1;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            t1 = t0;
            while (ctrl[1] < s)
            {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
                if ((long long) (t1 - t0) > timeout_ns) { go = 0; if (blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[6], 1ull); break; }
            }
            if (blockIdx.x == 0) { atomicAdd((unsigned long long*) &stats[4], (unsigned long long) (t1 - t0)); atomicAdd((unsigned long long*) &stats[5], 1ull); }
        }
    }
    __syncthreads();
    if (!go) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const float v = __ldcv(hout + i);
        if constexpr (std::is_same<T, __nv_bfloat16>::value) out[i] = __float2bfloat16(__bfloat162float(out[i]) + v);
        else out[i] = __float2half(__half2float(out[i]) + v);
    }
}

void ct_split(torch::Tensor ids, torch::Tensor w, torch::Tensor x, int64_t E, torch::Tensor slotof, torch::Tensor freq,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, torch::Tensor dseq, int64_t li,
              torch::Tensor ids_out, torch::Tensor dflag, torch::Tensor stats)
{
    c10::cuda::CUDAGuard g(ids.device());
    auto st = at::cuda::getCurrentCUDAStream(ids.device().index());
    TORCH_CHECK(ids.dtype() == at::kInt && w.dtype() == at::kFloat && x.is_contiguous() && ids.is_contiguous() && w.is_contiguous(), "ct_split dtypes");
    TORCH_CHECK(E <= 1024, "ct_split: E <= 1024");
    const int bsz = (int) x.size(0), H = (int) x.size(1), n = (int) ids.numel(), topk = n / bsz;
    if (x.dtype() == at::kBFloat16)
        ct_split_k<__nv_bfloat16><<<1, 256, 0, st>>>(ids.data_ptr<int>(), w.data_ptr<float>(), (const __nv_bfloat16*) x.data_ptr(), n, bsz, topk, H,
            (int) E, slotof.data_ptr<int>(), freq.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl, (float*) hx,
            (int*) hpicks, (long long*) dseq.data_ptr<int64_t>(), (int) li, ids_out.data_ptr<int>(), (long long*) dflag.data_ptr<int64_t>(),
            (long long*) stats.data_ptr<int64_t>());
    else
        ct_split_k<__half><<<1, 256, 0, st>>>(ids.data_ptr<int>(), w.data_ptr<float>(), (const __half*) x.data_ptr(), n, bsz, topk, H,
            (int) E, slotof.data_ptr<int>(), freq.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl, (float*) hx,
            (int*) hpicks, (long long*) dseq.data_ptr<int64_t>(), (int) li, ids_out.data_ptr<int>(), (long long*) dflag.data_ptr<int64_t>(),
            (long long*) stats.data_ptr<int64_t>());
}

void ct_combine(torch::Tensor out, torch::Tensor dflag, int64_t ctrl, int64_t hout, torch::Tensor stats, int64_t timeout_ns)
{
    c10::cuda::CUDAGuard g(out.device());
    auto st = at::cuda::getCurrentCUDAStream(out.device().index());
    TORCH_CHECK(out.is_contiguous() && (out.dtype() == at::kBFloat16 || out.dtype() == at::kHalf), "out bf16/fp16 contiguous");
    const int n = (int) out.numel();
    if (out.dtype() == at::kBFloat16)
        ct_combine_k<__nv_bfloat16><<<8, 256, 0, st>>>((__nv_bfloat16*) out.data_ptr(), n, (const long long*) dflag.data_ptr<int64_t>(),
            (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
    else
        ct_combine_k<__half><<<8, 256, 0, st>>>((__half*) out.data_ptr(), n, (const long long*) dflag.data_ptr<int64_t>(),
            (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ct_split", &ct_split);
    m.def("ct_combine", &ct_combine);
}
