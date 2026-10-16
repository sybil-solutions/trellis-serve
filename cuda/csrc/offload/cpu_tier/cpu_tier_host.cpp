// trellis-serve CPU expert tier, host side (ported from FreeToken-EXL3 ft_tier_ext.cpp, GLM-5.3 C05x lane).
//
// In GPU-cache offload decode, some of a layer's cache misses are computed by this AVX2 pool from the SAME pinned
// host records the GPU reads zero-copy (offload_store.HostExpertStore, record layout of kernels/offload_moe.py), in
// parallel with the GPU's zero-copy GEMMs for the remaining misses. The PCIe link only carries the GPU's share.
//
// Handoff through pinned + mapped host buffers written by the GPU split kernel (cpu_tier_dev.cu):
//   ctrl[0] job seq (GPU writes last)   ctrl[1] done seq (worker writes last)
//   ctrl[2] layer   ctrl[3] ntok   ctrl[4] npicks
//   hx    [ntok, H] fp32 expert input (bf16/fp16 values)   picks [npicks][3] = (token, expert, weight f32 bits)
//   hout  [ntok, H] fp32 weighted routed partial of the CPU picks (the GPU combine kernel adds it)
#include <torch/extension.h>
#include "ft_core.h"

namespace ct {

struct Stats { std::atomic<long> jobs{0}, experts{0}, tokens{0}, empty{0}; std::atomic<long long> busy_ns{0}; };

struct Tier
{
    Pool pool;
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    volatile int64_t* ctrl = nullptr;
    const float* hx = nullptr;
    const int32_t* picks = nullptr;
    float* hout = nullptr;
    volatile float* pol = nullptr;       // policy vector (pinned + mapped, read by ct_split): p[3] auto-calibrated
    double ema_b = 0.0;                  // us per CPU expert (EMA over jobs)
    int mode = 1, variant = 1, H = 0, I = 0;
    std::thread th;
    std::atomic<bool> quit{false}, running{false};
    Stats st;
};

Tier* T = nullptr;

void tier_init(int64_t threads, std::vector<int64_t> cpus, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks,
               torch::Tensor hout, int64_t mode, int64_t variant, int64_t H, int64_t I, int64_t nlayers)
{
    TORCH_CHECK(T == nullptr, "cpu tier already initialised");
    TORCH_CHECK(H % 512 == 0 && I % 128 == 0 && I / 128 <= 16 && H / 128 <= 32, "cpu tier: unsupported H / I");
    init_perm(); init_tables();
    T = new Tier();
    std::vector<int> c(cpus.begin(), cpus.end());
    if (c.empty()) c = cpu_order();
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    T->pool.start(int(threads), c);                 // pins the calling thread to c[0]; the worker thread takes that role
    sched_setaffinity(0, sizeof(saved), &saved);
    T->ctrl = reinterpret_cast<volatile int64_t*>(ctrl.data_ptr<int64_t>());
    T->hx = hx.data_ptr<float>();
    T->picks = picks.data_ptr<int32_t>();
    T->hout = hout.data_ptr<float>();
    T->keep = { ctrl, hx, picks, hout };
    T->mode = int(mode); T->variant = int(variant); T->H = int(H); T->I = int(I);
    T->layers.resize(nlayers);
}

// One layer from its record bank: host_base = address of expert 0's record; offs = the 6 field offsets
// (w13, w2, suh13, svh13, suh2, svh2) of kernels/offload_moe.RecordLayout.
void tier_add_layer(int64_t li, int64_t host_base, int64_t E, int64_t rec, std::vector<int64_t> offs)
{
    TORCH_CHECK(T && li >= 0 && li < (int64_t) T->layers.size() && offs.size() == 6, "tier_add_layer: bad args");
    const int H = T->H, I = T->I, rt13 = 2 * I / 16;
    Layer& L = T->layers[li];
    L.H = H; L.I = I; L.ex.resize(E);
    for (int e = 0; e < E; ++e)
    {
        const uint8_t* r = reinterpret_cast<const uint8_t*>(host_base) + size_t(e) * rec;
        const uint16_t* suh13 = reinterpret_cast<const uint16_t*>(r + offs[2]);
        const uint16_t* svh13 = reinterpret_cast<const uint16_t*>(r + offs[3]);
        L.ex[e].g = { r + offs[0], suh13, svh13, H, I, 0, rt13 };
        L.ex[e].u = { r + offs[0] + size_t(I / 16) * TILE_BYTES, suh13 + H, svh13 + I, H, I, 0, rt13 };
        L.ex[e].d = { r + offs[1], reinterpret_cast<const uint16_t*>(r + offs[4]), reinterpret_cast<const uint16_t*>(r + offs[5]), I, H, 0, 0 };
    }
}

static void forward(int li, const float* x, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route, float* out)
{
    if (T->variant == 0) moe_forward(T->pool, T->layers[li], x, ntok, route, out, T->mode);
    else moe_forward_band(T->pool, T->layers[li], x, ntok, route, out, T->mode);
}

void run_job(int li, int ntok, int np)
{
    std::vector<std::vector<std::pair<int, float>>> route(ntok);
    for (int k = 0; k < np; ++k)
    {
        const int t = T->picks[k * 3], e = T->picks[k * 3 + 1]; float w; std::memcpy(&w, &T->picks[k * 3 + 2], 4);
        if (t >= 0 && t < ntok) route[t].push_back({ e, w });
    }
    if (np == 0) { std::memset(T->hout, 0, size_t(ntok) * T->H * 4); T->st.empty++; return; }
    forward(li, T->hx, ntok, route, T->hout);
    T->st.experts += np; T->st.tokens += ntok;
}

void worker_main(int cpu0)
{
    Pool::pin(cpu0);
    int64_t last = T->ctrl[0];
    long idle = 0;
    while (!T->quit.load(std::memory_order_relaxed))
    {
        const int64_t s = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
        if (s == last)
        {
            if (g_pool_pause.load(std::memory_order_relaxed)) std::this_thread::sleep_for(std::chrono::microseconds(200));
            else if (++idle < 400000) _mm_pause();
            else std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
        }
        idle = 0;
        const double t0 = now();
        const int np = int(T->ctrl[4]);
        run_job(int(T->ctrl[2]), int(T->ctrl[3]), np);
        const double us = (now() - t0) * 1e6;
        T->st.busy_ns += (long long) (us * 1e3);
        if (T->pol && np > 0)
        {
            // per-expert cost model update: (busy - fixed) / experts, EMA; the split kernel reads p[3] next layer
            const double b = std::max(10.0, (us - T->pol[2]) / np);
            T->ema_b = T->ema_b > 0 ? 0.98 * T->ema_b + 0.02 * b : b;
            if (T->pol[8] > 0.5f) T->pol[3] = float(T->ema_b * T->pol[9]);
        }
        T->st.jobs++;
        last = s;
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 1), s, __ATOMIC_RELEASE);
    }
}

void tier_start(int64_t cpu0)
{
    TORCH_CHECK(T && !T->running, "init first / already running");
    T->running = true;
    T->th = std::thread(worker_main, int(cpu0));
}

void tier_stop()
{
    if (!T) return;
    T->quit = true;
    if (T->th.joinable()) T->th.join();
    T->pool.stop();
}

// Synchronous forward for tests / benchmarks (worker must not be running): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
torch::Tensor tier_forward(int64_t li, torch::Tensor x, torch::Tensor sel, torch::Tensor w)
{
    TORCH_CHECK(T && !T->running, "tier_forward: worker running");
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, T->H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j) if (sa[t][j] >= 0) route[t].push_back({ sa[t][j], wa[t][j] });
    forward(int(li), x.contiguous().data_ptr<float>(), m, route, out.data_ptr<float>());
    return out;
}

void tier_pause(bool p) { g_pool_pause.store(p ? 1 : 0); }

void tier_set(int64_t mode, int64_t variant) { if (T) { T->mode = int(mode); T->variant = int(variant); } }

// policy: pinned + mapped float [>= 10]; p[8] = 1 enables auto-calibration of p[3], p[9] = its safety factor
void tier_set_policy(torch::Tensor pol) { TORCH_CHECK(T && pol.dtype() == torch::kFloat); T->pol = pol.data_ptr<float>(); T->keep.push_back(pol); }

std::vector<int64_t> tier_stats()
{
    if (!T) return {};
    return { T->st.jobs.load(), T->st.experts.load(), T->st.tokens.load(), T->st.empty.load(), (int64_t) T->st.busy_ns.load(),
             (int64_t) (T->ema_b * 1000) };
}

}  // namespace ct

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tier_init", &ct::tier_init);
    m.def("tier_add_layer", &ct::tier_add_layer);
    m.def("tier_start", &ct::tier_start);
    m.def("tier_stop", &ct::tier_stop);
    m.def("tier_forward", &ct::tier_forward);
    m.def("tier_set", &ct::tier_set);
    m.def("tier_stats", &ct::tier_stats);
    m.def("tier_set_policy", &ct::tier_set_policy);
    m.def("tier_pause", &ct::tier_pause);
}
