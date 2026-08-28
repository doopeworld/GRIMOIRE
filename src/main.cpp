// =====================================================================
//  main.cpp  --  device probe, correctness check, roofline benchmark
// =====================================================================
#include "kernels.hpp"
#include "b70/moe.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <chrono>

using namespace b70;

namespace {

template <typename T>
T* upload(sycl::queue& q, const T* src, size_t n) {
    T* d = sycl::malloc_device<T>(n, q);
    q.memcpy(d, src, n * sizeof(T)).wait();
    return d;
}

struct DeviceWeight {
    QuantWeight w;
    uint8_t* payload = nullptr;
    void*    scales  = nullptr;
    uint8_t* zeros   = nullptr;
    void free(sycl::queue& q) {
        if (payload) sycl::free(payload, q);
        if (scales)  sycl::free(scales, q);
        if (zeros)   sycl::free(zeros, q);
    }
};

DeviceWeight to_device(sycl::queue& q, const PackedWeight& p) {
    DeviceWeight d;
    d.payload = upload(q, p.payload.data(), p.payload.size());
    if (!p.scales_raw.empty()) d.scales = upload(q, p.scales_raw.data(), p.scales_raw.size());
    if (!p.zeros.empty())      d.zeros  = upload(q, p.zeros.data(), p.zeros.size());
    d.w = p.view();
    d.w.payload = d.payload;
    d.w.scales  = d.scales;
    d.w.zeros   = static_cast<const uint8_t*>(d.zeros);
    return d;
}

} // namespace

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    const auto& dev = q.get_device();

    std::printf("=== b70 bare-metal inference engine ===\n");
    std::printf("device     : %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("driver     : %s\n", dev.get_info<sycl::info::device::driver_version>().c_str());
    std::printf("EUs        : %u\n", dev.get_info<sycl::info::device::max_compute_units>());
    std::printf("VRAM       : %.1f GiB\n",
                double(dev.get_info<sycl::info::device::global_mem_size>()) / (1 << 30));
    std::printf("SLM        : %zu KiB\n",
                size_t(dev.get_info<sycl::info::device::local_mem_size>()) / 1024);
    std::printf("sub-groups : ");
    for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) std::printf("%zu ", s);
    std::printf("\n");

    bool has16 = false;
    for (auto s : dev.get_info<sycl::info::device::sub_group_sizes>()) has16 |= (s == SG_SIZE);
    if (!has16) {
        std::printf("\nFATAL: this device does not expose a %d-wide sub-group.\n"
                    "Every kernel here is written against SIMD16 XMX geometry.\n", SG_SIZE);
        return 1;
    }

    // -----------------------------------------------------------------
    // Must exceed the last-level cache or the benchmark measures cache,
    // not GDDR6. 16384 x 16384 at int8 is 256 MB, comfortably past it.
    const int N = (argc > 1) ? std::atoi(argv[1]) : 16384;
    const int K = (argc > 2) ? std::atoi(argv[2]) : 16384;
    std::printf("\nGEMV %d x %d, decode path\n", N, K);

    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> W(size_t(N) * K), x(K);
    for (auto& v : W) v = nd(rng);
    for (auto& v : x) v = nd(rng) * 10.0f;

    float* d_x = upload(q, x.data(), x.size());
    float* d_y = sycl::malloc_device<float>(N, q);
    std::vector<float> y(N);

    std::printf("  %-10s %8s %10s %12s %12s %10s\n",
                "format", "GiB", "ms", "GB/s", "%peak", "rel-err");

    const Fmt formats[] = { Fmt::BF16, Fmt::FP8_E4M3, Fmt::FP8_E5M2,
                            Fmt::INT8, Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4 };

    for (Fmt f : formats) {
        PackedWeight p = quantize(W.data(), N, K, f);
        DeviceWeight d = to_device(q, p);

        // host reference through the SAME decode functions
        // Checking all N rows on the host costs more than the whole
        // benchmark at this size. Sample 256 rows; that is plenty to
        // catch a wrong kernel, which is never subtly wrong.
        QuantWeight hw = p.view();
        const int NCHK = (N < 256) ? N : 256;
        const int stride_chk = N / NCHK;
        std::vector<float> ref(NCHK);
        for (int i = 0; i < NCHK; ++i) {
            const int n = i * stride_chk;
            double a = 0.0;
            for (int k = 0; k < K; ++k) a += double(hw.at(n, k)) * x[k];
            ref[i] = float(a);
        }

        // Best of several trials. A single 20-iteration trial showed up
        // to 30% run-to-run spread on this hardware (mxfp4 read 255 and
        // 332 GB/s for the identical config), which is enough noise to
        // pick the wrong tuning parameters. The minimum is the honest
        // figure: it is the run least perturbed by clock and thermal
        // wander, not an average of a contaminated distribution.
        for (int i = 0; i < 3; ++i) launch_gemv(q, d.w, d_x, d_y);
        q.wait();                                      // warm up
        constexpr int ITER   = 50;
        constexpr int TRIALS = 3;
        double best_ms = 1e30;
        for (int t = 0; t < TRIALS; ++t) {
            auto a = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < ITER; ++i) launch_gemv(q, d.w, d_x, d_y);
            q.wait();
            auto b = std::chrono::high_resolution_clock::now();
            const double m = std::chrono::duration<double, std::milli>(b - a).count() / ITER;
            if (m < best_ms) best_ms = m;
        }
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t1 = t0;

        q.memcpy(y.data(), d_y, N * sizeof(float)).wait();
        double se = 0.0, sr = 0.0;
        for (int i = 0; i < NCHK; ++i) {
            const int n = i * stride_chk;
            se += double(y[n] - ref[i]) * (y[n] - ref[i]);
            sr += double(ref[i]) * ref[i];
        }

        const double ms   = best_ms;
        const double gb   = double(p.payload.size() + p.scales_raw.size() + p.zeros.size()) / 1e9;
        const double bw   = gb / (ms / 1e3);
        std::printf("  %-10s %8.3f %10.3f %12.1f %11.1f%% %10.2e  %s\n",
                    fmt_name(f), gb, ms, bw, 100.0 * bw / 608.0,
                    std::sqrt(se / sr),
                    std::sqrt(se / sr) < 1e-3 ? "" : " <-- CHECK");
        d.free(q);
    }

    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // MoE-sized GEMV. The 16384^2 case above is one enormous contiguous
    // stream -- the easiest possible shape. Real expert matrices are
    // small, and a decode step touches 8 of 256 experts, so the question
    // is whether the bandwidth survives at 2048x512 rather than 16384^2.
    // This is what actually predicts tokens/sec on a 35B-A3B.
    // -----------------------------------------------------------------
    std::printf("\nMoE expert dimensions (Qwen3.5-MoE 35B-A3B)\n");
    {
        struct Shape { const char* name; int N, K; };
        const Shape shapes[] = {
            { "gate/up  [1024,2048]", 1024, 2048 },   // gate+up fused
            { "down     [2048, 512]", 2048,  512 },
            { "1 expert  (all three)",   0,     0 },  // reported below
        };
        const Fmt mfs[] = { Fmt::MXFP4, Fmt::INT4, Fmt::INT8 };

        std::printf("  %-22s %-8s %10s %10s %10s\n",
                    "matrix", "format", "ms", "GB/s", "%peak");
        for (Fmt f : mfs) {
            double expert_ms = 0.0, expert_gb = 0.0;
            for (int si = 0; si < 2; ++si) {
                const int N = shapes[si].N, K = shapes[si].K;
                std::vector<float> Wm(size_t(N) * K), xm(K);
                std::mt19937 r2(5);
                std::normal_distribution<float> n2(0.0f, 0.02f);
                for (auto& q2 : Wm) q2 = n2(r2);
                for (auto& q2 : xm) q2 = n2(r2);

                PackedWeight pp = quantize(Wm.data(), N, K, f);
                DeviceWeight dd = to_device(q, pp);
                float* dx = upload(q, xm.data(), xm.size());
                float* dy = sycl::malloc_device<float>(N, q);

                for (int i = 0; i < 20; ++i) launch_gemv(q, dd.w, dx, dy);
                q.wait();
                double best = 1e30;
                for (int t = 0; t < 3; ++t) {
                    auto a0 = std::chrono::high_resolution_clock::now();
                    // Small matrices finish in microseconds, so time a
                    // large batch or the clock resolution dominates.
                    for (int i = 0; i < 500; ++i) launch_gemv(q, dd.w, dx, dy);
                    q.wait();
                    auto b0 = std::chrono::high_resolution_clock::now();
                    const double mm =
                        std::chrono::duration<double, std::milli>(b0 - a0).count() / 500;
                    if (mm < best) best = mm;
                }
                const double gb = double(pp.payload.size() + pp.scales_raw.size()
                                       + pp.zeros.size()) / 1e9;
                std::printf("  %-22s %-8s %10.4f %10.1f %9.1f%%\n",
                            shapes[si].name, fmt_name(f), best,
                            gb / (best / 1e3), 100.0 * (gb / (best / 1e3)) / 608.0);
                // one expert = gate+up (once) + down (once)
                expert_ms += best;
                expert_gb += gb;
                sycl::free(dx, q); sycl::free(dy, q); dd.free(q);
            }

            // Project a full decode step for the real model.
            const int TOPK = 8, LAYERS = 40;
            const double moe_ms = expert_ms * TOPK * LAYERS;
            const double moe_gb = expert_gb * TOPK * LAYERS;
            std::printf("  %-22s %-8s %10.3f %10.1f  -> %.0f tok/s (MoE only)\n",
                        "-> 8 experts x40 layers", fmt_name(f), moe_ms,
                        moe_gb / (moe_ms / 1e3), 1000.0 / moe_ms);
        }
        std::printf("\n  Compare against the 16384^2 figures above: any large drop\n"
                    "  is small-matrix overhead, not a bandwidth limit.\n");
    }

    // -----------------------------------------------------------------
    // FUSED MoE, the whole point of the project.
    //
    // The per-expert table above is what a generic framework does: one
    // kernel launch per expert per projection. At 10.4 us for a 1.1 MB
    // matrix that is ~3 us of memory work behind ~7 us of launch
    // latency -- the GPU idles two thirds of the step. It lands at
    // ~17.7% of peak, which matches the 20% measured from vLLM on this
    // same model.
    //
    // The fused kernel processes all top_k experts in ONE launch, with
    // the expert id carried as data. Same bytes moved, 12x fewer
    // launches. If the diagnosis is right the gap closes here.
    // -----------------------------------------------------------------
    std::printf("\nFUSED MoE block, one full layer (256 experts, top-8)\n");
    {
        MoeConfig mc;
        mc.hidden = 2048; mc.inter = 512; mc.num_experts = 256; mc.top_k = 8;
        mc.shared_inter = 0;

        const int H = mc.hidden, I = mc.inter, E = mc.num_experts;
        std::mt19937 r3(11);
        std::normal_distribution<float> n3(0.0f, 0.02f);

        std::printf("  %-8s %12s %12s %10s %14s\n",
                    "format", "ms/layer", "GB/s", "%peak", "40 layers");
        for (Fmt f : { Fmt::MXFP4, Fmt::INT4 }) {
            // gate_up is [E*2I][H], down is [E*H][I] -- expert-major, so
            // a routed expert is two contiguous runs, never a gather.
            std::vector<float> GU(size_t(E) * 2 * I * H);
            std::vector<float> DN(size_t(E) * H * I);
            for (auto& v : GU) v = n3(r3);
            for (auto& v : DN) v = n3(r3);

            PackedWeight pgu = quantize(GU.data(), E * 2 * I, H, f);
            PackedWeight pdn = quantize(DN.data(), E * H, I, f);
            DeviceWeight dgu = to_device(q, pgu);
            DeviceWeight ddn = to_device(q, pdn);

            MoeLayer L;
            L.cfg = mc;
            L.gate_up = dgu.w;
            L.down    = ddn.w;

            std::vector<float> xh(H);
            for (auto& v : xh) v = n3(r3);
            float* dx = upload(q, xh.data(), xh.size());
            float* dh = sycl::malloc_device<float>(size_t(mc.top_k) * I, q);
            float* dy = sycl::malloc_device<float>(H, q);

            std::vector<int32_t> exp_ids(mc.top_k);
            std::vector<float>   exp_w(mc.top_k);
            for (int i = 0; i < mc.top_k; ++i) {
                exp_ids[i] = (i * 37) % E;
                exp_w[i]   = 1.0f / mc.top_k;
            }
            int32_t* de = upload(q, exp_ids.data(), exp_ids.size());
            float*   dw = upload(q, exp_w.data(), exp_w.size());

            for (int i = 0; i < 20; ++i) {
                launch_moe_gate_up(q, L, de, dx, dh, {});
                launch_moe_down(q, L, de, dw, dh, dy, {});
            }
            q.wait();

            double best = 1e30;
            for (int t = 0; t < 3; ++t) {
                auto a0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 200; ++i) {
                    launch_moe_gate_up(q, L, de, dx, dh, {});
                    launch_moe_down(q, L, de, dw, dh, dy, {});
                }
                q.wait();
                auto b0 = std::chrono::high_resolution_clock::now();
                const double mm =
                    std::chrono::duration<double, std::milli>(b0 - a0).count() / 200;
                if (mm < best) best = mm;
            }

            const double gb = L.bytes_per_token() / 1e9;
            const double bw = gb / (best / 1e3);
            std::printf("  %-8s %12.4f %12.1f %9.1f%% %8.2f ms -> %4.0f tok/s\n",
                        fmt_name(f), best, bw, 100.0 * bw / 608.0,
                        best * 40.0, 1000.0 / (best * 40.0));

            sycl::free(dx, q); sycl::free(dh, q); sycl::free(dy, q);
            sycl::free(de, q); sycl::free(dw, q);
            dgu.free(q); ddn.free(q);
        }
        std::printf("\n  2 launches per layer instead of 24. Compare the %%peak\n"
                    "  column against the per-expert table above.\n");
    }

    std::printf("\nFlashDecoding, 32 heads / 8 kv heads / dim 128\n");
    const int NH = 32, NKV = 8, HD = 128, CAP = 8192;
    float* d_q  = sycl::malloc_device<float>(size_t(NH) * HD, q);
    uint8_t* d_kc = sycl::malloc_device<uint8_t>(size_t(NKV) * HD * CAP, q);
    uint8_t* d_vc = sycl::malloc_device<uint8_t>(size_t(NKV) * CAP * HD, q);
    float* d_o  = sycl::malloc_device<float>(size_t(NH) * HD, q);
    q.memset(d_q, 0, size_t(NH) * HD * sizeof(float));
    q.memset(d_kc, 0, size_t(NKV) * HD * CAP);
    q.memset(d_vc, 0, size_t(NKV) * CAP * HD);
    q.wait();

    const int EUS = int(dev.get_info<sycl::info::device::max_compute_units>());
    const int MAXSPLIT = 64;
    float* d_part  = sycl::malloc_device<float>(size_t(NH) * MAXSPLIT * HD, q);
    float* d_pm    = sycl::malloc_device<float>(size_t(NH) * MAXSPLIT, q);
    float* d_pl    = sycl::malloc_device<float>(size_t(NH) * MAXSPLIT, q);

    for (int seq : {512, 2048, 8192}) {
        AttnParams ap{d_q, d_kc, d_vc, d_o, seq, CAP, HD, NH, NKV,
                      1.0f / std::sqrt(float(HD)),
                      d_part, d_pm, d_pl, pick_splits(seq, NH, EUS)};
        launch_flash_decode(q, ap);
        launch_flash_merge(q, ap).wait();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) { launch_flash_decode(q, ap); launch_flash_merge(q, ap); }
        q.wait();
        auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / 20;
        const double gb = 2.0 * double(NKV) * seq * HD * 4 / 1e9;
        std::printf("  seq %5d  splits %2d  %8.3f ms  %8.1f GB/s  (%.1f%% of peak)\n",
                    seq, ap.splits, ms, gb / (ms / 1e3), 100.0 * (gb / (ms / 1e3)) / 608.0);
    }

    sycl::free(d_part, q); sycl::free(d_pm, q); sycl::free(d_pl, q);
    sycl::free(d_q, q); sycl::free(d_kc, q); sycl::free(d_vc, q); sycl::free(d_o, q);
    sycl::free(d_x, q); sycl::free(d_y, q);
    std::printf("\ndone\n");
    return 0;
}
