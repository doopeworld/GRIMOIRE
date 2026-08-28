// gemv_small.cpp -- the small-N projections are occupancy starved.
//
// The decode GEMV maps ONE sub-group to a row and 32 rows to a work-group.
// At N=64 (the deltanet a/b projection) that is 2 work-groups on a 256-EU
// card. The fix under test: for small N, give a whole work-group to a row
// block and split K across its sub-groups, reducing through SLM. Same
// GemvStep, so the dequant math is untouched; only the summation order and
// the launch geometry change.
#include "kernels.hpp"
#include "gemv_step.hpp"
#include <chrono>
#include <cstdio>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace b70;
using clk = std::chrono::high_resolution_clock;

// candidate: WG per row-block, sub-groups split K, SLM tree reduction
template <Fmt F, int EPL, int ROWS_W>
sycl::event gemv_wide(sycl::queue& q, const QuantWeight& w,
                      const float* x, float* y) {
    const int N = w.N, K = w.K;
    const int n_groups = (N + ROWS_W - 1) / ROWS_W;
    const int wg_threads = WG_SUBGROUPS * SG_SIZE;
    return q.submit([&](sycl::handler& h) {
        const QuantWeight wc = w;
        sycl::local_accessor<float, 1> part(WG_SUBGROUPS * ROWS_W, h);
        sycl::local_accessor<float, 1> nlut(16, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(n_groups) * wg_threads, wg_threads),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg   = it.get_sub_group();
                const int  lane = int(sg.get_local_id()[0]);
                const int  sgid = int(sg.get_group_id()[0]);
                const int  lid  = int(it.get_local_id(0));
                float* nl = nlut.template get_multi_ptr<sycl::access::decorated::no>().get();
                if constexpr (F == Fmt::MXFP4)
                    for (int b = lid; b < 16; b += wg_threads) nl[b] = e2m1_to_f32(uint8_t(b));
                if constexpr (F == Fmt::MXFP4) sycl::group_barrier(it.get_group());

                // K split across sub-groups; each slice is a whole number
                // of scale groups so the hoisted scale stays correct.
                const int per_sg = K / WG_SUBGROUPS;
                const int k_beg  = sgid * per_sg;
                const int k_end  = (sgid == WG_SUBGROUPS - 1) ? K : k_beg + per_sg;

                #pragma unroll
                for (int r = 0; r < ROWS_W; ++r) {
                    const int n = int(it.get_group(0)) * ROWS_W + r;
                    float acc = 0.0f;
                    if (n < N) {
                        const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                        for (int k = k_beg + lane * EPL; k + EPL <= k_end; k += SG_SIZE * EPL)
                            acc += GemvStep<F, EPL>::run(wc, row, x, nullptr, nullptr, nl, n, k);
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) part[r * WG_SUBGROUPS + sgid] = acc;
                }
                sycl::group_barrier(it.get_group());
                for (int r = lid; r < ROWS_W; r += wg_threads) {
                    const int n = int(it.get_group(0)) * ROWS_W + r;
                    if (n >= N) continue;
                    float s = 0.0f;                     // fixed order: deterministic
                    for (int i = 0; i < WG_SUBGROUPS; ++i) s += part[r * WG_SUBGROUPS + i];
                    y[n] = s;
                }
            });
    });
}

template <typename F> double bench(sycl::queue& q, int reps, F&& f) {
    for (int i = 0; i < 30; ++i) f();
    q.wait();
    const auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) f();
    q.wait();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::printf("device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-28s %9s %9s %9s %8s\n",
                "shape (int4, K=2048)", "cur us", "wide us", "GB/s cur", "maxrel");

    const int K = 2048;
    std::mt19937 rng(3);
    for (int N : {64, 256, 1024, 2048, 4096, 8192}) {
        const int groups = K / kInt4Group;
        const int64_t row_bytes = K / 2;
        std::vector<uint8_t> pack(size_t(N) * row_bytes);
        std::vector<bf16_t>  sc(size_t(N) * groups);
        std::vector<uint8_t> zr(size_t(N) * groups, 8);
        for (auto& b : pack) b = uint8_t(rng() & 0xFF);
        for (auto& s : sc) s = f32_to_bf16(0.01f);

        QuantWeight w{};
        w.fmt = Fmt::INT4; w.N = N; w.K = K;
        w.row_bytes = row_bytes; w.row_scales = groups;
        w.payload = sycl::malloc_device<uint8_t>(pack.size(), q);
        auto* dsc = sycl::malloc_device<bf16_t>(sc.size(), q);
        auto* dzr = sycl::malloc_device<uint8_t>(zr.size(), q);
        q.memcpy((void*)w.payload, pack.data(), pack.size()).wait();
        q.memcpy(dsc, sc.data(), sc.size() * sizeof(bf16_t)).wait();
        q.memcpy(dzr, zr.data(), zr.size()).wait();
        w.scales = dsc; w.zeros = dzr;

        float* x  = sycl::malloc_device<float>(K, q);
        float* y0 = sycl::malloc_device<float>(N, q);
        float* y1 = sycl::malloc_device<float>(N, q);
        std::vector<float> hx(K);
        for (auto& v : hx) v = float(rng() % 100) / 100.0f - 0.5f;
        q.memcpy(x, hx.data(), K * sizeof(float)).wait();

        const double t_cur  = bench(q, 500, [&]{ launch_gemv(q, w, x, y0, {}); });
        const double t_wide = bench(q, 500, [&]{ gemv_wide<Fmt::INT4, 16, 1>(q, w, x, y1); });

        std::vector<float> a(N), b(N);
        q.memcpy(a.data(), y0, N * sizeof(float)).wait();
        q.memcpy(b.data(), y1, N * sizeof(float)).wait();
        double mr = 0;
        for (int i = 0; i < N; ++i)
            mr = std::max<double>(mr, std::fabs(a[i] - b[i]) / std::fmax(1e-3f, std::fabs(a[i])));

        const double bytes = double(N) * K / 2.0;
        std::printf("N=%-6d                     %9.2f %9.2f %9.1f %8.1e\n",
                    N, t_cur, t_wide, bytes / t_cur * 1e-3, mr);

        sycl::free((void*)w.payload, q); sycl::free(dsc, q); sycl::free(dzr, q);
        sycl::free(x, q); sycl::free(y0, q); sycl::free(y1, q);
    }
    return 0;
}
