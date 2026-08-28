// One work-group scanning 248320 logits, vs a two-stage parallel argmax.
// Tie-break must stay "lowest index wins" so decoding is deterministic.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <climits>
#include <limits>
#include <vector>
#include <random>
constexpr int SG = 16;
using clk = std::chrono::high_resolution_clock;

template <typename F> double bench(sycl::queue& q, int reps, F&& f) {
    for (int i = 0; i < 20; ++i) f();
    q.wait();
    const auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) f();
    q.wait();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    const int n = 248320;
    float* lg = sycl::malloc_device<float>(n, q);
    int32_t* oi = sycl::malloc_device<int32_t>(1, q);
    float* ov = sycl::malloc_device<float>(1, q);
    std::vector<float> h(n);
    std::mt19937 rng(5);
    for (auto& v : h) v = float(rng() % 100000) / 1000.0f;
    h[173456] = 1e6f;                       // known winner
    q.memcpy(lg, h.data(), n * sizeof(float)).wait();

    // --- current: ONE work-group of 256 -------------------------------
    auto cur = [&]{
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float,1> bv(256/SG, hd);
            sycl::local_accessor<int32_t,1> bi(256/SG, hd);
            hd.parallel_for(sycl::nd_range<1>(256,256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lid = int(it.get_local_id(0)), sgid = int(sg.get_group_id()[0]);
                    const int lane = int(sg.get_local_id()[0]), nsg = 256/SG;
                    float best = -std::numeric_limits<float>::infinity(); int32_t bidx = 0;
                    for (int i = lid; i < n; i += 256)
                        if (lg[i] > best) { best = lg[i]; bidx = i; }
                    const float gm = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                    const int32_t gi = sycl::reduce_over_group(sg,
                        (best == gm) ? bidx : INT_MAX, sycl::minimum<int32_t>());
                    if (lane == 0) { bv[sgid] = gm; bi[sgid] = gi; }
                    sycl::group_barrier(it.get_group());
                    if (lid == 0) {
                        float m = bv[0]; int32_t x = bi[0];
                        for (int i = 1; i < nsg; ++i)
                            if (bv[i] > m || (bv[i] == m && bi[i] < x)) { m = bv[i]; x = bi[i]; }
                        *ov = m; *oi = x;
                    }
                });
        });
    };

    // --- two stage ----------------------------------------------------
    const int NG = 512;                      // stage-1 work-groups
    float* pv = sycl::malloc_device<float>(NG, q);
    int32_t* pi = sycl::malloc_device<int32_t>(NG, q);
    auto two = [&]{
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float,1> bv(256/SG, hd);
            sycl::local_accessor<int32_t,1> bi(256/SG, hd);
            hd.parallel_for(sycl::nd_range<1>(size_t(NG)*256, 256),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lid = int(it.get_local_id(0)), sgid = int(sg.get_group_id()[0]);
                    const int lane = int(sg.get_local_id()[0]), nsg = 256/SG;
                    const int g = int(it.get_group(0));
                    float best = -std::numeric_limits<float>::infinity(); int32_t bidx = INT_MAX;
                    for (int i = g*256 + lid; i < n; i += NG*256)
                        if (lg[i] > best) { best = lg[i]; bidx = i; }
                    const float gm = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                    const int32_t gi = sycl::reduce_over_group(sg,
                        (best == gm) ? bidx : INT_MAX, sycl::minimum<int32_t>());
                    if (lane == 0) { bv[sgid] = gm; bi[sgid] = gi; }
                    sycl::group_barrier(it.get_group());
                    if (lid == 0) {
                        float m = bv[0]; int32_t x = bi[0];
                        for (int i = 1; i < nsg; ++i)
                            if (bv[i] > m || (bv[i] == m && bi[i] < x)) { m = bv[i]; x = bi[i]; }
                        pv[g] = m; pi[g] = x;
                    }
                });
        });
        q.submit([&](sycl::handler& hd) {
            hd.parallel_for(sycl::nd_range<1>(SG,SG),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lane = int(it.get_local_id(0));
                    float best = -std::numeric_limits<float>::infinity(); int32_t bidx = INT_MAX;
                    for (int i = lane; i < NG; i += SG)
                        if (pv[i] > best || (pv[i] == best && pi[i] < bidx)) { best = pv[i]; bidx = pi[i]; }
                    const float gm = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                    const int32_t gi = sycl::reduce_over_group(sg,
                        (best == gm) ? bidx : INT_MAX, sycl::minimum<int32_t>());
                    if (lane == 0) { *ov = gm; *oi = gi; }
                });
        });
    };

    const double t1 = bench(q, 200, cur);
    int32_t r1 = 0; q.memcpy(&r1, oi, 4).wait();
    const double t2 = bench(q, 200, two);
    int32_t r2 = 0; q.memcpy(&r2, oi, 4).wait();

    std::printf("argmax over %d logits (1.0 MB)\n", n);
    std::printf("  current, ONE work-group : %8.2f us   -> idx %d\n", t1, r1);
    std::printf("  two-stage, %d groups    : %8.2f us   -> idx %d\n", NG, t2, r2);
    std::printf("  %s, speedup %.1fx\n", r1 == r2 ? "SAME INDEX" : "MISMATCH!", t1 / t2);
    return 0;
}
