// Why does a 256-element top-8 cost 21 us when an elementwise kernel costs 3?
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <climits>
#include <limits>
constexpr int SG = 16;
using clk = std::chrono::high_resolution_clock;

template <typename F> double us_per(int reps, sycl::queue& q, F&& body) {
    for (int i = 0; i < 50; ++i) body();      // warm
    q.wait();
    const auto t0 = clk::now();
    for (int i = 0; i < reps; ++i) body();
    q.wait();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    const int N = 256, K = 8, REP = 2000;
    float*   lg = sycl::malloc_device<float>(N, q);
    int32_t* oe = sycl::malloc_device<int32_t>(K, q);
    float*   ow = sycl::malloc_device<float>(K, q);
    q.memset(lg, 0, N * sizeof(float)).wait();

    printf("empty  16 threads        : %6.2f us\n", us_per(REP, q, [&]{
        q.submit([&](sycl::handler& h){ h.parallel_for(sycl::nd_range<1>(SG,SG),
            [=](sycl::nd_item<1>){}); }); }));

    printf("empty 256 threads        : %6.2f us\n", us_per(REP, q, [&]{
        q.submit([&](sycl::handler& h){ h.parallel_for(sycl::nd_range<1>(256,256),
            [=](sycl::nd_item<1>){}); }); }));

    printf("16thr + 1KB SLM stage    : %6.2f us\n", us_per(REP, q, [&]{
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> sl(size_t(N), h);
            h.parallel_for(sycl::nd_range<1>(SG,SG),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const int lane = int(it.get_local_id(0));
                    for (int e = lane; e < N; e += SG) sl[e] = lg[e];
                    sycl::group_barrier(it.get_group());
                    if (lane == 0) ow[0] = sl[0];
                }); }); }));

    printf("16thr full top-8 (as-is) : %6.2f us\n", us_per(REP, q, [&]{
        q.submit([&](sycl::handler& h){
            sycl::local_accessor<float,1> sl(size_t(N), h);
            h.parallel_for(sycl::nd_range<1>(SG,SG),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lane = int(it.get_local_id(0));
                    for (int e = lane; e < N; e += SG) sl[e] = lg[e];
                    sycl::group_barrier(it.get_group());
                    uint64_t mine = 0;
                    for (int s = 0; s < K; ++s) {
                        float cv = -std::numeric_limits<float>::infinity();
                        int ci = INT_MAX, cs = -1;
                        for (int e = lane, slot = 0; e < N; e += SG, ++slot) {
                            if (mine & (1ull << slot)) continue;
                            const float v = sl[e];
                            if (v > cv || (v == cv && e < ci)) { cv = v; ci = e; cs = slot; }
                        }
                        const float bv = sycl::reduce_over_group(sg, cv, sycl::maximum<float>());
                        const int bi = sycl::reduce_over_group(sg,
                            (cv == bv && ci != INT_MAX) ? ci : INT_MAX, sycl::minimum<int>());
                        if (ci == bi && cs >= 0) mine |= (1ull << cs);
                        if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                    }
                }); }); }));

    printf("16thr top-8, regs no SLM : %6.2f us\n", us_per(REP, q, [&]{
        q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::nd_range<1>(SG,SG),
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg = it.get_sub_group();
                    const int lane = int(it.get_local_id(0));
                    float r[N / SG];
                    #pragma unroll
                    for (int slot = 0; slot < N / SG; ++slot) r[slot] = lg[lane + slot * SG];
                    uint64_t mine = 0;
                    for (int s = 0; s < K; ++s) {
                        float cv = -std::numeric_limits<float>::infinity();
                        int ci = INT_MAX, cs = -1;
                        #pragma unroll
                        for (int slot = 0; slot < N / SG; ++slot) {
                            if (mine & (1ull << slot)) continue;
                            const int e = lane + slot * SG;
                            if (r[slot] > cv || (r[slot] == cv && e < ci)) { cv = r[slot]; ci = e; cs = slot; }
                        }
                        const float bv = sycl::reduce_over_group(sg, cv, sycl::maximum<float>());
                        const int bi = sycl::reduce_over_group(sg,
                            (cv == bv && ci != INT_MAX) ? ci : INT_MAX, sycl::minimum<int>());
                        if (ci == bi && cs >= 0) mine |= (1ull << cs);
                        if (lane == 0) { oe[s] = bi; ow[s] = bv; }
                    }
                }); }); }));

    sycl::free(lg,q); sycl::free(oe,q); sycl::free(ow,q);
    return 0;
}
