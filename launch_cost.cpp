// launch_cost.cpp -- what does ONE kernel launch cost on this driver?
//
// The fusion plan rests on a single number: the gap between kernels on an
// in-order queue. The decode loop shows ~23 us per kernel slot against a
// cost model that says ~5 us of work. If an EMPTY kernel costs ~17 us
// here, fusion is the lever. If an empty kernel is cheap, the gap is
// somewhere else and fusing 300 kernels would buy nothing.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using clk = std::chrono::high_resolution_clock;

template <typename F>
double timed(int reps, F&& f) {
    const auto t0 = clk::now();
    f(reps);
    const auto t1 = clk::now();
    return std::chrono::duration<double, std::micro>(t1 - t0).count() / reps;
}

int main() {
    sycl::queue qi(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    sycl::queue qo(sycl::gpu_selector_v);
    std::printf("device: %s\n\n", qi.get_device().get_info<sycl::info::device::name>().c_str());

    const int N = 2000;
    float* buf = sycl::malloc_device<float>(1 << 20, qi);
    qi.memset(buf, 0, (1 << 20) * sizeof(float)).wait();

    // 1. empty kernel, in-order, one wait at the end (same shape as decode)
    auto empty_batch = [&](sycl::queue& q) {
        return timed(N, [&](int reps) {
            for (int i = 0; i < reps; ++i)
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
                });
            q.wait();
        });
    };
    std::printf("empty kernel, in-order,     submit+drain : %7.2f us\n", empty_batch(qi));
    std::printf("empty kernel, out-of-order, submit+drain : %7.2f us\n", empty_batch(qo));

    // 2. submit cost alone (host side) vs the drain
    {
        const auto t0 = clk::now();
        for (int i = 0; i < N; ++i)
            qi.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
            });
        const auto t1 = clk::now();
        qi.wait();
        const auto t2 = clk::now();
        std::printf("   of which host submit                : %7.2f us\n",
                    std::chrono::duration<double, std::micro>(t1 - t0).count() / N);
        std::printf("   of which drain after last submit    : %7.2f us total\n",
                    std::chrono::duration<double, std::micro>(t2 - t1).count());
    }

    // 3. a realistic small elementwise kernel (the swiglu/gate_silu shape)
    for (int n : {2048, 8192, 65536}) {
        const double us = timed(N, [&](int reps) {
            for (int i = 0; i < reps; ++i)
                qi.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> id) {
                        buf[id[0]] = buf[id[0]] * 1.000001f + 1e-6f;
                    });
                });
            qi.wait();
        });
        std::printf("elementwise n=%6d, in-order            : %7.2f us\n", n, us);
    }

    // 4. round-trip: one kernel, one wait -- the fully serialized case
    const double rt = timed(200, [&](int reps) {
        for (int i = 0; i < reps; ++i) {
            qi.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
            }).wait();
        }
    });
    std::printf("empty kernel + wait each     (round trip) : %7.2f us\n", rt);

    sycl::free(buf, qi);
    return 0;
}
