// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// kgap_probe.cpp -- the DEVICE-side cost of one kernel slot on an in-order queue when the
// host is ahead of the GPU (10-28: Flash-Next two-card decode reaches every all-reduce
// ~100 us before its GPU drains, so the GPU runs the ~30 kernels of a layer back to back).
// launch_cost.cpp timed submit + drain, which is the host's pace, not the GPU's.
//
// A ~35 ms busy kernel chain goes first, so all N small kernels are queued before the GPU
// reaches them; the GPU time from the busy chain's end to the last small kernel's end,
// divided by N, is what one kernel slot costs the GPU.  Profiling timestamps only.
// Runs in well under a second; every kernel is tiny or a plain streaming loop.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <functional>
#include <vector>

namespace sx = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;

static uint64_t t_start(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_start>();
}
static uint64_t t_end(const sycl::event& e) {
    return e.get_profiling_info<sycl::info::event_profiling::command_end>();
}

int main() {
    sycl::queue q(sycl::gpu_selector_v,
                  sycl::property_list{sycl::property::queue::in_order{},
                                      sycl::property::queue::enable_profiling{}});
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const size_t BIG = size_t(1) << 26, SM = size_t(1) << 20;
    float* big = sycl::malloc_device<float>(BIG, q);
    float* sm  = sycl::malloc_device<float>(SM, q);
    float* sm2 = sycl::malloc_device<float>(SM, q);
    if (!big || !sm || !sm2) { std::printf("allocation failed\n"); return 1; }
    q.memset(big, 0, BIG * sizeof(float));
    q.memset(sm, 0, SM * sizeof(float));
    q.memset(sm2, 0, SM * sizeof(float)).wait();

    // 256 MB read + write per kernel, ~0.9 ms each
    auto busy = [&](int reps) {
        sycl::event e;
        for (int r = 0; r < reps; ++r)
            e = q.parallel_for(sycl::range<1>(BIG), [=](sycl::id<1> i) {
                big[i] = big[i] * 1.0000001f + 1e-7f;
            });
        return e;
    };

    const int N = 1000;
    struct Case { const char* name; std::function<sycl::event()> sub; };
    std::vector<Case> cases = {
        {"empty parallel_for(1)", [&] {
             return q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {}); }},
        {"64 WG x 256, one float each", [&] {
             return q.parallel_for(sycl::nd_range<1>(64 * 256, 256), [=](sycl::nd_item<1> it) {
                 const size_t g = it.get_global_id(0); sm[g] = sm[g] + 1.0f; }); }},
        {"512 WG x 256, one float each", [&] {
             return q.parallel_for(sycl::nd_range<1>(512 * 256, 256), [=](sycl::nd_item<1> it) {
                 const size_t g = it.get_global_id(0); sm[g] = sm[g] + 1.0f; }); }},
        {"1 WG x 1024, one float each", [&] {
             return q.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
                 const size_t g = it.get_global_id(0); sm[g] = sm[g] + 1.0f; }); }},
        {"memcpy 10 KB device->device", [&] { return q.memcpy(sm2, sm, 10240); }},
    };
    for (auto& c : cases) {
        c.sub(); q.wait();                         // compile + first touch
        const sycl::event b = busy(40);
        const auto h0 = clk::now();
        sycl::event first, last;
        for (int i = 0; i < N; ++i) { sycl::event e = c.sub(); if (i == 0) first = e; last = e; }
        const double host_us = std::chrono::duration<double, std::micro>(clk::now() - h0).count() / N;
        q.wait();
        const double gpu_us = double(t_end(last) - t_end(b)) / 1e3 / N;
        const double first_us = double(t_start(first)) / 1e3 - double(t_end(b)) / 1e3;
        const double k_us = double(t_end(first) - t_start(first)) / 1e3;
        std::printf("%-30s GPU %6.2f us a kernel slot (one kernel's own start->end %5.2f us; "
                    "host submit %5.2f us; first began %6.1f us after the busy chain)\n",
                    c.name, gpu_us, k_us, host_us, first_us);
    }

    // the same 64-WG kernel N times as one recorded graph (sycl_ext_oneapi_graph)
    try {
        sx::command_graph g(q.get_context(), q.get_device());
        g.begin_recording(q);
        for (int i = 0; i < N; ++i)
            q.parallel_for(sycl::nd_range<1>(64 * 256, 256), [=](sycl::nd_item<1> it) {
                const size_t gi = it.get_global_id(0); sm[gi] = sm[gi] + 1.0f; });
        g.end_recording(q);
        auto ex = g.finalize();
        q.ext_oneapi_graph(ex); q.wait();
        const sycl::event b = busy(40);
        const sycl::event ge = q.ext_oneapi_graph(ex);
        q.wait();
        std::printf("%-30s GPU %6.2f us a kernel slot\n", "graph of 1000 x (64 WG x 256)",
                    double(t_end(ge) - t_end(b)) / 1e3 / N);
    } catch (const sycl::exception& e) {
        std::printf("graph case skipped: %s\n", e.what());
    }
    sycl::free(big, q); sycl::free(sm, q); sycl::free(sm2, q);
    std::printf("KGAP_DONE\n");
    return 0;
}
