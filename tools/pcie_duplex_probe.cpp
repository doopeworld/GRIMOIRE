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

// pcie_duplex_probe.cpp -- does this card overlap the two PCIe directions?
//
// The TP prefill all-reduce moves each piece device->host on a copy queue while
// an add kernel reads the peer's piece host->device straight from pinned memory.
// If both run together the link is full duplex for us; if the time is the sum,
// something serializes them.  No device-side waiting anywhere: plain copies and
// plain kernels, each timed with q.wait(), so nothing can hang.
//
//   pcie_duplex_probe [MB=22] [reps=20]
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t mb = argc > 1 ? size_t(std::atoll(argv[1])) : 22;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const size_t bytes = mb << 20, n = bytes / 2;           // bf16 elements
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue qc{q.get_context(), q.get_device(), sycl::property::queue::in_order{}};
    std::printf("device %s, %zu MB per direction, %d reps\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), mb, reps);
    auto* dsrc = sycl::malloc_device<uint16_t>(n, q);
    auto* ddst = sycl::malloc_device<float>(n, q);
    auto* hout = sycl::malloc_host<uint16_t>(n, q);
    auto* hin = sycl::malloc_host<uint16_t>(n, q);
    q.memset(dsrc, 0, bytes); q.memset(ddst, 0, n * 4);
    for (size_t i = 0; i < n; ++i) hin[i] = uint16_t(0x3f80);  // bf16 1.0
    q.wait();
    auto d2h = [&] { return qc.memcpy(hout, dsrc, bytes); };
    auto zc_read = [&] {   // add kernel reading the peer piece from pinned memory (as tp_pipe_allreduce)
        const uint16_t* src = hin; float* dst = ddst;
        return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            dst[i] += sycl::bit_cast<float>(uint32_t(src[i]) << 16); });
    };
    auto h2d = [&] { return q.memcpy(dsrc, hin, bytes); };
    auto run = [&](const char* tag, auto&& f) {
        f(); q.wait(); qc.wait();                           // warm
        const auto t0 = clk::now();
        for (int r = 0; r < reps; ++r) { f(); q.wait(); qc.wait(); }
        const double t = ms(t0, clk::now()) / reps;
        std::printf("  %-36s %7.3f ms  (%5.2f GB/s per %zu MB)\n", tag, t, double(bytes) / 1e6 / t, mb);
        return t;
    };
    const double a = run("D2H copy (copy queue)", [&] { d2h(); });
    const double b = run("zero-copy read kernel (host->dev)", [&] { zc_read(); });
    const double c = run("H2D copy (main queue)", [&] { h2d(); });
    const double ab = run("D2H copy + zero-copy read together", [&] { d2h(); zc_read(); });
    const double ac = run("D2H copy + H2D copy together", [&] { d2h(); h2d(); });
    std::printf("overlap ratio (1.0 = full duplex, 0 = serialized):  read %.2f   copy %.2f\n",
                (a + b - ab) / std::min(a, b), (a + c - ac) / std::min(a, c));
    // Rate-limited device->host: a kernel with only `wgs` work-groups writes the bf16
    // piece straight into pinned memory (as a fused pack+send would), concurrently with
    // the zero-copy read kernel on the other queue.
    for (int wgs : {4, 8, 16, 32, 64}) {
        auto kd2h = [&] {
            const uint16_t* src = dsrc; uint16_t* dst = hout; const size_t N = n; const int W = wgs;
            return qc.parallel_for(sycl::nd_range<1>(size_t(W) * 256, 256), [=](sycl::nd_item<1> it) {
                const size_t stride = size_t(W) * 256;
                for (size_t i = it.get_global_id(0); i < N; i += stride) dst[i] = src[i]; });
        };
        char t1[64], t2[64];
        std::snprintf(t1, sizeof t1, "kernel D2H, %d WGs", wgs);
        std::snprintf(t2, sizeof t2, "kernel D2H %d WGs + zero-copy read", wgs);
        const double x = run(t1, [&] { kd2h(); });
        const double y = run(t2, [&] { kd2h(); zc_read(); });
        std::printf("     -> overlap %.2f, combined %.3f ms vs serialized %.3f\n", (x + b - y) / std::min(x, b), y, x + b);
    }
    sycl::free(dsrc, q); sycl::free(ddst, q); sycl::free(hout, q); sycl::free(hin, q);
    return 0;
}
