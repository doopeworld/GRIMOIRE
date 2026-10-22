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

// bar_write_probe.cpp -- can the CPU write the peer's piece straight into this card's VRAM?
//
// gpu1's x4 link (10-22): its device->host copy (posted writes, card->host) and a kernel
// reading the peer's piece from pinned host memory overlap only 24-33% -- the kernel's
// read requests travel card->host too, queued behind the writes.  If the CPU instead
// WRITES the piece into VRAM through the 32 GB BAR, both directions carry posted writes
// only.  Level Zero: a device allocation exported as dma-buf and mmap'd by the CPU of the
// same process -- one GPU driver, no P2P, nothing shared with the other card.  Uses
// SYCL's memory-export extension (opaque fd = the driver's dma-buf), no L0 headers.
//
// No device-side waiting anywhere; every timed loop has an internal time budget; the
// first rep of every variant is 1 MB.
//
//   bar_write_probe [MB=22] [reps=10]
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/memory_export.hpp>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

// CPU -> mapping with 32-byte non-temporal stores (write-combined, full 64-byte lines)
static void nt_copy(void* dst, const void* src, size_t bytes) {
    auto* d = static_cast<__m256i*>(dst);
    auto* s = static_cast<const __m256i*>(src);
    for (size_t i = 0; i < bytes / 32; ++i) _mm256_stream_si256(d + i, _mm256_load_si256(s + i));
    _mm_sfence();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const size_t mb = argc > 1 ? size_t(std::atoll(argv[1])) : 22;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 10;
    const size_t bytes = mb << 20, n = bytes / 4;
    const double budget_ms = 20000;                       // per variant
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    sycl::queue qc{q.get_context(), q.get_device(), sycl::property::queue::in_order{}};
    std::printf("device %s, %zu MB, %d reps\n", q.get_device().get_info<sycl::info::device::name>().c_str(), mb, reps);

    namespace sx = sycl::ext::oneapi::experimental;
    void* dbar = nullptr;
    try {
        dbar = sx::alloc_exportable_device_mem(65536, bytes, sx::external_mem_handle_type::opaque_fd, q);
    } catch (const std::exception& e) { std::printf("alloc_exportable_device_mem failed: %s -- stop\n", e.what()); return 1; }
    int fd = -1;
    try {
        fd = sx::export_device_mem_handle<sx::external_mem_handle_type::opaque_fd>(dbar, q);
    } catch (const std::exception& e) { std::printf("export_device_mem_handle failed: %s -- stop\n", e.what()); return 1; }
    std::printf("export: fd %d\n", fd);
    if (fd < 0) { std::printf("no fd -- stop\n"); return 1; }
    void* cpu = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (cpu == MAP_FAILED) { std::printf("mmap of the exported fd failed: %s -- stop\n", std::strerror(errno)); return 1; }
    std::printf("mmap ok at %p\n", cpu);

    float* dbarf = static_cast<float*>(dbar);
    float* dsrc = sycl::malloc_device<float>(n, q);                // ordinary VRAM, for D2H
    float* hout = sycl::malloc_host<float>(n, q);                  // D2H landing (pinned)
    float* hsrc = static_cast<float*>(std::aligned_alloc(64, bytes));  // CPU source (shm stand-in)
    int* bad = sycl::malloc_shared<int>(1, q);
    for (size_t i = 0; i < n; ++i) hsrc[i] = float(i % 1000003) * 0.5f;
    q.memset(dsrc, 0, bytes).wait();

    // 1) where does the exported allocation live?  VRAM reads ~400+ GB/s, system memory ~6.
    {
        float* acc = sycl::malloc_device<float>(1 << 16, q);
        auto rd = [&](const float* p) {
            const size_t N = n;
            return q.parallel_for(sycl::range<1>(1 << 16), [=](sycl::id<1> id) {
                float s = 0; for (size_t i = id[0]; i < N; i += (1 << 16)) s += p[i]; acc[id] = s; });
        };
        rd(dbarf).wait(); rd(dsrc).wait();
        auto t0 = clk::now(); rd(dbarf).wait(); const double a = ms(t0, clk::now());
        t0 = clk::now(); rd(dsrc).wait(); const double b = ms(t0, clk::now());
        std::printf("GPU read: exported buffer %.1f GB/s, ordinary device buffer %.1f GB/s\n",
                    bytes / 1e6 / a, bytes / 1e6 / b);
        sycl::free(acc, q);
    }
    // 2) CPU write bandwidth into the mapping, 1 MB first
    {
        auto t0 = clk::now(); nt_copy(cpu, hsrc, 1 << 20); const double t1 = ms(t0, clk::now());
        std::printf("CPU -> BAR 1 MB: %.3f ms (%.2f GB/s)\n", t1, (1 << 20) / 1e6 / t1);
        if (t1 > 50) { std::printf("too slow -- stop\n"); return 1; }
        double tot = 0; int done = 0;
        for (; done < reps && tot < budget_ms; ++done) { t0 = clk::now(); nt_copy(cpu, hsrc, bytes); tot += ms(t0, clk::now()); }
        std::printf("CPU -> BAR %zu MB: %.3f ms (%.2f GB/s), %d reps\n", mb, tot / done, bytes / 1e6 / (tot / done), done);
    }
    // 3) does the GPU see exactly what the CPU wrote?
    {
        *bad = 0;
        const float* p = dbarf; int* b = bad; const size_t N = n;
        q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            if (p[i] != float(i[0] % 1000003) * 0.5f) sycl::atomic_ref<int, sycl::memory_order::relaxed,
                sycl::memory_scope::device>(*b).fetch_add(1); }).wait();
        std::printf("GPU check of CPU-written data: %d wrong of %zu\n", *bad, n);
    }
    // 4) GPU D2H copy alone, CPU BAR write alone, both together
    {
        auto d2h = [&] { return qc.memcpy(hout, dsrc, bytes); };
        d2h().wait();
        double tot = 0; int done = 0;
        for (; done < reps && tot < budget_ms; ++done) { auto t0 = clk::now(); d2h().wait(); tot += ms(t0, clk::now()); }
        const double a = tot / done;
        tot = 0; done = 0;
        for (; done < reps && tot < budget_ms; ++done) { auto t0 = clk::now(); nt_copy(cpu, hsrc, bytes); tot += ms(t0, clk::now()); }
        const double b = tot / done;
        tot = 0; done = 0; double tc = 0;
        for (; done < reps && tot < budget_ms; ++done) {
            auto t0 = clk::now();
            sycl::event e = d2h();
            nt_copy(cpu, hsrc, bytes);
            const double tcpu = ms(t0, clk::now());
            e.wait();
            tot += ms(t0, clk::now()); tc += tcpu;
        }
        const double c = tot / done;
        std::printf("D2H alone %.3f ms (%.2f GB/s) | CPU->BAR alone %.3f ms (%.2f GB/s) | together %.3f ms (CPU part %.3f)\n",
                    a, bytes / 1e6 / a, b, bytes / 1e6 / b, c, tc / done);
        std::printf("overlap ratio (1.0 = full duplex, 0 = serialized): %.2f\n", (a + b - c) / std::min(a, b));
    }
    munmap(cpu, bytes);
    close(fd);
    sx::free_exportable_memory(dbar, q);
    sycl::free(dsrc, q); sycl::free(hout, q); sycl::free(bad, q); std::free(hsrc);
    return 0;
}
