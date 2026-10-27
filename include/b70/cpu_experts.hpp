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

// =====================================================================
//  cpu_experts.hpp -- NVFP4 routed experts computed by the HOST CPU.
//
//  Decode reads each routed expert once per token.  For an expert that
//  lives in system RAM the CPU reads it at DRAM speed (67 GB/s measured
//  on the Tower, 16 threads) while the GPU could only pull it over PCIe
//  (12.8 GB/s).  So during decode the GPU runs the VRAM-resident experts
//  and this pool runs the RAM-resident ones at the same time; the two
//  partial sums are added on the device.
//
//  Exact fp32 arithmetic on the checkpoint's NVFP4 values (no activation
//  quantization): E2M1 nibbles -> int8 (x2) through a 16-entry shuffle
//  table, one E4M3 scale per 16 elements, the projection's F32 scale per
//  row.  Same block layout as the GPU kernels (b70/tiered_moe.hpp).
// =====================================================================
#ifndef B70_CPU_EXPERTS_HPP
#define B70_CPU_EXPERTS_HPP

#include "b70/tiered_moe.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace b70 {

class CpuExperts {
public:
    ~CpuExperts() { stop(); }
    // threads <= 0: hardware_concurrency - 1.  The calling thread is worker 0.
    // cpus: the CPU each pool thread t >= 1 is pinned to (cpus[t]; cpus[0] is
    // the calling thread's, which is left unpinned); empty = not pinned.
    void start(int threads = 0, const std::vector<int>& cpus = {});
    void stop();
    bool running() const { return !workers_.empty() || nthreads_ == 1; }
    int  threads() const { return nthreads_; }

    // y[H] = sum_j w[j] * down_j(silu(gate_j x) * up_j x) for the n experts
    // in blocks[] (host memory).  x is H floats.  Overwrites y.
    void ffn(const NvExpertLayout& L, const uint8_t* const* blocks, const float* w,
             int n, const float* x, float* y);

private:
    void run(const std::function<void(int)>& job);   // all threads, then join
    void worker(int tid);
    int nthreads_ = 0;
    std::vector<int> cpus_;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> left_{0};
    std::atomic<bool> quit_{false};
    const std::function<void(int)>* job_ = nullptr;
    std::vector<float> gu_, h_;          // per-call scratch
};

// One NVFP4 row (K elements, K % 64 == 0) dotted with x.  Exposed for tests.
float cpu_nvfp4_dot(const uint8_t* pay, const uint8_t* scl, const float* x, int K);

// The CPUs for rank `rank` of `ranks` processes sharing this host's pool on
// a hybrid CPU (/sys/devices/cpu_core, cpu_atom): its share of the P-cores,
// then its share of the E-cores; the low-power E-cores (cpu_lowpower) are
// left out.  Empty when the CPU is not hybrid or sysfs is not readable.
std::vector<int> hybrid_pool_cpus(int ranks, int rank);

} // namespace b70
#endif
