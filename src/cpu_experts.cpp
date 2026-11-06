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
//  cpu_experts.cpp -- see b70/cpu_experts.hpp.
//
//  Host code only.  The whole file is hidden from the SYCL device pass
//  (x86 intrinsics have no meaning there); the functions carry their own
//  avx2/fma target so the rest of the engine keeps its baseline ISA.
// =====================================================================
#include "b70/cpu_experts.hpp"
#include "b70/formats.hpp"

#ifndef __SYCL_DEVICE_ONLY__
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

namespace b70 {
namespace {

// E4M3 value * 0.5: the shuffle table below holds E2M1 values times two
// (so they are exact int8), and the half goes back in with the scale.
struct E4M3Half {
    float v[256];
    E4M3Half() { for (int i = 0; i < 256; ++i) v[i] = 0.5f * e4m3_to_f32(uint8_t(i)); }
};
const float* e4m3_half() { static const E4M3Half t; return t.v; }

__attribute__((target("avx2,fma")))
inline float hsum8(__m256 v) {
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    __m128 sh = _mm_movehdup_ps(lo);
    __m128 s  = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, s);
    s  = _mm_add_ss(s, sh);
    return _mm_cvtss_f32(s);
}

} // namespace

// 64 elements per step: 32 payload bytes -> low/high nibbles -> int8 via
// pshufb -> interleaved back into element order by unpack (per 128-bit
// lane: unpacklo gives elements 0..15 / 32..47, unpackhi 16..31 / 48..63),
// four 16-element blocks, each scaled once by its E4M3 scale.
__attribute__((target("avx2,fma")))
float cpu_nvfp4_dot(const uint8_t* pay, const uint8_t* scl, const float* x, int K) {
    const __m256i lut = _mm256_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
                                         0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const float* sc = e4m3_half();
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int k = 0; k < K; k += 64) {
        const __m256i v  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pay + k / 2));
        const __m256i wl = _mm256_shuffle_epi8(lut, _mm256_and_si256(v, m4));
        const __m256i wh = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(v, 4), m4));
        const __m256i e0 = _mm256_unpacklo_epi8(wl, wh);
        const __m256i e1 = _mm256_unpackhi_epi8(wl, wh);
        const float* xk = x + k;
        const uint8_t* s = scl + k / 16;
#define B70_NV_BLOCK(b128, off, sidx, acc) {                                            \
            const __m128i bb = (b128);                                                 \
            const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bb));            \
            const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bb, 8))); \
            const __m256 p  = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xk + (off) + 8),     \
                                              _mm256_mul_ps(f0, _mm256_loadu_ps(xk + (off)))); \
            acc = _mm256_fmadd_ps(p, _mm256_set1_ps(sc[s[sidx]]), acc); }
        B70_NV_BLOCK(_mm256_castsi256_si128(e0), 0, 0, acc0)
        B70_NV_BLOCK(_mm256_castsi256_si128(e1), 16, 1, acc1)
        B70_NV_BLOCK(_mm256_extracti128_si256(e0, 1), 32, 2, acc0)
        B70_NV_BLOCK(_mm256_extracti128_si256(e1, 1), 48, 3, acc1)
#undef B70_NV_BLOCK
    }
    return hsum8(_mm256_add_ps(acc0, acc1));
}

namespace {
void pin_self(int cpu) {
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

std::vector<int> read_cpu_list(const char* path) {      // "0-5", "6-13,16"
    std::vector<int> v;
    std::ifstream f(path);
    std::string s;
    if (!(f >> s)) return v;
    std::stringstream ss(s);
    for (std::string part; std::getline(ss, part, ',');) {
        const size_t d = part.find('-');
        const int a = std::atoi(part.c_str());
        const int b = d == std::string::npos ? a : std::atoi(part.c_str() + d + 1);
        for (int c = a; c <= b; ++c) v.push_back(c);
    }
    return v;
}
} // namespace

// CpuExperts::state_: generation << 17 | job open << 16 | workers inside the job
constexpr uint64_t CX_OPEN = uint64_t(1) << 16, CX_INSIDE = CX_OPEN - 1;
constexpr int CX_GEN_SHIFT = 17;

static_assert(sizeof(cpu_set_t) <= 128, "ScopedCpuPin keeps the old mask in 128 bytes");

ScopedCpuPin::ScopedCpuPin(int cpu) {
    if (cpu < 0) return;
    cpu_set_t old;
    if (pthread_getaffinity_np(pthread_self(), sizeof old, &old) != 0) return;
    std::memcpy(old_, &old, sizeof old);
    cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s);
    on_ = pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
}

ScopedCpuPin::~ScopedCpuPin() {
    if (!on_) return;
    cpu_set_t old;
    std::memcpy(&old, old_, sizeof old);
    pthread_setaffinity_np(pthread_self(), sizeof old, &old);
}

void unpin_self() {
    cpu_set_t s; CPU_ZERO(&s);
    const int n = int(std::thread::hardware_concurrency());
    for (int c = 0; c < n && c < CPU_SETSIZE; ++c) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
}

std::vector<int> hybrid_pool_cpus(int ranks, int rank) {
    const std::vector<int> p = read_cpu_list("/sys/devices/cpu_core/cpus");
    const std::vector<int> e = read_cpu_list("/sys/devices/cpu_atom/cpus");
    std::vector<int> out;
    if (p.empty() || e.empty() || ranks < 1 || rank < 0 || rank >= ranks) return out;
    const std::vector<int> lp = read_cpu_list("/sys/devices/cpu_lowpower/cpus");
    auto share = [&](const std::vector<int>& all) {
        std::vector<int> v;
        for (int c : all) if (std::find(lp.begin(), lp.end(), c) == lp.end()) v.push_back(c);
        const size_t b = v.size() * size_t(rank) / size_t(ranks), en = v.size() * size_t(rank + 1) / size_t(ranks);
        for (size_t i = b; i < en; ++i) out.push_back(v[i]);
    };
    share(p);
    share(e);
    return out;
}

void CpuExperts::start(int threads, const std::vector<int>& cpus) {
    stop();
    if (threads <= 0) threads = std::max(1, int(std::thread::hardware_concurrency()) - 1);
    nthreads_ = threads;
    quit_ = false;
    // Only the pool's own threads are pinned: the caller (worker 0) is not,
    // or every thread it starts later would inherit the one-CPU mask.  cpus[0]
    // stays free for it.
    cpus_ = cpus;
    // The generation every worker starts from is read HERE, not in the new
    // thread: a worker that only got going after the first run() opened a job
    // would take that job's generation as already seen (10-27, cpu_expert_bench
    // with 14 threads: the caller then waited for it forever).
    const uint64_t g0 = state_.load(std::memory_order_acquire) >> CX_GEN_SHIFT;
    for (int t = 1; t < threads; ++t) workers_.emplace_back(&CpuExperts::worker, this, t, g0);
}

void CpuExperts::stop() {
    if (!workers_.empty()) {
        quit_ = true;
        for (auto& t : workers_) t.join();
        workers_.clear();
    }
    nthreads_ = 0;
}

// Spin while calls arrive every few hundred microseconds (decode), yield
// for a while after the last one, then back off to short sleeps so an idle
// server does not burn a core per worker.
void CpuExperts::worker(int tid, uint64_t seen) {
    if (tid < int(cpus_.size())) pin_self(cpus_[size_t(tid)]);
    using clk = std::chrono::steady_clock;
    auto last = clk::now();
    for (;;) {
        uint64_t s;
        int spins = 0;
        for (;;) {
            s = state_.load(std::memory_order_acquire);
            if ((s & CX_OPEN) && (s >> CX_GEN_SHIFT) != seen) {
                // Count in first, then read whether the job was still open, so run() knows who is inside
                // when it closes.  One step with no retry: 13 workers retrying a compare-exchange here cost
                // 6 us of every 85 us call on a quiet machine.  One that came too late counts out again.
                s = state_.fetch_add(1, std::memory_order_acq_rel);
                if ((s & CX_OPEN) && (s >> CX_GEN_SHIFT) != seen) break;
                state_.fetch_sub(1, std::memory_order_release);
            }
            if (quit_.load(std::memory_order_acquire)) return;
            if (++spins < 4096) { _mm_pause(); continue; }
            spins = 0;
            if (clk::now() - last < std::chrono::milliseconds(50)) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
        seen = s >> CX_GEN_SHIFT;
        (*job_)(tid);
        last = clk::now();
        state_.fetch_sub(1, std::memory_order_release);
    }
}

// The caller works through every item of the job itself, so a job never needs a given worker: one that is
// off its core when the job runs misses it and nothing waits for it.  Only workers inside hold items, and
// run() waits for those.  With every worker counted, one other busy process on the machine left the caller
// waiting for the last worker 96% of each call (11-02, cpu_expert_bench: 3,225 of 3,347 us).
void CpuExperts::run(const std::function<void(int)>& job) {
    job_ = &job;
    // Next generation, open.  Adds, because a worker that came too late may be counted in for a moment.
    state_.fetch_add((uint64_t(1) << CX_GEN_SHIFT) + CX_OPEN, std::memory_order_acq_rel);
    job(0);
    state_.fetch_and(~CX_OPEN, std::memory_order_acq_rel);
    const auto t = std::chrono::steady_clock::now();
    while (state_.load(std::memory_order_acquire) & CX_INSIDE) _mm_pause();
    wait_us_ += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t).count();
}

void CpuExperts::ffn(const NvExpertLayout& L, const uint8_t* const* blocks, const float* w,
                     int n, const float* x, float* y) {
    const int H = L.H, I = L.I;
    if (n <= 0 || nthreads_ <= 0) { std::fill(y, y + H, 0.0f); return; }
    h_.resize(size_t(n) * I);
    float* hb = h_.data();
    // Work is handed out in small chunks from a shared counter, not split
    // evenly: the Tower's cores are 6 P + 10 E, and an even split left the
    // P-cores idle behind the E-cores (0.76 ms/layer = 28 GB/s measured).
    // Pass 1: one item = 16 (gate row, up row) pairs of one expert, which
    // also applies silu(g) * u, so no serial step sits between the passes.
    constexpr int P1 = 16, P2 = 32;
    const int per_e = I / P1, n1 = n * per_e, n2 = (H + P2 - 1) / P2;
    std::atomic<int> next{0};
    run([&](int) {
        for (int c; (c = next.fetch_add(1, std::memory_order_relaxed)) < n1;) {
            const int j = c / per_e, i0 = (c % per_e) * P1;
            const uint8_t* b = blocks[j];
            const float* g3 = reinterpret_cast<const float*>(b + L.gsc);
            for (int i = i0; i < i0 + P1; ++i) {
                const float gv = g3[0] * cpu_nvfp4_dot(b + L.gu_p + size_t(i) * (H / 2),
                                                       b + L.gu_s + size_t(i) * (H / 16), x, H);
                const float uv = g3[1] * cpu_nvfp4_dot(b + L.gu_p + size_t(I + i) * (H / 2),
                                                       b + L.gu_s + size_t(I + i) * (H / 16), x, H);
                hb[size_t(j) * I + i] = gv / (1.0f + std::exp(-gv)) * uv;
            }
        }
    });
    // Pass 2: one item = 32 outputs, all experts, one expert's rows at a time.
    next.store(0, std::memory_order_relaxed);
    run([&](int) {
        for (int c; (c = next.fetch_add(1, std::memory_order_relaxed)) < n2;) {
            const int o0 = c * P2, o1 = std::min(H, o0 + P2);
            float acc[P2] = {};
            for (int j = 0; j < n; ++j) {
                const uint8_t* b = blocks[j];
                const float f = w[j] * reinterpret_cast<const float*>(b + L.gsc)[2];
                const float* hj = hb + size_t(j) * I;
                for (int o = o0; o < o1; ++o)
                    acc[o - o0] += f * cpu_nvfp4_dot(b + L.dn_p + size_t(o) * (I / 2),
                                                     b + L.dn_s + size_t(o) * (I / 16), hj, I);
            }
            for (int o = o0; o < o1; ++o) y[o] = acc[o - o0];
        }
    });
}

// One call for several rows (11-06, the Flash-Next MTP verify: tiered_moe_cpu_rows called ffn()
// once per candidate row, each call two pool jobs, and re-read from RAM every expert two rows share).
// Pass 1 walks the DISTINCT experts: each gate / up row is dotted with the input of every row that
// routes to that expert while it is in cache.  Pass 2 walks the outputs: for a chunk of 32, every row,
// its experts in its own order -- ffn()'s sum for that row, operation for operation.
void CpuExperts::ffn_rows(const NvExpertLayout& L, int rows, const uint8_t* const* blocks,
                          const float* w, const int* off, const float* x, float* y) {
    const int H = L.H, I = L.I;
    if (rows <= 0) return;
    const int total = off[rows];
    if (total <= 0 || nthreads_ <= 0) { std::fill(y, y + size_t(rows) * H, 0.0f); return; }
    fr_.resize(size_t(total));
    ui_.resize(size_t(total));
    ub_.clear();
    for (int r = 0; r < rows; ++r)
        for (int f = off[r]; f < off[r + 1]; ++f) {
            fr_[size_t(f)] = r;
            int u = 0;
            while (u < int(ub_.size()) && ub_[size_t(u)] != blocks[f]) ++u;
            if (u == int(ub_.size())) ub_.push_back(blocks[f]);
            ui_[size_t(f)] = u;
        }
    const int nu = int(ub_.size());
    uo_.assign(size_t(nu) + 1, 0);              // each distinct expert's slots, in slot order
    for (int f = 0; f < total; ++f) ++uo_[size_t(ui_[size_t(f)]) + 1];
    for (int u = 0; u < nu; ++u) uo_[size_t(u) + 1] += uo_[size_t(u)];
    ul_.resize(size_t(total));
    {
        std::vector<int> at(uo_.begin(), uo_.end() - 1);
        for (int f = 0; f < total; ++f) ul_[size_t(at[size_t(ui_[size_t(f)])]++)] = f;
    }
    h_.resize(size_t(total) * I);
    float* hb = h_.data();
    const int* fr = fr_.data(); const int* uo = uo_.data(); const int* ul = ul_.data();
    const uint8_t* const* ub = ub_.data();
    constexpr int P1 = 16, P2 = 32;
    const int per_e = I / P1, n1 = nu * per_e, n2 = (H + P2 - 1) / P2;
    std::atomic<int> next{0};
    run([&](int) {
        for (int c; (c = next.fetch_add(1, std::memory_order_relaxed)) < n1;) {
            const int u = c / per_e, i0 = (c % per_e) * P1;
            const uint8_t* b = ub[u];
            const float* g3 = reinterpret_cast<const float*>(b + L.gsc);
            for (int i = i0; i < i0 + P1; ++i)
                for (int s = uo[u]; s < uo[u + 1]; ++s) {
                    const int f = ul[s];
                    const float* xr = x + size_t(fr[f]) * H;
                    const float gv = g3[0] * cpu_nvfp4_dot(b + L.gu_p + size_t(i) * (H / 2),
                                                           b + L.gu_s + size_t(i) * (H / 16), xr, H);
                    const float uv = g3[1] * cpu_nvfp4_dot(b + L.gu_p + size_t(I + i) * (H / 2),
                                                           b + L.gu_s + size_t(I + i) * (H / 16), xr, H);
                    hb[size_t(f) * I + i] = gv / (1.0f + std::exp(-gv)) * uv;
                }
        }
    });
    next.store(0, std::memory_order_relaxed);
    run([&](int) {
        for (int c; (c = next.fetch_add(1, std::memory_order_relaxed)) < n2;) {
            const int o0 = c * P2, o1 = std::min(H, o0 + P2);
            for (int r = 0; r < rows; ++r) {
                float acc[P2] = {};
                for (int j = off[r]; j < off[r + 1]; ++j) {
                    const uint8_t* b = blocks[j];
                    const float f = w[j] * reinterpret_cast<const float*>(b + L.gsc)[2];
                    const float* hj = hb + size_t(j) * I;
                    for (int o = o0; o < o1; ++o)
                        acc[o - o0] += f * cpu_nvfp4_dot(b + L.dn_p + size_t(o) * (I / 2),
                                                         b + L.dn_s + size_t(o) * (I / 16), hj, I);
                }
                float* yr = y + size_t(r) * H;
                for (int o = o0; o < o1; ++o) yr[o] = acc[o - o0];
            }
        }
    });
}

} // namespace b70
#endif // !__SYCL_DEVICE_ONLY__
