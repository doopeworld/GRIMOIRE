// GRIMOIRE -- microbenchmark of the Inkling verify expert kernels (ink_moe_small_esimd, tools/inkling2_xexp.cpp).
// SPDX-License-Identifier: GPL-3.0-or-later
//
//   bench_ink_sm U OWN ITERS [ALU_GU ALU_DN]
// U used experts of 128 (the verify grouping: 8 rows x top-8 = 64 (row, slot) pairs spread over U experts), OWN = -1
// (one card: every used expert) or 0 (two cards: this card's parity half), random NVFP4 payload with unit scales.
// Prints the device time and the payload + scale bandwidth of the gate_up (2304 x 2304) and down (2304 x 1152) calls.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <algorithm>

struct InkSmArgs {
    const uint8_t* pay; const uint8_t* scl; int Ne, K, rows_total;
    const uint16_t* A; int slots;
    void* out;
    const int* gi; int e0, en;
    const float* amax; const float* den;
    const uint32_t* lut;
    float gcap, ucap;
    int U;
    int alu = 1;
    int own = -1;
};
sycl::event ink_moe_small_esimd(sycl::queue& q, const InkSmArgs& g, int fmt, int epi);

int main(int argc, char** argv) {
    const int U = argc > 1 ? std::atoi(argv[1]) : 25;
    const int own = argc > 2 ? std::atoi(argv[2]) : 0;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 200;
    const int alu_gu = argc > 4 ? std::atoi(argv[4]) : 31, alu_dn = argc > 5 ? std::atoi(argv[5]) : 23;
    const int E = 128, h = 2304, I = 1152, TK = 8, M = 8, NT = M * TK;
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    std::printf("%s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const size_t gu_pay = size_t(E) * 2 * I * h / 2, gu_scl = size_t(E) * 2 * I * h / 16;
    const size_t dn_pay = size_t(E) * h * I / 2, dn_scl = size_t(E) * h * I / 16;
    auto* pgu = sycl::malloc_device<uint8_t>(gu_pay, q); auto* sgu = sycl::malloc_device<uint8_t>(gu_scl, q);
    auto* pdn = sycl::malloc_device<uint8_t>(dn_pay, q); auto* sdn = sycl::malloc_device<uint8_t>(dn_scl, q);
    auto fill = [&](uint8_t* p, size_t n, int mode) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            uint32_t x = uint32_t(i[0]) * 2654435761u; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
            p[i[0]] = mode ? uint8_t(0x30 + (x & 7)) : uint8_t(x & 0x77);   // scales: E4M3 ~0.5..0.9; payload: |nibble| < 4
        });
    };
    fill(pgu, gu_pay, 0); fill(sgu, gu_scl, 1); fill(pdn, dn_pay, 0); fill(sdn, dn_scl, 1);
    auto* amax = sycl::malloc_device<float>(E, q); auto* den = sycl::malloc_device<float>(1, q);
    q.fill(amax, 1.0f, E); q.fill(den, 1.0f, 1);
    auto* A = sycl::malloc_device<uint16_t>(size_t(NT) * h, q);
    auto* act = sycl::malloc_device<uint16_t>(size_t(NT) * I, q);
    auto* dbuf = sycl::malloc_device<float>(size_t(NT) * h, q);
    q.fill(A, uint16_t(0x2c00), size_t(NT) * h);   // fp16 0.0625
    q.fill(act, uint16_t(0x2c00), size_t(NT) * I);
    auto* lut = sycl::malloc_device<uint32_t>(256, q);
    q.fill(lut, 0u, 256);
    // grouping table: U distinct experts, the 64 pairs spread as evenly as the verify grouping gives them
    std::vector<int> gi(600, 0), ids(E);
    for (int e = 0; e < E; ++e) ids[e] = e;
    std::mt19937 rng(7);
    std::shuffle(ids.begin(), ids.end(), rng);
    std::vector<int> pick(ids.begin(), ids.begin() + U);
    std::sort(pick.begin(), pick.end());
    gi[0] = U;
    int at = 0;
    for (int u = 0; u < U; ++u) { gi[1 + u] = pick[u]; gi[129 + u] = at; at += NT / U + (u < NT % U ? 1 : 0); }
    gi[129 + U] = at;
    int* gid = sycl::malloc_device<int>(600, q);
    q.memcpy(gid, gi.data(), 600 * sizeof(int)).wait();
    const int mine = own >= 0 ? (U + 1 - own) / 2 : U;
    InkSmArgs ag{pgu, sgu, 2 * I, h, E * 2 * I, A, NT, act, gid, 0, E, amax, den, lut, 30.f, 25.f, U, alu_gu, own};
    InkSmArgs ad{pdn, sdn, h, I, E * h, act, NT, dbuf, gid, 0, E, amax, den, lut, 0.f, 0.f, U, alu_dn, own};
    auto timeit = [&](const InkSmArgs& a, int epi, const char* name, size_t bytes_per_expert) {
        for (int i = 0; i < 10; ++i) ink_moe_small_esimd(q, a, 0, epi);
        q.wait();
        double dev = 0.0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            sycl::event ev = ink_moe_small_esimd(q, a, 0, epi);
            ev.wait();
            dev += double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() - ev.get_profiling_info<sycl::info::event_profiling::command_start>());
        }
        const double wall = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        const double us = dev / iters / 1000.0;
        const double gb = double(bytes_per_expert) * mine / (us * 1e3);
        std::printf("%-8s experts %2d  device %7.1f us  %6.1f GB/s  (wall %.1f us a call incl. wait)\n", name, mine, us, gb, wall);
    };
    timeit(ag, 1, "gate_up", size_t(2 * I) * h / 2 + size_t(2 * I) * h / 16);
    timeit(ad, 2, "down", size_t(h) * I / 2 + size_t(h) * I / 16);
    // back to back (no host wait between): the pair as the verify runs it; median device time of the two kernels
    {
        for (int i = 0; i < 10; ++i) { ink_moe_small_esimd(q, ag, 0, 1); ink_moe_small_esimd(q, ad, 0, 2); }
        q.wait();
        std::vector<sycl::event> e1, e2;
        for (int i = 0; i < iters; ++i) { e1.push_back(ink_moe_small_esimd(q, ag, 0, 1)); e2.push_back(ink_moe_small_esimd(q, ad, 0, 2)); }
        q.wait();
        auto dt = [](sycl::event& e) { return double(e.get_profiling_info<sycl::info::event_profiling::command_end>() - e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0; };
        std::vector<double> g1, g2, pr;
        for (int i = 0; i < iters; ++i) { g1.push_back(dt(e1[i])); g2.push_back(dt(e2[i])); pr.push_back(g1.back() + g2.back()); }
        auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
        const double bytes = double(mine) * (size_t(2 * I) * h / 2 + size_t(2 * I) * h / 16 + size_t(h) * I / 2 + size_t(h) * I / 16);
        std::printf("pair median: gate_up %.1f us, down %.1f us, sum %.1f us = %.1f GB/s\n", med(g1), med(g2), med(pr), bytes / (med(pr) * 1e3));
    }
    return 0;
}
