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
//  test_cpu_ffn_rows.cpp -- CpuExperts::ffn_rows (several rows in one
//  call) against ffn() row by row: the outputs must be BIT-IDENTICAL.
//  Flash-Next-sized NVFP4 experts (H 2560, I 640) with random payloads,
//  rows that share experts, rows with none.  Host only, no GPU.
//
//    test_cpu_ffn_rows [threads] [reps]
//  (the times it prints are this machine's, not the Tower's)
// =====================================================================
#include "b70/cpu_experts.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int threads = argc > 1 ? std::atoi(argv[1]) : 0;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int H = 2560, I = 640, E = 24;
    const b70::NvExpertLayout L = b70::NvExpertLayout::make(H, I);
    std::mt19937 rng(1234);
    std::vector<uint8_t> pool(static_cast<size_t>(E) * L.bytes, 0);
    for (int e = 0; e < E; ++e) {
        uint8_t* b = pool.data() + static_cast<size_t>(e) * L.bytes;
        for (size_t i = L.gu_p; i < L.gu_s; ++i) b[i] = uint8_t(rng());
        for (size_t i = L.gu_s; i < L.dn_p; ++i) b[i] = uint8_t(0x28 + rng() % 24);   // E4M3 0.25 .. 1.9
        for (size_t i = L.dn_p; i < L.dn_s; ++i) b[i] = uint8_t(rng());
        for (size_t i = L.dn_s; i < L.gsc; ++i) b[i] = uint8_t(0x28 + rng() % 24);
        const float g3[3] = {0.02f + 0.01f * float(e % 5), 0.03f, 0.05f};
        std::memcpy(b + L.gsc, g3, sizeof g3);
    }
    auto blk = [&](int e) { return static_cast<const uint8_t*>(pool.data() + static_cast<size_t>(e) * L.bytes); };

    b70::CpuExperts cx;
    cx.start(threads);
    std::printf("pool: %d threads, %d experts of %.2f MB\n", cx.threads(), E, double(L.bytes) / 1e6);

    struct Case { const char* name; std::vector<std::vector<int>> routes; };
    std::vector<Case> cases;
    cases.push_back({"4 rows, 2 shared experts", {{0, 1, 2, 3}, {0, 1, 4, 5}, {6, 0, 7}, {1, 8}}});
    cases.push_back({"4 rows, one with none", {{3, 9}, {}, {9, 10, 3}, {11}}});
    cases.push_back({"1 row", {{5, 6, 7, 8, 9}}});
    cases.push_back({"4 rows, none at all", {{}, {}, {}, {}}});
    {
        Case c{"16 rows, 0-10 of 24 experts each", {}};
        for (int r = 0; r < 16; ++r) {
            std::vector<int> v;
            const int n = int(rng() % 11);
            while (int(v.size()) < n) {
                const int e = int(rng() % E);
                bool dup = false;
                for (int x : v) dup = dup || x == e;
                if (!dup) v.push_back(e);
            }
            c.routes.push_back(v);
        }
        cases.push_back(c);
    }

    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    for (const Case& c : cases) {
        const int R = int(c.routes.size());
        std::vector<float> x(static_cast<size_t>(R) * H), yref(x.size(), -1.0f), y(x.size(), -2.0f);
        for (float& v : x) v = nd(rng);
        std::vector<const uint8_t*> blocks;
        std::vector<float> w;
        std::vector<int> off(static_cast<size_t>(R) + 1, 0);
        for (int r = 0; r < R; ++r) {
            off[static_cast<size_t>(r)] = int(blocks.size());
            for (int e : c.routes[static_cast<size_t>(r)]) {
                blocks.push_back(blk(e));
                w.push_back(0.05f + 0.1f * float(rng() % 9));
            }
        }
        off[static_cast<size_t>(R)] = int(blocks.size());
        auto per_row = [&]() {
            for (int r = 0; r < R; ++r) {
                const int o = off[static_cast<size_t>(r)], n = off[static_cast<size_t>(r) + 1] - o;
                cx.ffn(L, blocks.data() + o, w.data() + o, n, x.data() + static_cast<size_t>(r) * H,
                       yref.data() + static_cast<size_t>(r) * H);
            }
        };
        auto grouped = [&]() { cx.ffn_rows(L, R, blocks.data(), w.data(), off.data(), x.data(), y.data()); };
        per_row();
        grouped();
        const bool same = std::memcmp(y.data(), yref.data(), y.size() * sizeof(float)) == 0;
        double maxd = 0.0, maxr = 0.0;
        for (size_t i = 0; i < y.size(); ++i) {
            maxd = std::max(maxd, double(std::fabs(y[i] - yref[i])));
            maxr = std::max(maxr, double(std::fabs(yref[i])));
        }
        auto time_us = [&](const auto& f) {
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) f();
            return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        const double t_row = time_us(per_row), t_grp = time_us(grouped);
        std::printf("%-36s %s  max|diff| %.3g (max|y| %.3g)  row by row %.0f us, one call %.0f us\n",
                    c.name, same ? "IDENTICAL" : "DIFFERS", maxd, maxr, t_row, t_grp);
        if (!same) ++fails;
    }
    cx.stop();
    std::printf("%s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
