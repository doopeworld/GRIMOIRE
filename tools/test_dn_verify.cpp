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

// test_dn_verify -- bit-for-bit check of the fused speculative-verify DeltaNet path
// (launch_deltanet_verify_rows + launch_deltanet_replay) against the per-row loop it
// replaces (launch_deltanet_step per row, state updated in place, snapshot after each row).
// Qwen3.8-27B geometry: 48 v-heads, 16 k-heads, k_dim = v_dim = 128; two conversations of
// 7 and 4 rows on two state slots.  Prints mismatching elements (expect 0 everywhere).
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace b70;

int main() {
    const int H = 48, NK = 16, KD = 128, VD = 128;
    const int rows_g[2] = {7, 4};
    const int M = rows_g[0] + rows_g[1];
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    const size_t sn = size_t(H) * VD * KD;
    std::vector<float> st0(2 * sn), hq(size_t(M) * NK * KD), hk(hq.size()), hv(size_t(M) * H * VD),
        ha(size_t(M) * H), hb(size_t(M) * H);
    for (auto& x : st0) x = 0.05f * nd(rng);
    auto l2 = [&](std::vector<float>& t) {
        for (size_t r = 0; r < t.size() / KD; ++r) {
            double s = 0;
            for (int j = 0; j < KD; ++j) { t[r * KD + j] = nd(rng); s += double(t[r * KD + j]) * t[r * KD + j]; }
            for (int j = 0; j < KD; ++j) t[r * KD + j] = float(t[r * KD + j] / std::sqrt(s));
        }
    };
    l2(hq); l2(hk);
    for (auto& x : hv) x = nd(rng);
    for (auto& x : ha) x = 0.8f + 0.2f * ud(rng);
    for (auto& x : hb) x = ud(rng);

    auto dev = [&](const std::vector<float>& h) {
        float* d = sycl::malloc_device<float>(h.size(), q);
        q.memcpy(d, h.data(), h.size() * 4).wait();
        return d;
    };
    float *dq = dev(hq), *dk = dev(hk), *dv = dev(hv), *da = dev(ha), *db = dev(hb);
    float* st_ref = dev(st0);
    float* st_fus = dev(st0);
    float* out_ref = sycl::malloc_device<float>(size_t(M) * H * VD, q);
    float* out_fus = sycl::malloc_device<float>(size_t(M) * H * VD, q);
    std::vector<float> snaps(size_t(M) * sn);   // state after each row (of its own slot)

    // reference: the per-row loop
    for (int r = 0, g = 0, t = 0; r < M; ++r) {
        if (t == rows_g[g]) { ++g; t = 0; }
        DeltaNetParams sp{};
        sp.q = dq + size_t(r) * NK * KD; sp.k = dk + size_t(r) * NK * KD; sp.v = dv + size_t(r) * H * VD;
        sp.a = da + size_t(r) * H; sp.beta = db + size_t(r) * H; sp.out = out_ref + size_t(r) * H * VD;
        sp.state = st_ref + size_t(g) * sn;
        sp.n_heads = H; sp.k_dim = KD; sp.v_dim = VD; sp.n_k_heads = NK;
        launch_deltanet_step(q, sp, {});
        q.memcpy(snaps.data() + size_t(r) * sn, sp.state, sn * 4).wait();
        ++t;
    }
    // fused verify: two groups, state read-only
    DeltaNetParams fp{};
    fp.q = dq; fp.k = dk; fp.v = dv; fp.a = da; fp.beta = db; fp.out = out_fus; fp.state = st_fus;
    fp.n_heads = H; fp.k_dim = KD; fp.v_dim = VD; fp.n_k_heads = NK;
    if (!deltanet_verify_ok(fp)) { std::printf("deltanet_verify_ok = false\n"); return 1; }
    DnGroups g{};
    g.n = 2; g.row0[0] = 0; g.nrows[0] = rows_g[0]; g.slot[0] = 0;
    g.row0[1] = rows_g[0]; g.nrows[1] = rows_g[1]; g.slot[1] = 1;
    const int64_t srow = deltanet_saved_row_floats(fp);
    float* saved = sycl::malloc_device<float>(size_t(M) * srow, q);
    launch_deltanet_verify_rows(q, fp, g, int64_t(sn), saved, {});
    q.wait();

    auto diff = [&](const float* a, const float* b, size_t n, double& mx) {
        size_t bad = 0; mx = 0;
        for (size_t i = 0; i < n; ++i)
            if (std::memcmp(&a[i], &b[i], 4)) { ++bad; mx = std::max(mx, double(std::fabs(a[i] - b[i]))); }
        return bad;
    };
    std::vector<float> o1(size_t(M) * H * VD), o2(o1.size()), s_now(2 * sn);
    q.memcpy(o1.data(), out_ref, o1.size() * 4);
    q.memcpy(o2.data(), out_fus, o2.size() * 4);
    q.memcpy(s_now.data(), st_fus, s_now.size() * 4);
    q.wait();
    double mx;
    size_t bad = diff(o1.data(), o2.data(), o1.size(), mx);
    std::printf("outputs, %d rows: %zu of %zu differ (max |diff| %.3g)\n", M, bad, o1.size(), mx);
    bad = diff(s_now.data(), st0.data(), s_now.size(), mx);
    std::printf("state untouched by the verify: %zu of %zu differ\n", bad, s_now.size());
    // replay every prefix of each group on a copy of its initial state
    size_t worst = 0;
    for (int gi = 0; gi < 2; ++gi)
        for (int n = 1; n <= rows_g[gi]; ++n) {
            float* st = sycl::malloc_device<float>(sn, q);
            q.memcpy(st, st0.data() + size_t(gi) * sn, sn * 4).wait();
            DeltaNetParams rp = fp;
            rp.state = st;
            launch_deltanet_replay(q, rp, saved + size_t(g.row0[gi]) * srow, n, {});
            std::vector<float> got(sn);
            q.memcpy(got.data(), st, sn * 4).wait();
            const size_t b2 = diff(got.data(), snaps.data() + size_t(g.row0[gi] + n - 1) * sn, sn, mx);
            if (b2) std::printf("  group %d replay %d rows: %zu differ (max %.3g)\n", gi, n, b2, mx);
            worst = std::max(worst, b2);
            sycl::free(st, q);
        }
    std::printf("replay of every prefix: %s\n", worst ? "MISMATCH" : "bit-identical to the per-row snapshots");
    return 0;
}
