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

// bench_verify_attn -- speculative-verify attention, Qwen3.8-27B geometry (24 query
// heads, 4 KV heads, head_dim 256, FP8 E4M3 cache): the per-row decode kernel the
// engine runs today (launch_flash_decode_rows + merge) against the matrix-engine kernel
// (launch_verify_attn_dpas + merge), both checked against a double-precision reference.
// Rows r = 0..M-1 sit at consecutive positions; the last one sees L keys.
//   bench_verify_attn [L ...]     (default 512 2048 8192; M = 7 and M = 1 for each)
// Build (grimoire-dev, from the repo root; the kernel library needs -doubleGRF):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -Xsycl-target-backend=intel_gpu_bmg_g31 \
//        "-options -doubleGRF" -O3 -std=c++20 -fPIC -shared -I include -I src \
//        src/attn_verify_dpas.cpp -o bin/libgrimoire_attn.so
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_verify_attn.cpp src/attention.cpp -Lbin -lgrimoire_attn \
//        '-Wl,-rpath,$ORIGIN' -o bin/bench_verify_attn
// MEASURED 2026-10-08, one B70, per layer (old -> new), 7 rows: 512 keys 232 -> 82 us,
// 2K 283 -> 57, 4K 661 -> 86, 8K 1215 -> 128, 16K 3917 -> 231; 1 row: 95 -> 23,
// 70 -> 44, 101 -> 57, 246 -> 99, 478 -> 146.  Max error vs fp64: old ~2e-6, new 2-5e-4.
#include "kernels.hpp"
#include "attn_verify.hpp"
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using namespace b70;

static float e4m3_val(uint8_t b) {
    const uint16_t hb = uint16_t(((b & 0x7F) << 7) | ((b & 0x80) << 8));
    return float(sycl::bit_cast<sycl::half>(hb)) * 256.0f;
}

int main(int argc, char** argv) {
    const int H = 24, KVH = 4, HD = 256, G = H / KVH;
    const float scale = 1.0f / 16.0f;
    std::vector<int> Ls;
    for (int i = 1; i < argc; ++i) Ls.push_back(std::atoi(argv[i]));
    if (Ls.empty()) Ls = {512, 2048, 8192};
    int CAP = 16384;
    for (int L : Ls) CAP = std::max(CAP, (L + 15) / 16 * 16);
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s   GRIMOIRE_VERIFY_KEYS_PER_SPLIT=%s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                std::getenv("GRIMOIRE_VERIFY_KEYS_PER_SPLIT") ? std::getenv("GRIMOIRE_VERIFY_KEYS_PER_SPLIT") : "(unset: thread target)");

    // nearest-E4M3 encoder over the 127 non-negative finite codes
    std::vector<std::pair<float, uint8_t>> codes;
    for (int b = 0; b < 0x7F; ++b) codes.push_back({e4m3_val(uint8_t(b)), uint8_t(b)});
    std::sort(codes.begin(), codes.end());
    auto enc = [&](float x) -> uint8_t {
        const float a = std::fabs(x);
        auto it = std::lower_bound(codes.begin(), codes.end(), std::make_pair(a, uint8_t(0)));
        if (it == codes.end()) it = codes.end() - 1;
        else if (it != codes.begin() && (a - (it - 1)->first) < (it->first - a)) --it;
        return uint8_t(it->second | (x < 0 ? 0x80 : 0));
    };
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<uint8_t> hk(size_t(KVH) * HD * CAP), hv(size_t(KVH) * CAP * HD);
    for (auto& b : hk) b = enc(nd(rng) * 1.5f);
    for (auto& b : hv) b = enc(nd(rng));
    const int MMAX = 7;
    std::vector<float> hq(size_t(MMAX) * H * HD);
    for (auto& x : hq) x = nd(rng);

    const int64_t kv_stride = int64_t(KVH) * HD * CAP;
    uint8_t* dk = sycl::malloc_device<uint8_t>(hk.size(), q);
    uint8_t* dv = sycl::malloc_device<uint8_t>(hv.size(), q);
    float* dq = sycl::malloc_device<float>(hq.size(), q);
    float* dout = sycl::malloc_device<float>(hq.size(), q);
    const size_t pcount = size_t(MMAX) * H * MAX_SPLITS;
    float* dpart = sycl::malloc_device<float>(pcount * HD, q);
    float* dpm = sycl::malloc_device<float>(pcount, q);
    float* dpl = sycl::malloc_device<float>(pcount, q);
    int32_t* dlens = sycl::malloc_device<int32_t>(MMAX, q);
    q.memcpy(dk, hk.data(), hk.size()).wait();
    q.memcpy(dv, hv.data(), hv.size()).wait();
    q.memcpy(dq, hq.data(), hq.size() * 4).wait();

    // decoded caches for the reference
    std::vector<float> kd(hk.size()), vd(hv.size());
    for (size_t i = 0; i < hk.size(); ++i) kd[i] = e4m3_val(hk[i]);
    for (size_t i = 0; i < hv.size(); ++i) vd[i] = e4m3_val(hv[i]);

    AttnParams base{};
    base.q = dq; base.k_cache = dk; base.v_cache = dv; base.out = dout;
    base.seq_len = 1; base.seq_cap = CAP; base.head_dim = HD; base.num_heads = H; base.num_kv_heads = KVH;
    base.softmax_scale = scale; base.partials = dpart; base.part_m = dpm; base.part_l = dpl; base.splits = 8;

    // clocks up before the first timed case
    for (int i = 0; i < 200; ++i) q.memset(dpart, 0, pcount * HD * 4);
    q.wait();
    std::printf("%6s %3s | %10s %10s | %10s %10s | %8s\n", "keys", "M", "old us", "old err", "new us",
                "new err", "speedup");
    for (int L : Ls) {
        for (int M : {7, 1}) {
            std::vector<int32_t> lens(M);
            for (int r = 0; r < M; ++r) lens[r] = L - M + 1 + r;
            q.memcpy(dlens, lens.data(), M * 4).wait();
            // reference (fp64, up to 16K keys; longer contexts are timed only)
            std::vector<double> ref(size_t(M) * H * HD);
            double refmax = 0;
            const bool do_ref = L <= 16384;
            for (int r = 0; do_ref && r < M; ++r)
                for (int h = 0; h < H; ++h) {
                    const int kvh = h / G;
                    const float* qr = &hq[(size_t(r) * H + h) * HD];
                    std::vector<double> sc(lens[r]);
                    double mx = -1e300;
                    for (int k = 0; k < lens[r]; ++k) {
                        double s = 0;
                        for (int d = 0; d < HD; ++d) s += double(qr[d]) * kd[(size_t(kvh) * HD + d) * CAP + k];
                        sc[k] = s * scale;
                        mx = std::max(mx, sc[k]);
                    }
                    double den = 0;
                    for (auto& s : sc) { s = std::exp(s - mx); den += s; }
                    double* o = &ref[(size_t(r) * H + h) * HD];
                    for (int k = 0; k < lens[r]; ++k) {
                        const float* vr = &vd[(size_t(kvh) * CAP + k) * HD];
                        for (int d = 0; d < HD; ++d) o[d] += sc[k] / den * vr[d];
                    }
                    for (int d = 0; d < HD; ++d) refmax = std::max(refmax, std::fabs(o[d]));
                }
            auto err = [&]() {
                if (!do_ref) return -1.0;
                std::vector<float> got(size_t(M) * H * HD);
                q.memcpy(got.data(), dout, got.size() * 4).wait();
                double e = 0;
                for (size_t i = 0; i < got.size(); ++i) e = std::max(e, std::fabs(got[i] - ref[i]));
                return e / refmax;
            };
            auto timeit = [&](auto&& fn) {
                for (int i = 0; i < 3; ++i) fn();
                q.wait();
                const int N = 50;
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < N; ++i) fn();
                q.wait();
                return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / N;
            };
            // old: per-row decode kernel + merge
            RowSlots rs{};
            int smax_old = 1;
            for (int r = 0; r < M; ++r) {
                rs.slot[r] = 0; rs.pos[r] = lens[r] - 1;
                smax_old = std::max(smax_old, flash_rows_splits(lens[r], 8));
            }
            q.memset(dout, 0, hq.size() * 4).wait();
            const double t_old = timeit([&] {
                launch_flash_decode_rows(q, base, M, rs, kv_stride, dlens, smax_old);
                launch_flash_merge_rows(q, base, M, dlens, smax_old);
            });
            const double e_old = err();
            // new: matrix-engine kernel + merge
            VerifyGroups vg{};
            vg.n = 1; vg.row0[0] = 0; vg.nrows[0] = M; vg.slot[0] = 0;
            const int smax_new = verify_dpas_splits(base, vg, L);
            q.memset(dout, 0, hq.size() * 4).wait();
            const double t_new = timeit([&] {
                launch_verify_attn_dpas(q, base, vg, kv_stride, dlens, smax_new);
                launch_verify_merge(q, base, M, smax_new);
            });
            const double e_new = err();
            std::printf("%6d %3d | %10.1f %10.2e | %10.1f %10.2e | %7.1fx   (splits old %d new %d)\n", L, M, t_old,
                        e_old, t_new, e_new, t_old / t_new, smax_old, smax_new);
        }
    }
    return 0;
}
