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

// test_conv_norm -- bit-for-bit checks of two 2026-10-09 verify-path changes:
//  1. the DeltaNet conv over a conversation's 7 verify rows in ONE
//     launch_causal_conv1d_split_prefill call, against one call per row (outputs), and
//     the commit's ring rebuild (pre-verify ring + accepted rows' inputs) against the
//     per-row ring snapshots, for every accepted count;
//  2. launch_rmsnorm_residual_batched (registers) against a verbatim copy of the loop
//     kernel it replaced, M = 7 and 33, with and without residuals.
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
using namespace b70;

// the kernel launch_rmsnorm_residual_batched ran before 2026-10-09 (prefill.cpp)
static sycl::event old_norm(sycl::queue& q, float* h, const float* r0, const float* r1,
                            const bf16_t* weight, float* out, int tokens, int hidden, float eps,
                            sycl_bf16* out_bf, float weight_offset) {
    constexpr int WG = 256;
    return q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> partial(WG / SG_SIZE, hd);
        hd.parallel_for(
            sycl::nd_range<1>(size_t(tokens) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
                const auto sg = it.get_sub_group();
                const int t = int(it.get_group(0));
                const int lid = int(it.get_local_id(0));
                const int sgid = int(sg.get_group_id()[0]);
                const int lane = int(sg.get_local_id()[0]);
                float* ht = h + int64_t(t) * hidden;
                float* ot = out ? out + int64_t(t) * hidden : nullptr;
                sycl_bf16* obt = out_bf ? out_bf + int64_t(t) * hidden : nullptr;
                const float* a = r0 ? r0 + int64_t(t) * hidden : nullptr;
                const float* b = r1 ? r1 + int64_t(t) * hidden : nullptr;
                float ss = 0.0f;
                for (int i = lid; i < hidden; i += WG) {
                    float v = ht[i];
                    if (a) v += a[i];
                    if (b) v += b[i];
                    if (a || b) ht[i] = v;
                    ss = sycl::fma(v, v, ss);
                }
                ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
                float* pt = partial.template get_multi_ptr<sycl::access::decorated::no>().get();
                if (lane == 0) pt[sgid] = ss;
                sycl::group_barrier(it.get_group());
                float total = 0.0f;
                for (int i = 0; i < WG / SG_SIZE; ++i) total += pt[i];
                const float scale = sycl::rsqrt(total / float(hidden) + eps);
                for (int i = lid; i < hidden; i += WG) {
                    const float v=ht[i]*scale*(weight_offset+bf16_to_f32(weight[i]));
                    if(ot)ot[i]=v;if(obt)obt[i]=sycl_bf16(v);
                }
            });
    });
}

template <typename T>
static size_t ndiff(const std::vector<T>& a, const std::vector<T>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto bf = [](float f) { uint32_t u; std::memcpy(&u, &f, 4); return bf16_t(u >> 16); };
    auto up = [&](const auto& h) {
        using T = typename std::decay_t<decltype(h)>::value_type;
        T* d = sycl::malloc_device<T>(h.size(), q);
        q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
        return d;
    };
    auto down = [&](auto* d, size_t n) {
        using T = std::remove_cv_t<std::remove_pointer_t<decltype(d)>>;
        std::vector<T> h(n);
        q.memcpy(h.data(), d, n * sizeof(T)).wait();
        return h;
    };

    // ---- 1. conv: Qwen3.8 linear-attention geometry ----
    const int QS = 16 * 128, VS = 48 * 128, C = 2 * QS + VS, KER = 4, HIST = KER - 1, M = 7;
    std::vector<float> x(size_t(M) * C), ring0(size_t(C) * HIST);
    std::vector<bf16_t> w(size_t(C) * KER);
    for (auto& v : x) v = nd(rng);
    for (auto& v : ring0) v = nd(rng);
    for (auto& v : w) v = bf(0.3f * nd(rng));
    float* dx = up(x);
    bf16_t* dw = up(w);
    float* ring_ref = up(ring0);
    float* ring_new = up(ring0);
    float *qa = sycl::malloc_device<float>(size_t(M) * QS, q), *ka = sycl::malloc_device<float>(size_t(M) * QS, q),
          *va = sycl::malloc_device<float>(size_t(M) * VS, q);
    float *qb = sycl::malloc_device<float>(size_t(M) * QS, q), *kb = sycl::malloc_device<float>(size_t(M) * QS, q),
          *vb = sycl::malloc_device<float>(size_t(M) * VS, q);
    std::vector<std::vector<float>> snaps;
    for (int r = 0; r < M; ++r) {
        ConvParams cp{dx + size_t(r) * C, dw, ring_ref, nullptr, C, KER};
        launch_causal_conv1d_split_prefill(q, cp, 1, qa + size_t(r) * QS, ka + size_t(r) * QS, va + size_t(r) * VS,
                                           nullptr, QS, VS);
        snaps.push_back(down(ring_ref, ring0.size()));
    }
    ConvParams cp{dx, dw, ring_new, nullptr, C, KER};
    launch_causal_conv1d_split_prefill(q, cp, M, qb, kb, vb, nullptr, QS, VS);
    q.wait();
    size_t bad = ndiff(down(qa, size_t(M) * QS), down(qb, size_t(M) * QS)) +
                 ndiff(down(ka, size_t(M) * QS), down(kb, size_t(M) * QS)) +
                 ndiff(down(va, size_t(M) * VS), down(vb, size_t(M) * VS));
    std::printf("conv outputs, 7 rows in one call vs one call per row: %zu differ\n", bad);
    float* dst = sycl::malloc_device<float>(ring0.size(), q);
    const float* base = up(ring0);
    size_t rb_bad = 0;
    for (int cnt = 1; cnt <= M; ++cnt) {
        const float* inputs = dx;
        const int channels = C, hist = HIST;
        q.parallel_for(sycl::range<1>(size_t(channels) * hist), [=](sycl::id<1> id) {
            const int c = int(id[0]) / hist, j = int(id[0]) % hist;
            if (cnt >= hist) dst[int64_t(c) * hist + j] = inputs[int64_t(cnt - hist + j) * channels + c];
            else if (j < hist - cnt) dst[int64_t(c) * hist + j] = base[int64_t(c) * hist + j + cnt];
            else dst[int64_t(c) * hist + j] = inputs[int64_t(j - (hist - cnt)) * channels + c];
        }).wait();
        const size_t b = ndiff(down(dst, ring0.size()), snaps[size_t(cnt - 1)]);
        if (b) std::printf("  ring rebuild after %d rows: %zu differ\n", cnt, b);
        rb_bad += b;
    }
    std::printf("ring rebuild for every accepted count: %s\n", rb_bad ? "MISMATCH" : "bit-identical to the per-row snapshots");

    // ---- 2. norm ----
    const int HID = 5120;
    for (int tokens : {7, 33})
        for (int res = 0; res < 3; ++res) {
            std::vector<float> h(size_t(tokens) * HID), r0(h.size()), r1(h.size());
            std::vector<bf16_t> nw(HID);
            for (auto& v : h) v = nd(rng);
            for (auto& v : r0) v = nd(rng);
            for (auto& v : r1) v = nd(rng);
            for (auto& v : nw) v = bf(0.1f * nd(rng));
            float *h1 = up(h), *h2 = up(h), *a = up(r0), *b = up(r1);
            bf16_t* dnw = up(nw);
            float *o1 = sycl::malloc_device<float>(h.size(), q), *o2 = sycl::malloc_device<float>(h.size(), q);
            sycl_bf16 *ob1 = sycl::malloc_device<sycl_bf16>(h.size(), q), *ob2 = sycl::malloc_device<sycl_bf16>(h.size(), q);
            const float* ra = res >= 1 ? a : nullptr;
            const float* rbp = res >= 2 ? b : nullptr;
            old_norm(q, h1, ra, rbp, dnw, o1, tokens, HID, 1e-6f, ob1, 1.0f).wait();
            launch_rmsnorm_residual_batched(q, h2, ra, rbp, dnw, o2, tokens, HID, 1e-6f, ob2, {}, 1.0f).wait();
            const size_t d = ndiff(down(h1, h.size()), down(h2, h.size())) + ndiff(down(o1, h.size()), down(o2, h.size())) +
                             ndiff(down(ob1, h.size()), down(ob2, h.size()));
            std::printf("norm M=%-2d residuals=%d: %zu differ (h, out, out_bf)\n", tokens, res, d);
        }
    return 0;
}
