// bench_native_moe_grouped.cpp -- device check + timing of the grouped native
// Q4_K MoE prefill tile (launch_moe_native_qk_grouped) against a double
// precision host reference: bf16 activations, Q4_K weights decoded with
// decode_elem and rounded to bf16 (what the tile feeds the DPAS), fp64 sums.
// Ornith geometry: gate_up K=2048 Ne=1024 (SwiGLU -> bf16 H), down K=512 Ne=2048
// (fp32 out). E experts with ragged row counts (partial tiles, empty expert).
// Build (repo root; the kernel lives in the 256-GRF library):
//   icpx ... src/gemm_fast.cpp -shared -fPIC -o bin/libgrimoire_gemm.so   (build_grimoire_only_b70.sh)
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_moe_grouped.cpp -Lbin -lgrimoire_gemm '-Wl,-rpath,$ORIGIN' -o bin/bench_native_moe_grouped
// Set GRIMOIRE_MOE_NATIVE_ESIMD_GROUPED=0 for the joint_matrix reference kernel.
#include <sycl/sycl.hpp>
#include "kernels.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;

static uint16_t bf16_bits_rne(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return uint16_t((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static float bf16_val(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

static int run(sycl::queue& q, bool swiglu, int E, const std::vector<int>& cnts) {
    const int K = swiglu ? 2048 : 512, Ne = swiglu ? 1024 : 2048, FI = Ne / 2;
    std::mt19937 rng(5 + swiglu);
    QuantWeight w; w.fmt = Fmt::Q4_K; w.N = E * Ne; w.K = K; w.row_bytes = int64_t(K / 256) * 144;
    std::vector<uint8_t> hw(size_t(w.N) * w.row_bytes);
    for (auto& b : hw) b = uint8_t(rng());
    for (int64_t r = 0; r < w.N; ++r)
        for (int j = 0; j < K / 256; ++j) {
            uint8_t* p = hw.data() + r * w.row_bytes + j * 144;
            p[0] = 0; p[1] = 0x20 + (rng() % 6); p[2] = 0; p[3] = 0x1e + (rng() % 6);
        }
    uint8_t* dw = sycl::malloc_device<uint8_t>(hw.size(), q);
    q.memcpy(dw, hw.data(), hw.size()).wait();
    w.payload = dw;
    if (!moe_native_qk_grouped_supported(w, Ne)) { std::printf("unsupported\n"); return 1; }
    std::vector<int32_t> off(E), cnt(E);
    int rows = 0;
    for (int e = 0; e < E; ++e) { off[e] = rows; cnt[e] = cnts[e % cnts.size()]; rows += cnt[e]; }
    std::vector<int32_t> te, tm;
    for (int e = 0; e < E; ++e) for (int mb = 0; mb * moe_grouped_rows() < cnt[e]; ++mb) { te.push_back(e); tm.push_back(mb); }
    const int T = int(te.size());
    std::vector<uint16_t> ha(size_t(rows) * K);
    for (auto& b : ha) b = bf16_bits_rne((float(rng() % 2001) - 1000.0f) / 1000.0f);
    sycl_bf16* dA = sycl::malloc_device<sycl_bf16>(ha.size(), q);
    q.memcpy(dA, ha.data(), ha.size() * 2);
    int32_t *dte = sycl::malloc_device<int32_t>(T, q), *dtm = sycl::malloc_device<int32_t>(T, q);
    int32_t *doff = sycl::malloc_device<int32_t>(E, q), *dcnt = sycl::malloc_device<int32_t>(E, q);
    q.memcpy(dte, te.data(), T * 4); q.memcpy(dtm, tm.data(), T * 4);
    q.memcpy(doff, off.data(), E * 4); q.memcpy(dcnt, cnt.data(), E * 4);
    const size_t outsz = swiglu ? size_t(rows) * FI * 2 : size_t(rows) * Ne * 4;
    void* dout = sycl::malloc_device<uint8_t>(outsz, q);
    q.memset(dout, 0xFF, outsz).wait();
    launch_moe_native_qk_grouped(q, w, Ne, swiglu, dA, dout, dte, dtm, doff, dcnt, T).wait();
    std::vector<uint8_t> out(outsz); q.memcpy(out.data(), dout, outsz).wait();
    // reference on sampled (row, col)
    double maxrel = 0; int bad = 0, checked = 0;
    auto wval = [&](int64_t r, int k) {   // weight row r, value k, rounded to bf16 as the tile does
        return double(bf16_val(bf16_bits_rne(decode_elem<Fmt::Q4_K>(hw.data() + r * w.row_bytes, k, 1.0f))));
    };
    for (int e = 0; e < E; ++e) {
        for (int smp = 0; smp < 6 && cnt[e] > 0; ++smp) {
            const int r = smp == 0 ? 0 : smp == 1 ? cnt[e] - 1 : int(rng() % cnt[e]);
            const uint16_t* a = ha.data() + size_t(off[e] + r) * K;
            const int ncol = swiglu ? FI : Ne;
            for (int cs = 0; cs < 4; ++cs) {
                const int c = cs == 0 ? 0 : cs == 1 ? ncol - 1 : int(rng() % ncol);
                double g = 0, u = 0;
                if (swiglu) {
                    for (int k = 0; k < K; ++k) {
                        g += double(bf16_val(a[k])) * wval(int64_t(e) * Ne + c, k);
                        u += double(bf16_val(a[k])) * wval(int64_t(e) * Ne + FI + c, k);
                    }
                    const double ref = g / (1.0 + std::exp(-g)) * u;
                    const float got = bf16_val(reinterpret_cast<const uint16_t*>(out.data())[size_t(off[e] + r) * FI + c]);
                    const double err = std::fabs(ref - got) / (std::fabs(ref) * 1e-2 + 0.05);
                    maxrel = std::max(maxrel, err); if (!(err < 1.0)) ++bad;
                } else {
                    for (int k = 0; k < K; ++k) g += double(bf16_val(a[k])) * wval(int64_t(e) * Ne + c, k);
                    const float got = reinterpret_cast<const float*>(out.data())[size_t(off[e] + r) * Ne + c];
                    const double err = std::fabs(g - got) / (std::fabs(g) * 1e-3 + 0.05);
                    maxrel = std::max(maxrel, err); if (!(err < 1.0)) ++bad;
                }
                ++checked;
            }
        }
    }
    // rows past the last expert row must be untouched (0xFF fill == NaN pattern / all ones)
    // timing
    for (int i = 0; i < 3; ++i) launch_moe_native_qk_grouped(q, w, Ne, swiglu, dA, dout, dte, dtm, doff, dcnt, T);
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    const int iters = 10;
    for (int i = 0; i < iters; ++i) launch_moe_native_qk_grouped(q, w, Ne, swiglu, dA, dout, dte, dtm, doff, dcnt, T);
    q.wait();
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    const double flop = 2.0 * rows * K * Ne;
    std::printf("%s E=%d rows=%d tiles=%d : %9.1f us  %6.1f TFLOP/s  checked %d bad %d worst-err-ratio %.3f\n",
                swiglu ? "gate_up+swiglu" : "down          ", E, rows, T, us, flop / (us * 1e6), checked, bad, maxrel);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    // ragged: partial tiles, a 1-row expert, an empty expert, a 2-tile expert
    const std::vector<int> ragged = {300, 1, 256, 0, 190, 511, 33, 128};
    bad += run(q, true, 16, ragged);
    bad += run(q, false, 16, ragged);
    // timing-sized: ~Ornith prefill (4096 tokens x top-8 / 256 experts = 128 rows each)
    bad += run(q, true, 64, {128, 140, 110, 160});
    bad += run(q, false, 64, {128, 140, 110, 160});
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
