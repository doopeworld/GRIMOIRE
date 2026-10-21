// bench_native_moe.cpp -- device check + timing of the native Q4_K/Q6_K fused
// MoE decode pair (launch_moe_gate_up_shared / launch_moe_down_shared) on
// random raw GGUF blocks, Ornith geometry (H 2048, I 512, 256 experts, top 8,
// shared expert). Correctness: a double-precision host MoE built from
// decode_elem<F> for the routed + shared experts.
// Build (repo root), link the engine's MoE translation unit:
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 -O3 -std=c++20 -I include -I src \
//        tools/bench_native_moe.cpp src/moe_kernels.cpp -o bin/bench_native_moe
// Set GRIMOIRE_MOE_NATIVE_ESIMD=0 to time/check the reference (SIMT) path.
#include <sycl/sycl.hpp>
#include "kernels.hpp"
#include "b70/moe.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace b70;

static void fill(std::vector<uint8_t>& v, Fmt F, int64_t rows, int64_t rb, std::mt19937& rng) {
    const int bb = F == Fmt::Q4_K ? 144 : 210;
    for (auto& b : v) b = uint8_t(rng());
    for (int64_t r = 0; r < rows; ++r)
        for (int64_t j = 0; j < rb / bb; ++j) {
            uint8_t* p = v.data() + r * rb + j * bb;
            if (F == Fmt::Q4_K) { p[0] = 0; p[1] = 0x25; p[2] = 0; p[3] = 0x24; }
            else { p[208] = 0; p[209] = 0x25; }
        }
}

static QuantWeight upload(sycl::queue& q, Fmt F, int N, int K, std::vector<uint8_t>& host,
                          std::mt19937& rng) {
    const int bb = F == Fmt::Q4_K ? 144 : 210;
    QuantWeight w; w.fmt = F; w.N = N; w.K = K; w.row_bytes = int64_t(K / 256) * bb;
    host.resize(size_t(N) * w.row_bytes);
    fill(host, F, N, w.row_bytes, rng);
    uint8_t* d = sycl::malloc_device<uint8_t>(host.size(), q);
    q.memcpy(d, host.data(), host.size()).wait();
    w.payload = d;
    return w;
}

static double dotrow(Fmt F, const uint8_t* row, const float* x, int K) {
    double s = 0;
    for (int k = 0; k < K; ++k)
        s += double(F == Fmt::Q4_K ? decode_elem<Fmt::Q4_K>(row, k, 1.0f)
                                   : decode_elem<Fmt::Q6_K>(row, k, 1.0f)) * x[k];
    return s;
}

static int run(sycl::queue& q, Fmt Fgu, Fmt Fdn, int iters) {
    constexpr int H = 2048, I = 512, E = 256, TK = 8;
    std::mt19937 rng(77 + int(Fgu) * 3 + int(Fdn));
    MoeLayer L; L.cfg.hidden = H; L.cfg.inter = I; L.cfg.num_experts = E; L.cfg.top_k = TK;
    std::vector<uint8_t> hgu, hdn, hsgu, hsdn;
    L.gate_up = upload(q, Fgu, E * 2 * I, H, hgu, rng);
    L.down = upload(q, Fdn, E * H, I, hdn, rng);
    QuantWeight sgu = upload(q, Fgu, 2 * I, H, hsgu, rng);
    QuantWeight sdn = upload(q, Fdn, H, I, hsdn, rng);
    std::vector<float> x(H);
    for (auto& v : x) v = (float(rng() % 2001) - 1000.0f) / 1000.0f;
    std::vector<uint16_t> gw(H);
    for (auto& v : gw) v = 0x3c00 + uint16_t(rng() % 64);   // bf16 ~ 0.0078..0.0087 *
    float* dx = sycl::malloc_device<float>(H, q); q.memcpy(dx, x.data(), H * 4);
    uint16_t* dgw = sycl::malloc_device<uint16_t>(H, q); q.memcpy(dgw, gw.data(), H * 2);
    int32_t* dex = sycl::malloc_device<int32_t>(TK, q);
    float* dwt = sycl::malloc_device<float>(TK, q);
    float* dh = sycl::malloc_device<float>((TK + 1) * I, q);
    float* dgo = sycl::malloc_device<float>(1, q);
    float* dy = sycl::malloc_device<float>(H, q);
    q.wait();
    // correctness on a fixed routing
    std::vector<int32_t> ex(TK); std::vector<float> wt(TK);
    for (int i = 0; i < TK; ++i) { ex[i] = (i * 37 + 11) % E; wt[i] = 0.05f + 0.01f * i; }
    q.memcpy(dex, ex.data(), TK * 4); q.memcpy(dwt, wt.data(), TK * 4); q.wait();
    launch_moe_gate_up_shared(q, L, sgu, dgw, dex, dx, dh, dgo).wait();
    launch_moe_down_shared(q, L, sdn, dex, dwt, dgo, dh, dy).wait();
    std::vector<float> h((TK + 1) * I), y(H), go(1);
    q.memcpy(h.data(), dh, h.size() * 4); q.memcpy(y.data(), dy, H * 4); q.memcpy(go.data(), dgo, 4); q.wait();
    // host reference
    double sgate = 0;
    for (int k = 0; k < H; ++k) { uint32_t b = uint32_t(gw[k]) << 16; float f; memcpy(&f, &b, 4); sgate += double(f) * x[k]; }
    const double gsig = 1.0 / (1.0 + std::exp(-sgate));
    double maxrel_h = 0, maxrel_y = 0; int bad = 0;
    std::vector<double> href((TK + 1) * I);
    for (int s = 0; s <= TK; ++s) {
        const bool sh = s == TK;
        const uint8_t* base = sh ? hsgu.data() : hgu.data();
        const int64_t rb = L.gate_up.row_bytes;
        const int64_t r0 = sh ? 0 : int64_t(ex[s]) * 2 * I;
        for (int i = 0; i < I; i += 5) {   // sample rows
            const double g = dotrow(Fgu, base + (r0 + i) * rb, x.data(), H);
            const double u = dotrow(Fgu, base + (r0 + I + i) * rb, x.data(), H);
            const double hv = g / (1.0 + std::exp(-g)) * u;
            href[s * I + i] = hv;
            const double err = std::fabs(hv - h[s * I + i]) / (std::fabs(hv) + 1e-3);
            maxrel_h = std::max(maxrel_h, err);
            if (!(err < 1e-3)) ++bad;
        }
    }
    {
        const double err = std::fabs(gsig - go[0]) / gsig;
        if (!(err < 1e-4)) { ++bad; std::printf("shared gate mismatch %g vs %g\n", gsig, go[0]); }
    }
    // down: needs full h from device (use device h as input to the reference: isolates the down kernel)
    for (int o = 0; o < H; o += 37) {
        double acc = 0;
        for (int s = 0; s <= TK; ++s) {
            const bool sh = s == TK;
            const uint8_t* base = sh ? hsdn.data() : hdn.data();
            const int64_t rb = L.down.row_bytes;
            const int64_t r = sh ? o : int64_t(ex[s]) * H + o;
            const double w = sh ? gsig : wt[s];
            acc += w * dotrow(Fdn, base + r * rb, h.data() + s * I, I);
        }
        const double err = std::fabs(acc - y[o]) / (std::fabs(acc) + 1e-3);
        maxrel_y = std::max(maxrel_y, err);
        if (!(err < 1e-3)) ++bad;
    }
    // timing: random routing every iteration (cold experts)
    std::vector<std::vector<int32_t>> routes(iters, std::vector<int32_t>(TK));
    for (auto& r : routes) for (auto& e : r) e = rng() % E;
    double t_gu = 0, t_dn = 0;
    for (int it = 0; it < iters; ++it) {
        q.memcpy(dex, routes[it].data(), TK * 4).wait();
        auto t0 = std::chrono::steady_clock::now();
        launch_moe_gate_up_shared(q, L, sgu, dgw, dex, dx, dh, dgo).wait();
        auto t1 = std::chrono::steady_clock::now();
        launch_moe_down_shared(q, L, sdn, dex, dwt, dgo, dh, dy).wait();
        auto t2 = std::chrono::steady_clock::now();
        if (it >= 2) { t_gu += std::chrono::duration<double, std::micro>(t1 - t0).count();
                       t_dn += std::chrono::duration<double, std::micro>(t2 - t1).count(); }
    }
    t_gu /= (iters - 2); t_dn /= (iters - 2);
    const double bgu = double(TK + 1) * 2 * I * L.gate_up.row_bytes, bdn = double(TK + 1) * H * L.down.row_bytes;
    std::printf("gu %s dn %s : gate_up %7.2f us (%5.1f GB/s)  down %7.2f us (%5.1f GB/s)  maxrel h %.1e y %.1e bad %d\n",
                Fgu == Fmt::Q4_K ? "Q4_K" : "Q6_K", Fdn == Fmt::Q4_K ? "Q4_K" : "Q6_K",
                t_gu, bgu / (t_gu * 1e3), t_dn, bdn / (t_dn * 1e3), maxrel_h, maxrel_y, bad);
    return bad;
}

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    int bad = 0;
    bad += run(q, Fmt::Q4_K, Fmt::Q4_K, 40);
    bad += run(q, Fmt::Q4_K, Fmt::Q6_K, 40);
    std::printf(bad ? "FAIL\n" : "OK\n");
    return bad ? 1 : 0;
}
