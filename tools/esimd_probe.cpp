// esimd_probe.cpp -- verify, on the B70, the ESIMD building blocks the next
// kernels rely on, against a CPU reference:
//   1. load_2d of a VNNI-packed bf16 B tile (16 k x 16 n, stored [k/2][n][2])
//   2. load_2d of a row-major bf16 A tile (8 rows x 16 k)
//   3. xmx::dpas<8, 8, float, float, bf16, bf16>: C (8x16 fp32, row-major)
//   4. row reductions on the result (hmax per row)
//   5. result reused as the A operand of a second dpas (P V pattern)
// Build (256-GRF, AOT for the B70):
//   icpx -fsycl -fsycl-targets=intel_gpu_bmg_g31 \
//     -Xsycl-target-backend=intel_gpu_bmg_g31 "-options -cl-intel-256-GRF-per-thread" \
//     -O2 -std=c++20 tools/esimd_probe.cpp -o bin/esimd_probe
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using bf16 = sycl::ext::oneapi::bfloat16;

static float bf(float x) { return float(bf16(x)); }

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    constexpr int M = 8, K = 32, N = 16;           // two K-steps of 16
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.0f, 1.0f);
    std::vector<float> a(M * K), b(K * N), v(N * N);
    for (auto& x : a) x = bf(U(rng));
    for (auto& x : b) x = bf(U(rng));
    for (auto& x : v) x = bf(U(rng));
    // A row-major [M][K]; B VNNI [K/2][N][2]; V (for P V) VNNI [16/2][16][2]
    std::vector<bf16> ha(M * K), hb(K * N), hv(N * N);
    for (int i = 0; i < M * K; ++i) ha[i] = bf16(a[i]);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) hb[((k / 2) * N + n) * 2 + (k & 1)] = bf16(b[k * N + n]);
    for (int k = 0; k < N; ++k)
        for (int n = 0; n < N; ++n) hv[((k / 2) * N + n) * 2 + (k & 1)] = bf16(v[k * N + n]);
    bf16* da = sycl::malloc_device<bf16>(M * K, q);
    bf16* db = sycl::malloc_device<bf16>(K * N, q);
    bf16* dv = sycl::malloc_device<bf16>(N * N, q);
    float* dc = sycl::malloc_device<float>(M * N, q);
    float* dm = sycl::malloc_device<float>(M, q);
    float* dpv = sycl::malloc_device<float>(M * N, q);
    q.memcpy(da, ha.data(), ha.size() * 2);
    q.memcpy(db, hb.data(), hb.size() * 2);
    q.memcpy(dv, hv.data(), hv.size() * 2).wait();
    q.parallel_for(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
        esimd::simd<float, M * N> c = 0.0f;
        for (int k0 = 0; k0 < K; k0 += 16) {
            // A: 8 rows x 16 bf16 at column k0 of a [M][K] surface
            esimd::simd<bf16, M * 16> ta = esimd::load_2d<bf16, 16, M>(
                da, K * 2 - 1, M - 1, K * 2 - 1, k0, 0);
            // B: 8 rows (k-pairs) x 16 dwords (n, 2 bf16 each) at row k0/2
            esimd::simd<uint32_t, 8 * 16> tb32 = esimd::load_2d<uint32_t, 16, 8>(
                reinterpret_cast<const uint32_t*>(db), N * 4 - 1, K / 2 - 1, N * 4 - 1, 0, k0 / 2);
            esimd::simd<bf16, 16 * 16> tb = tb32.template bit_cast_view<bf16>();
            c = xmx::dpas<8, M, float, float, bf16, bf16>(c, tb, ta);
        }
        esimd::block_store<float, M * N>(dc, c);
        esimd::simd<float, M> mx;
        #pragma unroll
        for (int r = 0; r < M; ++r)
            mx[r] = esimd::hmax<float>(esimd::simd<float, 16>(c.template select<16, 1>(16 * r)));
        esimd::block_store<float, M>(dm, mx);
        // result (8x16 fp32 row-major) as the A operand of P V
        esimd::simd<bf16, M * 16> p = c;
        esimd::simd<uint32_t, 8 * 16> tv32 = esimd::load_2d<uint32_t, 16, 8>(
            reinterpret_cast<const uint32_t*>(dv), N * 4 - 1, 8 - 1, N * 4 - 1, 0, 0);
        esimd::simd<bf16, 16 * 16> tv = tv32.template bit_cast_view<bf16>();
        esimd::simd<float, M * N> pv = 0.0f;
        pv = xmx::dpas<8, M, float, float, bf16, bf16>(pv, tv, p);
        esimd::block_store<float, M * N>(dpv, pv);
    }).wait();
    std::vector<float> hc(M * N), hm(M), hpv(M * N);
    q.memcpy(hc.data(), dc, hc.size() * 4);
    q.memcpy(hm.data(), dm, hm.size() * 4);
    q.memcpy(hpv.data(), dpv, hpv.size() * 4).wait();
    double e1 = 0, e2 = 0, e3 = 0;
    for (int m = 0; m < M; ++m) {
        float rmax = -1e30f;
        for (int n = 0; n < N; ++n) {
            double ref = 0;
            for (int k = 0; k < K; ++k) ref += double(a[m * K + k]) * b[k * N + n];
            e1 = std::max(e1, std::fabs(ref - hc[m * N + n]));
            rmax = std::max(rmax, hc[m * N + n]);
        }
        e2 = std::max(e2, double(std::fabs(rmax - hm[m])));
        for (int n = 0; n < N; ++n) {
            double ref = 0;
            for (int k = 0; k < N; ++k) ref += double(bf(hc[m * N + k])) * v[k * N + n];
            e3 = std::max(e3, std::fabs(ref - hpv[m * N + n]));
        }
    }
    std::printf("A*B (dpas, load_2d A row-major, B VNNI): max|err| %.3e\n", e1);
    std::printf("row hmax:                              max|err| %.3e\n", e2);
    std::printf("P*V (C reused as A):                    max|err| %.3e\n", e3);
    const bool ok = e1 < 1e-3 && e2 == 0.0 && e3 < 1e-3;
    std::printf("%s\n", ok ? "ESIMD PROBE PASS" : "ESIMD PROBE FAIL");
    return ok ? 0 : 1;
}
