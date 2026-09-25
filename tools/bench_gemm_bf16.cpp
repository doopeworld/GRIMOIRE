// bench_gemm_bf16 -- candidate prompt-processing GEMM for the B70.
//
//   C[M][N] (fp32) = A[M][K] (bf16, row-major) * B[K][N] (bf16, VNNI-packed)
//
// Shape of the kernel: every sub-group owns an MC1 x NC1 output tile and
// loads its A and B fragments straight from global memory with
// joint_matrix_load (2-D block loads on Xe2) -- no shared-local staging, no
// per-element copies, no barriers in the K loop.  A work-group is a grid of
// (MC2/MC1) x (NC2/NC1) sub-groups, so neighbours share A rows and B columns
// through the cache.  B is expected pre-packed ([K/2][N][2]); for quantized
// weights a separate dequantize pass writes that layout once per layer.
//
// Usage: bench_gemm_bf16 [M N K]   (M multiple of 256, N of 256, K of 32)
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace mx = sycl::ext::oneapi::experimental::matrix;
using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int SG = 16, TM = 8, TN = 16, TK = 16;

template <int MC1, int NC1, int KC1, int MC2, int NC2>
struct Gemm {
    static sycl::event run(sycl::queue& q, const bf16* A, const bf16* B, float* C,
                           int M, int N, int K) {
        const sycl::range<2> global(size_t(M / MC1), size_t(N / NC1) * SG);
        const sycl::range<2> local(size_t(MC2 / MC1), size_t(NC2 / NC1) * SG);
#ifdef BENCH_PROP
        sycl::ext::oneapi::experimental::properties props{
            sycl::ext::intel::experimental::grf_size<256>};
        return q.parallel_for(sycl::nd_range<2>(global, local), props,
#else
        return q.parallel_for(sycl::nd_range<2>(global, local),
#endif
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            auto sg = it.get_sub_group();
            const int m0 = int(it.get_global_id(0)) * MC1;
            const int n0 = int(it.get_global_id(1) / SG) * NC1;
            auto pA = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(A);
            auto pB = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(B);
            mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, TM, TN>
                acc[MC1 / TM][NC1 / TN];
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    mx::joint_matrix_fill(sg, acc[m][n], 0.0f);
            for (int k = 0; k < K; k += KC1) {
                mx::joint_matrix<sycl::sub_group, bf16, mx::use::a, TM, TK,
                                 mx::layout::row_major> a[MC1 / TM][KC1 / TK];
                mx::joint_matrix<sycl::sub_group, bf16, mx::use::b, TK, TN,
                                 mx::layout::ext_intel_packed> b[NC1 / TN][KC1 / TK];
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK; ++kk) {
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        mx::joint_matrix_load(sg, a[m][kk],
                            pA + size_t(m0 + m * TM) * K + k + kk * TK, K);
                    #pragma unroll
                    for (int n = 0; n < NC1 / TN; ++n)
                        mx::joint_matrix_load(sg, b[n][kk],
                            pB + size_t(k + kk * TK) / 2 * (size_t(N) * 2) + size_t(n0 + n * TN) * 2,
                            size_t(N) * 2);
                }
                #pragma unroll
                for (int kk = 0; kk < KC1 / TK; ++kk)
                    #pragma unroll
                    for (int m = 0; m < MC1 / TM; ++m)
                        #pragma unroll
                        for (int n = 0; n < NC1 / TN; ++n)
                            mx::joint_matrix_mad(sg, acc[m][n], a[m][kk], b[n][kk], acc[m][n]);
            }
            auto pC = sycl::address_space_cast<sycl::access::address_space::global_space,
                                               sycl::access::decorated::no>(C);
            #pragma unroll
            for (int m = 0; m < MC1 / TM; ++m)
                #pragma unroll
                for (int n = 0; n < NC1 / TN; ++n)
                    mx::joint_matrix_store(sg, acc[m][n],
                        pC + size_t(m0 + m * TM) * N + n0 + n * TN, N, mx::layout::row_major);
        });
    }
};

template <class G>
static void bench(const char* name, sycl::queue& q, const bf16* A, const bf16* B, float* C,
                  int M, int N, int K, const std::vector<bf16>& hA,
                  const std::vector<bf16>& hBrm) {
    q.memset(C, 0, size_t(M) * N * sizeof(float)).wait();
    G::run(q, A, B, C, M, N, K).wait();                       // warm-up + JIT-free check
    // Correctness: 64 random output elements against a host fp64 reference.
    std::vector<float> row(N);
    std::mt19937 rng(7);
    double max_rel = 0;
    for (int t = 0; t < 64; ++t) {
        const int i = int(rng() % M), j = int(rng() % N);
        float got;
        q.memcpy(&got, C + size_t(i) * N + j, sizeof(float)).wait();
        double ref = 0;
        for (int k = 0; k < K; ++k)
            ref += double(float(hA[size_t(i) * K + k])) * double(float(hBrm[size_t(k) * N + j]));
        max_rel = std::max(max_rel, std::fabs(got - ref) / (std::fabs(ref) + 1.0));
    }
    const int iters = 10;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < iters; ++r) G::run(q, A, B, C, M, N, K);
    q.wait();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / iters;
    std::printf("%-26s %8.3f ms  %7.1f TFLOP/s  max_rel_err %.2e  %s\n", name, s * 1e3,
                2.0 * M * N * K / s / 1e12, max_rel, max_rel < 1e-3 ? "OK" : "WRONG");
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const int M = argc > 3 ? std::atoi(argv[1]) : 6144;
    const int N = argc > 3 ? std::atoi(argv[2]) : 17408;
    const int K = argc > 3 ? std::atoi(argv[3]) : 5120;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s  M=%d N=%d K=%d\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(), M, N, K);
    std::vector<bf16> hA(size_t(M) * K), hB(size_t(K) * N), hBv(size_t(K) * N);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto& v : hA) v = bf16(d(rng));
    for (auto& v : hB) v = bf16(d(rng));
    for (int k = 0; k < K; ++k)                       // VNNI pack: [k/2][n][k%2]
        for (int n = 0; n < N; ++n)
            hBv[(size_t(k / 2) * N + n) * 2 + (k & 1)] = hB[size_t(k) * N + n];
    bf16* A = sycl::malloc_device<bf16>(hA.size(), q);
    bf16* B = sycl::malloc_device<bf16>(hBv.size(), q);
    float* C = sycl::malloc_device<float>(size_t(M) * N, q);
    q.memcpy(A, hA.data(), hA.size() * sizeof(bf16));
    q.memcpy(B, hBv.data(), hBv.size() * sizeof(bf16)).wait();
    bench<Gemm<32, 64, 32, 256, 256>>("sg32x64 k32 wg256x256", q, A, B, C, M, N, K, hA, hB);
    bench<Gemm<32, 64, 32, 128, 256>>("sg32x64 k32 wg128x256", q, A, B, C, M, N, K, hA, hB);
    bench<Gemm<32, 64, 16, 256, 256>>("sg32x64 k16 wg256x256", q, A, B, C, M, N, K, hA, hB);
    bench<Gemm<64, 32, 32, 256, 128>>("sg64x32 k32 wg256x128", q, A, B, C, M, N, K, hA, hB);
    bench<Gemm<32, 32, 32, 256, 128>>("sg32x32 k32 wg256x128", q, A, B, C, M, N, K, hA, hB);
    sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
    return 0;
}
