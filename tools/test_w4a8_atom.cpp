// =====================================================================
//  test_w4a8_atom.cpp -- does cutlass' NATIVE s8 x s4 DPAS atom work
//  with our 2-D block loads and our packed 4-bit weight layout?
//
//  This is the kill-or-proceed check for W4A8 prefill.  Three questions,
//  and nothing else:
//    1. Does XE_DPAS_TT<8,int32_t,int8_t,int4_t,int32_t> instantiate
//       through TiledMMAHelper with our production 128x256 / 4x8 policy?
//    2. Does get_block_2d_copy_B load an int4_t tensor into a fragment
//       that `reorder` can feed to that atom's B operand?
//    3. Is the result CORRECT?
//
//  The math is pure integer, so the test is BIT-EXACT equality against a
//  host reference -- a far stronger check than the cosine used elsewhere
//  in this project, and the kind that would have caught the all-zeros
//  BesTLA result immediately.
//
//  Deliberately NOT tested here: scales, group boundaries, performance.
//  Correctness of the atom plumbing first.
// =====================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <string>
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include "xe2_grouped_raw_launcher.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

using namespace cute;

class W4A8Policy : public MoE::xe_gemm_policy_base {
 public:
  using WGTile   = cute::Shape<cute::_128, cute::_256, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_4, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<32, 8, 16>;
};

// Small-M tiles.  The production policy is 128x256: at M=4 its A-tensor block
// loads still fetch 128 rows per tile -- ~89 MB/layer of pure padding, about
// equal to the weight traffic itself -- plus a 128-row epilogue write.  That
// is why the W4A8 GEMM measured 51 ms for the FFN at M=4 against the M=1
// GEMV's 19.  The DPAS atom is natively M=8, so an 8- or 16-row tile should
// make the batch weight-bound: eight tokens for the price of one.
class W4A8_M8 : public MoE::xe_gemm_policy_base {
 public:
  using WGTile   = cute::Shape<cute::_8, cute::_256, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_1, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<32, 8, 16>;
};
class W4A8_M16 : public MoE::xe_gemm_policy_base {
 public:
  using WGTile   = cute::Shape<cute::_16, cute::_256, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_2, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<32, 8, 16>;
};
class W4A8_M32 : public MoE::xe_gemm_policy_base {
 public:
  using WGTile   = cute::Shape<cute::_32, cute::_256, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_4, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<32, 8, 16>;
};
class W4A8_M128 : public MoE::xe_gemm_policy_base {
 public:
  using WGTile   = cute::Shape<cute::_128, cute::_256, cute::_32>;
  using SGLayout = cute::Layout<cute::Shape<cute::_4, cute::_8, cute::_1>,
                                cute::Stride<cute::_8, cute::_1, cute::_0>>;
  using GmemTiledCopyD = cute::XE_STORE_2D<32, 8, 16>;
};

class W4A8Kernel;

// ---------------------------------------------------------------------
//  Bench mode: production shape, 64 DISTINCT weight matrices so the B
//  stream comes from DRAM exactly as it does in a real prefill.  No
//  scales yet -- this measures the ceiling of the s8xs4 DPAS plus the
//  2-D block loads, which is the number that decides whether the rest of
//  the W4A8 work is worth doing.  Gate to beat: 99.5 TFLOP/s (MXFP4
//  p128x256, same shape, same streaming conditions, bin/bench_sweep).
// ---------------------------------------------------------------------
class W4A8Bench;

template <class MMA, class CA, class CB>
static void run_tiles(sycl::queue& q, const int8_t* A, const uint8_t* B,
                      int32_t* D, int M, int N, int K, int groups) {
  MMA mma{};
  const int threads = size(mma);
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, groups, 1);
  namespace sx = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  sx::properties props{sx::sub_group_size<16>, ix::grf_size<256>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<W4A8Bench>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto item) {
          MMA mm{};
          auto At = MoE::make_moe_tensor<int8_t, 'R'>(const_cast<int8_t*>(A), M, K);
          auto Bt = MoE::make_moe_tensor<cute::int4_t, 'R'>(
              reinterpret_cast<cute::int4_t*>(const_cast<uint8_t*>(B)), N, K);
          auto Dt = MoE::make_moe_tensor<int32_t, 'R'>(D, M, N);
          auto wg_tile = mm.tile_mnk();
          const int TM = int(get<0>(wg_tile)), TN = int(get<1>(wg_tile));
          const int mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN;
          const int total = mt * nt;
          const int local_id = int(item.get_local_linear_id());
          for (int t = int(item.get_group_linear_id()); t < total;
               t += int(item.get_group_range(1))) {
            const int mi = t / nt, ni = t % nt;
            Tensor cA = make_identity_tensor(At.shape());
            Tensor cB = make_identity_tensor(Bt.shape());
            Tensor cC = make_identity_tensor(Dt.shape());
            Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(mi, _));
            Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(ni, _));
            Tensor gC = local_tile(cC, wg_tile, make_coord(mi,ni,0), Step<_1,_1,X>{});
            auto copy_a = MoE::get_block_2d_copy_A<CA>(mm, At);
            auto copy_b = MoE::get_block_2d_copy_B<CB>(mm, Bt);
            auto thr_mma = mm.get_slice(local_id);
            auto thr_copy_a = copy_a.get_slice(local_id);
            auto thr_copy_b = copy_b.get_slice(local_id);
            auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
            auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
            auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
            auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));
            Tensor tAgA = thr_copy_a.partition_S(gA);
            Tensor tBgB = thr_copy_b.partition_S(gB);
            auto tCrC = thr_mma.partition_sg_fragment_C(gC);
            clear(tCrC);
            auto prefetch_a = make_block_2d_prefetch(copy_a);
            auto prefetch_b = make_block_2d_prefetch(copy_b);
            auto pA = prefetch_a.get_slice(local_id).partition_S(gA);
            auto pB = prefetch_b.get_slice(local_id).partition_S(gB);
            const int k_tiles = K / int(get<2>(wg_tile));
            int kp = 0;
            CUTE_UNROLL
            for (; kp < 3; ++kp) { prefetch(prefetch_a, pA(_,_,_,kp));
                                   prefetch(prefetch_b, pB(_,_,_,kp)); }
            for (int kt = 0; kt < k_tiles; ++kt, ++kp) {
              barrier_arrive(2);
              copy(copy_a, tAgA(_,_,_,kt), tArA);
              copy(copy_b, tBgB(_,_,_,kt), tBrB);
              if (kp < k_tiles) { prefetch(prefetch_a, pA(_,_,_,kp));
                                  prefetch(prefetch_b, pB(_,_,_,kp)); }
              reorder(tArA, tCrA);
              reorder(tBrB, tCrB);
              cute::gemm(mm, tCrA, tCrB, tCrC);
              barrier_wait(2);
            }
            constexpr int ATOM_N = get<2>(typename MMA::ThrLayoutVMNK{}.shape());
            constexpr int ATOM_M = get<1>(typename MMA::ThrLayoutVMNK{}.shape());
            const int SG_M = TM / ATOM_M, SG_N = TN / ATOM_N;
            const int sg = int(cutlass::get_sub_group_id());
            const int lane = int(cutlass::get_sub_group_local_id());
            const int m0 = mi*TM + (sg / ATOM_N) * SG_M;
            const int n0 = ni*TN + (sg % ATOM_N) * SG_N;
            for (int sn = 0; sn < SG_N / 16; ++sn)
              for (int sm = 0; sm < SG_M; ++sm) {
                const int m = m0 + sm, n = n0 + sn*16 + lane;
                if (m < M && n < N) D[size_t(m)*N + n] = tCrC(sn*SG_M + sm);
              }
          }
        });
  });
}

// ---------------------------------------------------------------------
//  The same loop, but rescaling at every K-GROUP boundary the way real
//  g128 weights require: int32 sums are only valid while the weight scale
//  is constant, so at each group the accumulator is converted to float,
//  multiplied by weight_scale[n, k/G], added into a float accumulator, and
//  cleared.  This is the cost that decides whether the 193 TFLOP/s ceiling
//  survives contact with a real quantized artifact.
// ---------------------------------------------------------------------
template<typename DT> class W4A8BenchG;

template <class MMA, class CA, class CB, int G, typename DT = float>
static void run_tiles_grouped(sycl::queue& q, const int8_t* A, const uint8_t* B,
                              const float* WS, const float* AS, DT* D,
                              int M, int N, int K, int groups) {
  MMA mma{};
  const int threads = size(mma);
  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, groups, 1);
  namespace sx = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  sx::properties props{sx::sub_group_size<16>, ix::grf_size<256>};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<W4A8BenchG<DT>>(
        sycl::nd_range<3>{global * local, local}, props, [=](auto item) {
          MMA mm{};
          auto At = MoE::make_moe_tensor<int8_t, 'R'>(const_cast<int8_t*>(A), M, K);
          auto Bt = MoE::make_moe_tensor<cute::int4_t, 'R'>(
              reinterpret_cast<cute::int4_t*>(const_cast<uint8_t*>(B)), N, K);
          auto Dt = MoE::make_moe_tensor<DT, 'R'>(D, M, N);
          auto wg_tile = mm.tile_mnk();
          const int TM = int(get<0>(wg_tile)), TN = int(get<1>(wg_tile));
          const int KT = int(get<2>(wg_tile));
          const int mt = (M + TM - 1) / TM, nt = (N + TN - 1) / TN;
          const int total = mt * nt;
          const int local_id = int(item.get_local_linear_id());
          const int ktiles_per_group = G / KT;
          for (int t = int(item.get_group_linear_id()); t < total;
               t += int(item.get_group_range(1))) {
            const int mi = t / nt, ni = t % nt;
            Tensor cA = make_identity_tensor(At.shape());
            Tensor cB = make_identity_tensor(Bt.shape());
            Tensor cC = make_identity_tensor(Dt.shape());
            Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(mi, _));
            Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(ni, _));
            Tensor gC = local_tile(cC, wg_tile, make_coord(mi,ni,0), Step<_1,_1,X>{});
            auto copy_a = MoE::get_block_2d_copy_A<CA>(mm, At);
            auto copy_b = MoE::get_block_2d_copy_B<CB>(mm, Bt);
            auto thr_mma = mm.get_slice(local_id);
            auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
            auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
            auto tArA = copy_a.get_slice(local_id).partition_sg_fragment_D(gA(_,_,0));
            auto tBrB = copy_b.get_slice(local_id).partition_sg_fragment_D(gB(_,_,0));
            Tensor tAgA = copy_a.get_slice(local_id).partition_S(gA);
            Tensor tBgB = copy_b.get_slice(local_id).partition_S(gB);
            auto tCrC = thr_mma.partition_sg_fragment_C(gC);
            auto prefetch_a = make_block_2d_prefetch(copy_a);
            auto prefetch_b = make_block_2d_prefetch(copy_b);
            auto pA = prefetch_a.get_slice(local_id).partition_S(gA);
            auto pB = prefetch_b.get_slice(local_id).partition_S(gB);

            constexpr int ATOM_N = get<2>(typename MMA::ThrLayoutVMNK{}.shape());
            constexpr int ATOM_M = get<1>(typename MMA::ThrLayoutVMNK{}.shape());
            const int SG_M = TM / ATOM_M, SG_N = TN / ATOM_N;
            const int sg = int(cutlass::get_sub_group_id());
            const int lane = int(cutlass::get_sub_group_local_id());
            const int m0 = mi*TM + (sg / ATOM_N) * SG_M;
            const int n0 = ni*TN + (sg % ATOM_N) * SG_N;
            constexpr int NACC = 64;              // SG_M * SG_N/16 upper bound
            float facc[NACC];
            CUTE_UNROLL
            for (int i = 0; i < NACC; ++i) facc[i] = 0.0f;

            const int k_tiles = K / KT;
            int kp = 0;
            CUTE_UNROLL
            for (; kp < 3; ++kp) { prefetch(prefetch_a, pA(_,_,_,kp));
                                   prefetch(prefetch_b, pB(_,_,_,kp)); }
            clear(tCrC);
            for (int kt = 0; kt < k_tiles; ++kt, ++kp) {
              barrier_arrive(2);
              copy(copy_a, tAgA(_,_,_,kt), tArA);
              copy(copy_b, tBgB(_,_,_,kt), tBrB);
              if (kp < k_tiles) { prefetch(prefetch_a, pA(_,_,_,kp));
                                  prefetch(prefetch_b, pB(_,_,_,kp)); }
              reorder(tArA, tCrA);
              reorder(tBrB, tCrB);
              cute::gemm(mm, tCrA, tCrB, tCrC);
              barrier_wait(2);
              if ((kt + 1) % ktiles_per_group == 0) {
                const int g = (kt + 1) / ktiles_per_group - 1;
                const int kg = K / G;
                CUTE_UNROLL
                for (int sn = 0; sn < SG_N/16; ++sn) {
                  const int n = n0 + sn*16 + lane;
                  const float ws = WS[size_t(n)*kg + g];
                  CUTE_UNROLL
                  for (int sm = 0; sm < SG_M; ++sm)
                    facc[sn*SG_M+sm] = sycl::fma(float(tCrC(sn*SG_M+sm)), ws,
                                                 facc[sn*SG_M+sm]);
                }
                clear(tCrC);
              }
            }
            for (int sn = 0; sn < SG_N/16; ++sn)
              for (int sm = 0; sm < SG_M; ++sm) {
                const int m = m0 + sm, n = n0 + sn*16 + lane;
                if (m < M && n < N)
                  D[size_t(m)*N + n] = DT(facc[sn*SG_M+sm] * AS[m]);
              }
          }
        });
  });
}

// ---------------------------------------------------------------------
//  smallm: the Phase A experiment.  Time the FFN of all 64 layers at the
//  batch sizes a speculative verify actually uses, once per tile height.
//  Reference: the batched int4 GEMV is 19.07 ms at M=1 and 35.42 at M=4 for
//  the same weights.  A weight-bound tile should sit near 19 ms at every M.
// ---------------------------------------------------------------------
template <class P>
static double ffn_pass(sycl::queue& q, int M, const std::vector<uint8_t*>& Wg,
                       const std::vector<uint8_t*>& Wd, const int8_t* A,
                       const float* WSg, const float* WSd, const float* AS,
                       float* D, int H, int I) {
  double best = 1e18;
  for (int rep = 0; rep < 2; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (size_t l = 0; l < Wg.size(); ++l) {
      grimoire_moe_raw::launch_dense_w4a8<P, 128, float>(
          q, A, Wg[l], WSg, AS, D, M, 2 * I, H);
      grimoire_moe_raw::launch_dense_w4a8<P, 128, float>(
          q, A, Wd[l], WSd, AS, D, M, H, I);
    }
    q.wait_and_throw();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (rep) best = std::min(best, ms);
  }
  return best;
}

// Bit-exact check for a given tile policy, at the batch sizes a verify uses.
// Integer math, so equality is exact -- no tolerance to hide behind.
template <class P>
static int verify_tile(sycl::queue& q, const char* name) {
  const int N = 512, K = 512;
  int fails = 0;
  for (int M : {1, 2, 4, 8, 16}) {
    std::mt19937 r(31);
    std::vector<int8_t> hA(size_t(M)*K), hBu(size_t(N)*K);
    std::vector<uint8_t> hB(size_t(N)*K/2);
    for (auto& v : hA)  v = int8_t(int(r()&0xF)-8);
    for (auto& v : hBu) v = int8_t(int(r()&0xF)-8);
    for (size_t i=0;i<hB.size();++i)
      hB[i] = uint8_t((uint8_t(hBu[2*i])&0x0F)|(uint8_t(hBu[2*i+1])<<4));
    const int kg = K/128;
    std::vector<float> hWS(size_t(N)*kg, 0.0f);
    std::vector<float> hAS(size_t(M), 0.0f);
    for (size_t i=0;i<hWS.size();++i) hWS[i] = 0.5f + 0.25f*float(i%3);
    for (int m=0;m<M;++m) hAS[m] = 0.125f * float(m+1);
    int8_t*  a = sycl::malloc_device<int8_t>(hA.size(), q);
    uint8_t* b = sycl::malloc_device<uint8_t>(hB.size(), q);
    float*   ws= sycl::malloc_device<float>(hWS.size(), q);
    float*   as= sycl::malloc_device<float>(hAS.size(), q);
    float*   d = sycl::malloc_device<float>(size_t(M)*N, q);
    q.memcpy(a,hA.data(),hA.size()).wait();
    q.memcpy(b,hB.data(),hB.size()).wait();
    q.memcpy(ws,hWS.data(),hWS.size()*sizeof(float)).wait();
    q.memcpy(as,hAS.data(),hAS.size()*sizeof(float)).wait();
    q.memset(d,0,size_t(M)*N*sizeof(float)).wait();
    grimoire_moe_raw::launch_dense_w4a8<P,128,float>(q,a,b,ws,as,d,M,N,K);
    q.wait_and_throw();
    std::vector<float> got(size_t(M)*N);
    q.memcpy(got.data(), d, got.size()*sizeof(float)).wait();
    size_t bad = 0;
    for (int m=0;m<M;++m) for (int n2=0;n2<N;++n2) {
      double acc = 0;
      for (int g=0; g<kg; ++g) {
        int32_t s32 = 0;
        for (int k=g*128;k<(g+1)*128;++k)
          s32 += int32_t(hA[size_t(m)*K+k])*int32_t(hBu[size_t(n2)*K+k]);
        acc += double(s32) * double(hWS[size_t(n2)*kg+g]);
      }
      acc *= double(hAS[m]);
      const double diff = std::fabs(acc - double(got[size_t(m)*N+n2]));
      if (diff > 1e-3 * std::max(1.0, std::fabs(acc))) ++bad;
    }
    printf("  %-10s M=%-3d %s\n", name, M, bad ? "FAIL" : "exact");
    if (bad) ++fails;
    sycl::free(a,q); sycl::free(b,q); sycl::free(ws,q);
    sycl::free(as,q); sycl::free(d,q);
  }
  return fails;
}

// ---------------------------------------------------------------------
//  The non-FFN weights are still MXFP4 (only the FFN was converted to int4),
//  so a batched verify needs the same tile fix on launch_dense_mxfp4.  Shapes
//  are the real ones: la_qkv, q+gate, z/out, and lm_head -- which is 675 MB
//  and the single biggest weight in the model.
// ---------------------------------------------------------------------
template <class P>
static double mx_pass(sycl::queue& q, int M, const std::vector<uint8_t*>& W,
                      const uint8_t* SC, const cutlass::bfloat16_t* A,
                      cutlass::bfloat16_t* D, int N, int K) {
  double best = 1e18;
  for (int rep = 0; rep < 2; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    for (auto* w : W)
      grimoire_moe_raw::launch_dense_mxfp4<P, cutlass::bfloat16_t>(
          q, A, w, SC, D, M, N, K);
    q.wait_and_throw();
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    if (rep) best = std::min(best, ms);
  }
  return best;
}

// Same real shapes, but through the W4A8 int8xint4 path with the M16 tile --
// i.e. what a verify step would cost if the DN and attention projections were
// converted to int4 as well, not just the FFN.
static int w4small(sycl::queue& q) {
  struct Sh { int N, K, L; const char* name; double gemv_ms; };
  const Sh shapes[] = {
    {10240,  5120, 48, "la_qkv  x48", 3.44},
    {12288,  5120, 16, "q+gate  x16", 1.20},
    { 6144,  5120, 48, "z       x48", 1.72},
    { 5120,  6144, 48, "out     x48", 1.89},
    {248320, 5120,  1, "lm_head x1 ", 1.15},
  };
  std::mt19937 rng(13);
  printf("%-12s %8s %8s %8s %8s %8s   %s\n",
         "shape", "M=1", "M=2", "M=4", "M=8", "M=16", "decode GEMV");
  for (const auto& sh : shapes) {
    const size_t bw = size_t(sh.N) * sh.K / 2;
    const int kg = sh.K / 128;
    std::vector<uint8_t*> W(sh.L);
    { std::vector<uint8_t> t(bw);
      for (auto& b : t) b = uint8_t(rng());
      for (int l = 0; l < sh.L; ++l) { W[l] = sycl::malloc_device<uint8_t>(bw, q);
        q.memcpy(W[l], t.data(), bw).wait(); } }
    float* WS = sycl::malloc_device<float>(size_t(sh.N) * kg, q);
    float* AS = sycl::malloc_device<float>(16, q);
    int8_t* A = sycl::malloc_device<int8_t>(size_t(16) * sh.K, q);
    float*  D = sycl::malloc_device<float>(size_t(16) * sh.N, q);
    { std::vector<float> h(size_t(sh.N) * kg, 0.01f);
      q.memcpy(WS, h.data(), h.size()*sizeof(float)).wait(); }
    { std::vector<float> h(16, 0.002f);
      q.memcpy(AS, h.data(), h.size()*sizeof(float)).wait(); }
    q.memset(A, 3, size_t(16) * sh.K).wait();
    printf("%-12s", sh.name); fflush(stdout);
    for (int M : {1, 2, 4, 8, 16}) {
      double best = 1e18;
      for (int rep = 0; rep < 2; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (auto* w : W)
          grimoire_moe_raw::launch_dense_w4a8<W4A8_M16, 128, float>(
              q, A, w, WS, AS, D, M, sh.N, sh.K);
        q.wait_and_throw();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (rep) best = std::min(best, ms);
      }
      printf(" %8.2f", best);
    }
    printf("   %8.2f\n", sh.gemv_ms);
    for (auto* w : W) sycl::free(w, q);
    sycl::free(WS,q); sycl::free(AS,q); sycl::free(A,q); sycl::free(D,q);
  }
  return 0;
}

static int mxsmall(sycl::queue& q) {
  struct Sh { int N, K, L; const char* name; };
  const Sh shapes[] = {
    {10240,  5120, 48, "la_qkv  x48"},
    {12288,  5120, 16, "q+gate  x16"},
    { 6144,  5120, 48, "z       x48"},
    {248320, 5120,  1, "lm_head x1 "},
  };
  std::mt19937 rng(11);
  printf("%-12s %9s %9s %9s %9s %9s\n", "shape", "M=1", "M=2", "M=4", "M=8", "M=16");
  for (const auto& sh : shapes) {
    const size_t bw = size_t(sh.N) * sh.K / 2, sw = size_t(sh.N) * sh.K / 32;
    std::vector<uint8_t*> W(sh.L);
    { std::vector<uint8_t> t(bw);
      for (auto& b : t) b = uint8_t(rng());
      for (int l = 0; l < sh.L; ++l) { W[l] = sycl::malloc_device<uint8_t>(bw, q);
        q.memcpy(W[l], t.data(), bw).wait(); } }
    uint8_t* SC = sycl::malloc_device<uint8_t>(sw, q);
    q.memset(SC, 127, sw).wait();
    auto* A = sycl::malloc_device<cutlass::bfloat16_t>(size_t(16) * sh.K, q);
    auto* D = sycl::malloc_device<cutlass::bfloat16_t>(size_t(16) * sh.N, q);
    q.memset(A, 0x3c, size_t(16) * sh.K * 2).wait();
    printf("%-12s", sh.name); fflush(stdout);
    for (int M : {1, 2, 4, 8, 16})
      printf(" %9.2f", mx_pass<W4A8_M16>(q, M, W, SC, A, D, sh.N, sh.K));
    printf("   M16 tile\n%-12s", "");
    for (int M : {1, 2, 4, 8, 16})
      printf(" %9.2f", mx_pass<W4A8_M128>(q, M, W, SC, A, D, sh.N, sh.K));
    printf("   M128 tile\n");
    for (auto* w : W) sycl::free(w, q);
    sycl::free(SC,q); sycl::free(A,q); sycl::free(D,q);
  }
  return 0;
}

static int smallm(sycl::queue& q) {
  {
    printf("correctness of the small tiles first:\n");
    int f = 0;
    f += verify_tile<W4A8_M8>(q,  "M8x256");
    f += verify_tile<W4A8_M16>(q, "M16x256");
    printf("  -> %s\n\n", f ? "FAILURES, timings below are meaningless" : "all exact");
    if (f) return 1;
  }
  const int H = 5120, I = 17408, L = 64;
  printf("FFN of %d layers: gate_up [%d x %d] + down [%d x %d]\n", L, 2*I, H, H, I);
  printf("reference: batched int4 GEMV 19.07 ms at M=1, 35.42 at M=4\n\n");
  std::mt19937 rng(3);
  std::vector<uint8_t*> Wg(L), Wd(L);
  { std::vector<uint8_t> t(size_t(2*I)*H/2);
    for (auto& b : t) b = uint8_t(rng());
    for (int l = 0; l < L; ++l) { Wg[l] = sycl::malloc_device<uint8_t>(t.size(), q);
      q.memcpy(Wg[l], t.data(), t.size()).wait(); } }
  { std::vector<uint8_t> t(size_t(H)*I/2);
    for (auto& b : t) b = uint8_t(rng());
    for (int l = 0; l < L; ++l) { Wd[l] = sycl::malloc_device<uint8_t>(t.size(), q);
      q.memcpy(Wd[l], t.data(), t.size()).wait(); } }
  const int MMAX = 32;
  int8_t* A  = sycl::malloc_device<int8_t>(size_t(MMAX)*I, q);
  float*  D  = sycl::malloc_device<float>(size_t(MMAX)*2*I, q);
  float*  AS = sycl::malloc_device<float>(MMAX, q);
  float*  WSg= sycl::malloc_device<float>(size_t(2*I)*(H/128), q);
  float*  WSd= sycl::malloc_device<float>(size_t(H)*(I/128), q);
  { std::vector<int8_t> h(size_t(MMAX)*I);
    for (auto& v : h) v = int8_t(int(rng()&0xF)-8);
    q.memcpy(A, h.data(), h.size()).wait(); }
  { std::vector<float> h(size_t(2*I)*(H/128), 0.01f);
    q.memcpy(WSg, h.data(), h.size()*sizeof(float)).wait(); }
  { std::vector<float> h(size_t(H)*(I/128), 0.01f);
    q.memcpy(WSd, h.data(), h.size()*sizeof(float)).wait(); }
  { std::vector<float> h(MMAX, 0.002f);
    q.memcpy(AS, h.data(), h.size()*sizeof(float)).wait(); }

  printf("%-10s %9s %9s %9s %9s\n", "tile", "M=2", "M=4", "M=8", "M=16");
  printf("%-10s", "M8x256"); fflush(stdout);
  for (int M : {2,4,8,16}) printf(" %9.2f", ffn_pass<W4A8_M8>(q,M,Wg,Wd,A,WSg,WSd,AS,D,H,I));
  printf("\n%-10s", "M16x256"); fflush(stdout);
  for (int M : {2,4,8,16}) printf(" %9.2f", ffn_pass<W4A8_M16>(q,M,Wg,Wd,A,WSg,WSd,AS,D,H,I));
  printf("\n%-10s", "M32x256"); fflush(stdout);
  for (int M : {2,4,8,16}) printf(" %9.2f", ffn_pass<W4A8_M32>(q,M,Wg,Wd,A,WSg,WSd,AS,D,H,I));
  printf("\n%-10s", "M128x256"); fflush(stdout);
  for (int M : {2,4,8,16}) printf(" %9.2f", ffn_pass<W4A8_M128>(q,M,Wg,Wd,A,WSg,WSd,AS,D,H,I));
  printf("\n");
  for (int l = 0; l < L; ++l) { sycl::free(Wg[l], q); sycl::free(Wd[l], q); }
  sycl::free(A,q); sycl::free(D,q); sycl::free(AS,q);
  sycl::free(WSg,q); sycl::free(WSd,q);
  return 0;
}

static int bench_shape(sycl::queue& q, int N, int K, const char* name, double mxfp4_ref);

// Reproduce the ENGINE's gate_up call exactly: bf16 output, real shape, and
// the small M the first prefill actually uses.  This is the one combination
// no earlier test covered, and it is where the engine died with DEVICE_LOST.
static int bf16out(sycl::queue& q) {
  using Op  = XE_DPAS_TT<8, int32_t, int8_t, cute::int4_t, int32_t>;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>,
      Layout<typename W4A8Policy::WGTile>, typename W4A8Policy::SGLayout>::TiledMMA;
  using CA = typename W4A8Policy::GmemTiledCopyA;
  using CB = typename W4A8Policy::GmemTiledCopyB;
  struct Case { int M, N, K; };
  const Case cases[] = { {32,256,256}, {32,34816,5120}, {4096,34816,5120} };
  int fails = 0;
  for (const auto& c : cases) {
    printf("  bf16 out  M=%-5d N=%-6d K=%-5d ... ", c.M, c.N, c.K); fflush(stdout);
    std::mt19937 r(5);
    const int kg = c.K / 128;
    std::vector<uint8_t> hB(size_t(c.N)*c.K/2);
    for (auto& v : hB) v = uint8_t(r());
    std::vector<int8_t> hA(size_t(c.M)*c.K);
    for (auto& v : hA) v = int8_t(int(r()&0xF)-8);
    std::vector<float> hWS(size_t(c.N)*kg), hAS(size_t(c.M), 0.002f);
    for (auto& v : hWS) v = 0.01f;
    int8_t*  a = sycl::malloc_device<int8_t>(hA.size(), q);
    uint8_t* b = sycl::malloc_device<uint8_t>(hB.size(), q);
    float*   ws= sycl::malloc_device<float>(hWS.size(), q);
    float*   as= sycl::malloc_device<float>(hAS.size(), q);
    auto*    d = sycl::malloc_device<cutlass::bfloat16_t>(size_t(c.M)*c.N, q);
    if (!a||!b||!ws||!as||!d) { printf("ALLOC FAILED\n"); ++fails; continue; }
    q.memcpy(a,hA.data(),hA.size()).wait();
    q.memcpy(b,hB.data(),hB.size()).wait();
    q.memcpy(ws,hWS.data(),hWS.size()*sizeof(float)).wait();
    q.memcpy(as,hAS.data(),hAS.size()*sizeof(float)).wait();
    q.memset(d,0,size_t(c.M)*c.N*sizeof(cutlass::bfloat16_t)).wait();
    run_tiles_grouped<MMA,CA,CB,128,cutlass::bfloat16_t>(
        q,a,b,ws,as,d,c.M,c.N,c.K,32);
    q.wait_and_throw();
    printf("survived\n");
    sycl::free(a,q); sycl::free(b,q); sycl::free(ws,q); sycl::free(as,q); sycl::free(d,q);
  }
  printf("\nbf16-out: %s\n", fails?"FAILURES":"all survived");
  return fails;
}

// PARTIAL M TILES.  The engine's first prefill call is M=32 against a TM=128
// tile, and that is where the in-engine path died with DEVICE_LOST.  The
// original validation used M=128 (exactly one tile) and M=4096 (32 full
// tiles), so a partial M tile was never covered.
static int partial(sycl::queue& q) {
  const int N = 256, K = 256;
  int fails = 0;
  for (int M : {32, 64, 96, 100, 128, 130}) {
    std::mt19937 r(77);
    std::vector<int8_t> hA(size_t(M)*K), hBu(size_t(N)*K);
    std::vector<uint8_t> hB(size_t(N)*K/2);
    for (auto& v : hA)  v = int8_t(int(r()&0xF)-8);
    for (auto& v : hBu) v = int8_t(int(r()&0xF)-8);
    for (size_t i=0;i<hB.size();++i)
      hB[i] = uint8_t((uint8_t(hBu[2*i])&0x0F)|(uint8_t(hBu[2*i+1])<<4));
    int8_t*  a = sycl::malloc_device<int8_t>(hA.size(), q);
    uint8_t* b = sycl::malloc_device<uint8_t>(hB.size(), q);
    int32_t* d = sycl::malloc_device<int32_t>(size_t(M)*N, q);
    q.memcpy(a, hA.data(), hA.size()).wait();
    q.memcpy(b, hB.data(), hB.size()).wait();
    q.memset(d, 0, size_t(M)*N*sizeof(int32_t)).wait();
    printf("  M=%-4d ... ", M); fflush(stdout);
    using Op  = XE_DPAS_TT<8, int32_t, int8_t, cute::int4_t, int32_t>;
    using MMA = typename TiledMMAHelper<MMA_Atom<Op>,
        Layout<typename W4A8Policy::WGTile>, typename W4A8Policy::SGLayout>::TiledMMA;
    run_tiles<MMA, typename W4A8Policy::GmemTiledCopyA,
                   typename W4A8Policy::GmemTiledCopyB>(q, a, b, d, M, N, K, 4);
    q.wait_and_throw();
    std::vector<int32_t> got(size_t(M)*N);
    q.memcpy(got.data(), d, got.size()*sizeof(int32_t)).wait();
    size_t bad = 0;
    for (int m=0;m<M;++m) for (int n2=0;n2<N;++n2) {
      int32_t acc=0;
      for (int k=0;k<K;++k) acc += int32_t(hA[size_t(m)*K+k])*int32_t(hBu[size_t(n2)*K+k]);
      if (got[size_t(m)*N+n2]!=acc) ++bad;
    }
    printf("%s (%zu mismatches)\n", bad?"FAIL":"ok", bad);
    if (bad) ++fails;
    sycl::free(a,q); sycl::free(b,q); sycl::free(d,q);
  }
  printf("\npartial-M: %s\n", fails?"FAILURES ABOVE":"all clean");
  return fails;
}

static int bench(sycl::queue& q) {
  bench_shape(q, 34816, 5120,  "ffn-gate-up", 99.5);
  bench_shape(q,  5120, 17408, "ffn-down",    99.5);
  bench_shape(q, 10240, 5120,  "dn-qkv",      99.8);
  return 0;
}

static int bench_shape(sycl::queue& q, int N, int K, const char* name, double mxfp4_ref) {
  const int M = 4096, L = 64;
  printf("\n=== %s  N=%d K=%d M=%d, %d distinct matrices\n", name, N, K, M, L);
  using Op  = XE_DPAS_TT<8, int32_t, int8_t, cute::int4_t, int32_t>;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>,
      Layout<typename W4A8Policy::WGTile>, typename W4A8Policy::SGLayout>::TiledMMA;
  using CA = typename W4A8Policy::GmemTiledCopyA;
  using CB = typename W4A8Policy::GmemTiledCopyB;
  int8_t*  A = sycl::malloc_device<int8_t>(size_t(M)*K, q);
  int32_t* D = sycl::malloc_device<int32_t>(size_t(M)*N, q);
  std::vector<uint8_t*> W(L);
  std::mt19937 rng(9);
  std::vector<uint8_t> tmp(size_t(N)*K/2);
  for (int i = 0; i < L; ++i) {
    W[i] = sycl::malloc_device<uint8_t>(tmp.size(), q);
    for (auto& b : tmp) b = uint8_t(rng());
    q.memcpy(W[i], tmp.data(), tmp.size()).wait();
  }
  { std::vector<int8_t> ha(size_t(M)*K);
    for (auto& v : ha) v = int8_t(int(rng() & 0xF) - 8);
    q.memcpy(A, ha.data(), ha.size()).wait(); }
  static bool validated = false;
  // Validate THIS kernel -- the one with prefetch and the grid-stride loop --
  // on a small shape before believing any timing from it.  A fast kernel that
  // computes the wrong thing has cost this project two false results already.
  if (!validated) {
    validated = true;
    const int vM=128, vN=256, vK=256;
    std::mt19937 vr(4321);
    std::vector<int8_t> vA(size_t(vM)*vK), vBu(size_t(vN)*vK);
    std::vector<uint8_t> vB(size_t(vN)*vK/2);
    for (auto& v : vA)  v = int8_t(int(vr()&0xF)-8);
    for (auto& v : vBu) v = int8_t(int(vr()&0xF)-8);
    for (size_t i=0;i<vB.size();++i)
      vB[i] = uint8_t((uint8_t(vBu[2*i])&0x0F) | (uint8_t(vBu[2*i+1])<<4));
    int8_t*  a = sycl::malloc_device<int8_t>(vA.size(), q);
    uint8_t* b = sycl::malloc_device<uint8_t>(vB.size(), q);
    int32_t* d = sycl::malloc_device<int32_t>(size_t(vM)*vN, q);
    q.memcpy(a, vA.data(), vA.size()).wait();
    q.memcpy(b, vB.data(), vB.size()).wait();
    q.memset(d, 0, size_t(vM)*vN*sizeof(int32_t)).wait();
    run_tiles<MMA,CA,CB>(q, a, b, d, vM, vN, vK, 4);
    q.wait_and_throw();
    std::vector<int32_t> got(size_t(vM)*vN);
    q.memcpy(got.data(), d, got.size()*sizeof(int32_t)).wait();
    size_t bad = 0;
    for (int m=0;m<vM;++m) for (int n=0;n<vN;++n) {
      int32_t acc=0;
      for (int k=0;k<vK;++k) acc += int32_t(vA[size_t(m)*vK+k])*int32_t(vBu[size_t(n)*vK+k]);
      if (got[size_t(m)*vN+n] != acc) ++bad;
    }
    printf("  bench-kernel validation: %zu mismatches -> %s\n", bad,
           bad ? "INVALID, timing below is meaningless" : "bit-exact");
    sycl::free(a,q); sycl::free(b,q); sycl::free(d,q);
    if (bad) return 1;
  }
  const int eus = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  MMA probe{}; const int groups = eus * 512 / size(probe);
  printf("  %d work-groups of %d threads (%d Xe-cores reported)\n",
         groups, int(size(probe)), eus);
  double best = 1e18;
  for (int r = 0; r < 3; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < L; ++i)
      run_tiles<MMA,CA,CB>(q, A, W[i], D, M, N, K, groups);
    q.wait_and_throw();
    double ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-t0).count() / L;
    if (r) best = std::min(best, ms);
  }
  const double tflops = 2.0*M*N*K / (best*1e-3) / 1e12;
  printf("\n  %.3f ms/layer   %.1f TFLOP/s\n", best, tflops);
  printf("  MXFP4 same shape/conditions: %.1f TFLOP/s  -> %.2fx\n",
         mxfp4_ref, tflops/mxfp4_ref);
  // ---- now with real g128 group rescaling + per-row activation scale ----
  {
    const int kg = K / 128;
    float* WS = sycl::malloc_device<float>(size_t(N)*kg, q);
    float* AS = sycl::malloc_device<float>(size_t(M), q);
    float* Df = sycl::malloc_device<float>(size_t(M)*N, q);
    { std::vector<float> h(size_t(N)*kg);
      for (auto& v : h) v = 0.01f + 0.001f*float(rng()%97);
      q.memcpy(WS, h.data(), h.size()*sizeof(float)).wait(); }
    { std::vector<float> h(size_t(M), 0.002f);
      q.memcpy(AS, h.data(), h.size()*sizeof(float)).wait(); }
    printf("\n  with g128 rescale + per-row act scale:\n");
    double bg = 1e18;
    for (int r = 0; r < 3; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < L; ++i)
        run_tiles_grouped<MMA,CA,CB,128>(q, A, W[i], WS, AS, Df, M, N, K, groups);
      q.wait_and_throw();
      double ms = std::chrono::duration<double,std::milli>(
          std::chrono::steady_clock::now()-t0).count() / L;
      if (r) bg = std::min(bg, ms);
    }
    const double tf = 2.0*M*N*K / (bg*1e-3) / 1e12;
    printf("    %.3f ms/layer   %.1f TFLOP/s   (%.2fx MXFP4)\n", bg, tf, tf/mxfp4_ref);
    printf("    rescale overhead vs raw: %.1f%%\n", 100.0*(bg-best)/best);
    sycl::free(WS,q); sycl::free(AS,q); sycl::free(Df,q);
  }
  for (auto* w : W) sycl::free(w, q);
  sycl::free(A,q); sycl::free(D,q);
  return 0;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int M = 128, N = 256;
  const int K = (argc > 1) ? std::atoi(argv[1]) : 256;

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  if (argc > 1 && std::string(argv[1]) == "w4small") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    return w4small(q);
  }
  if (argc > 1 && std::string(argv[1]) == "mxsmall") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    return mxsmall(q);
  }
  if (argc > 1 && std::string(argv[1]) == "smallm") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    return smallm(q);
  }
  if (argc > 1 && std::string(argv[1]) == "bf16") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    return bf16out(q);
  }
  if (argc > 1 && std::string(argv[1]) == "partial") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    printf("partial M tiles against TM=128, N=256 K=256\n\n");
    return partial(q);
  }
  if (argc > 1 && std::string(argv[1]) == "bench") {
    printf("device : %s\n",
           q.get_device().get_info<sycl::info::device::name>().c_str());
    return bench(q);
  }
  printf("device : %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("M=%d N=%d K=%d  (one work-group tile)\n\n", M, N, K);

  // ---- host data ----------------------------------------------------
  std::mt19937 rng(1234);
  std::uniform_int_distribution<int> da(-8, 7), db(-8, 7);
  std::vector<int8_t>  hA(size_t(M) * K);
  std::vector<int8_t>  hBv(size_t(N) * K);          // unpacked, for reference
  std::vector<uint8_t> hB(size_t(N) * K / 2);       // packed 2 nibbles/byte
  for (auto& v : hA) v = int8_t(da(rng));
  for (size_t i = 0; i < hBv.size(); ++i) hBv[i] = int8_t(db(rng));
  for (size_t i = 0; i < hB.size(); ++i)
    hB[i] = uint8_t((uint8_t(hBv[2*i]) & 0x0F) | (uint8_t(hBv[2*i+1]) << 4));

  std::vector<int32_t> ref(size_t(M) * N, 0);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k)
        acc += int32_t(hA[size_t(m)*K + k]) * int32_t(hBv[size_t(n)*K + k]);
      ref[size_t(m)*N + n] = acc;
    }

  int8_t*   dA = sycl::malloc_device<int8_t>(hA.size(), q);
  uint8_t*  dB = sycl::malloc_device<uint8_t>(hB.size(), q);
  int32_t*  dD = sycl::malloc_device<int32_t>(size_t(M) * N, q);
  q.memcpy(dA, hA.data(), hA.size()).wait();
  q.memcpy(dB, hB.data(), hB.size()).wait();
  q.memset(dD, 0, size_t(M) * N * sizeof(int32_t)).wait();

  // ---- the atom under test -------------------------------------------
  using Op  = XE_DPAS_TT<8, int32_t, int8_t, cute::int4_t, int32_t>;
  using MMA = typename TiledMMAHelper<MMA_Atom<Op>,
      Layout<typename W4A8Policy::WGTile>,
      typename W4A8Policy::SGLayout>::TiledMMA;
  MMA mma{};
  const int threads = size(mma);
  printf("TiledMMA instantiated: %d threads/work-group\n", threads);

  using CA = typename W4A8Policy::GmemTiledCopyA;
  using CB = typename W4A8Policy::GmemTiledCopyB;

  sycl::range<3> local(1, 1, threads);
  sycl::range<3> global(1, 1, 1);
  namespace sx = sycl::ext::oneapi::experimental;
  namespace ix = sycl::ext::intel::experimental;
  sx::properties props{sx::sub_group_size<16>, ix::grf_size<256>};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<W4A8Kernel>(
        sycl::nd_range<3>{global * local, local}, props,
        [=](auto item) {
          MMA mm{};
          auto A = MoE::make_moe_tensor<int8_t, 'R'>(const_cast<int8_t*>(dA), M, K);
          auto B = MoE::make_moe_tensor<cute::int4_t, 'R'>(
              reinterpret_cast<cute::int4_t*>(const_cast<uint8_t*>(dB)), N, K);

          auto wg_tile = mm.tile_mnk();
          Tensor cA = make_identity_tensor(A.shape());
          Tensor cB = make_identity_tensor(B.shape());
          Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(0, _));
          Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(0, _));

          const int local_id = int(item.get_local_linear_id());
          auto copy_a = MoE::get_block_2d_copy_A<CA>(mm, A);
          auto copy_b = MoE::get_block_2d_copy_B<CB>(mm, B);
          auto thr_mma    = mm.get_slice(local_id);
          auto thr_copy_a = copy_a.get_slice(local_id);
          auto thr_copy_b = copy_b.get_slice(local_id);

          auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
          auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));
          auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
          auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));
          Tensor tAgA = thr_copy_a.partition_S(gA);
          Tensor tBgB = thr_copy_b.partition_S(gB);

          auto D = MoE::make_moe_tensor<int32_t, 'R'>(dD, M, N);
          Tensor cC = make_identity_tensor(D.shape());
          Tensor gC = local_tile(cC, wg_tile, make_coord(0,0,0), Step<_1,_1,X>{});
          auto tCrC = thr_mma.partition_sg_fragment_C(gC);
          clear(tCrC);

          const int k_tiles = K / int(get<2>(wg_tile));
          for (int kt = 0; kt < k_tiles; ++kt) {
            copy(copy_a, tAgA(_,_,_,kt), tArA);
            copy(copy_b, tBgB(_,_,_,kt), tBrB);
            reorder(tArA, tCrA);
            reorder(tBrB, tCrB);
            cute::gemm(mm, tCrA, tCrB, tCrC);
          }

          // Manual store, using the (sn,sm) -> (m,n) mapping the bias
          // epilogue in gemm_xe2.hpp documents.  Doing it by hand also
          // VERIFIES that mapping, which the real epilogue will need to
          // apply the per-row activation scale.
          constexpr int ATOM_M = get<1>(typename MMA::ThrLayoutVMNK{}.shape());
          constexpr int ATOM_N = get<2>(typename MMA::ThrLayoutVMNK{}.shape());
          constexpr int SG_M = 128 / ATOM_M;
          constexpr int SG_N = 256 / ATOM_N;
          const int sg   = int(cutlass::get_sub_group_id());
          const int lane = int(cutlass::get_sub_group_local_id());
          const int m0 = (sg / ATOM_N) * SG_M;
          const int n0 = (sg % ATOM_N) * SG_N;
          CUTE_UNROLL
          for (int sn = 0; sn < SG_N / 16; ++sn)
            CUTE_UNROLL
            for (int sm = 0; sm < SG_M; ++sm) {
              const int m = m0 + sm;
              const int n = n0 + sn * 16 + lane;
              dD[size_t(m) * N + n] = tCrC(sn * SG_M + sm);
            }
        });
  }).wait();

  std::vector<int32_t> got(size_t(M) * N);
  q.memcpy(got.data(), dD, got.size() * sizeof(int32_t)).wait();

  size_t bad = 0; int first = -1;
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i] != ref[i]) { if (!bad) first = int(i); ++bad; }

  const bool allzero = [&]{ for (auto v : got) if (v) return false; return true; }();
  // Is `got` a PERMUTATION of `ref`?  If so the arithmetic is right and only
  // the accumulator -> (m,n) store mapping is wrong, which is a much smaller
  // problem than a broken atom.
  std::vector<int32_t> sg_(got), sr_(ref);
  std::sort(sg_.begin(), sg_.end()); std::sort(sr_.begin(), sr_.end());
  const bool perm = (sg_ == sr_);
  printf("  same multiset as reference: %s\n", perm ? "YES -> store mapping is wrong, math is RIGHT"
                                                     : "no  -> arithmetic differs");
  printf("\nresult: %zu / %zu mismatches%s\n", bad, got.size(),
              allzero ? "   *** OUTPUT IS ALL ZEROS ***" : "");
  if (bad) {
    printf("  first at %d: got %d want %d\n", first, got[first], ref[first]);
    for (int i = 0; i < 8; ++i)
      printf("  [%d] got %8d  want %8d\n", i, got[i], ref[i]);
  } else {
    printf("  BIT-EXACT against the host reference.\n");
    printf("  sample: D[0]=%d D[1]=%d D[N+1]=%d\n", got[0], got[1], got[N+1]);
  }
  sycl::free(dA,q); sycl::free(dB,q); sycl::free(dD,q);
  return bad ? 1 : 0;
}
