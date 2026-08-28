// GRIMOIRE INT4 vs MXFP4 dense GEMM on Qwen's real prefill shapes, M=4096.
// Random (non-compressible) buffers: constant fills read unrealistically fast on
// the B70 because of lossless memory compression.
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>
#include "xe2_grouped_raw_launcher.hpp"

using bf16 = cutlass::bfloat16_t;

// Local copy of the production policy (the real one lives in the bridge's
// anonymous namespace and is not visible from here).
class bp128x256 : public MoE::xe_gemm_policy_base { public:
 using WGTile=cute::Shape<cute::_128,cute::_256,cute::_32>;
 using SGLayout=cute::Layout<cute::Shape<cute::_4,cute::_8,cute::_1>,
   cute::Stride<cute::_8,cute::_1,cute::_0>>;
 using GmemTiledCopyD=cute::XE_STORE_2D<16,8,32>;
};

struct Shape { int n, k; const char* name; };
static const Shape g_shapes[] = { {34816,5120,"ffn-gate-up"},
                                  {5120,17408,"ffn-down"},
                                  {10240,5120,"dn-qkv"} };
enum { kM = 4096 };

typedef std::chrono::steady_clock clk;

int main() {
  sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
  printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
  printf("%-13s %7s %7s | %9s %8s | %14s %8s | %s\n",
         "shape","N","K","mxfp4 ms","TFLOP/s","int4 ms","TFLOP/s","winner");
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> ux(-1.0f, 1.0f);
  std::uniform_real_distribution<float> us(0.01f, 0.05f);

  for (int si = 0; si < 3; ++si) {
    const Shape z = g_shapes[si];
    const size_t bw = size_t(z.n) * z.k / 2;
    bf16* x  = sycl::malloc_device<bf16>(size_t(kM) * z.k, q);
    unsigned char* w  = sycl::malloc_device<unsigned char>(bw, q);
    unsigned char* sm = sycl::malloc_device<unsigned char>(size_t(z.n)*z.k/32, q);
    bf16* y  = sycl::malloc_device<bf16>(size_t(kM) * z.n, q);

    { std::vector<unsigned char> h(bw);
      for (size_t i = 0; i < h.size(); ++i) h[i] = (unsigned char)(rng() & 0xff);
      q.memcpy(w, h.data(), bw).wait();
      std::vector<unsigned char> hs(size_t(z.n)*z.k/32, 127);
      q.memcpy(sm, hs.data(), hs.size()).wait();
      std::vector<bf16> hx(size_t(kM)*z.k);
      for (size_t i = 0; i < hx.size(); ++i) hx[i] = bf16(ux(rng));
      q.memcpy(x, hx.data(), hx.size()*sizeof(bf16)).wait(); }

    double mx = 1e18;
    for (int r = 0; r < 6; ++r) {
      clk::time_point t0 = clk::now();
      grimoire_moe_raw::launch_dense_mxfp4<bp128x256>(q, x, w, sm, y, kM, z.n, z.k);
      q.wait_and_throw();
      double ms = std::chrono::duration<double, std::milli>(clk::now()-t0).count();
      if (r && ms < mx) mx = ms;
    }

    double bi = 1e18; int bg = 0;
    const int groups[3] = {32, 64, 128};
    for (int gi = 0; gi < 3; ++gi) {
      const int g = groups[gi];
      bf16* sb = sycl::malloc_device<bf16>(size_t(z.n)*(z.k/g), q);
      std::vector<bf16> hs(size_t(z.n)*(z.k/g));
      for (size_t i = 0; i < hs.size(); ++i) hs[i] = bf16(us(rng));
      q.memcpy(sb, hs.data(), hs.size()*sizeof(bf16)).wait();
      double t = 1e18;
      try {
        for (int r = 0; r < 6; ++r) {
          clk::time_point t0 = clk::now();
          if (g == 32)  grimoire_moe_raw::launch_dense_int4<bp128x256,32 >(q,x,w,sb,y,kM,z.n,z.k);
          if (g == 64)  grimoire_moe_raw::launch_dense_int4<bp128x256,64 >(q,x,w,sb,y,kM,z.n,z.k);
          if (g == 128) grimoire_moe_raw::launch_dense_int4<bp128x256,128>(q,x,w,sb,y,kM,z.n,z.k);
          q.wait_and_throw();
          double ms = std::chrono::duration<double, std::milli>(clk::now()-t0).count();
          if (r && ms < t) t = ms;
        }
      } catch (sycl::exception const& e) { printf("  [g=%d failed: %s]\n", g, e.what()); }
      if (t < bi) { bi = t; bg = g; }
      sycl::free(sb, q);
    }

    const double fl = 2.0 * kM * z.n * z.k;
    char i4[48];
    if (bg) snprintf(i4, sizeof i4, "%9.3f(g%d)", bi, bg); else snprintf(i4, sizeof i4, "%14s", "FAILED");
    printf("%-13s %7d %7d | %9.3f %8.1f | %14s %8.1f | %s\n",
           z.name, z.n, z.k, mx, fl/mx*1e-9, i4,
           bg ? fl/bi*1e-9 : 0.0, (bg && bi < mx) ? "INT4" : "MXFP4");
    sycl::free(x,q); sycl::free(w,q); sycl::free(sm,q); sycl::free(y,q);
  }
  return 0;
}
