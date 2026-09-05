// A/B the decode GEMV: current launch_gemv vs a streaming rewrite.
//
// Diagnosis being tested (2026-09-05): the current kernel re-reads the
// activation vector x from global memory for every row -- gemv_step.hpp's own
// comment says "64 bytes of L2 per 8 bytes of DRAM, 4x more than the algorithm
// needs". Measured, the real GEMV moves <=135 GB/s while bench_fmt does the
// SAME MXFP4 dequant at 586 GB/s.
//
// The rewrite:
//   * x staged ONCE into SLM per work-group, tiled when it does not fit
//   * one 16 B load per lane == exactly one MXFP4 block of 32 == one scale byte
//   * lanes sub-group contiguous, so a step retires 256 B
//   * UNROLL independent chains in explicit registers (indexed arrays cost 8%)
#include "kernels.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cmath>
using namespace b70;
typedef std::chrono::steady_clock clk;

static constexpr int SGSZ    = 16;
static constexpr int WG      = 256;          // 16 sub-groups
static constexpr int SGS     = WG / SGSZ;
static constexpr int UNROLL  = 4;
static constexpr int ELPL    = 32;           // elements per lane per chunk
static constexpr int ELPS    = SGSZ * ELPL;  // 512 elements per sub-group step
static constexpr int KT      = 4096;         // x tile in SLM (16 KB)

sycl::event gemv_stream(sycl::queue& q, const QuantWeight& w,
                        const float* x, float* y) {
    const int N = w.N, K = w.K;
    constexpr int RPSG = 4;                        // consecutive rows per sub-group
    const int rows_wg  = SGS * RPSG;
    const int n_wg     = (N + rows_wg - 1) / rows_wg;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float,1> nlut(16, h);
        sycl::local_accessor<float,1> slut(256, h);
        const QuantWeight wc = w;
        h.parallel_for(sycl::nd_range<1>(size_t(n_wg) * WG, WG),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SGSZ)]] {
            const int lid = int(it.get_local_id(0));
            const int sg  = lid / SGSZ, lane = lid % SGSZ;
            if (lid < 16) {
                const float m[8] = {0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};
                nlut[lid] = (lid & 8) ? -m[lid & 7] : m[lid & 7];
            }
            for (int i = lid; i < 256; i += WG)
                slut[i] = sycl::exp2(float(i) - 127.0f);
            sycl::group_barrier(it.get_group());

            const int n0 = (int(it.get_group(0)) * SGS + sg) * RPSG;

            // ONE ROW AT A TIME, fully streamed. Rows are adjacent in memory,
            // so consecutive rows continue the same linear run instead of the
            // old kernel's interleave-4-rows-at-each-k, which broke contiguity
            // every 8 bytes. x comes from global (L1-resident, 20 KB) rather
            // than SLM: staging it in SLM cost more than it saved.
            for (int r = 0; r < RPSG; ++r) {
                const int n = n0 + r;
                if (n >= N) break;
                const uint8_t* row = wc.payload + int64_t(n) * wc.row_bytes;
                const uint8_t* sc  = static_cast<const uint8_t*>(wc.scales)
                                   + int64_t(n) * wc.row_scales;
                float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
                int k = 0;
                for (; k + ELPS * UNROLL <= K; k += ELPS * UNROLL) {
                    float* av[UNROLL] = {&a0,&a1,&a2,&a3};
                    #pragma unroll
                    for (int u = 0; u < UNROLL; ++u) {
                        const int ke = k + u * ELPS + lane * ELPL;
                        const sycl::vec<uint32_t,4> v =
                            *reinterpret_cast<const sycl::vec<uint32_t,4>*>(row + (ke >> 1));
                        const float sfac = slut[sc[ke / kMXBlock]];
                        float t = 0.f;
                        #pragma unroll
                        for (int wd = 0; wd < 4; ++wd) {
                            const uint32_t word = v[wd];
                            #pragma unroll
                            for (int b = 0; b < 8; ++b)
                                t = sycl::fma(nlut[(word >> (4*b)) & 0xF],
                                              x[ke + wd*8 + b], t);
                        }
                        *av[u] += t * sfac;
                    }
                }
                for (; k + ELPS <= K; k += ELPS) {
                    const int ke = k + lane * ELPL;
                    const sycl::vec<uint32_t,4> v =
                        *reinterpret_cast<const sycl::vec<uint32_t,4>*>(row + (ke >> 1));
                    const float sfac = slut[sc[ke / kMXBlock]];
                    float t = 0.f;
                    #pragma unroll
                    for (int wd = 0; wd < 4; ++wd) {
                        const uint32_t word = v[wd];
                        #pragma unroll
                        for (int b = 0; b < 8; ++b)
                            t = sycl::fma(nlut[(word >> (4*b)) & 0xF],
                                          x[ke + wd*8 + b], t);
                    }
                    a0 += t * sfac;
                }
                float acc = a0+a1+a2+a3;
                for (int kk = k + lane; kk < K; kk += SGSZ) {
                    const uint8_t byte = row[kk >> 1];
                    const uint8_t nib = (kk & 1) ? (byte >> 4) : (byte & 0xF);
                    acc += nlut[nib] * slut[sc[kk / kMXBlock]] * x[kk];
                }
                const float tot = sycl::reduce_over_group(
                    it.get_sub_group(), acc, sycl::plus<float>());
                if (lane == 0) y[n] = tot;
            }
        });
    });
}

struct S { int n, k; const char* name; };
static const S SH[] = {{34816,5120,"ffn-gate-up"},{10240,5120,"la-qkv"},{5120,17408,"ffn-down"}};

int main() {
    sycl::queue q{sycl::gpu_selector_v,{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("%-13s %10s %10s %10s %10s  %s\n",
                "shape","cur GB/s","new GB/s","speedup","maxdiff","");
    for (const auto& z : SH) {
        QuantWeight w{}; w.fmt=Fmt::MXFP4; w.N=z.n; w.K=z.k;
        w.row_bytes=bytes_per_row(Fmt::MXFP4,z.k); w.row_scales=z.k/kMXBlock;
        const size_t pb=size_t(z.n)*w.row_bytes, sb=size_t(z.n)*w.row_scales;
        auto* pay=sycl::malloc_device<uint8_t>(pb,q);
        auto* scl=sycl::malloc_device<uint8_t>(sb,q);
        auto* x=sycl::malloc_device<float>(z.k,q);
        auto* y1=sycl::malloc_device<float>(z.n,q);
        auto* y2=sycl::malloc_device<float>(z.n,q);
        { std::vector<uint8_t> h(pb); uint32_t r=1u;
          for(size_t i=0;i<pb;++i){ r=r*1664525u+1013904223u; h[i]=uint8_t(r>>24); }
          q.memcpy(pay,h.data(),pb).wait();
          std::vector<uint8_t> s(sb); for(size_t i=0;i<sb;++i) s[i]=uint8_t(125+(i%5));
          q.memcpy(scl,s.data(),sb).wait();
          std::vector<float> hx(z.k); for(int i=0;i<z.k;++i) hx[i]=0.01f*float((i%7)-3);
          q.memcpy(x,hx.data(),size_t(z.k)*4).wait(); }
        w.payload=pay; w.scales=scl;

        auto time=[&](auto&& fn){ fn(); q.wait(); double b=1e18;
            for(int i=0;i<6;++i){ auto t=clk::now(); fn(); q.wait();
              double ms=std::chrono::duration<double,std::milli>(clk::now()-t).count();
              if(i&&ms<b)b=ms; } return b; };
        double m1=time([&]{ launch_gemv(q,w,x,y1,{}); });
        double m2=time([&]{ gemv_stream(q,w,x,y2); });

        std::vector<float> h1(z.n),h2(z.n);
        q.memcpy(h1.data(),y1,size_t(z.n)*4).wait();
        q.memcpy(h2.data(),y2,size_t(z.n)*4).wait();
        double md=0; for(int i=0;i<z.n;++i) md=std::max(md,double(std::fabs(h1[i]-h2[i])));
        const double g1=double(pb+sb)/m1*1e-6, g2=double(pb+sb)/m2*1e-6;
        std::printf("%-13s %10.1f %10.1f %9.2fx %10.4f  %s\n",
                    z.name,g1,g2,g2/g1,md, md<0.05?"MATCH":"*** MISMATCH ***");
        sycl::free(pay,q);sycl::free(scl,q);sycl::free(x,q);sycl::free(y1,q);sycl::free(y2,q);
    }
    return 0;
}
