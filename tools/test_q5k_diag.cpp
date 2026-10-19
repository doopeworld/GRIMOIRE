// Component-level B70 diagnostic for the packed Q5_K lane (one block, K=256).
// Compares per-lane header decode, low-nibble sum, qh-bit sum and dot/sum
// between the production load16-based path and plain scalar byte reads.
#include "b70/native_qk_gemv.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cmath>

using namespace b70;
struct Comp { float ds, mins, lo, hi, dot, sum; };

template <int V>
static inline Comp lane_comp(const uint8_t* p, const float* x, int lane) {
    Comp c{};
    const NativeQKBytes16 header = native_qk_load16(p);
    int sc, mn;
    if (V == 0) {
        if (lane < 4) { sc = native_qk_byte(header, lane + 4) & 63; mn = native_qk_byte(header, lane + 8) & 63; }
        else { sc = (native_qk_byte(header, lane + 8) & 15) | ((native_qk_byte(header, lane) >> 2) & 48);
               mn = (native_qk_byte(header, lane + 8) >> 4) | ((native_qk_byte(header, lane + 4) >> 2) & 48); }
    } else {
        const uint8_t* s = p + 4;
        if (lane < 4) { sc = s[lane] & 63; mn = s[lane + 4] & 63; }
        else { sc = (s[lane + 4] & 15) | ((s[lane - 4] >> 2) & 48); mn = (s[lane + 4] >> 4) | ((s[lane] >> 2) & 48); }
    }
    const float d = f16_to_f32(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
    const float dm = f16_to_f32(uint16_t(p[2]) | (uint16_t(p[3]) << 8));
    c.ds = d * float(sc); c.mins = dm * float(mn);
    const uint8_t* qs = p + 48 + (lane / 2) * 32;
    const NativeQKBytes16 a = native_qk_load16(qs), b = native_qk_load16(qs + 16);
    const NativeQKBytes16 qh = native_qk_load16(p + 16);
    float dot = 0, sum = 0, lo = 0, hi = 0;
    for (int i = 0; i < 16; ++i) {
        int q0, q1, h0, h1;
        if (V == 0 || V == 1) {
            q0 = (native_qk_byte(a, i) >> ((lane & 1) * 4)) & 15;
            q1 = (native_qk_byte(b, i) >> ((lane & 1) * 4)) & 15;
        } else {
            q0 = (qs[i] >> ((lane & 1) * 4)) & 15;
            q1 = (qs[16 + i] >> ((lane & 1) * 4)) & 15;
        }
        if (V == 0 || V == 3) { h0 = (native_qk_byte(qh, i) >> lane) & 1; h1 = (p[32 + i] >> lane) & 1; }
        else { h0 = (p[16 + i] >> lane) & 1; h1 = (p[32 + i] >> lane) & 1; }
        lo += float(q0 + q1); hi += float(h0 + h1);
        q0 |= h0 << 4; q1 |= h1 << 4;
        const float x0 = x[lane * 32 + i], x1 = x[lane * 32 + 16 + i];
        dot = __builtin_fmaf(float(q0), x0, dot);
        dot = __builtin_fmaf(float(q1), x1, dot);
        sum += x0 + x1;
    }
    c.lo = lo; c.hi = hi; c.dot = dot; c.sum = sum;
    return c;
}

template <int V>
static void run(sycl::queue& q, const uint8_t* p, const float* x, int N, int RB, Comp* out) {
    q.parallel_for(sycl::nd_range<1>(N * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int n = it.get_group(0), lane = it.get_local_id(0);
        Comp c{};
        if (lane < 8) c = lane_comp<V>(p + n * RB, x, lane);
        out[n * 16 + lane] = c;
    }).wait_and_throw();
}

int main() {
    constexpr int K = 256, N = 3;
    sycl::queue q{sycl::gpu_selector_v};
    int total_bad = 0;
    for (int off : {0, 1}) {
        const int RB = 176 + off;
        auto* p = sycl::malloc_shared<uint8_t>(N * RB + 64, q);
        auto* x = sycl::malloc_shared<float>(K, q);
        auto* out = sycl::malloc_shared<Comp>(N * 16, q);
        for (int k = 0; k < K; ++k) x[k] = float((k * 37) % 113 - 56) / 71.0f;
        for (int n = 0; n < N; ++n) {
            uint8_t* blk = p + off + n * RB;
            for (int i = 0; i < 176; ++i) blk[i] = uint8_t((i * 73 + n * 19) & 255);
            blk[0] = 0; blk[1] = 0x28; blk[2] = 0; blk[3] = 0x24;
        }
        for (int V = 0; V < 4; ++V) {
            if (V == 0) run<0>(q, p + off, x, N, RB, out);
            if (V == 1) run<1>(q, p + off, x, N, RB, out);
            if (V == 2) run<2>(q, p + off, x, N, RB, out);
            if (V == 3) run<3>(q, p + off, x, N, RB, out);
            int bad = 0;
            for (int n = 0; n < N; ++n) for (int lane = 0; lane < 8; ++lane) {
                const uint8_t* blk = p + off + n * RB;
                const int s0 = lane < 4 ? blk[4 + lane] & 63 : (blk[4 + lane + 4] & 15) | ((blk[4 + lane - 4] >> 2) & 48);
                const int m0 = lane < 4 ? blk[4 + lane + 4] & 63 : (blk[4 + lane + 4] >> 4) | ((blk[4 + lane] >> 2) & 48);
                float rlo = 0, rhi = 0, rdot = 0, rsum = 0;
                for (int i = 0; i < 32; ++i) {
                    const int lo = (blk[48 + (lane / 2) * 32 + i] >> ((lane & 1) * 4)) & 15;
                    const int hq = (blk[16 + i] >> lane) & 1;
                    rlo += lo; rhi += hq; rdot += float(lo | (hq << 4)) * x[lane * 32 + i]; rsum += x[lane * 32 + i];
                }
                const Comp& c = out[n * 16 + lane];
                const bool ok = c.ds == f16_to_f32(0x2800) * float(s0) && c.mins == f16_to_f32(0x2400) * float(m0) &&
                                c.lo == rlo && c.hi == rhi && std::fabs(c.dot - rdot) < 1e-2f && std::fabs(c.sum - rsum) < 1e-3f;
                if (!ok) {
                    ++bad;
                    std::printf("off=%d V=%d n=%d lane=%d ds=%g/%g mins=%g/%g lo=%g/%g hi=%g/%g dot=%g/%g sum=%g/%g\n", off, V, n, lane,
                                c.ds, f16_to_f32(0x2800) * float(s0), c.mins, f16_to_f32(0x2400) * float(m0), c.lo, rlo, c.hi, rhi,
                                c.dot, rdot, c.sum, rsum);
                }
            }
            std::printf("off=%d variant=%d bad=%d\n", off, V, bad);
            total_bad += bad;
        }
        sycl::free(out, q); sycl::free(x, q); sycl::free(p, q);
    }

    {   // raw dump: what does the device read at p[16..47] for lane 1, row 0?
        auto* p = sycl::malloc_shared<uint8_t>(256, q);
        auto* o = sycl::malloc_shared<int>(16 * 32, q);
        for (int i = 0; i < 176; ++i) p[i] = uint8_t((i * 73) & 255);
        q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int lane = it.get_local_id(0);
            for (int i = 0; i < 32; ++i) o[lane * 32 + i] = (p[16 + i] >> lane) & 1;
        }).wait_and_throw();
        for (int lane = 0; lane < 8; ++lane) {
            int bad = 0, cnt = 0;
            for (int i = 0; i < 32; ++i) { const int r = (p[16 + i] >> lane) & 1; bad += r != o[lane * 32 + i]; cnt += o[lane*32+i]; }
            std::printf("rawdump lane=%d bitmismatch=%d count=%d\n", lane, bad, cnt);
        }
    }
    return total_bad ? 1 : 0;
}
