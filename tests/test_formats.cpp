// =====================================================================
//  test_formats.cpp  --  validates the decode path that the GPU kernels
//  use, on the host, with a plain C++ compiler. No GPU required.
//
//  Checks:
//    1. E8M0 / E2M1 / E4M3 / E5M2 codec conformance against the OCP spec
//    2. dequant(quantize(W)) error is within the theoretical bound for
//       every format
//    3. a full fp32 GEMV against dequantized weights, per format
// =====================================================================
#include "b70/weights.hpp"
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace b70;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

// ---------------------------------------------------------------------
static void test_e8m0() {
    std::printf("E8M0 shared exponent\n");
    CHECK(e8m0_to_f32(127) == 1.0f, "2^0 wrong: %g", e8m0_to_f32(127));
    CHECK(e8m0_to_f32(128) == 2.0f, "2^1 wrong: %g", e8m0_to_f32(128));
    CHECK(e8m0_to_f32(126) == 0.5f, "2^-1 wrong: %g", e8m0_to_f32(126));
    CHECK(e8m0_to_f32(254) == std::ldexp(1.0f, 127), "2^127 wrong");
    CHECK(e8m0_to_f32(1)   == std::ldexp(1.0f, -126), "2^-126 wrong");
    CHECK(e8m0_to_f32(0)   == std::ldexp(1.0f, -127), "2^-127 subnormal wrong");
    CHECK(std::isnan(e8m0_to_f32(255)), "0xFF must be NaN");
    // every code must be an exact power of two
    for (int x = 1; x < 255; ++x) {
        float v = e8m0_to_f32(uint8_t(x));
        int e; float m = std::frexp(v, &e);
        CHECK(m == 0.5f, "code %d not a power of two (%g)", x, v);
    }
}

static void test_e2m1() {
    std::printf("E2M1 (MXFP4 element)\n");
    const float expect[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (int i = 0; i < 8; ++i) {
        CHECK(e2m1_to_f32(uint8_t(i)) == expect[i], "code %d = %g", i, e2m1_to_f32(uint8_t(i)));
        CHECK(e2m1_to_f32(uint8_t(i | 8)) == -expect[i], "neg code %d", i);
    }
    // exact values must round-trip
    for (int i = 0; i < 16; ++i) {
        float v = e2m1_to_f32(uint8_t(i));
        uint8_t r = f32_to_e2m1(v);
        CHECK(e2m1_to_f32(r) == v, "roundtrip code %d -> %g -> %g", i, v, e2m1_to_f32(r));
    }
    // round-to-nearest-even on the midpoints
    CHECK(e2m1_to_f32(f32_to_e2m1(0.75f)) == 1.0f, "tie 0.75 -> %g", e2m1_to_f32(f32_to_e2m1(0.75f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(1.25f)) == 1.0f, "tie 1.25 -> %g", e2m1_to_f32(f32_to_e2m1(1.25f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(1.75f)) == 2.0f, "tie 1.75 -> %g", e2m1_to_f32(f32_to_e2m1(1.75f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(2.50f)) == 2.0f, "tie 2.50 -> %g", e2m1_to_f32(f32_to_e2m1(2.50f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(3.50f)) == 4.0f, "tie 3.50 -> %g", e2m1_to_f32(f32_to_e2m1(3.50f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(5.00f)) == 4.0f, "tie 5.00 -> %g", e2m1_to_f32(f32_to_e2m1(5.00f)));
    CHECK(e2m1_to_f32(f32_to_e2m1(99.0f)) == 6.0f, "saturate");
}

static void test_fp8() {
    std::printf("FP8 E4M3 / E5M2\n");
    // spec landmarks
    CHECK(e4m3_to_f32(0x7E) == 448.0f, "e4m3 max = %g", e4m3_to_f32(0x7E));
    CHECK(std::isnan(e4m3_to_f32(0x7F)), "e4m3 0x7F must be NaN");
    CHECK(std::isnan(e4m3_to_f32(0xFF)), "e4m3 0xFF must be NaN");
    CHECK(e4m3_to_f32(0x38) == 1.0f, "e4m3 1.0 = %g", e4m3_to_f32(0x38));
    CHECK(e4m3_to_f32(0x01) == std::ldexp(1.0f, -9), "e4m3 min subnormal");
    CHECK(e4m3_to_f32(0x08) == std::ldexp(1.0f, -6), "e4m3 min normal");

    CHECK(e5m2_to_f32(0x7B) == 57344.0f, "e5m2 max = %g", e5m2_to_f32(0x7B));
    CHECK(std::isinf(e5m2_to_f32(0x7C)), "e5m2 0x7C must be inf");
    CHECK(std::isnan(e5m2_to_f32(0x7D)), "e5m2 0x7D must be NaN");
    CHECK(e5m2_to_f32(0x3C) == 1.0f, "e5m2 1.0 = %g", e5m2_to_f32(0x3C));

    // exhaustive roundtrip over all finite codes
    int bad4 = 0, bad5 = 0;
    for (int b = 0; b < 256; ++b) {
        float v = e4m3_to_f32(uint8_t(b));
        if (std::isnan(v)) continue;
        if (f32_to_e4m3(v) != uint8_t(b) && !(v == 0.0f)) ++bad4;
    }
    for (int b = 0; b < 256; ++b) {
        float v = e5m2_to_f32(uint8_t(b));
        if (std::isnan(v) || std::isinf(v)) continue;
        if (f32_to_e5m2(v) != uint8_t(b) && !(v == 0.0f)) ++bad5;
    }
    CHECK(bad4 == 0, "e4m3 roundtrip failures: %d", bad4);
    CHECK(bad5 == 0, "e5m2 roundtrip failures: %d", bad5);
}

static void test_bf16() {
    std::printf("BF16 conversion\n");
    CHECK(bf16_to_f32(f32_to_bf16(1.0f)) == 1.0f, "1.0");
    CHECK(bf16_to_f32(f32_to_bf16(-2.5f)) == -2.5f, "-2.5");
    // bf16 has an 8-bit exponent, so scaling by a power of two is exact.
    // This is the property that makes MX shared scales free of error when
    // folded into the dequantized tile.
    for (int e = -60; e <= 60; ++e) {
        float s = std::ldexp(1.0f, e);
        float v = 1.3125f;   // exactly representable in bf16
        CHECK(bf16_to_f32(f32_to_bf16(v * s)) == v * s, "scale 2^%d not exact", e);
    }
}

// ---------------------------------------------------------------------
struct ErrStat { double rel_rms; double max_abs; };

static ErrStat roundtrip(const std::vector<float>& src, int N, int K, Fmt f) {
    PackedWeight p = quantize(src.data(), N, K, f);
    QuantWeight  w = p.view();
    double se = 0.0, sr = 0.0, mx = 0.0;
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k) {
            const double a = src[size_t(n) * K + k];
            const double b = w.at(n, k);
            se += (a - b) * (a - b);
            sr += a * a;
            mx = std::max(mx, std::fabs(a - b));
        }
    return { std::sqrt(se / sr), mx };
}

static void test_roundtrip() {
    std::printf("\nQuantization round-trip, 256x1024 gaussian weights\n");
    const int N = 256, K = 1024;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> src(size_t(N) * K);
    for (auto& v : src) v = nd(rng);

    // Theoretical relative-RMS ceilings. A uniform quantizer with m
    // mantissa levels has rel error ~ 1/(2*sqrt(3)*levels); block formats
    // beat per-tensor formats because outliers only poison 32 neighbours.
    struct Row { Fmt f; double bound; };
    const Row rows[] = {
        { Fmt::BF16,     0.005 },
        { Fmt::FP8_E4M3, 0.050 },
        { Fmt::FP8_E5M2, 0.100 },
        { Fmt::INT8,     0.020 },
        // INT4 group=128: a gaussian group spans about +/-3.2 sigma, so the
        // step is 6.4/15 = 0.43 sigma and the uniform-quantizer RMS is
        // step/sqrt(12) = 0.123 sigma. 0.13 is theory plus a little slack.
        { Fmt::INT4,     0.130 },
        { Fmt::MXFP8,    0.040 },
        { Fmt::MXFP4,    0.160 },
    };
    std::printf("  %-10s %8s %12s %12s %10s\n", "format", "bits/el", "rel-RMS", "max-abs", "bound");
    for (const Row& r : rows) {
        ErrStat e = roundtrip(src, N, K, r.f);
        std::printf("  %-10s %8.3f %12.6f %12.6f %10.3f  %s\n",
                    fmt_name(r.f), bits_per_elem(r.f), e.rel_rms, e.max_abs, r.bound,
                    e.rel_rms <= r.bound ? "ok" : "OVER");
        CHECK(e.rel_rms <= r.bound, "%s rel-RMS %.5f exceeds %.5f",
              fmt_name(r.f), e.rel_rms, r.bound);
    }
}

// ---------------------------------------------------------------------
// An outlier channel is the case that separates block-scaled formats
// from per-tensor ones. MX formats must barely notice it.
static void test_outliers() {
    std::printf("\nOutlier robustness, one 100x spike per row\n");
    const int N = 64, K = 512;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> src(size_t(N) * K);
    for (auto& v : src) v = nd(rng);
    for (int n = 0; n < N; ++n) src[size_t(n) * K + (n * 7) % K] = 2.0f;

    const Fmt fs[] = { Fmt::INT8, Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4 };
    for (Fmt f : fs) {
        ErrStat e = roundtrip(src, N, K, f);
        std::printf("  %-10s rel-RMS %.6f\n", fmt_name(f), e.rel_rms);
    }
    // per-tensor INT8 must degrade more than per-block MXFP8 here
    ErrStat i8 = roundtrip(src, N, K, Fmt::INT8);
    ErrStat m8 = roundtrip(src, N, K, Fmt::MXFP8);
    CHECK(m8.rel_rms < i8.rel_rms,
          "block scaling should beat per-channel under outliers (mxfp8 %.5f vs int8 %.5f)",
          m8.rel_rms, i8.rel_rms);
}

// ---------------------------------------------------------------------
// End-to-end GEMV: the operation the decode kernel performs.
static void test_gemv() {
    std::printf("\nGEMV y = W x, N=512 K=2048, relative L2 error vs fp32\n");
    const int N = 512, K = 2048;
    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    std::vector<float> W(size_t(N) * K), x(K), ref(N);
    for (auto& v : W) v = nd(rng);
    for (auto& v : x) v = nd(rng) * 50.0f;
    for (int n = 0; n < N; ++n) {
        double a = 0.0;
        for (int k = 0; k < K; ++k) a += double(W[size_t(n) * K + k]) * x[k];
        ref[n] = float(a);
    }
    const Fmt fs[] = { Fmt::BF16, Fmt::FP8_E4M3, Fmt::FP8_E5M2,
                       Fmt::INT8, Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4 };
    for (Fmt f : fs) {
        PackedWeight p = quantize(W.data(), N, K, f);
        QuantWeight  w = p.view();
        double se = 0.0, sr = 0.0;
        for (int n = 0; n < N; ++n) {
            double a = 0.0;
            for (int k = 0; k < K; ++k) a += double(w.at(n, k)) * x[k];
            se += (a - ref[n]) * (a - ref[n]);
            sr += double(ref[n]) * ref[n];
        }
        const double rel = std::sqrt(se / sr);
        std::printf("  %-10s %.6f   (%.2f GB per 1B params)\n",
                    fmt_name(f), rel, bits_per_elem(f) / 8.0);
        CHECK(rel < 0.25, "%s GEMV error %.4f too large", fmt_name(f), rel);
    }
}

int main() {
    std::printf("=== b70 format layer tests ===\n\n");
    test_e8m0();
    test_e2m1();
    test_fp8();
    test_bf16();
    test_roundtrip();
    test_outliers();
    test_gemv();
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
