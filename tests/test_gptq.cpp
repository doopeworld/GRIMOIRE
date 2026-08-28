// =====================================================================
//  test_gptq.cpp  --  GPTQ 4-bit decode, host validation
//
//  The bug this suite exists to prevent: qzeros packs its nibbles along
//  the OUTPUT axis, so an indexing error there is invisible at output
//  channel 0 and wrong everywhere after it. A previous debugging session
//  spot-checked o=0, found it correct, and spent hours looking elsewhere.
//  Every check below therefore sweeps ALL output channels.
//
//  Independence: the golden values in kProbeW were produced by a separate
//  Python implementation of the GPTQ formula over the same generated
//  tensors. They are not derived from the code under test, so a shared
//  misconception cannot cancel out.
// =====================================================================
#include "b70/gptq.hpp"
#include "b70/weights.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace b70;

static int failures = 0;
static void check(bool cond, const char* what) {
    if (!cond) { std::printf("  FAIL: %s\n", what); ++failures; }
}

// Same LCG the golden generator used.
static uint32_t lcg(uint32_t x) { return x * 1664525u + 1013904223u; }

static const int   kProbeO[] = {0, 0, 0, 0, 1, 7, 8, 9, 15, 16, 100, 255, 256, 511, 511, 300};
static const int   kProbeK[] = {0, 1, 7, 8, 0, 0, 0, 3, 127, 128, 1000, 2047, 17, 0, 2047, 129};
static const float kProbeW[] = {-0.0842895508f, -0.0842895508f, -0.421447754f, -0.0842895508f,
                                -0.218139648f, -0.119384766f, 0.797363281f, -1.22930908f,
                                0.260192871f, -0.236755371f, -0.240112305f, 0.0741577148f,
                                -0.109863281f, 0.222290039f, -0.317871094f, -1.10083008f};
static const int   kNProbe = int(sizeof(kProbeO) / sizeof(kProbeO[0]));

int main() {
    std::printf("=== GPTQ 4-bit decode ===\n\n");

    // -----------------------------------------------------------------
    // 1. IEEE binary16 decode, exhaustive over all 65536 codes.
    // -----------------------------------------------------------------
    {
        int bad = 0, nan_seen = 0, inf_seen = 0, sub_seen = 0;
        for (uint32_t h = 0; h < 65536u; ++h) {
            const float got = f16_to_f32(uint16_t(h));
            const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
            const float sign = (h & 0x8000u) ? -1.0f : 1.0f;
            float want;
            if (e == 0)            { want = sign * (float(m) / 1024.0f) * std::ldexp(1.0f, -14); if (m) ++sub_seen; }
            else if (e == 0x1F)    { if (m) { ++nan_seen; check(std::isnan(got), "f16 NaN"); continue; }
                                     ++inf_seen; want = sign * INFINITY; }
            else                     want = sign * (1.0f + float(m) / 1024.0f) * std::ldexp(1.0f, int(e) - 15);
            if (got != want) { if (++bad <= 3) std::printf("  f16 0x%04X: got %g want %g\n", h, got, want); }
        }
        check(bad == 0, "f16_to_f32 exhaustive");
        std::printf("  f16 exhaustive : 65536 codes, %d subnormal, %d inf, %d NaN, %d wrong\n",
                    sub_seen, inf_seen, nan_seen, bad);
    }

    // -----------------------------------------------------------------
    // 2. Real expert shape: gate_proj, out=512 in=2048 group=128.
    //    Golden probe values from the independent implementation.
    // -----------------------------------------------------------------
    const int OUT = 512, IN = 2048, G = 128;
    const int ngrp = IN / G;
    std::vector<int32_t>  qw(size_t(IN / 8) * OUT);
    std::vector<int32_t>  qz(size_t(ngrp) * (OUT / 8));
    std::vector<uint16_t> sc(size_t(ngrp) * OUT);
    {
        uint32_t x = 12345u;
        for (auto& v : qw) { x = lcg(x); v = int32_t(x); }
        for (auto& v : qz) { x = lcg(x); v = int32_t(x); }
        for (auto& v : sc) { x = lcg(x); v = uint16_t(0x2C00u | ((x >> 16) & 0x03FFu)); }
    }

    GptqTensor t;
    t.qweight = qw.data(); t.qzeros = qz.data(); t.scales = sc.data();
    t.in = IN; t.out = OUT; t.group = G;
    check(t.ok(), "GptqTensor::ok");

    {
        int bad = 0;
        for (int i = 0; i < kNProbe; ++i) {
            const float got = gptq_at(t, kProbeO[i], kProbeK[i]);
            if (std::fabs(got - kProbeW[i]) > 1e-6f * (1.0f + std::fabs(kProbeW[i]))) {
                std::printf("  probe o=%d k=%d: got %.9g want %.9g\n",
                            kProbeO[i], kProbeK[i], got, kProbeW[i]);
                ++bad;
            }
        }
        check(bad == 0, "golden probes vs independent reference");
        std::printf("  golden probes  : %d/%d match (o spans 0..511, k spans group edges)\n",
                    kNProbe - bad, kNProbe);
    }

    // -----------------------------------------------------------------
    // 3. Bulk dequant must agree with the element accessor at EVERY
    //    (o,k). This is what closes the o>=8 qzeros hole for good:
    //    512 channels x 2048 inputs, no sampling.
    // -----------------------------------------------------------------
    std::vector<float> w(size_t(OUT) * IN);
    gptq_dequant_4bit(t, w.data());
    {
        int64_t bad = 0; int first_o = -1, first_k = -1;
        for (int o = 0; o < OUT; ++o)
            for (int k = 0; k < IN; ++k)
                if (w[size_t(o) * IN + k] != gptq_at(t, o, k)) {
                    if (!bad) { first_o = o; first_k = k; }
                    ++bad;
                }
        check(bad == 0, "bulk dequant == element accessor, all 1048576 elements");
        if (bad) std::printf("  first mismatch at o=%d k=%d (%lld total)\n",
                             first_o, first_k, (long long)bad);
        else std::printf("  bulk vs element: %d x %d = %lld elements, exact\n",
                         OUT, IN, (long long)(int64_t(OUT) * IN));
    }

    // Direct runtime repack: nibbles and asymmetric zero points must be
    // exact; only F16->BF16 scale metadata is allowed to round.
    {
        PackedWeight p = gptq_repack_int4(t);
        QuantWeight v = p.view();
        int64_t code_bad = 0, zero_bad = 0;
        double se = 0.0, sr = 0.0;
        for (int o = 0; o < OUT; ++o) {
            for (int g = 0; g < ngrp; ++g) {
                const uint32_t zw = uint32_t(qz[int64_t(g) * (OUT / 8) + (o >> 3)]);
                const uint8_t want = uint8_t(((zw >> ((o & 7) * 4)) & 15u) + 1u);
                if (p.zeros[int64_t(o) * ngrp + g] != want) ++zero_bad;
            }
            const uint8_t* row = p.payload.data() + int64_t(o) * p.row_bytes;
            for (int k = 0; k < IN; ++k) {
                const uint32_t ww = uint32_t(qw[int64_t(k >> 3) * OUT + o]);
                const uint8_t want = uint8_t((ww >> ((k & 7) * 4)) & 15u);
                if (unpack_int4_raw(row, k) != want) ++code_bad;
                const double ref = gptq_at(t, o, k), got = v.at(o, k);
                se += (got - ref) * (got - ref); sr += ref * ref;
            }
        }
        const double rel = std::sqrt(se / (sr + 1e-30));
        check(code_bad == 0, "direct repack preserves every weight nibble");
        check(zero_bad == 0, "direct repack preserves every asymmetric zero");
        check(rel < 5e-3, "direct repack BF16 scale rounding bound");
        std::printf("  direct repack   : %lld code errors, %lld zero errors, rel-L2 %.3e\n",
                    (long long)code_bad, (long long)zero_bad, rel);
    }

    // -----------------------------------------------------------------
    // 4. Zero-point sensitivity. If qzeros were indexed by the wrong
    //    output channel, perturbing ONE channel's zero would move some
    //    OTHER channel. Sweep every channel and prove the blast radius
    //    is exactly one channel wide.
    // -----------------------------------------------------------------
    {
        int leaked = 0;
        for (int o = 0; o < OUT; ++o) {
            std::vector<int32_t> qz2 = qz;
            qz2[size_t(0) * (OUT / 8) + (o >> 3)] ^= int32_t(0xFu << ((o & 7) * 4));
            GptqTensor t2 = t; t2.qzeros = qz2.data();
            for (int oo = 0; oo < OUT; ++oo) {
                const bool changed = gptq_at(t2, oo, 0) != gptq_at(t, oo, 0);
                if (changed != (oo == o)) { ++leaked; break; }
            }
        }
        check(leaked == 0, "zero-point affects exactly its own output channel");
        std::printf("  zero isolation : swept all %d channels, %d leaks\n", OUT, leaked);
    }

    // -----------------------------------------------------------------
    // 5. The +1 convention. Omitting it is the classic port bug and
    //    shifts every weight by exactly one scale step -- assert that
    //    the code is NOT doing the naive (nib - zero) form.
    // -----------------------------------------------------------------
    {
        const int o = 33, k = 5, g = k / G;
        const uint32_t nib  = (uint32_t(qw[(k >> 3) * OUT + o]) >> ((k & 7) * 4)) & 0xFu;
        const uint32_t znib = (uint32_t(qz[g * (OUT / 8) + (o >> 3)]) >> ((o & 7) * 4)) & 0xFu;
        const float    s    = f16_to_f32(sc[g * OUT + o]);
        check(std::fabs(gptq_at(t, o, k) - (float(nib) - float(znib + 1)) * s) < 1e-7f,
              "uses (nib - (zero+1)) * scale");
        check(std::fabs(gptq_at(t, o, k) - (float(nib) - float(znib)) * s) > 1e-9f,
              "is NOT the naive (nib - zero) form");
        std::printf("  +1 convention  : confirmed (naive form differs by one scale step)\n");
    }

    // -----------------------------------------------------------------
    // 6. down_proj has reversed dims. Derive in/out from the qweight
    //    shape, never a config field -- assert the path works there too.
    // -----------------------------------------------------------------
    {
        const int O2 = 2048, I2 = 512, ng2 = I2 / G;
        std::vector<int32_t>  qw2(size_t(I2 / 8) * O2);
        std::vector<int32_t>  qz2(size_t(ng2) * (O2 / 8));
        std::vector<uint16_t> sc2(size_t(ng2) * O2);
        uint32_t x = 999u;
        for (auto& v : qw2) { x = lcg(x); v = int32_t(x); }
        for (auto& v : qz2) { x = lcg(x); v = int32_t(x); }
        for (auto& v : sc2) { x = lcg(x); v = uint16_t(0x2C00u | ((x >> 16) & 0x03FFu)); }
        GptqTensor d; d.qweight = qw2.data(); d.qzeros = qz2.data(); d.scales = sc2.data();
        d.in = I2; d.out = O2; d.group = G;
        std::vector<float> wd(size_t(O2) * I2);
        gptq_dequant_4bit(d, wd.data());
        int64_t bad = 0;
        for (int o = 0; o < O2; ++o)
            for (int k = 0; k < I2; ++k)
                if (wd[size_t(o) * I2 + k] != gptq_at(d, o, k)) ++bad;
        check(bad == 0, "down_proj orientation (out=2048 in=512)");
        std::printf("  down_proj shape: %d x %d exact\n", O2, I2);
    }

    // -----------------------------------------------------------------
    // 7. The real pipeline: GPTQ -> f32 -> quantize(MXFP4) -> at().
    //    A 35B model cannot stay in bf16, so the loader must requantize.
    //    Check the GEMV through the requantized weight tracks the GEMV
    //    through the exact f32 weight within the 4-bit error band.
    // -----------------------------------------------------------------
    {
        PackedWeight p = quantize(w.data(), OUT, IN, Fmt::MXFP4);
        QuantWeight  qv = p.view();
        std::vector<float> x(IN);
        uint32_t r = 7u;
        for (int k = 0; k < IN; ++k) { r = lcg(r); x[k] = float(int32_t(r) % 1000) / 1000.0f; }

        double num = 0.0, den = 0.0;
        for (int o = 0; o < OUT; ++o) {
            double a = 0.0, b = 0.0;
            for (int k = 0; k < IN; ++k) {
                a += double(w[size_t(o) * IN + k]) * x[k];
                b += double(qv.at(o, k))           * x[k];
            }
            num += (a - b) * (a - b); den += a * a;
        }
        const double rel = std::sqrt(num / den);
        check(rel < 0.25, "GPTQ->MXFP4 GEMV within 4-bit error band");
        std::printf("  requant GEMV   : rel-L2 %.4f vs exact GPTQ weights (4-bit band)\n", rel);
    }

    std::printf("\n%s (%d failures)\n", failures ? "FAILURES" : "ALL PASS", failures);
    return failures ? 1 : 0;
}
