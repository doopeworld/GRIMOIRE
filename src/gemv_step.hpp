// =====================================================================
//  gemv_step.hpp  --  per-format inner accumulation, shared by the dense
//  decode GEMV and the fused MoE expert kernels.
//
//  This lives in a header rather than inside gemv_decode.cpp because
//  moe_kernels.cpp needs the IDENTICAL decode path. Two copies would
//  drift, and a divergence between the dense and MoE dequant would be
//  invisible until the model produced subtly wrong tokens.
//
//  Every specialization takes the SLM lookup tables:
//      lut   FP8 byte    -> float   (E4M3 / E5M2)
//      slut  E8M0 byte   -> 2^(x-127)
//      nlut  E2M1 nibble -> float
//  Callers that do not use a given format may pass nullptr for the
//  tables it does not read.
// =====================================================================
#ifndef B70_GEMV_STEP_HPP
#define B70_GEMV_STEP_HPP

#include "kernels.hpp"

namespace b70 {

// ---------------------------------------------------------------------
// FP8 decode is a branchy bit-assembly per element -- subnormal check,
// exponent rebuild, sign insert. Measured on a B70 that costs ~3x the
// memory time: fp8 read HALF the bytes of bf16 and still ran 44% slower,
// pinned at 106 GB/s while direct-load formats hit 310.
//
// A 256-entry table collapses the whole thing to one indexed load. The
// table is built once per work-group into SLM (1 KB) and stays resident.
template <Fmt F, int EPL>
struct GemvStep;

template <int EPL> struct GemvStep<Fmt::BF16, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int, int k0) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + k0;
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i)
            a = sycl::fma(bf16_to_f32(bf16_t{p[i]}), x[k0 + i], a);
        return a;
    }
};

template <int EPL> struct GemvStep<Fmt::INT8, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const float s = static_cast<const float*>(w.scales)[n];
        const int8_t* p = reinterpret_cast<const int8_t*>(row) + k0;
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i)
            a = sycl::fma(float(p[i]), x[k0 + i], a);
        return a * s;                       // one multiply per 16 elements
    }
};

template <int EPL> struct GemvStep<Fmt::FP8_E4M3, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const float s = static_cast<const float*>(w.scales)[n];
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i)
            a = sycl::fma(lut[row[k0 + i]], x[k0 + i], a);
        return a * s;
    }
};

template <int EPL> struct GemvStep<Fmt::FP8_E5M2, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const float s = static_cast<const float*>(w.scales)[n];
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i)
            a = sycl::fma(lut[row[k0 + i]], x[k0 + i], a);
        return a * s;
    }
};

template <int EPL> struct GemvStep<Fmt::MXFP8, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const uint8_t X = static_cast<const uint8_t*>(w.scales)
                          [int64_t(n) * w.row_scales + k0 / kMXBlock];
        const float s = slut[X];
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL; ++i)
            a = sycl::fma(lut[row[k0 + i]], x[k0 + i], a);
        return a * s;
    }
};

// E2M1 decoded in the ALU instead of through the SLM magnitude table.
// The table costs one indexed SLM gather PER NIBBLE -- two per weight byte,
// ~740 G gathers/s at the rate decode actually runs.  E2M1 has 8 magnitudes
// and a sign, so the float can be assembled from the nibble directly:
//   e!=0 -> (126+e)<<23 | m<<22   =  (1 + m/2) * 2^(e-1)
//   e==0 -> m ? 0.5 : 0
// Verified against e2m1_to_f32 for all 16 nibbles.
static inline float e2m1_alu(uint32_t nib) {
    const uint32_t e = (nib >> 1) & 3u, m = nib & 1u, sgn = (nib >> 3) & 1u;
    uint32_t bits = e ? (((126u + e) << 23) | (m << 22)) : (m ? 0x3F000000u : 0u);
    return sycl::bit_cast<float>(bits | (sgn << 31));
}

template <int EPL> struct GemvStep<Fmt::MXFP4, EPL> {
    // Hoisted-x entry point.  `xv` points at EPL activations already in
    // registers, loaded ONCE for all ROWS_PER_SG rows of the sub-group.
    // The generic run() takes `x` and `row` as unqualified pointers, so the
    // compiler cannot prove they do not alias and reloads every x[k] for
    // each of the 4 rows: 64 bytes of L2 per 8 bytes of DRAM, 4x more than
    // the algorithm needs.
    // OPT bit 1 selects the ALU E2M1 decode over the SLM table.
    template <int OPT>
    static inline float run_xv(const QuantWeight& w, const uint8_t* row,
                               const float* xv, const float* slut,
                               const float* nlut, int n, int k0) {
        const uint8_t X = static_cast<const uint8_t*>(w.scales)
                          [int64_t(n) * w.row_scales + k0 / kMXBlock];
        const float s = slut[X];
        const uint8_t* p = row + (k0 >> 1);
        float a = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL / 2; ++i) {
            uint8_t byte;
            if constexpr (EPL == 16) {
                // one 8-byte transaction, then shift -- same as run()
                byte = uint8_t((*reinterpret_cast<const uint64_t*>(p)) >> (8 * i));
            } else {
                byte = p[i];
            }
            if constexpr (OPT & 2) {
                a = sycl::fma(e2m1_alu(byte & 0x0Fu),        xv[2 * i],     a);
                a = sycl::fma(e2m1_alu((byte >> 4) & 0x0Fu), xv[2 * i + 1], a);
            } else {
                a = sycl::fma(nlut[byte & 0x0F],        xv[2 * i],     a);
                a = sycl::fma(nlut[(byte >> 4) & 0x0F], xv[2 * i + 1], a);
            }
        }
        return a * s;
    }

    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const uint8_t X = static_cast<const uint8_t*>(w.scales)
                          [int64_t(n) * w.row_scales + k0 / kMXBlock];
        const float s = slut[X];
        // One byte carries two E2M1 nibbles. The magnitude table lives in
        // SLM rather than as a function-local array, which the compiler
        // otherwise places in private memory and reloads per element.
        const uint8_t* p = row + (k0 >> 1);
        float a = 0.0f;
        if constexpr (EPL == 16) {
            // k0 is a multiple of 16, hence p is naturally 8-byte aligned.
            // Make the transaction width explicit: the scalar form below
            // compiled to eight byte-load instructions on Xe2 and left the
            // memory pipeline under-filled during M=1 decode.
            const uint64_t packed=*reinterpret_cast<const uint64_t*>(p);
            #pragma unroll
            for(int i=0;i<8;++i){
                const uint8_t byte=uint8_t(packed>>(8*i));
                a=sycl::fma(nlut[byte&0x0F],x[k0+2*i],a);
                a=sycl::fma(nlut[(byte>>4)&0x0F],x[k0+2*i+1],a);
            }
        } else if constexpr (EPL == 32) {
            // EPL=32 is exactly ONE MX block, so k0 is a multiple of 32 and
            // p is 16-byte aligned -- the widest load the lane can issue and
            // one scale lookup for the whole transaction.  Until 2026-08-26
            // this case fell into the scalar byte loop below, which is why
            // the B70_EPL sweep read 32 as a 40% REGRESSION: it was timing an
            // unvectorized path against the vectorized EPL=16 one, not
            // measuring transaction width.
            const sycl::vec<uint32_t, 4> v =
                *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(p);
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                const uint32_t word = v[wi];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint8_t byte = uint8_t(word >> (8 * i));
                    const int kk = k0 + 8 * wi + 2 * i;
                    a = sycl::fma(nlut[byte & 0x0F],        x[kk],     a);
                    a = sycl::fma(nlut[(byte >> 4) & 0x0F], x[kk + 1], a);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < EPL / 2; ++i) {
                const uint8_t byte = p[i];
                a = sycl::fma(nlut[byte & 0x0F],        x[k0 + 2 * i],     a);
                a = sycl::fma(nlut[(byte >> 4) & 0x0F], x[k0 + 2 * i + 1], a);
            }
        }
        return a * s;
    }
};

template <int EPL> struct GemvStep<Fmt::INT4, EPL> {
    static inline float run(const QuantWeight& w, const uint8_t* row,
                            const float* x, const float* lut,
                            const float* slut, const float* nlut, int n, int k0) {
        const int64_t gi = int64_t(n) * w.row_scales + k0 / kInt4Group;
        const float   s  = bf16_to_f32(static_cast<const bf16_t*>(w.scales)[gi]);
        const uint8_t zu = w.zeros[gi];
        const float   z  = float(zu);
        const uint8_t* p = row + (k0 >> 1);
        // Asymmetric dequant is (q - z) * s. Factor it: accumulate
        // sum(q*x) and sum(x) separately, apply s and z once per group.
        float aq = 0.0f, ax = 0.0f;
        #pragma unroll
        for (int i = 0; i < EPL / 2; ++i) {
            const uint8_t byte = p[i];
            const float x0 = x[k0 + 2 * i], x1 = x[k0 + 2 * i + 1];
            const uint8_t q0=byte&0x0F, q1=(byte>>4)&0x0F;
            const float v0=zu==0xff ? float(q0&8 ? int(q0)-16 : int(q0)) : float(q0);
            const float v1=zu==0xff ? float(q1&8 ? int(q1)-16 : int(q1)) : float(q1);
            aq = sycl::fma(v0, x0, aq);
            aq = sycl::fma(v1, x1, aq);
            ax += x0 + x1;
        }
        return (zu==0xff ? aq : aq-z*ax) * s;
    }
};


} // namespace b70
#endif
