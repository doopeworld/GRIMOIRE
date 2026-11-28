// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// onednn_gemm.cpp -- see onednn_gemm.hpp.  Host code only: one oneDNN engine
// and stream per SYCL queue (the engine's own context, so no second context
// and nothing shared between cards), user-managed scratchpad, deterministic
// mode (measured: no cost on these shapes).
//
// Row pieces.  One primitive per exact M made the first prompt of each new
// length ~0.5 s slower (two cards, code probe TTFT 1.39 vs 0.98 s): MEASURED
// 2026-10-25 (tools/bench/onednn/dnnl_create_probe.cpp, one B70), a primitive
// for a NEW row count costs 17-48 ms of kernel selection on the 34816-wide
// weights and 94-597 ms when it needs a kernel variant not compiled yet, but
// ~0.1 ms on the 5120-wide ones.  So a WIDE weight (N >= 32768) at 512+ rows
// is cut into pieces of 1024, 768, 512 and 256 rows (largest first), the last
// < 256 rows through a staging buffer padded to 64, 128 or 256 rows: six
// primitives per weight, whatever prompt lengths it sees.  From 512 rows up a
// wide GEMM's time is linear in the rows, so the pieces cost <= 10%
// (gemm_fast_probe wall time, FP8 512 / 2153 / 4096 rows: 1.44 / 6.29 / 11.09
// ms pieces, 1.45 / 6.17 / 11.15 exact).  Narrow weights and short calls keep
// one primitive per M: there an extra piece re-reads the whole weight and a
// short piece leaves the GPU half idle (FP8 1000 x 5120 x 17408: 2.17 ms in
// pieces, 1.41 exact; BF16 300 rows on the wide weight: 1.63 vs 0.97).
// The line is at 32768, above the TP ranks' 17408-wide gate|up: on two cards
// (Qwen3.8-27B-FP8) eight prompts of new lengths ran 0.6-2.3% FASTER with one
// primitive per M than in pieces at 16384 -- there only the first request's
// kernel variants cost (the server's warm-up pays those).
// Run-time M (DNNL_RUNTIME_DIM_VAL) is not an option: oneDNN serves it with
// ocl:ref:any, ~700x slower.  GRIMOIRE_ONEDNN_EXACT_M=1: one primitive per M
// for every weight.
//
// Numerics differ from the dequantize path in the last bits: that path rounds
// e4m3 * row_scale to bf16 before the GEMM; oneDNN multiplies the exact e4m3
// value (it fits bf16) and applies the scale to the fp32 sum.
#include "onednn_gemm.hpp"
#include "hadamard128.hpp"
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/ext/intel/math.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

namespace b70 {

namespace {

using dt  = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;

constexpr int kPieces[] = {1024, 768, 512, 256};
constexpr int kTails[]  = {64, 128, 256};
constexpr int kWideN = 32768, kPieceMinM = 512;

// On by default since 2026-10-25 (two cards, Qwen3.8-27B-FP8: prefill +5-26%, decode and
// answers unchanged; regress_all_g0.sh 40/44 texts identical, the other four a rounding
// change on 5987-token prompts or not oneDNN's).  GRIMOIRE_ONEDNN=0 turns it off.
bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("GRIMOIRE_ONEDNN");
        return !(v && *v && std::atoi(v) == 0);
    }();
    return on;
}

bool exact_m() {
    static const bool on = std::getenv("GRIMOIRE_ONEDNN_EXACT_M") != nullptr;
    return on;
}

struct Key {
    int fmt, M, N, K, acc, a8, rot, swi;
    bool operator<(const Key& o) const {
        return std::tie(fmt, M, N, K, acc, a8, rot, swi) <
               std::tie(o.fmt, o.M, o.N, o.K, o.acc, o.a8, o.rot, o.swi);
    }
};

struct Prim {
    dnnl::matmul       prim;
    dnnl::memory::desc src, wei, dst, scr, scl, ascl, bin;
    bool a8=false;
    bool rot=false;   // a8 source in the 128-point Hadamard domain (onednn_gemm_a8rot)
    bool swi=false;   // SwiGLU's gate half: bf16 dst = swish(acc) * bin (onednn_gemm_a8rot_swiglu)
};

struct Ctx {
    dnnl::engine         eng;
    dnnl::stream         strm;
    std::map<Key, Prim>  prims;
    uint8_t*             scratch = nullptr;
    size_t               scratch_cap = 0;
    sycl_bf16*           xs = nullptr;       // the padded tail's rows, in and out
    size_t               xs_cap = 0;
    float*               ys = nullptr;
    size_t               ys_cap = 0;
    int8_t*              x8 = nullptr;
    size_t               x8_cap = 0;
    float*               as = nullptr;
    size_t               as_cap = 0;
    float*               us = nullptr;       // SwiGLU: the up half's rows of one piece (f32)
    size_t               us_cap = 0;
    sycl_bf16*           hs = nullptr;       // SwiGLU: the staged tail's output rows
    size_t               hs_cap = 0;
};

std::mutex& mu() { static std::mutex m; return m; }

Ctx& ctx_for(sycl::queue& q) {                  // caller holds mu()
    static std::map<const sycl::queue*, Ctx> all;
    auto it = all.find(&q);
    if (it == all.end()) {
        Ctx c;
        c.eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        c.strm = dnnl::sycl_interop::make_stream(c.eng, q);
        it = all.emplace(&q, std::move(c)).first;
    }
    return it->second;
}

// Grow a device buffer; nothing in flight may still use the old one.
template <class T>
void grow(sycl::queue& q, T*& p, size_t& cap, size_t need, const char* what) {
    if (need <= cap) return;
    q.wait();
    if (p) sycl::free(p, q);
    p = static_cast<T*>(sycl::malloc_device(need * sizeof(T), q));
    cap = p ? need : 0;
    if (!p) throw std::runtime_error(std::string("oneDNN matmul: ") + what + " allocation failed");
}

const Prim& prim_for(sycl::queue& q, Ctx& c, const QuantWeight& w, int M, bool acc,
                     bool rot = false, bool swi = false) {
    static const int requested=[] {
        const char* e=std::getenv("GRIMOIRE_ONEDNN_W8A8");return e?std::atoi(e):0;
    }();
    // Mode 2 limits activation quantization to large projections. Mode 3
    // further excludes narrow outputs fed by unnormalized FFN activations.
    const bool selected=requested==1 || (requested==2 && (w.N>=32768 || w.K>=16384)) ||
                        (requested==3 && w.N>=32768);
    // rot: the caller (gemm_fast.cpp, INT4 prompts) rebuilt W as rotated int8 and wants the
    // source rotated the same way -- always int8 x int8.
    const bool a8=rot || (selected&&w.fmt==Fmt::INT8&&M>=32&&w.N>=256&&w.K%64==0);
    const Key k{int(w.fmt), M, w.N, w.K, acc ? 1 : 0, a8?1:0, rot?1:0, swi?1:0};
    auto it = c.prims.find(k);
    if (it != c.prims.end()) return it->second;
    if (swi && (acc || !rot)) throw std::runtime_error("oneDNN SwiGLU primitive: rotated int8 source only, no sum");
    const auto t0 = std::chrono::steady_clock::now();
    const bool fp8 = w.fmt == Fmt::FP8_E4M3;
    // INT8 (2026-10-29): GRIMOIRE's symmetric per-row int8 -- the same layout as its FP8, one
    // byte a weight and one f32 scale a row -- as oneDNN weights decompression: s8 weights,
    // bf16 source, fpmath bf16 applied to integers.  Before, INT8 had no oneDNN or small-M
    // path: Qwen3.8-27B on two cards pp4096 2,448 / 4 users 46 tok/s against FP8 3,475 / 89.
    const bool s8 = w.fmt == Fmt::INT8;
    Prim p;
    p.a8=a8;
    p.rot=rot;
    p.swi=swi;
    p.src = dnnl::memory::desc({M, w.K}, a8?dt::s8:dt::bf16, tag::ab);
    // W is [N][K] row-major: oneDNN's [K][N] weights in the transposed tag.
    p.wei = dnnl::memory::desc({w.K, w.N}, fp8 ? dt::f8_e4m3 : s8 ? dt::s8 : dt::bf16, tag::ba);
    p.dst = dnnl::memory::desc({M, w.N}, swi ? dt::bf16 : dt::f32, tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_deterministic(true);
    if (fp8 || s8) {
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);    // one f32 scale per output column
        p.scl = dnnl::memory::desc({w.N}, dt::f32, tag::a);
    }
    if(a8){
        attr.set_scales_mask(DNNL_ARG_SRC,1<<0);
        p.ascl=dnnl::memory::desc({M},dt::f32,tag::a);
    }else if (s8) attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    if (acc) {
        dnnl::post_ops po;
        po.append_sum(1.0f);
        attr.set_post_ops(po);
    }
    if (swi) {
        // SwiGLU's gate half (2026-11-17): after the scales, swish(g) = g / (1 + exp(-g)), times the up half's
        // f32 rows (post-op 1's src), rounded once to the bf16 dst -- the f32 gate|up rows never reach memory
        dnnl::post_ops po;
        po.append_eltwise(dnnl::algorithm::eltwise_swish, 1.0f, 0.0f);
        p.bin = dnnl::memory::desc({M, w.N}, dt::f32, tag::ab);
        po.append_binary(dnnl::algorithm::binary_mul, p.bin);
        attr.set_post_ops(po);
    }
    dnnl::matmul::primitive_desc pd(c.eng, p.src, p.wei, p.dst, attr);
    if(a8&&std::string(pd.impl_info_str()).find(":ref:")!=std::string::npos)
        throw std::runtime_error(swi ? "oneDNN W8A8 SwiGLU: only a reference implementation for this shape"
                                     : "oneDNN W8A8: only a reference implementation is available for this shape");
    p.scr  = pd.scratchpad_desc();
    p.prim = dnnl::matmul(pd);
    grow(q, c.scratch, c.scratch_cap, p.scr.get_size(), "scratchpad");
    static const bool verbose = std::getenv("GRIMOIRE_ONEDNN_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "  oneDNN matmul %s M=%d N=%d K=%d%s: %s, scratchpad %zu B, "
                     "created in %.1f ms\n", swi ? "W8A8 swiglu" : a8?"W8A8":fp8 ? "fp8" : s8 ? "s8" : "bf16", M, w.N, w.K,
                     acc ? " (+=)" : "", pd.impl_info_str(), p.scr.get_size(),
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count());
    return c.prims.emplace(k, std::move(p)).first->second;
}

// One primitive over exactly its own row count, x and y at those rows.  `preq`: an a8 primitive whose
// source rows are already in c.x8 / c.as (the call before quantized the same x rows); `bin`: a SwiGLU
// primitive's up-half rows.
sycl::event exec(sycl::queue& q,Ctx& c, const Prim& p, const QuantWeight& w, const sycl_bf16* x, void* y,
                 const std::vector<sycl::event>& deps, const float* bin = nullptr, bool preq = false) {
    using namespace dnnl::sycl_interop;
    std::unordered_map<int, dnnl::memory> args;
    std::vector<sycl::event> ready=deps;
    void* source=const_cast<sycl_bf16*>(x);
    if(p.a8&&preq){
        source=c.x8;
        args[DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC]=make_memory(p.ascl,c.eng,memory_kind::usm,c.as);
    }else if(p.a8){
        const int M=int(p.src.get_dims()[0]),K=w.K;
        grow(q,c.x8,c.x8_cap,size_t(M)*K,"int8 activations");
        grow(q,c.as,c.as_cap,size_t(M),"activation scales");
        int8_t* out=c.x8;float* scales=c.as;
        constexpr size_t WG=256;
        if(p.rot){
            // Each 128-wide block of a row through the unnormalised 128-point Sylvester
            // Hadamard transform (gemm_fast.cpp rotated the weight rows the same way and
            // folded the 1/128), then one int8 scale per row.  The rotation spreads an outlier
            // channel over its block, which is what makes a per-row int8 scale accurate
            // (exl3xpu's int8 prompt linears: NLL +0.17%).  A sub-group owns a block, a lane 8
            // consecutive values; two passes so the values never leave registers.
            const auto quant=q.submit([&](sycl::handler& h){
                h.depends_on(ready);
                h.parallel_for(sycl::nd_range<1>(size_t(M)*WG,WG),
                    [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                    auto sg=it.get_sub_group();
                    const int row=int(it.get_group(0));
                    const int lane=int(sg.get_local_id()[0]),sgid=int(sg.get_group_id()[0]);
                    const int nsg=int(sg.get_group_range()[0]),nb=K/128;
                    const sycl_bf16* xr=x+size_t(row)*K;
                    auto block=[&](int b,float (&v)[8]){
                        const sycl_bf16* s=xr+b*128+lane*8;
                        #pragma unroll
                        for(int t=0;t<8;++t)v[t]=float(s[t]);
                        hadamard128_sg(sg,lane,v);
                    };
                    float local=0.f;
                    for(int b=sgid;b<nb;b+=nsg){
                        float v[8];block(b,v);
                        #pragma unroll
                        for(int t=0;t<8;++t)local=sycl::fmax(local,sycl::fabs(v[t]));
                    }
                    const float maximum=sycl::reduce_over_group(it.get_group(),local,sycl::maximum<float>());
                    const float scale=maximum>0.f?sycl::ext::intel::math::fdiv_rn(maximum,127.f):1.f;
                    const float inverse=sycl::ext::intel::math::fdiv_rn(1.f,scale);
                    if(it.get_local_id(0)==0)scales[row]=scale;
                    for(int b=sgid;b<nb;b+=nsg){
                        float v[8];block(b,v);
                        int8_t* o=out+size_t(row)*K+b*128+lane*8;
                        #pragma unroll
                        for(int t=0;t<8;++t)
                            o[t]=int8_t(sycl::clamp(sycl::rint(v[t]*inverse),-127.f,127.f));
                    }
                });
            });
            ready={quant};source=c.x8;
            args[DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC]=make_memory(p.ascl,c.eng,memory_kind::usm,c.as);
        }else{
        const auto quant=q.submit([&](sycl::handler& h){
            h.depends_on(ready);
            h.parallel_for(sycl::nd_range<1>(size_t(M)*WG,WG),[=](sycl::nd_item<1> it){
                const int row=int(it.get_group(0)),lid=int(it.get_local_id(0));
                float local=0.f;
                for(int k=lid;k<K;k+=WG)local=sycl::fmax(local,sycl::fabs(float(x[size_t(row)*K+k])));
                const float maximum=sycl::reduce_over_group(it.get_group(),local,sycl::maximum<float>());
                const float scale=maximum>0.f?sycl::ext::intel::math::fdiv_rn(maximum,127.f):1.f;
                const float inverse=sycl::ext::intel::math::fdiv_rn(1.f,scale);
                if(lid==0)scales[row]=scale;
                for(int k=lid;k<K;k+=WG){
                    const float v=sycl::rint(float(x[size_t(row)*K+k])*inverse);
                    out[size_t(row)*K+k]=int8_t(sycl::clamp(v,-127.f,127.f));
                }
            });
        });
        ready={quant};source=c.x8;
        args[DNNL_ARG_ATTR_SCALES|DNNL_ARG_SRC]=make_memory(p.ascl,c.eng,memory_kind::usm,c.as);
        }
    }
    args[DNNL_ARG_SRC]     = make_memory(p.src, c.eng, memory_kind::usm,source);
    args[DNNL_ARG_WEIGHTS] = make_memory(p.wei, c.eng, memory_kind::usm,
                                         const_cast<uint8_t*>(w.payload));
    args[DNNL_ARG_DST]     = make_memory(p.dst, c.eng, memory_kind::usm, y);
    if (p.scr.get_size())
        args[DNNL_ARG_SCRATCHPAD] = make_memory(p.scr, c.eng, memory_kind::usm, c.scratch);
    if (w.fmt == Fmt::FP8_E4M3 || w.fmt == Fmt::INT8)
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
            make_memory(p.scl, c.eng, memory_kind::usm, const_cast<void*>(w.scales));
    if (p.swi)
        args[DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1] =
            make_memory(p.bin, c.eng, memory_kind::usm, const_cast<float*>(bin));
    return execute(p.prim, c.strm, args, ready);
}

} // namespace

// ---- INT4 (2026-11-14) ------------------------------------------------------------------------
namespace {
struct I4T { sycl_bf16* s = nullptr; uint8_t* z = nullptr; bool ok = false; };
struct I4Prim { dnnl::matmul prim; dnnl::memory::desc src, wei, dst, scr, smd, zmd; };

int i4_group(const QuantWeight& w) { return 1 << w.int4_gshift(); }

bool i4_shape_ok(const QuantWeight& w) {
    if (w.fmt != Fmt::INT4 || !w.payload || !w.scales || !w.zeros || w.N <= 0 || w.K <= 0) return false;
    const int g = i4_group(w);
    return (g == 64 || g == 128) && int64_t(w.row_scales) * g == w.K && w.row_bytes == int64_t(w.K) / 2 &&
           w.K % 128 == 0;
}

// caller holds mu()
const I4T& i4_transposed(sycl::queue& q, const QuantWeight& w) {
    static std::map<const void*, I4T> all;
    auto it = all.find(w.scales);
    if (it != all.end()) return it->second;
    I4T t;
    const int N = w.N, RS = w.row_scales;
    const size_t n = size_t(N) * size_t(RS);
    t.s = sycl::malloc_device<sycl_bf16>(n, q);
    t.z = sycl::malloc_device<uint8_t>(n, q);
    int* bad = sycl::malloc_shared<int>(1, q);
    if (t.s && t.z && bad) {
        *bad = 0;
        const uint16_t* s = static_cast<const uint16_t*>(w.scales);
        const uint8_t* z = w.zeros;
        uint16_t* so = reinterpret_cast<uint16_t*>(t.s);
        uint8_t* zo = t.z;
        q.parallel_for(sycl::range<2>(size_t(RS), size_t(N)), [=](sycl::id<2> id) {
            const size_t g = id[0], c = id[1];
            const uint8_t zz = z[c * RS + g];
            so[g * N + c] = s[c * RS + g];
            zo[g * N + c] = zz;
            if (zz == 0xff) {   // signed s4 group (GPTQ experts): no oneDNN form
                sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*bad);
                a.store(1);
            }
        }).wait();
        t.ok = *bad == 0;
    }
    if (bad) sycl::free(bad, q);
    if (!t.ok) {
        if (t.s) sycl::free(t.s, q);
        if (t.z) sycl::free(t.z, q);
        t.s = nullptr; t.z = nullptr;
    }
    return all.emplace(w.scales, t).first->second;
}

const I4Prim& i4_prim(Ctx& c, sycl::queue& q, const QuantWeight& w, int M, bool acc) {
    static std::map<std::tuple<int, int, int, int, int>, I4Prim> prims;
    const int g = i4_group(w);
    const auto k = std::make_tuple(M, w.N, w.K, g, acc ? 1 : 0);
    auto it = prims.find(k);
    if (it != prims.end()) return it->second;
    const auto t0 = std::chrono::steady_clock::now();
    I4Prim p;
    p.src = dnnl::memory::desc({M, w.K}, dt::bf16, tag::ab);
    p.wei = dnnl::memory::desc({w.K, w.N}, dt::u4, tag::ba);
    p.dst = dnnl::memory::desc({M, w.N}, dt::f32, tag::ab);
    p.smd = dnnl::memory::desc({w.K / g, w.N}, dt::bf16, tag::ab);
    p.zmd = dnnl::memory::desc({w.K / g, w.N}, dt::u8, tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_deterministic(true);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {g, 1}, dt::bf16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {g, 1}, dt::u8);
    attr.set_fpmath_mode(dnnl::fpmath_mode::bf16, true);
    if (acc) {
        dnnl::post_ops po;
        po.append_sum(1.0f);
        attr.set_post_ops(po);
    }
    dnnl::matmul::primitive_desc pd(c.eng, p.src, p.wei, p.dst, attr);
    if (std::string(pd.impl_info_str()).find(":ref:") != std::string::npos)
        throw std::runtime_error("oneDNN INT4: only a reference implementation for this shape");
    p.scr = pd.scratchpad_desc();
    p.prim = dnnl::matmul(pd);
    grow(q, c.scratch, c.scratch_cap, p.scr.get_size(), "scratchpad");
    static const bool verbose = std::getenv("GRIMOIRE_ONEDNN_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "  oneDNN matmul int4 g%d M=%d N=%d K=%d%s: %s, scratchpad %zu B, created in %.1f ms\n",
                     g, M, w.N, w.K, acc ? " (+=)" : "", pd.impl_info_str(), p.scr.get_size(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return prims.emplace(k, std::move(p)).first->second;
}
} // namespace

bool onednn_int4_ok(sycl::queue& q, const QuantWeight& w, int M) {
    static const bool on = [] { const char* v = std::getenv("GRIMOIRE_I4_ONEDNN"); return !(v && *v == '0'); }();
    if (!on || !enabled() || M <= 0 || !i4_shape_ok(w)) return false;
    std::lock_guard<std::mutex> lk(mu());
    // first use of the weight: its transposed scales / zeros are built here, so a weight oneDNN cannot
    // take (signed s4 groups, no memory) is known before anything is enqueued for it
    return i4_transposed(q, w).ok;
}

sycl::event onednn_gemm_int4(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y, int M,
                             bool accumulate, const std::vector<sycl::event>& deps) {
    using namespace dnnl::sycl_interop;
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    const I4T& t = i4_transposed(q, w);
    if (!t.ok) throw std::runtime_error("onednn_gemm_int4: weight not supported (check onednn_int4_ok first)");
    const size_t K = size_t(w.K), N = size_t(w.N);
    auto run = [&](const I4Prim& p, const sycl_bf16* xin, float* yout, const std::vector<sycl::event>& d) {
        std::unordered_map<int, dnnl::memory> args;
        args[DNNL_ARG_SRC] = make_memory(p.src, c.eng, memory_kind::usm, const_cast<sycl_bf16*>(xin));
        args[DNNL_ARG_WEIGHTS] = make_memory(p.wei, c.eng, memory_kind::usm, const_cast<uint8_t*>(w.payload));
        args[DNNL_ARG_DST] = make_memory(p.dst, c.eng, memory_kind::usm, yout);
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = make_memory(p.smd, c.eng, memory_kind::usm, t.s);
        args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = make_memory(p.zmd, c.eng, memory_kind::usm, t.z);
        if (p.scr.get_size())
            args[DNNL_ARG_SCRATCHPAD] = make_memory(p.scr, c.eng, memory_kind::usm, c.scratch);
        return execute(p.prim, c.strm, args, d);
    };
    // Fixed pieces of 1024 / 512 / 256 / 128 / 64 / 32 rows and a tail of < 32 through 32 staging rows (the
    // tail's rows in -- and its output rows when accumulating -- a 32-row primitive, its rows out), so a row count
    // never seen before builds nothing: a oneDNN JIT build costs up to ~0.1 s.
    static constexpr int kI4Pieces[] = {1024, 512, 256, 128, 64, 32};
    std::vector<sycl::event> d = deps;
    sycl::event e;
    int r0 = 0;
    for (const int P : kI4Pieces)
        while (M - r0 >= P) {
            e = run(i4_prim(c, q, w, P, accumulate), x + size_t(r0) * K, y + size_t(r0) * N, d);
            d = {e};
            r0 += P;
        }
    const int rem = M - r0;
    if (rem > 0) {
        constexpr int T = 32;
        if (size_t(T) * K > c.xs_cap) {
            grow(q, c.xs, c.xs_cap, size_t(T) * K, "staging");
            q.memset(c.xs, 0, c.xs_cap * sizeof(sycl_bf16)).wait();
        }
        grow(q, c.ys, c.ys_cap, size_t(T) * N, "staging");
        e = q.memcpy(c.xs, x + size_t(r0) * K, size_t(rem) * K * sizeof(sycl_bf16), d);
        if (accumulate) e = q.memcpy(c.ys, y + size_t(r0) * N, size_t(rem) * N * sizeof(float), {e});
        e = run(i4_prim(c, q, w, T, accumulate), c.xs, c.ys, {e});
        e = q.memcpy(y + size_t(r0) * N, c.ys, size_t(rem) * N * sizeof(float), {e});
    }
    return e;
}

// =====================================================================
// GRIMOIRE_I4A8G=<32|64|128> (opt-in, 2026-11-22): INT4 prompt GEMMs as int8 activations x the INT4
// weights as stored.  Each activation row is quantized to int8 with one f32 scale per G-wide K group;
// oneDNN's matmul takes the u4 weights with their own group scales / zero points, s8 x int4 on the
// DPAS units (fpmath strict: no implicit down-conversion; the s8 x u4 kernels are the integer ones, an f32
// fallback would be several times slower than the bf16 path, and this one is faster).  Against I4A8 (one
// scale per row, weights requantized to rotated int8 per call) the weights are exact and an outlier only
// coarsens its own group; there is no per-call weight conversion (~75 ms a prompt), so it also pays below
// I4A8's 1536 rows.
// MEASURED 2026-11-22, one B70, M 4096, standalone probe linked against oneAPI 2026.1's oneDNN 3.11
// (i4a8g_probe / i4a8g_check): gate_up 34816x5120 12.1 -> 6.5 ms, down 5120x17408 4.5 -> 2.8 ms against
// the bf16 x u4 primitive above; output against a float64 reference of the same int8 / u4 operands: rms
// rel 5e-8..9e-8.  End to end (the image, oneDNN 3.14): see gemm_fast.cpp.
// =====================================================================
namespace {
struct I4GPrim { dnnl::matmul prim; dnnl::memory::desc src, wei, dst, scr, smd, zmd, asd; };

const I4GPrim& i4g_prim(Ctx& c, sycl::queue& q, const QuantWeight& w, int M, int ga, bool acc) {
    static std::map<std::tuple<int, int, int, int, int, int>, I4GPrim> prims;
    const int g = i4_group(w);
    const auto k = std::make_tuple(M, w.N, w.K, g, ga, acc ? 1 : 0);
    auto it = prims.find(k);
    if (it != prims.end()) return it->second;
    const auto t0 = std::chrono::steady_clock::now();
    I4GPrim p;
    p.src = dnnl::memory::desc({M, w.K}, dt::s8, tag::ab);
    p.wei = dnnl::memory::desc({w.K, w.N}, dt::u4, tag::ba);
    p.dst = dnnl::memory::desc({M, w.N}, dt::f32, tag::ab);
    p.smd = dnnl::memory::desc({w.K / g, w.N}, dt::bf16, tag::ab);
    p.zmd = dnnl::memory::desc({w.K / g, w.N}, dt::u8, tag::ab);
    p.asd = dnnl::memory::desc({M, w.K / ga}, dt::f32, tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_deterministic(true);
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {g, 1}, dt::bf16);
    attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {g, 1}, dt::u8);
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, ga}, dt::f32);
    attr.set_fpmath_mode(dnnl::fpmath_mode::strict, true);
    if (acc) {
        dnnl::post_ops po;
        po.append_sum(1.0f);
        attr.set_post_ops(po);
    }
    dnnl::matmul::primitive_desc pd(c.eng, p.src, p.wei, p.dst, attr);
    if (std::string(pd.impl_info_str()).find(":ref:") != std::string::npos)
        throw std::runtime_error("oneDNN INT4 a8g: only a reference implementation for this shape");
    p.scr = pd.scratchpad_desc();
    p.prim = dnnl::matmul(pd);
    grow(q, c.scratch, c.scratch_cap, p.scr.get_size(), "scratchpad");
    static const bool verbose = std::getenv("GRIMOIRE_ONEDNN_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "  oneDNN matmul int4 x s8 g%d/a%d M=%d N=%d K=%d%s: %s, scratchpad %zu B, created in %.1f ms\n",
                     g, ga, M, w.N, w.K, acc ? " (+=)" : "", pd.impl_info_str(), p.scr.get_size(),
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    return prims.emplace(k, std::move(p)).first->second;
}

// x [M][K] bf16 -> int8 rows + one f32 scale per (row, G-wide group): scale = max|x| / 127 of the group,
// q = rint(x / scale).  One sub-group of 16 lanes per group, G / 16 (<= 8) elements a lane.
sycl::event quant_groups(sycl::queue& q, const sycl_bf16* x, int M, int K, int G, int8_t* x8, float* as,
                         const std::vector<sycl::event>& deps) {
    const int NG = K / G, E = G / 16;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(size_t(M) * NG * 16, 16), [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(16)]] {
            const size_t grp = it.get_group(0);
            const int lane = int(it.get_local_id(0));
            const size_t base = grp * size_t(G) + size_t(lane) * E;     // rows are K wide, groups tile them
            float v[8];
            float mx = 0.0f;
            for (int i = 0; i < E; ++i) { v[i] = float(x[base + i]); mx = sycl::fmax(mx, sycl::fabs(v[i])); }
            mx = sycl::reduce_over_group(it.get_sub_group(), mx, sycl::maximum<float>());
            const float s = mx > 0.0f ? mx / 127.0f : 1.0f;
            const float inv = 1.0f / s;
            for (int i = 0; i < E; ++i)
                x8[base + i] = int8_t(sycl::clamp(sycl::rint(v[i] * inv), -127.0f, 127.0f));
            if (lane == 0) as[grp] = s;
        });
    });
}
} // namespace

int onednn_i4a8g_group() {
    static const int g = [] {
        const char* e = std::getenv("GRIMOIRE_I4A8G");
        const int v = e && *e ? std::atoi(e) : 0;
        if (e && *e && v != 0 && v != 32 && v != 64 && v != 128)
            std::fprintf(stderr, "  GRIMOIRE_I4A8G=%s ignored (32, 64 or 128)\n", e);
        return v == 32 || v == 64 || v == 128 ? v : 0;
    }();
    return g;
}

namespace {
// The piece sizes onednn_gemm_int4_a8g runs (its tail rows run as one 32-row piece), as bits of kPiecesG.
constexpr int kPiecesG[] = {1024, 512, 256, 128, 64, 32};
unsigned a8g_pieces(int M) {
    unsigned bits = 0;
    int r0 = 0;
    for (int i = 0; i < 6; ++i)
        while (M - r0 >= kPiecesG[i]) { bits |= 1u << i; r0 += kPiecesG[i]; }
    if (M > r0) bits |= 1u << 5;
    return bits;
}
} // namespace

// Whether every piece primitive a call of M rows (made in calls of at most `chunk` rows; 0 = one call) runs is a
// fast kernel on this GPU / oneDNN, checked BEFORE any of its work is enqueued: where oneDNN offers only the
// reference kernel for a piece, the shape keeps its previous path for good (a piece refused mid-call would fail
// the request with the output half written).  Pieces are built on first need, so a short first prompt does not
// pay for all six (~2 s of primitive creation).
bool onednn_int4_a8g_ok(sycl::queue& q, const QuantWeight& w, bool accumulate, int M, int chunk) {
    const int ga = onednn_i4a8g_group();
    if (ga == 0 || w.K % ga != 0 || M <= 0 || !onednn_int4_ok(q, w, 32)) return false;
    static std::map<std::tuple<int, int, int, int>, unsigned> built;   // pieces built per shape; ~0u = refused
    std::lock_guard<std::mutex> lk(mu());
    unsigned& have = built[std::make_tuple(w.N, w.K, i4_group(w), accumulate ? 1 : 0)];
    if (have == ~0u) return false;
    const unsigned need = chunk > 0 && M > chunk ? a8g_pieces(chunk) | a8g_pieces(M % chunk) : a8g_pieces(M);
    try {
        Ctx& c = ctx_for(q);
        for (int i = 0; i < 6; ++i)
            if (need & ~have & (1u << i)) {
                (void)i4g_prim(c, q, w, kPiecesG[i], ga, accumulate);
                have |= 1u << i;
            }
    } catch (const std::exception& ex) {
        have = ~0u;
        std::fprintf(stderr, "  i4a8g off for N=%d K=%d (%s); this shape keeps the bf16 / INT4 path\n", w.N, w.K, ex.what());
        return false;
    }
    return true;
}

sycl::event onednn_gemm_int4_a8g(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y, int M,
                                 bool accumulate, const std::vector<sycl::event>& deps) {
    using namespace dnnl::sycl_interop;
    const int ga = onednn_i4a8g_group();
    if (ga == 0 || w.K % ga != 0) throw std::runtime_error("onednn_gemm_int4_a8g: GRIMOIRE_I4A8G off or K % group");
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    const I4T& t = i4_transposed(q, w);
    if (!t.ok) throw std::runtime_error("onednn_gemm_int4_a8g: weight not supported (check onednn_int4_ok first)");
    const size_t K = size_t(w.K), N = size_t(w.N), NG = K / size_t(ga);
    static bool said = false;
    if (!said) {
        said = true;
        std::fprintf(stderr, "  i4a8g: INT4 prompt GEMMs as int8 activations (one scale per %d values) x INT4 "
                             "weights as stored\n", ga);
    }
    constexpr int T = 32;                         // staged tail rows
    const int Mp = (M + T - 1) / T * T;
    grow(q, c.x8, c.x8_cap, size_t(Mp) * K, "int8 activations");
    grow(q, c.as, c.as_cap, size_t(Mp) * NG, "activation scales");
    sycl::event e = quant_groups(q, x, M, int(K), ga, c.x8, c.as, deps);
    auto run = [&](const I4GPrim& p, const int8_t* xin, const float* sin, float* yout,
                   const std::vector<sycl::event>& d) {
        std::unordered_map<int, dnnl::memory> args;
        args[DNNL_ARG_SRC] = make_memory(p.src, c.eng, memory_kind::usm, const_cast<int8_t*>(xin));
        args[DNNL_ARG_WEIGHTS] = make_memory(p.wei, c.eng, memory_kind::usm, const_cast<uint8_t*>(w.payload));
        args[DNNL_ARG_DST] = make_memory(p.dst, c.eng, memory_kind::usm, yout);
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC] = make_memory(p.asd, c.eng, memory_kind::usm, const_cast<float*>(sin));
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = make_memory(p.smd, c.eng, memory_kind::usm, t.s);
        args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] = make_memory(p.zmd, c.eng, memory_kind::usm, t.z);
        if (p.scr.get_size())
            args[DNNL_ARG_SCRATCHPAD] = make_memory(p.scr, c.eng, memory_kind::usm, c.scratch);
        return execute(p.prim, c.strm, args, d);
    };
    // The same fixed pieces as onednn_gemm_int4, so a new prompt length builds no primitive.  The tail runs
    // as a 32-row primitive over the int8 rows past M (the buffers are padded to whole 32-row pieces; those
    // rows hold earlier values, and their output rows are never copied out), its output staged.
    std::vector<sycl::event> d = {e};
    int r0 = 0;
    for (const int P : kPiecesG)
        while (M - r0 >= P) {
            e = run(i4g_prim(c, q, w, P, ga, accumulate), c.x8 + size_t(r0) * K, c.as + size_t(r0) * NG,
                    y + size_t(r0) * N, d);
            d = {e};
            r0 += P;
        }
    const int rem = M - r0;
    if (rem > 0) {
        grow(q, c.ys, c.ys_cap, size_t(T) * N, "staging");
        if (accumulate) { e = q.memcpy(c.ys, y + size_t(r0) * N, size_t(rem) * N * sizeof(float), d); d = {e}; }
        e = run(i4g_prim(c, q, w, T, ga, accumulate), c.x8 + size_t(r0) * K, c.as + size_t(r0) * NG, c.ys, d);
        e = q.memcpy(y + size_t(r0) * N, c.ys, size_t(rem) * N * sizeof(float), {e});
    }
    return e;
}

bool onednn_gemm_ok(const QuantWeight& w, int M) {
    if (!enabled() || M <= 0 || !w.payload) return false;
    if (w.fmt == Fmt::FP8_E4M3 || w.fmt == Fmt::INT8)
        return w.scales && w.row_scales == 1 && w.row_bytes == int64_t(w.K);
    if (w.fmt == Fmt::BF16)
        return w.row_bytes == int64_t(w.K) * 2;
    return false;
}

namespace {
sycl::event gemm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                      int M, bool accumulate, const std::vector<sycl::event>& deps, bool rot);
}

sycl::event onednn_gemm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                        int M, bool accumulate, const std::vector<sycl::event>& deps) {
    return gemm_impl(q, w, x, y, M, accumulate, deps, false);
}

// w: int8 [N][K] rows already in the 128-point Hadamard domain, f32 scale per row with the
// transform's 1/128 folded in (gemm_fast.cpp i4a8).  x is rotated + quantized here.
sycl::event onednn_gemm_a8rot(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                              int M, bool accumulate, const std::vector<sycl::event>& deps) {
    if (w.fmt != Fmt::INT8 || w.K % 128 != 0 || !onednn_gemm_ok(w, M))
        throw std::runtime_error("onednn_gemm_a8rot: needs INT8 [N][K] with per-row scales, K % 128 == 0");
    return gemm_impl(q, w, x, y, M, accumulate, deps, true);
}

namespace {
sycl::event gemm_impl(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                      int M, bool accumulate, const std::vector<sycl::event>& deps, bool rot) {
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    // rot (INT4 prompts as int8, gemm_fast.cpp): EVERY M through the fixed pieces and the staged
    // tails, so a prompt length never seen before builds no primitive.  Exact-M primitives cost a
    // build per new length and shape: one B70, 8 users' ~60-token prompts admitted together
    // (~480 rows), TTFT 555 ms with them against 486 on the bf16 path (2026-11-12).
    if (!rot && (exact_m() || w.N < kWideN || M < kPieceMinM))
        return exec(q,c, prim_for(q, c, w, M, accumulate, rot), w, x, y, deps);
    const size_t K = size_t(w.K), N = size_t(w.N);
    std::vector<sycl::event> d = deps;
    sycl::event e;
    int r0 = 0;
    for (const int P : kPieces) {
        while (M - r0 >= P) {            // after the 1024s at most one of each smaller piece
            const Prim& p = prim_for(q, c, w, P, accumulate, rot);
            e = exec(q,c, p, w, x + size_t(r0) * K, y + size_t(r0) * N, d);
            d = {e};
            r0 += P;
        }
    }
    const int rem = M - r0;
    if (rem > 0) {
        // The tail through staging rows: its rows in, a T-row primitive, its rows out.  The
        // staging rows past `rem` hold an earlier call's rows (zeros at first); their output
        // rows are never read, and a GEMM row depends on its own input row only.
        int T = kTails[2];
        for (const int t : kTails) if (rem <= t) { T = t; break; }
        const Prim& p = prim_for(q, c, w, T, accumulate, rot);
        if (size_t(T) * K > c.xs_cap) {
            grow(q, c.xs, c.xs_cap, size_t(T) * K, "staging");
            q.memset(c.xs, 0, c.xs_cap * sizeof(sycl_bf16)).wait();
        }
        grow(q, c.ys, c.ys_cap, size_t(T) * N, "staging");
        e = q.memcpy(c.xs, x + size_t(r0) * K, size_t(rem) * K * sizeof(sycl_bf16), d);
        if (accumulate)
            e = q.memcpy(c.ys, y + size_t(r0) * N, size_t(rem) * N * sizeof(float), {e});
        e = exec(q,c, p, w, c.xs, c.ys, {e});
        e = q.memcpy(y + size_t(r0) * N, c.ys, size_t(rem) * N * sizeof(float), {e});
    }
    return e;
}
} // namespace

// SwiGLU (2026-11-17): w = the rotated int8 gate|up rows [2 FI][K] (gate first, as swiglu_rows reads them).
// Per piece of rows: x quantized once; the up half into f32 rows (c.us); the gate half with swish and the
// multiply by those rows as post-ops, written as bf16 h [M][FI].  Before, the whole f32 gate|up block went
// through memory and a separate SwiGLU kernel (Qwen3.8-27B, 4K prompt: ~1.26 GB a layer).  Same pieces and
// staged tails as gemm_impl.  Throws before its first launch when oneDNN has no fast kernel for a shape.
sycl::event onednn_gemm_a8rot_swiglu(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, sycl_bf16* h,
                                     int M, const std::vector<sycl::event>& deps) {
    if (w.fmt != Fmt::INT8 || w.K % 128 != 0 || w.N % 2 != 0 || !onednn_gemm_ok(w, M))
        throw std::runtime_error("onednn_gemm_a8rot_swiglu: needs INT8 [2 FI][K] with per-row scales, K % 128 == 0");
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    const int FI = w.N / 2;
    const size_t K = size_t(w.K);
    QuantWeight gw = w; gw.N = FI;
    QuantWeight uw = w; uw.N = FI;
    uw.payload = w.payload + size_t(FI) * K;
    uw.scales = static_cast<const float*>(w.scales) + FI;
    auto piece = [&](int P, const sycl_bf16* xin, sycl_bf16* hout, const std::vector<sycl::event>& d) {
        const Prim& pu = prim_for(q, c, uw, P, false, true, false);
        const Prim& pg = prim_for(q, c, gw, P, false, true, true);
        grow(q, c.us, c.us_cap, size_t(P) * FI, "SwiGLU up rows");
        const sycl::event e = exec(q, c, pu, uw, xin, c.us, d);           // quantizes xin into c.x8 / c.as
        return exec(q, c, pg, gw, xin, hout, {e}, c.us, true);            // the same quantized rows
    };
    std::vector<sycl::event> d = deps;
    sycl::event e;
    int r0 = 0;
    for (const int P : kPieces)
        while (M - r0 >= P) {
            e = piece(P, x + size_t(r0) * K, h + size_t(r0) * FI, d);
            d = {e};
            r0 += P;
        }
    const int rem = M - r0;
    if (rem > 0) {
        int T = kTails[2];
        for (const int t : kTails) if (rem <= t) { T = t; break; }
        if (size_t(T) * K > c.xs_cap) {
            grow(q, c.xs, c.xs_cap, size_t(T) * K, "staging");
            q.memset(c.xs, 0, c.xs_cap * sizeof(sycl_bf16)).wait();
        }
        grow(q, c.hs, c.hs_cap, size_t(T) * FI, "SwiGLU staging");
        e = q.memcpy(c.xs, x + size_t(r0) * K, size_t(rem) * K * sizeof(sycl_bf16), d);
        e = piece(T, c.xs, c.hs, {e});
        e = q.memcpy(h + size_t(r0) * FI, c.hs, size_t(rem) * FI * sizeof(sycl_bf16), {e});
    }
    return e;
}

} // namespace b70
