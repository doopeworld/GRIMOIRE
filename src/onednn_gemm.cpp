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
    int fmt, M, N, K, acc, a8;
    bool operator<(const Key& o) const {
        return std::tie(fmt, M, N, K, acc, a8) < std::tie(o.fmt, o.M, o.N, o.K, o.acc, o.a8);
    }
};

struct Prim {
    dnnl::matmul       prim;
    dnnl::memory::desc src, wei, dst, scr, scl, ascl;
    bool a8=false;
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

const Prim& prim_for(sycl::queue& q, Ctx& c, const QuantWeight& w, int M, bool acc) {
    static const bool requested=[] {
        const char* e=std::getenv("GRIMOIRE_ONEDNN_W8A8");return e&&std::atoi(e)!=0;
    }();
    const bool a8=requested&&w.fmt==Fmt::INT8&&M>=32&&w.N>=256&&w.K%64==0;
    const Key k{int(w.fmt), M, w.N, w.K, acc ? 1 : 0, a8?1:0};
    auto it = c.prims.find(k);
    if (it != c.prims.end()) return it->second;
    const auto t0 = std::chrono::steady_clock::now();
    const bool fp8 = w.fmt == Fmt::FP8_E4M3;
    // INT8 (2026-10-29): GRIMOIRE's symmetric per-row int8 -- the same layout as its FP8, one
    // byte a weight and one f32 scale a row -- as oneDNN weights decompression: s8 weights,
    // bf16 source, fpmath bf16 applied to integers.  Before, INT8 had no oneDNN or small-M
    // path: Qwen3.8-27B on two cards pp4096 2,448 / 4 users 46 tok/s against FP8 3,475 / 89.
    const bool s8 = w.fmt == Fmt::INT8;
    Prim p;
    p.a8=a8;
    p.src = dnnl::memory::desc({M, w.K}, a8?dt::s8:dt::bf16, tag::ab);
    // W is [N][K] row-major: oneDNN's [K][N] weights in the transposed tag.
    p.wei = dnnl::memory::desc({w.K, w.N}, fp8 ? dt::f8_e4m3 : s8 ? dt::s8 : dt::bf16, tag::ba);
    p.dst = dnnl::memory::desc({M, w.N}, dt::f32, tag::ab);
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
    dnnl::matmul::primitive_desc pd(c.eng, p.src, p.wei, p.dst, attr);
    if(a8&&std::string(pd.impl_info_str()).find(":ref:")!=std::string::npos)
        throw std::runtime_error("oneDNN W8A8: only a reference implementation is available for this shape");
    p.scr  = pd.scratchpad_desc();
    p.prim = dnnl::matmul(pd);
    grow(q, c.scratch, c.scratch_cap, p.scr.get_size(), "scratchpad");
    static const bool verbose = std::getenv("GRIMOIRE_ONEDNN_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "  oneDNN matmul %s M=%d N=%d K=%d%s: %s, scratchpad %zu B, "
                     "created in %.1f ms\n", a8?"W8A8":fp8 ? "fp8" : s8 ? "s8" : "bf16", M, w.N, w.K,
                     acc ? " (+=)" : "", pd.impl_info_str(), p.scr.get_size(),
                     std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0).count());
    return c.prims.emplace(k, std::move(p)).first->second;
}

// One primitive over exactly its own row count, x and y at those rows.
sycl::event exec(sycl::queue& q,Ctx& c, const Prim& p, const QuantWeight& w, const sycl_bf16* x, float* y,
                 const std::vector<sycl::event>& deps) {
    using namespace dnnl::sycl_interop;
    std::unordered_map<int, dnnl::memory> args;
    std::vector<sycl::event> ready=deps;
    void* source=const_cast<sycl_bf16*>(x);
    if(p.a8){
        const int M=int(p.src.get_dims()[0]),K=w.K;
        grow(q,c.x8,c.x8_cap,size_t(M)*K,"int8 activations");
        grow(q,c.as,c.as_cap,size_t(M),"activation scales");
        int8_t* out=c.x8;float* scales=c.as;
        constexpr size_t WG=256;
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
    args[DNNL_ARG_SRC]     = make_memory(p.src, c.eng, memory_kind::usm,source);
    args[DNNL_ARG_WEIGHTS] = make_memory(p.wei, c.eng, memory_kind::usm,
                                         const_cast<uint8_t*>(w.payload));
    args[DNNL_ARG_DST]     = make_memory(p.dst, c.eng, memory_kind::usm, y);
    if (p.scr.get_size())
        args[DNNL_ARG_SCRATCHPAD] = make_memory(p.scr, c.eng, memory_kind::usm, c.scratch);
    if (w.fmt == Fmt::FP8_E4M3 || w.fmt == Fmt::INT8)
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
            make_memory(p.scl, c.eng, memory_kind::usm, const_cast<void*>(w.scales));
    return execute(p.prim, c.strm, args, ready);
}

} // namespace

bool onednn_gemm_ok(const QuantWeight& w, int M) {
    if (!enabled() || M <= 0 || !w.payload) return false;
    if (w.fmt == Fmt::FP8_E4M3 || w.fmt == Fmt::INT8)
        return w.scales && w.row_scales == 1 && w.row_bytes == int64_t(w.K);
    if (w.fmt == Fmt::BF16)
        return w.row_bytes == int64_t(w.K) * 2;
    return false;
}

sycl::event onednn_gemm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                        int M, bool accumulate, const std::vector<sycl::event>& deps) {
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    if (exact_m() || w.N < kWideN || M < kPieceMinM)
        return exec(q,c, prim_for(q, c, w, M, accumulate), w, x, y, deps);
    const size_t K = size_t(w.K), N = size_t(w.N);
    std::vector<sycl::event> d = deps;
    sycl::event e;
    int r0 = 0;
    for (const int P : kPieces) {
        while (M - r0 >= P) {            // after the 1024s at most one of each smaller piece
            const Prim& p = prim_for(q, c, w, P, accumulate);
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
        const Prim& p = prim_for(q, c, w, T, accumulate);
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

} // namespace b70
