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
// and nothing shared between cards), one matmul primitive per (format, M, N,
// K, accumulate), user-managed scratchpad, deterministic mode (measured: no
// cost on these shapes).
//
// Numerics differ from the dequantize path in the last bits: that path rounds
// e4m3 * row_scale to bf16 before the GEMM; oneDNN multiplies the exact e4m3
// value (it fits bf16) and applies the scale to the fp32 sum.
#include "onednn_gemm.hpp"
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace b70 {

namespace {

using dt  = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;

bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("GRIMOIRE_ONEDNN");
        return v && std::atoi(v) != 0;
    }();
    return on;
}

struct Key {
    int fmt, M, N, K, acc;
    bool operator<(const Key& o) const {
        return std::tie(fmt, M, N, K, acc) < std::tie(o.fmt, o.M, o.N, o.K, o.acc);
    }
};

struct Prim {
    dnnl::matmul       prim;
    dnnl::memory::desc src, wei, dst, scr, scl;
};

struct Ctx {
    dnnl::engine         eng;
    dnnl::stream         strm;
    std::map<Key, Prim>  prims;
    void*                scratch = nullptr;
    size_t               scratch_cap = 0;
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

const Prim& prim_for(Ctx& c, const QuantWeight& w, int M, bool acc) {
    const Key k{int(w.fmt), M, w.N, w.K, acc ? 1 : 0};
    auto it = c.prims.find(k);
    if (it != c.prims.end()) return it->second;
    const bool fp8 = w.fmt == Fmt::FP8_E4M3;
    Prim p;
    p.src = dnnl::memory::desc({M, w.K}, dt::bf16, tag::ab);
    // W is [N][K] row-major: oneDNN's [K][N] weights in the transposed tag.
    p.wei = dnnl::memory::desc({w.K, w.N}, fp8 ? dt::f8_e4m3 : dt::bf16, tag::ba);
    p.dst = dnnl::memory::desc({M, w.N}, dt::f32, tag::ab);
    dnnl::primitive_attr attr;
    attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    attr.set_deterministic(true);
    if (fp8) {
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);    // one f32 scale per output column
        p.scl = dnnl::memory::desc({w.N}, dt::f32, tag::a);
    }
    if (acc) {
        dnnl::post_ops po;
        po.append_sum(1.0f);
        attr.set_post_ops(po);
    }
    dnnl::matmul::primitive_desc pd(c.eng, p.src, p.wei, p.dst, attr);
    p.scr  = pd.scratchpad_desc();
    p.prim = dnnl::matmul(pd);
    static const bool verbose = std::getenv("GRIMOIRE_ONEDNN_VERBOSE") != nullptr;
    if (verbose)
        std::fprintf(stderr, "  oneDNN matmul %s M=%d N=%d K=%d%s: %s, scratchpad %zu B\n",
                     fp8 ? "fp8" : "bf16", M, w.N, w.K, acc ? " (+=)" : "",
                     pd.impl_info_str(), p.scr.get_size());
    return c.prims.emplace(k, std::move(p)).first->second;
}

} // namespace

bool onednn_gemm_ok(const QuantWeight& w, int M) {
    if (!enabled() || M <= 0 || !w.payload) return false;
    if (w.fmt == Fmt::FP8_E4M3)
        return w.scales && w.row_scales == 1 && w.row_bytes == int64_t(w.K);
    if (w.fmt == Fmt::BF16)
        return w.row_bytes == int64_t(w.K) * 2;
    return false;
}

sycl::event onednn_gemm(sycl::queue& q, const QuantWeight& w, const sycl_bf16* x, float* y,
                        int M, bool accumulate, const std::vector<sycl::event>& deps) {
    std::lock_guard<std::mutex> lk(mu());
    Ctx& c = ctx_for(q);
    const Prim& p = prim_for(c, w, M, accumulate);
    const size_t sb = p.scr.get_size();
    if (sb > c.scratch_cap) {
        q.wait();                                  // nothing in flight may still use the old one
        if (c.scratch) sycl::free(c.scratch, q);
        c.scratch = sycl::malloc_device(sb, q);
        c.scratch_cap = c.scratch ? sb : 0;
        if (!c.scratch) throw std::runtime_error("oneDNN matmul: scratchpad allocation failed");
    }
    using namespace dnnl::sycl_interop;
    std::unordered_map<int, dnnl::memory> args;
    args[DNNL_ARG_SRC]     = make_memory(p.src, c.eng, memory_kind::usm,
                                         const_cast<sycl_bf16*>(x));
    args[DNNL_ARG_WEIGHTS] = make_memory(p.wei, c.eng, memory_kind::usm,
                                         const_cast<uint8_t*>(w.payload));
    args[DNNL_ARG_DST]     = make_memory(p.dst, c.eng, memory_kind::usm, y);
    if (sb) args[DNNL_ARG_SCRATCHPAD] = make_memory(p.scr, c.eng, memory_kind::usm, c.scratch);
    if (w.fmt == Fmt::FP8_E4M3)
        args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
            make_memory(p.scl, c.eng, memory_kind::usm, const_cast<void*>(w.scales));
    return execute(p.prim, c.strm, args, deps);
}

} // namespace b70
