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

// dnnl_create_probe -- what a oneDNN matmul costs to CREATE on the B70, per row count, for the
// FP8 (row-scaled) and BF16 weights onednn_gemm.cpp serves: primitive_desc + primitive, timed on
// the host, then one execution (its first run) and a second one.  Tells whether a new prompt
// length compiles a new kernel or reuses one (oneDNN's kernel cache), i.e. whether row buckets
// are needed.  No GPU busy-waits; each case is one tiny GEMM.
// Usage: dnnl_create_probe M1 M2 ...   (N=34816 K=5120 and N=5120 K=17408 for each M)
#include <dnnl.hpp>
#include <dnnl_sycl.hpp>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using dt  = dnnl::memory::data_type;
using tag = dnnl::memory::format_tag;

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    auto eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
    const int shapes[2][2] = {{34816, 5120}, {5120, 17408}};
    const size_t maxM = 4096;
    void* x = sycl::malloc_device(maxM * 17408 * 2, q);
    void* y = sycl::malloc_device(maxM * 34816 * 4, q);
    void* w = sycl::malloc_device(size_t(34816) * 5120 * 2, q);
    void* s = sycl::malloc_device(34816 * 4, q);
    q.memset(x, 0, maxM * 17408 * 2); q.memset(w, 0, size_t(34816) * 5120 * 2);
    q.memset(s, 0, 34816 * 4).wait();
    for (int f = 0; f < 2; ++f) {
        const bool fp8 = f == 0;
        for (int a = 1; a < argc; ++a) {
            const int M = std::atoi(argv[a]);
            if (M <= 0 || size_t(M) > maxM) continue;
            for (auto& sh : shapes) {
                const int N = sh[0], K = sh[1];
                auto t0 = std::chrono::steady_clock::now();
                dnnl::memory::desc smd({M, K}, dt::bf16, tag::ab);
                dnnl::memory::desc wmd({K, N}, fp8 ? dt::f8_e4m3 : dt::bf16, tag::ba);
                dnnl::memory::desc dmd({M, N}, dt::f32, tag::ab);
                dnnl::primitive_attr attr;
                attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
                attr.set_deterministic(true);
                if (fp8) attr.set_scales_mask(DNNL_ARG_WEIGHTS, 1 << 1);
                dnnl::matmul::primitive_desc pd(eng, smd, wmd, dmd, attr);
                const double t_pd = ms_since(t0);
                dnnl::matmul prim(pd);
                const double t_prim = ms_since(t0) - t_pd;
                void* scr = pd.scratchpad_desc().get_size()
                                ? sycl::malloc_device(pd.scratchpad_desc().get_size(), q) : nullptr;
                using namespace dnnl::sycl_interop;
                std::unordered_map<int, dnnl::memory> args{
                    {DNNL_ARG_SRC, make_memory(smd, eng, memory_kind::usm, x)},
                    {DNNL_ARG_WEIGHTS, make_memory(wmd, eng, memory_kind::usm, w)},
                    {DNNL_ARG_DST, make_memory(dmd, eng, memory_kind::usm, y)}};
                if (scr) args[DNNL_ARG_SCRATCHPAD] = make_memory(pd.scratchpad_desc(), eng, memory_kind::usm, scr);
                if (fp8) args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
                    make_memory(dnnl::memory::desc({N}, dt::f32, tag::a), eng, memory_kind::usm, s);
                auto t1 = std::chrono::steady_clock::now();
                execute(prim, strm, args, {}).wait();
                const double t_run1 = ms_since(t1);
                t1 = std::chrono::steady_clock::now();
                execute(prim, strm, args, {}).wait();
                const double t_run2 = ms_since(t1);
                std::printf("%s M=%4d N=%5d K=%5d  pd %7.2f ms  prim %7.2f ms  run1 %7.2f  run2 %6.2f ms  %s\n",
                            fp8 ? "fp8 " : "bf16", M, N, K, t_pd, t_prim, t_run1, t_run2,
                            pd.impl_info_str());
                if (scr) sycl::free(scr, q);
            }
        }
    }
    sycl::free(x, q); sycl::free(y, q); sycl::free(w, q); sycl::free(s, q);
    return 0;
}
