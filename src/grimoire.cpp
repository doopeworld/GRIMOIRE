// =====================================================================
//  grimoire.cpp  --  the engine
//
//  Upload, forward pass, and generation for Qwen3.5 dense and MoE.
//
//  LOAD-TIME QUANTIZATION IS THE POINT
//  -----------------------------------
//  The checkpoint quantizes its experts and leaves everything else in
//  bf16. Measured on a B70, per decoded token:
//
//      deltanet projections   2.02 GB   3.73 ms   <- largest single cost
//      lm_head                1.02 GB   1.88 ms
//      routed experts (mxfp4) 0.60 GB   1.92 ms
//
//  The two bf16 items cost more than twice what the experts cost. Both
//  are quantized here at load, once, on the CPU: 3.04 GB becomes 0.80
//  GB and ~5.6 ms becomes ~1.4 ms. No kernel work involved -- the int4
//  GEMV already runs at 478 GB/s.
//
//  Experts are NOT re-quantized. compressed-tensors MXFP4 is already
//  byte-identical to the kernel's layout, so those 15.9 GiB are
//  memcpy'd straight from the mmapped shard to VRAM.
// =====================================================================
#include "kernels.hpp"
#include "b70/engine.hpp"
#include "b70/grimoire_api.hpp"
#include "b70/http_request.hpp"
#include "b70/qwen35.hpp"
#include "b70/dflash_config.hpp"
#include "b70/qwen4_exp.hpp"
#include "b70/nvfp4.hpp"
#include "b70/tensor_layout.hpp"
#include "b70/gptq.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <memory>
#include <functional>
#include <type_traits>
#include <dlfcn.h>

namespace sycl_ext = sycl::ext::oneapi::experimental;
#include <cstdio>
#include <map>
#include <set>
#include <array>
#include <cctype>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <algorithm>
#include <numeric>
#include <array>
#include <cstdlib>
#include <cerrno>
#include <thread>
#include <tuple>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace b70 {

// The standalone native profile must never load a framework-backed bridge.
// Generic SYCL/XMX kernels remain available when bridges are disabled.
static void* open_bridge(const char* path,int flags) {
#ifdef GRIMOIRE_NATIVE_ONLY
    (void)path; (void)flags;
    return nullptr;
#else
    return ::dlopen(path,flags);
#endif
}

using Xe2GroupedW4A16 = void (*)(sycl::queue*, const void*, const unsigned char*,
    const void*, void*, int, int, const int*, const int*, int, int, int*);

static Xe2GroupedW4A16 load_xe2_grouped() {
    static Xe2GroupedW4A16 fn = nullptr;
    static bool attempted = false;
    static void* handle = nullptr;
    if (attempted) return fn;
    attempted = true;
    const char* env = std::getenv("GRIMOIRE_XE2_BRIDGE");
    const char* paths[] = {env, "src/libgrimoire_xe2_bridge.so",
                           "/work/src/libgrimoire_xe2_bridge.so",
                           "/bridge/libgrimoire_xe2_bridge.so"};
    for (const char* path : paths) {
        if (!path || !*path) continue;
        handle = open_bridge(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) continue;
        fn = reinterpret_cast<Xe2GroupedW4A16>(
            dlsym(handle, "grimoire_xe2_grouped_w4a16"));
        if (fn) break;
        dlclose(handle); handle = nullptr;
    }
    if (!fn) std::fprintf(stderr,
        "  Xe2 grouped GEMM unavailable; using exact fallback (%s)\n", dlerror());
    return fn;
}

using Xe2DenseMXFP4 = void (*)(sycl::queue*, const void*, const unsigned char*,
    const unsigned char*, void*, int, int, int);
using Xe2GroupedMXFP4 = void (*)(sycl::queue*, const void*, const unsigned char*,
    const unsigned char*, void*, int, int, const int*, const int*, int, int*);

static Xe2DenseMXFP4 load_xe2_dense_mxfp4() {
    static Xe2DenseMXFP4 fn = nullptr;
    static bool attempted = false;
    static void* handle = nullptr;
    if (attempted) return fn;
    attempted = true;
    const char* env = std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[] = {env, "src/libgrimoire_xe2_grouped.so",
                           "/work/src/libgrimoire_xe2_grouped.so",
                           "/bridge/libgrimoire_xe2_grouped.so"};
    for (const char* path : paths) {
        if (!path || !*path) continue;
        handle = open_bridge(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) continue;
        fn = reinterpret_cast<Xe2DenseMXFP4>(
            dlsym(handle, "grimoire_xe2_dense_mxfp4_bf16"));
        if (fn) break;
        dlclose(handle); handle = nullptr;
    }
    if (!fn) std::fprintf(stderr,
        "  Xe2 native MXFP4 GEMM unavailable; using internal XMX fallback\n");
    return fn;
}

static Xe2DenseMXFP4 load_xe2_dense_mxfp4_f32() {
    static Xe2DenseMXFP4 fn=nullptr; static bool attempted=false;
    static void* handle=nullptr; if(attempted)return fn; attempted=true;
    const char* env=std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_grouped.so",
        "/work/src/libgrimoire_xe2_grouped.so","/bridge/libgrimoire_xe2_grouped.so"};
    for(const char* path:paths){if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        fn=reinterpret_cast<Xe2DenseMXFP4>(
            dlsym(handle,"grimoire_xe2_dense_mxfp4_f32"));
        if(fn)break;dlclose(handle);handle=nullptr;}
    return fn;
}

// W4A8: int8 activations x symmetric int4 weights on the native s8xs4 DPAS.
// Measured 1.69-1.87x the MXFP4 GEMM on production shapes.
using Xe2DenseW4A8 = void (*)(sycl::queue*, const void*, const unsigned char*,
    const float*, const float*, void*, int, int, int);

// GRIMOIRE_W4A8=1 turns on the int8xint4 prefill path.  Off by default: it
// costs ~9 GB of VRAM and, until the weights come from the BF16 original
// rather than the MXFP4 artifact, it quantizes a quantization.
static const float kK2GateBeta = 0.6931472f;   // log 2
static bool w4a8_enabled() {
    static const bool v = []{ const char* e = std::getenv("GRIMOIRE_W4A8");
        return e && *e && std::atoi(e) != 0; }();
    return v;
}

static Xe2DenseW4A8 load_xe2_dense_w4a8(const char* sym) {
    void* handle=nullptr;
    const char* env=std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_grouped.so",
        "/work/src/libgrimoire_xe2_grouped.so","/bridge/libgrimoire_xe2_grouped.so"};
    for(const char* path:paths){if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        auto fn=reinterpret_cast<Xe2DenseW4A8>(dlsym(handle,sym));
        if(fn)return fn;
        dlclose(handle);handle=nullptr;}
    return nullptr;
}

// GRIMOIRE_W4A8 FREES a converted weight's MXFP4 payload, so a weight may
// only be converted when the tiles that consume int4 actually exist.  The
// decode selection chain always falls back to the _big symbols, so those two
// are the real requirement.  A STALE libgrimoire_xe2_grouped.so still
// resolves the older mxfp4 symbols but not these -- build_b70.sh does not
// rebuild the bridges -- and mmb()/mmbb() would then fall through to the
// freed payload with a live function pointer, which is a DEVICE_LOST and a
// power cycle, not a clean error.  Refuse the conversion and stay on MXFP4.
static bool w4a8_tiles_available() {
    static const bool v = [] {
        const bool ok = load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32")
                     && load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16");
        if (!ok)
            std::fprintf(stderr,
                "  W4A8 requested but the w4a8 tiles are missing from the "
                "bridge -- staying on MXFP4 (run tools/build_bridges_b70.sh)\n");
        return ok;
    }();
    return v;
}

static Xe2GroupedMXFP4 load_xe2_grouped_sym(const char* sym) {
    void* handle=nullptr;
    const char* env=std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_grouped.so",
        "/work/src/libgrimoire_xe2_grouped.so","/bridge/libgrimoire_xe2_grouped.so"};
    for(const char* path:paths){if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        auto fn=reinterpret_cast<Xe2GroupedMXFP4>(dlsym(handle,sym));
        if(fn)return fn;
        dlclose(handle);handle=nullptr;}
    return nullptr;
}

static Xe2GroupedMXFP4 load_xe2_grouped_mxfp4() {
    static Xe2GroupedMXFP4 fn = nullptr;
    static bool attempted = false;
    static void* handle = nullptr;
    if (attempted) return fn;
    attempted = true;
    const char* env = std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[] = {env, "src/libgrimoire_xe2_grouped.so",
                           "/work/src/libgrimoire_xe2_grouped.so",
                           "/bridge/libgrimoire_xe2_grouped.so"};
    for (const char* path : paths) {
        if (!path || !*path) continue;
        handle = open_bridge(path, RTLD_NOW | RTLD_LOCAL);
        if (!handle) continue;
        fn = reinterpret_cast<Xe2GroupedMXFP4>(
            dlsym(handle, "grimoire_xe2_grouped_mxfp4_bf16"));
        if (fn) break;
        dlclose(handle); handle = nullptr;
    }
    return fn;
}

using Xe2FusedGateUpMXFP4 = void (*)(sycl::queue*, const void*,
    const unsigned char*, const unsigned char*, void*, int, int, const int*,
    int, int32_t*);
static Xe2FusedGateUpMXFP4 load_xe2_fused_gate_up_mxfp4() {
    static Xe2FusedGateUpMXFP4 fn=nullptr; static bool attempted=false;
    static void* handle=nullptr; if(attempted)return fn; attempted=true;
    const char* env=std::getenv("GRIMOIRE_XE2_FUSED_MOE_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_fused_moe.so",
        "/bridge/libgrimoire_xe2_fused_moe.so"};
    for(const char* path:paths){if(!path||!*path)continue;
      handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
      fn=reinterpret_cast<Xe2FusedGateUpMXFP4>(dlsym(handle,
        "grimoire_xe2_fused_moe_gate_up_mxfp4_silu_bf16"));
      if(fn)break;dlclose(handle);handle=nullptr;}
    return fn;
}

using Xe2ChunkPrefill = void (*)(sycl::queue*, const void*, const void*,
    const void*, void*, int, int, int, int, int, const int*, const int*, float,
    bool);
using Xe2DFlashPagedF16 = int (*)(
    sycl::queue*, const void*, const void*, const void*, void*, int, int, int,
    int, int, int, int, const int*, const int*, const int*, const int*, float,
    int, int, bool);

static Xe2DFlashPagedF16 load_xe2_dflash_paged_f16() {
    static Xe2DFlashPagedF16 fn=nullptr;
    static bool attempted=false;
    static void* handle=nullptr;
    if(attempted)return fn;
    attempted=true;
    const char* env=std::getenv("GRIMOIRE_XE2_ATTN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_attention_raw.so",
        "/grimoire/src/libgrimoire_xe2_attention_raw.so",
        "/bridge/libgrimoire_xe2_attention_raw.so",
        "/opt/grimoire/lib/libgrimoire_xe2_attention_raw.so"};
    for(const char* path:paths){
        if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);
        if(!handle)continue;
        fn=reinterpret_cast<Xe2DFlashPagedF16>(
            dlsym(handle,"grimoire_xe2_dflash_paged_f16"));
        if(fn)break;
        fn=nullptr;
        dlclose(handle);
        handle=nullptr;
    }
    if(!fn)std::fprintf(stderr,
        "  Muse DFlash: raw Xe2 FP16 paged FA2 bridge unavailable\n");
    return fn;
}

// ---- BesTLA (Intel Neural Compressor) prefill GEMM -------------------------
// Measured 2026-08-25 on Qwen3.8-27B: 143 TFLOP/s vs our cutlass MXFP4's ~100, i.e.
// 6.46 ms/layer saved across gate/up/down at M=4096 (~413 ms over 64 layers).
// PREFILL ONLY: at M=1 BesTLA manages 99 GB/s against our GEMV's 386-393, so decode
// must keep the existing MXFP4 path.  Weights come from the int4-AutoRound checkpoint
// (quantized once from BF16); GRIMOIRE's own MXFP4 artifact still serves decode.
using BestlaInitAll = int(*)(const char*,int,int*,int*,int*,int*);
using BestlaFfn     = int(*)(sycl::queue*,int,int,const void*,void**,int);
static BestlaInitAll g_bestla_init=nullptr;
static BestlaFfn     g_bestla_ffn =nullptr;
static int g_bestla_ready=0, g_bestla_min_m=64;

static bool load_bestla(int nlayers){
    static bool attempted=false;
    if(attempted) return g_bestla_ready!=0;
    attempted=true;
    const char* ckpt=std::getenv("GRIMOIRE_BESTLA_CKPT");
    if(!ckpt||!*ckpt) return false;
    if(const char* m=std::getenv("GRIMOIRE_BESTLA_MIN_M")) g_bestla_min_m=std::atoi(m);
    const char* env=std::getenv("GRIMOIRE_BESTLA_LIB");
    const char* paths[]={env,"src/libgrimoire_bestla.so",
        "/grimoire/src/libgrimoire_bestla.so"};
    void* h=nullptr;
    for(const char* pth:paths){ if(!pth||!*pth)continue;
        h=open_bridge(pth,RTLD_NOW|RTLD_GLOBAL); if(h)break; }
    if(!h){ std::fprintf(stderr,"  BesTLA: %s\n",dlerror()); return false; }
    g_bestla_init=reinterpret_cast<BestlaInitAll>(dlsym(h,"grimoire_bestla_init_all"));
    g_bestla_ffn =reinterpret_cast<BestlaFfn>(dlsym(h,"grimoire_bestla_ffn"));
    if(!g_bestla_init||!g_bestla_ffn){ std::fprintf(stderr,"  BesTLA: symbols missing\n"); return false; }
    int Ng=0,Kg=0,Nd=0,Kd=0;
    const int rc=g_bestla_init(ckpt,nlayers,&Ng,&Kg,&Nd,&Kd);
    if(rc){ std::fprintf(stderr,"  BesTLA: init failed rc=%d\n",rc); return false; }
    std::fprintf(stderr,"  BesTLA prefill FFN: gate/up %dx%d down %dx%d, M>=%d\n",
                 Ng,Kg,Nd,Kd,g_bestla_min_m);
    g_bestla_ready=1;
    return true;
}

static Xe2ChunkPrefill load_xe2_chunk_prefill() {
    static Xe2ChunkPrefill fn=nullptr; static bool attempted=false;
    static void* handle=nullptr;
    if(attempted)return fn; attempted=true;
    // Prefer vLLM's FlashAttention-2 (_vllm_fa2_C::varlen_fwd) when the FA2 bridge
    // is present: measured 2.713 ms vs 10.87 ms per layer at M=4096 on Qwen3.8-27B
    // (76.0 vs 19.0 TFLOP/s), i.e. ~130 ms over the 16 full-attention layers.
    // NOTE: this is a DIFFERENT vLLM library from libattn_kernels_xe_2.so, which is
    // what libgrimoire_xe2_attention_bridge.so calls and which is SLOWER than our own.
    // Set GRIMOIRE_DISABLE_FA2=1 to fall back.
    // OPT-IN ONLY.  Standalone the FA2 kernel is 2.713 ms/layer vs our 10.87, but
    // in-model it measures 226 ms vs 173.9 for attention_raw: it runs on torch's XPU
    // stream while GRIMOIRE runs its own queue, so each call needs a wait() before and
    // a stream synchronize() after -- 32 full pipeline stalls per forward pass that
    // cost more than the kernel saves.  Enable with GRIMOIRE_ENABLE_FA2=1 to work on
    // it; the fix is to make the op run on GRIMOIRE's queue instead of torch's.
    if(std::getenv("GRIMOIRE_ENABLE_FA2")){
        const char* fenv=std::getenv("GRIMOIRE_XE2_FA2_BRIDGE");
        const char* fpaths[]={fenv,"src/libgrimoire_xe2_fa2.so",
            "/grimoire/src/libgrimoire_xe2_fa2.so","/bridge/libgrimoire_xe2_fa2.so"};
        for(const char* path:fpaths){
            if(!path||!*path)continue;
            void* fh=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!fh)continue;
            auto av=reinterpret_cast<int(*)()>(dlsym(fh,"grimoire_xe2_fa2_available"));
            auto pf=reinterpret_cast<Xe2ChunkPrefill>(
                dlsym(fh,"grimoire_xe2_fa2_prefill_bf16"));
            if(av&&pf&&av()){ handle=fh; fn=pf;
                std::fprintf(stderr,"  full attention: vLLM FlashAttention-2\n");
                return fn; }
            dlclose(fh);
        }
    }
    const char* env=std::getenv("GRIMOIRE_XE2_ATTN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_attention_bridge.so",
        "/work/src/libgrimoire_xe2_attention_bridge.so",
        "/bridge/libgrimoire_xe2_attention_bridge.so"};
    for(const char* path:paths){
        if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        fn=reinterpret_cast<Xe2ChunkPrefill>(
            dlsym(handle,"grimoire_xe2_chunk_prefill_bf16"));
        if(fn)break;dlclose(handle);handle=nullptr;
    }
    if(!fn)std::fprintf(stderr,"  Xe2 chunk prefill unavailable; using fallback\n");
    return fn;
}

using Xe2ChunkGdn = void (*)(sycl::queue*, void*, const void*, const void*,
    const void*, const float*, const float*, const float*, const void*, float*,
    int, int, int, int, int, const int*, const int*, const bool*);

static Xe2ChunkGdn load_xe2_chunk_gdn() {
    static Xe2ChunkGdn fn=nullptr;static bool attempted=false;static void* handle=nullptr;
    if(attempted)return fn;attempted=true;
    const char* env=std::getenv("GRIMOIRE_XE2_ATTN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_attention_bridge.so",
        "/work/src/libgrimoire_xe2_attention_bridge.so","/bridge/libgrimoire_xe2_attention_bridge.so"};
    for(const char* path:paths){if(!path||!*path)continue;handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);
        if(!handle)continue;fn=reinterpret_cast<Xe2ChunkGdn>(dlsym(handle,"grimoire_xe2_chunk_gdn_bf16"));
        if(fn)break;dlclose(handle);handle=nullptr;}
    if(!fn)std::fprintf(stderr,"  Xe2 chunk GDN unavailable; using fallback\n");
    return fn;
}

using Xe2ChunkGdnRaw = void (*)(sycl::queue*,void*,const void*,const void*,const void*,
    void*,void*,void*,const float*,float*,const float*,const void*,float*,int,
    const int*,const int*,const bool*,int,int,int,int,int);
static Xe2ChunkGdnRaw load_xe2_chunk_gdn_raw(){
    static Xe2ChunkGdnRaw fn=nullptr;static bool attempted=false;static void* handle=nullptr;
    if(attempted)return fn;attempted=true;
    const char* env=std::getenv("GRIMOIRE_XE2_GDN_RAW_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_gdn_raw.so","/work/src/libgrimoire_xe2_gdn_raw.so",
        "/bridge/libgrimoire_xe2_gdn_raw.so"};
    for(const char* path:paths){if(!path||!*path)continue;handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);
        if(!handle)continue;fn=reinterpret_cast<Xe2ChunkGdnRaw>(dlsym(handle,"grimoire_xe2_chunk_gdn_raw_bf16"));
        if(fn)break;dlclose(handle);handle=nullptr;}
    return fn;
}

struct OneDnnW4Api {
    void* (*create)(sycl::queue*,int,int,int,int,int) = nullptr;
    size_t (*scratch_size)(void*) = nullptr;
    void (*execute)(void*,const void*,const void*,const void*,const int8_t*,void*,void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    explicit operator bool() const { return create && scratch_size && execute && destroy; }
};

// oneDNN MXFP4 W4A16 matmul. This is the same class of kernel vLLM XPU calls
// for every linear layer (csrc/xpu/onednn/onednn_matmul.cpp -> int4_gemm_w4a16).
// The bridge's MXPlan already maps GRIMOIRE's native [N,K] row-major E2M1
// payload and its contiguous E8M0 scales straight in -- no repacking.
// Measured 2026-09-05 at M=1, N=10240, K=5120 with incompressible weights:
// 48.45 us = 535 GiB/s (~575 GB/s, 94% of the 608 GB/s part), where the
// hand-written GEMV on the same shape does 126-370 GB/s.
struct OneDnnMXApi {
    void* (*create)(sycl::queue*,int,int,int) = nullptr;
    size_t (*scratch_size)(void*) = nullptr;
    void (*execute)(void*,const void*,const void*,const void*,void*,void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    explicit operator bool() const { return create && scratch_size && execute && destroy; }
};

// Signed-int4 oneDNN matmul. The u4 API above needs a zero point of 8 and
// therefore decodes GRIMOIRE's sign-extended nibbles rotated by 8; this one
// declares the payload as s4 and maps it in unchanged. See the SPlan comment
// in xe2_onednn_bridge.cpp.
// vLLM's w4a8 oneDNN matmul, ported in xe2_onednn_bridge.cpp.
struct OneDnnW4A8Api {
    void* (*create)(sycl::queue*,int,int,int,int) = nullptr;
    size_t (*scratch_size)(void*) = nullptr;
    void (*execute)(void*,const void*,const void*,const void*,const void*,void*,void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    explicit operator bool() const { return create && scratch_size && execute && destroy; }
};

struct OneDnnS4Api {
    void* (*create)(sycl::queue*,int,int,int,int,int) = nullptr;
    size_t (*scratch_size)(void*) = nullptr;
    void (*execute)(void*,const void*,const void*,const void*,void*,void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    explicit operator bool() const { return create && scratch_size && execute && destroy; }
};

struct OneDnnBF16Api {
    void* (*create)(sycl::queue*,int,int,int) = nullptr;
    size_t (*scratch_size)(void*) = nullptr;
    void (*execute)(void*,const void*,const void*,void*,void*) = nullptr;
    void (*destroy)(void*) = nullptr;
    explicit operator bool() const { return create && scratch_size && execute && destroy; }
};

using OneDnnF16Api=OneDnnBF16Api;

static OneDnnF16Api load_onednn_f16() {
    static OneDnnF16Api api{};static bool attempted=false;static void* handle=nullptr;
    if(attempted)return api;attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so",
        "/work/src/libgrimoire_onednn.so","/bridge/libgrimoire_onednn.so",
        "/opt/grimoire/lib/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue;
        handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(
            dlsym(handle,"grimoire_onednn_f16_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(
            dlsym(handle,"grimoire_onednn_f16_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(
            dlsym(handle,"grimoire_onednn_f16_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(
            dlsym(handle,"grimoire_onednn_f16_destroy"));
        if(api)break;
        api={};dlclose(handle);handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  Muse DFlash: oneDNN FP16 linear unavailable\n");
    return api;
}

static OneDnnBF16Api load_onednn_bf16() {
    static OneDnnBF16Api api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_bf16_f32_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_bf16_f32_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_bf16_f32_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_bf16_f32_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    return api;
}

static OneDnnMXApi load_onednn_mx() {
    static OneDnnMXApi api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so","/opt/grimoire/lib/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  oneDNN MXFP4 W4A16 unavailable\n");
    return api;
}

static OneDnnW4A8Api load_onednn_w4a8() {
    static OneDnnW4A8Api api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so","/opt/grimoire/lib/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_w4a8_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_w4a8_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_w4a8_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_w4a8_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  oneDNN W4A8 unavailable\n");
    return api;
}

static OneDnnS4Api load_onednn_s4() {
    static OneDnnS4Api api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so","/opt/grimoire/lib/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_s4a16_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_s4a16_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_s4a16_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_s4a16_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  oneDNN s4 W4A16 unavailable\n");
    return api;
}

static OneDnnW4Api load_onednn_w4() {
    static OneDnnW4Api api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=open_bridge(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_w4a16_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_w4a16_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_w4a16_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_w4a16_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  raw oneDNN W4A16 unavailable; using XMX fallback\n");
    return api;
}

// Fixed split count for graph capture: the launch geometry must not
// change between tokens, so the attention kernel always runs this many
// chunks and skips the ones past the current sequence end.
// Fixed split count for graph capture. The launch geometry cannot vary
// between replays, so this is a compromise: too high wastes work at
// short context, too low leaves the machine idle at long context. 8 is
// the crossover for 32 heads on 256 EUs.
constexpr int GRAPH_SPLITS = 8;

// forward decls from ops.cpp / other kernels
sycl::event launch_rmsnorm_residual(sycl::queue&, float*, const float*, const bf16_t*,
                                    float*, int, float, const std::vector<sycl::event>&);
sycl::event launch_rmsnorm_residual2(sycl::queue&, float*, const float*, const float*,
                                     const bf16_t*, float*, int, float,
                                     const std::vector<sycl::event>&);
sycl::event launch_rope(sycl::queue&, float*, int, int, int, float, float,
                        const std::vector<sycl::event>&);
sycl::event launch_swiglu(sycl::queue&, const float*, const float*, float*, int,
                          const std::vector<sycl::event>&);
sycl::event launch_l2norm_heads(sycl::queue&, float*, int, int,
                                const std::vector<sycl::event>&);
sycl::event launch_argmax(sycl::queue&, const float*, int, int32_t*, float*,
                          const std::vector<sycl::event>&);
sycl::event launch_embed(sycl::queue&, const bf16_t*, int, float*, int,
                         const std::vector<sycl::event>&);
sycl::event launch_deltanet_gates(sycl::queue&, const float*, const float*,
                                  const bf16_t*, const bf16_t*, float*, float*,
                                  int, const std::vector<sycl::event>&);
sycl::event launch_gate_silu(sycl::queue&, const float*, const float*, float*, int,
                             const std::vector<sycl::event>&);
sycl::event launch_rmsnorm_gate_silu(sycl::queue&, float*, const float*, const bf16_t*,
                                     int, int, float, const std::vector<sycl::event>&);
sycl::event launch_qk_norm_rope(sycl::queue&, float*, float*, const bf16_t*,
                                const bf16_t*, int, int, int, const int32_t*,
                                float, float, float, const std::vector<sycl::event>&);
// ONE OWNER for "can this device actually RUN a joint_matrix kernel".
//
// The aspect alone is not proof.  On 2026-09-16 this container was
// rescheduled from a Xeon @ 2.80GHz to a Xeon @ 2.10GHz whose OpenCL CPU
// runtime ADVERTISES ext_intel_matrix -- so every predicate that trusted
// the aspect flipped, batched prefill turned itself on for the first time
// on a CPU device, and the joint_matrix kernels reached a JIT that cannot
// compile them.  Four gates went red with no code change behind them.
//
// The aspect answers "does this device CLAIM matrix support".  What every
// caller actually needs is "can it run the kernel", and a CPU that claims
// it and then crashes its own JIT answers yes to one and no to the other.
//
// A GPU is trusted outright: a B70 IS a GPU, and the original reason this
// predicate leads with is_gpu() stands -- a driver that under-reports must
// not cost the card its batched prefill.  This only refuses the converse,
// a non-GPU that over-reports.  GRIMOIRE_TRUST_MATRIX_ASPECT=1 restores
// the old behaviour for a CPU device that genuinely can.
//
// Three callers used to compute this separately -- Grimoire::prefill's
// guard, its NOXMX escape hatch, and the capability banner -- and when the
// host changed they disagreed: the banner printed "batched" while the
// engine had fallen back, which is precisely the drift the banner exists
// to expose.  Keep it in one place.
static bool device_can_matrix(const sycl::queue& q) {
    static const bool trust =
        std::getenv("GRIMOIRE_TRUST_MATRIX_ASPECT") != nullptr;
    const auto& d = q.get_device();
    if (d.is_gpu()) return true;
    return trust && d.has(sycl::aspect::ext_intel_matrix);
}

sycl::event launch_rmsnorm_heads(sycl::queue&, float*, const bf16_t*, int, int,
                                 float, bool, const std::vector<sycl::event>&);
sycl::event launch_scale(sycl::queue&, float*, float, int,
                         const std::vector<sycl::event>&);
sycl::event launch_logit_softcap(sycl::queue&, float*, float, int,
                                 const std::vector<sycl::event>&);
sycl::event launch_add(sycl::queue&, float*, const float*, int,
                       const std::vector<sycl::event>&);
sycl::event launch_add_f16_round(sycl::queue&, float*, const float*, int,
                                 const std::vector<sycl::event>&);
sycl::event launch_kv_append(sycl::queue&, const float*, const float*, float*,
                             float*, int, int, int, int,
                             const std::vector<sycl::event>&);
sycl::event launch_router_topk(sycl::queue&, const float*, int, int, int32_t*,
                               float*, bool, const std::vector<sycl::event>&);
sycl::event launch_scale_by_sigmoid(sycl::queue&, float*, const float*, int,
                                    const std::vector<sycl::event>&);
sycl::event launch_rope_dev(sycl::queue&, float*, int, int, const int32_t*,
                            float, float, const std::vector<sycl::event>&);
// gemma-4 full-attention layers.  NOT interchangeable with launch_rope_dev's
// partial_rope: different frequencies and a different dimension pairing.
sycl::event launch_rope_proportional(sycl::queue&, float*, int, int,
                                     const int32_t*, float, float,
                                     const std::vector<sycl::event>&, float);
sycl::event launch_geglu(sycl::queue&, const float*, const float*, float*, int,
                         const std::vector<sycl::event>&);
sycl::event launch_geglu_batched(sycl::queue&, const float*, float*, int, int,
                                 const std::vector<sycl::event>&);
sycl::event launch_kv_append_dev(sycl::queue&, const float*, const float*,
                                 uint8_t*, uint8_t*, const int32_t*, int, int, int,
                                 const std::vector<sycl::event>&);
sycl::event launch_incr_pos(sycl::queue&, int32_t*,
                            const std::vector<sycl::event>&);
sycl::event launch_incr_pos2(sycl::queue&, int32_t*, int32_t*,
                             const std::vector<sycl::event>&);
sycl::event launch_split_qgate(sycl::queue&, const float*, float*, float*,
                               int, int, const std::vector<sycl::event>&);
sycl::event launch_gate_sigmoid_mul(sycl::queue&, float*, const float*, int,
                                    const std::vector<sycl::event>&);
sycl::event launch_causal_conv1d_l2norm(sycl::queue&, const ConvParams&, int, int,
                                        const std::vector<sycl::event>&);
sycl::event launch_gemv_int4sym(sycl::queue&, const uint8_t*, const float*,
                                const float*, float*, int, int,
                                const std::vector<sycl::event>&);
sycl::event launch_gemv_int4sym_batch(sycl::queue&, const uint8_t*, const float*,
                                const float*, float*, int, int, int,
                                const std::vector<sycl::event>&);
sycl::event launch_gemv_int4sym_batch_nored(sycl::queue&, const uint8_t*, const float*,
                                const float*, float*, int, int, int,
                                const std::vector<sycl::event>&);
sycl::event launch_mxfp4_to_int4sym(sycl::queue&, const uint8_t*, const uint8_t*,
                                    int64_t, int64_t, uint8_t*, float*, int, int,
                                    const std::vector<sycl::event>&);
sycl::event launch_f16_to_int4sym(sycl::queue&, const sycl::half*, uint8_t*,
                                  float*, int, int,
                                  const std::vector<sycl::event>&);
sycl::event launch_quantize_rows_int8(sycl::queue&, const float*, int8_t*,
                                    float*, int, int,
                                    const std::vector<sycl::event>&);
sycl::event launch_quantize_rows_int8_bf16(sycl::queue&, const sycl_bf16*, int8_t*,
                                    float*, int, int,
                                    const std::vector<sycl::event>&);
sycl::event launch_probe(sycl::queue&, const float*, int, float*,
                         const std::vector<sycl::event>&);


// Counts NVFP4 tensors actually DECODED; defined beside the other gate
// counters further down.  It is declared HERE, before the anonymous
// namespace opens, because a declaration inside that namespace names a
// DIFFERENT, internal-linkage symbol -- which compiles, and then fails
// to link against the definition the gate reads.
extern long g_nvfp4_tensors;
extern long g_prefix_tokens_reused;
extern long g_prefix_tokens_reused_calls;
extern long g_prefix_bytes_copied;
extern long g_batch_decode_steps;
extern long g_batch_decode_rows;

// ---------------------------------------------------------------------
// Upload helpers
// ---------------------------------------------------------------------
namespace {

// Straight copy: tensor is already in the layout the kernel reads.
// Validated raw upload. Reports the tensor by NAME on any problem --
// a null TensorRef, a zero-length span, or a byte count that is not a
// whole number of elements. Previously a bad ref produced a segfault
// with no indication of which of ~600 tensors was at fault.
template <typename T>
T* dev_copy_t(sycl::queue& q, const Qwen35Model& ck, const TensorRef& r,
              const char* what, bool* ok) {
    if (!r.ok()) {
        std::printf("\n  MISSING tensor: %s\n", what);
        std::fflush(stdout);
        *ok = false;
        return nullptr;
    }
    const size_t src_bytes = size_t(ck.bytes(r));
    const size_t count = std::is_same_v<T, bf16_t> ? size_t(r.t.numel())
                                                    : src_bytes / sizeof(T);
    const size_t bytes = count * sizeof(T);
    if (src_bytes == 0 || (!std::is_same_v<T, bf16_t> && (src_bytes % sizeof(T)) != 0)) {
        std::printf("\n  BAD tensor %s: %zu bytes, element size %zu\n",
                    what, bytes, sizeof(T));
        std::fflush(stdout);
        *ok = false;
        return nullptr;
    }
    // Print the resolved location BEFORE touching it. If this is the
    // last line before a fault, the numbers say which shard and offset.
    if (std::getenv("GRIMOIRE_VERBOSE")) {
        std::printf("[shard %d/%zu off %llu len %zu] ", r.shard, ck.shard_count(),
                    (unsigned long long)r.t.begin, bytes);
        std::fflush(stdout);
    }
    T* d = sycl::malloc_device<T>(bytes / sizeof(T), q);
    if (!d) {
        std::printf("\n  device alloc FAILED for %s (%.2f MiB)\n",
                    what, double(bytes) / 1048576.0);
        std::fflush(stdout);
        *ok = false;
        return nullptr;
    }

    // STAGE THROUGH HOST MEMORY. The source is a file-backed mmap, and
    // Level Zero cannot fault in pages it does not own -- copying
    // straight from the mapping segfaults on any page the CPU has not
    // already touched. embed_tokens happened to survive because
    // MADV_SEQUENTIAL had paged it in; the first tensor in an untouched
    // shard did not. A plain memcpy forces residency first.
    // Read via pread instead of dereferencing the mapping. Whatever is
    // wrong with the mmap in this process, a file read cannot fault.
    std::vector<uint8_t> stage;
    try { stage.resize(bytes); }
    catch (const std::bad_alloc&) {
        std::printf("\n  host staging alloc of %.1f MiB failed for %s\n",
                    double(bytes) / 1048576.0, what);
        std::fflush(stdout);
        *ok = false; return nullptr;
    }
    std::string rerr;
    if constexpr (std::is_same_v<T, bf16_t>) {
        if (r.t.dtype == STDtype::BF16) {
            if (!ck.read_raw(r, stage.data(), rerr)) {
                std::printf("\n  read failed for %s: %s\n", what, rerr.c_str());
                *ok = false; return nullptr;
            }
        } else {
            std::vector<float> f32(count);
            if (!(r.native ? ck.read_native_f32(r,f32.data(),rerr) :
                  ck.shards[r.shard]->read_f32(r.t, f32.data(), rerr))) {
                std::printf("\n  conversion failed for %s: %s\n", what, rerr.c_str());
                *ok = false; return nullptr;
            }
            bf16_t* out = reinterpret_cast<bf16_t*>(stage.data());
            for (size_t i = 0; i < count; ++i) out[i] = f32_to_bf16(f32[i]);
        }
    } else if (!ck.read_raw(r, stage.data(), rerr)) {
        std::printf("\n  read failed for %s: %s\n", what, rerr.c_str());
        std::fflush(stdout);
        *ok = false;
        return nullptr;
    }
    // Stage through PINNED USM host memory for the device copy.  A plain
    // std::vector is pageable, and over a USB4/Thunderbolt link the copy
    // engine cannot DMA pageable host memory to the attached card -- it throws
    // OUT_OF_DEVICE_MEMORY (error 39) at the device boundary.  malloc_host is
    // DMA-able from either card.  Chunk it so the pinned buffer stays modest.
    {
        const size_t CH = size_t(64) << 20;   // 64 MB pinned window
        uint8_t* pin = sycl::malloc_host<uint8_t>(std::min(bytes, CH), q);
        if (pin) {
            for (size_t off = 0; off < bytes; off += CH) {
                const size_t n = std::min(CH, bytes - off);
                std::memcpy(pin, stage.data() + off, n);
                q.memcpy(reinterpret_cast<uint8_t*>(d) + off, pin, n).wait();
            }
            sycl::free(pin, q);
        } else {
            q.memcpy(d, stage.data(), bytes).wait();
        }
    }
    return d;
}

template <typename T>
T* dev_copy(sycl::queue& q, const void* src, size_t bytes) {
    if (!src || !bytes) return nullptr;
    T* d = nullptr;
    try { d = sycl::malloc_device<T>(bytes / sizeof(T), q); }
    catch (sycl::exception& e) {
        if(std::getenv("GRIMOIRE_PIPE_DIAG")){std::printf("[dc] malloc_device %.1f MiB THREW: %s\n", double(bytes)/1048576.0, e.what());std::fflush(stdout);}
        throw;
    }
    if (!d) {
        std::printf("\n  FATAL: device allocation of %.1f MiB failed\n",
                    double(bytes) / 1048576.0);
        std::fflush(stdout);
        return nullptr;
    }
    // Pinned USM host staging + chunking -- pageable host memory cannot be
    // DMA'd to a USB4/Thunderbolt-attached card (error 39 at the boundary).
    static const bool PDIAG = std::getenv("GRIMOIRE_PIPE_DIAG") != nullptr;
    if (PDIAG) { std::printf("[dc] malloc_device %.1f MiB OK; ", double(bytes)/1048576.0); std::fflush(stdout); }
    const size_t CH = size_t(64) << 20;
    uint8_t* pin = nullptr;
    try { pin = sycl::malloc_host<uint8_t>(std::min(bytes, CH), q); }
    catch (sycl::exception& e) { if(PDIAG){std::printf("malloc_host THREW: %s\n", e.what());std::fflush(stdout);} throw; }
    if (PDIAG) { std::printf("malloc_host OK; "); std::fflush(stdout); }
    if (pin) {
        const uint8_t* s8 = static_cast<const uint8_t*>(src);
        for (size_t off = 0; off < bytes; off += CH) {
            const size_t n = std::min(CH, bytes - off);
            std::memcpy(pin, s8 + off, n);
            try { q.memcpy(reinterpret_cast<uint8_t*>(d) + off, pin, n).wait(); }
            catch (sycl::exception& e) { if(PDIAG){std::printf("memcpy THREW: %s\n", e.what());std::fflush(stdout);} throw; }
        }
        if (PDIAG) { std::printf("memcpy OK\n"); std::fflush(stdout); }
        sycl::free(pin, q);
    } else {
        q.memcpy(d, src, bytes).wait();
    }
    return d;
}

// Quantize a bf16 [N][K] tensor to `fmt` on the host, then upload.
// This is where the 3 GB of unquantized weights get dealt with.
struct DevQuant {
    QuantWeight w;
    // TP stores only [row_begin,row_begin+w.N) of a logical full_N-row
    // matrix.  Non-TP weights keep full_N==0.
    int full_N = 0;
    int row_begin = 0;
    uint8_t* payload = nullptr;
    void*    scales  = nullptr;
    uint8_t* zeros   = nullptr;
    void*    od_scales = nullptr;
    void*    od_scales_fp16 = nullptr;
    bool     od_w4 = false;
    sycl::half* fp16 = nullptr;
    // Symmetric int4 g128 companion.  Same 4.25 bits/weight as MXFP4 g32, so
    // when it exists the MXFP4 payload is FREED, not kept alongside -- the
    // duplicate copies cost 8.5 GB and the context room with them.
    uint8_t* i4  = nullptr;
    float*   i4s = nullptr;
    bool has_i4() const { return i4 && i4s; }
    int output_rows() const { return full_N ? full_N : w.N; }
    bool tp_sharded() const { return full_N > 0; }
    void release(sycl::queue& q) {
        if (i4)  sycl::free(i4, q);
        if (i4s) sycl::free(i4s, q);
        if (payload) sycl::free(payload, q);
        if (scales)  sycl::free(scales, q);
        if (zeros)   sycl::free(zeros, q);
        if (od_scales) sycl::free(od_scales, q);
        if (od_scales_fp16) sycl::free(od_scales_fp16, q);
        if (fp16) sycl::free(fp16, q);
    }
};

static size_t scale_value_bytes(Fmt f) {
    if(f==Fmt::INT4)return sizeof(bf16_t);
    if(f==Fmt::FP8_E4M3||f==Fmt::FP8_E5M2||f==Fmt::INT8)return sizeof(float);
    return 1;
}

// Derive [N][K] from the tensor's own shape. Guessing dimensions from
// config is how you read past the end of an mmap: this model sets
// attn_output_gate, so q_proj carries query AND gate rows and is twice
// the size config arithmetic predicts. Reading the shape removes the
// whole class of error.
// Concatenate two [N][K] tensors row-wise into one [2N][K] and quantize.
// Fusing at load costs nothing at runtime and removes a launch per use.
DevQuant concat_upload_t(sycl::queue& q, const Qwen35Model& ck,
                         const TensorRef& ra, const TensorRef& rb,
                         Fmt fmt, const char* what, bool* ok);

bool read_compressed_int4_ref(const Qwen35Model& ck, const TensorRef& r,
                              PackedWeight& p, std::string& err);

// One float out of a 1-element tensor.  A global quantisation scale is
// stored as a tensor like any other, and reading it with the matrix
// reader would work but says nothing about the expectation that it is
// scalar -- a per-row scale silently read as "the first row's" is the
// kind of thing that produces a plausible model.
bool read_scalar_f32(const Qwen35Model& ck, const TensorRef& r,
                     float& out, std::string& err);


bool read_matrix_f32(const Qwen35Model& ck, const TensorRef& r,
                     float* dst, std::string& err) {
    if (r.native) return ck.read_native_f32(r,dst,err);
    // compressed-tensors INT4 must be decoded, not read.  The loader
    // rewrites this tensor's LOGICAL shape to [N][K] while the payload
    // still holds K/8 int32 words per row, so falling through to the
    // plain branch below asks read_f32 for N*K values from a buffer with
    // N*K/8 -- an overread past the end of the mapping, with the group
    // scales ignored on top.  Every earlier caller happened to hold a
    // format this never reached; the FFN fold reaches it.
    if (r.compressed_int4) {
        PackedWeight p;
        if (!read_compressed_int4_ref(ck, r, p, err)) {
            if (err.empty()) err = "compressed INT4 tensor could not be decoded";
            return false;
        }
        const QuantWeight w = p.view();
        // at() is the reference dequant the GPU tiles must agree with bit
        // for bit, so decoding through it cannot drift from the kernels.
        for (int n = 0; n < w.N; ++n)
            for (int k = 0; k < w.K; ++k)
                dst[int64_t(n) * w.K + k] = w.at(n, k);
        return true;
    }
    // NVFP4 must be DECODED, like compressed INT4 above and for the same
    // reason: the logical shape says [N][K] while the payload holds K/2
    // bytes per row, so the plain read below would overread the mapping
    // by 2x and ignore both scales.  Decoding here is also what keeps
    // the merged-linear trap closed -- each TensorRef carries its OWN
    // global scale, so gate and up are dequantized with theirs before
    // concat_upload_t joins them.  b70/nvfp4.hpp has the derivation.
    if (r.nvfp4) {
        ++g_nvfp4_tensors;
        const int N = int(r.t.shape[0]), K = int(r.t.shape[1]);
        std::vector<uint8_t> pk(size_t(N) * K / 2);
        std::vector<uint8_t> sc(size_t(N) * K / kNVFP4Block);
        TensorRef scr; scr.shard = r.scales_shard; scr.t = r.scales_t;
        TensorRef gsr; gsr.shard = r.gscale_shard; gsr.t = r.gscale_t;
        float g = 1.0f;
        if (!ck.read_raw(r, pk.data(), err) ||
            !ck.read_raw(scr, sc.data(), err)) return false;
        if (!read_scalar_f32(ck, gsr, g, err)) {
            err = "NVFP4 weight_global_scale could not be read: " + err;
            return false;
        }
        // The dequant formula DIVIDES by this (see nvfp4.hpp).  A stored
        // zero used to silently zero the weight under the old multiply
        // convention; under divide it would produce inf/nan that
        // propagates through the whole model instead.  Refuse rather
        // than load a checkpoint that cannot mean this.
        if (g == 0.0f) {
            err = "NVFP4 weight_global_scale is zero, which cannot be a "
                  "valid quantization scale";
            return false;
        }
        nvfp4_dequant(pk.data(), sc.data(), g, N, K, dst);
        return true;
    }
    if (!r.gptq) {
        if (!ck.shards[r.shard]->read_f32(r.t, dst, err)) return false;
        if (r.row_scaled) {
            const int N = int(r.t.shape[0]), K = int(r.t.shape[1]);
            std::vector<float> scale(N);
            if (!ck.shards[r.scales_shard]->read_f32(r.scales_t, scale.data(), err))
                return false;
            for (int n = 0; n < N; ++n)
                for (int k = 0; k < K; ++k)
                    dst[int64_t(n) * K + k] *= scale[n];
        }
        return true;
    }
    std::vector<uint8_t> qw(size_t(r.t.end - r.t.begin));
    std::vector<uint8_t> qz(size_t(r.qzeros_t.end - r.qzeros_t.begin));
    std::vector<uint8_t> sc(size_t(r.scales_t.end - r.scales_t.begin));
    if (!ck.shards[r.shard]->read_raw(r.t, qw.data(), err) ||
        !ck.shards[r.qzeros_shard]->read_raw(r.qzeros_t, qz.data(), err) ||
        !ck.shards[r.scales_shard]->read_raw(r.scales_t, sc.data(), err))
        return false;
    GptqTensor g;
    g.qweight = reinterpret_cast<const int32_t*>(qw.data());
    g.qzeros  = reinterpret_cast<const int32_t*>(qz.data());
    g.scales  = reinterpret_cast<const uint16_t*>(sc.data());
    g.out = int(r.t.shape[0]); g.in = int(r.t.shape[1]); g.group = r.gptq_group;
    if (!g.ok()) { err = "invalid GPTQ/AutoRound tensor geometry"; return false; }
    gptq_dequant_4bit(g, dst);
    return true;
}

bool read_scalar_f32(const Qwen35Model& ck, const TensorRef& r,
                     float& out, std::string& err) {
    if (!r.ok()) { err = "scale tensor is absent"; return false; }
    int64_t n = 1;
    for (int64_t d : r.t.shape) n *= d;
    if (n != 1) {
        err = "expected a single global scale, got a tensor with " +
              std::to_string(n) + " elements";
        return false;
    }
    return read_matrix_f32(ck, r, &out, err);
}


DevQuant concat_upload_many_bf16_t(sycl::queue& q, const Qwen35Model& ck,
                                    const std::vector<TensorRef>& refs,
                                    const char* what, bool* ok) {
    DevQuant d;
    if(refs.empty()){*ok=false;return d;}
    int K=-1,N=0;
    for(const auto& r:refs){
        if(!r.ok()||r.t.shape.size()!=2||(K>=0&&int(r.t.shape[1])!=K)){
            std::printf("\n  cannot concatenate %s (shape mismatch)\n",what);
            *ok=false;return d;
        }
        K=int(r.t.shape[1]);N+=int(r.t.shape[0]);
    }
    std::vector<float> f32(size_t(N)*K);
    size_t off=0;
    std::string err;
    for(const auto& r:refs){
        const size_t count=size_t(r.t.shape[0])*K;
        if(!read_matrix_f32(ck,r,f32.data()+off,err)){
            std::printf("\n  read failed for %s: %s\n",what,err.c_str());
            *ok=false;return d;
        }
        off+=count;
    }
    PackedWeight p=quantize(f32.data(),N,K,Fmt::BF16);
    d.payload=dev_copy<uint8_t>(q,p.payload.data(),p.payload.size());
    d.w=p.view();d.w.payload=d.payload;
    if(!d.payload)*ok=false;
    return d;
}

DevQuant upload_f16_t(sycl::queue& q,const Qwen35Model& ck,const TensorRef& r,
                      const char* what,bool* ok){
    DevQuant d;
    if(!r.ok()||r.t.shape.size()!=2){
        std::printf("\n  invalid FP16 tensor: %s\n",what);*ok=false;return d;
    }
    const int N=int(r.t.shape[0]),K=int(r.t.shape[1]);
    std::vector<float> f32(size_t(N)*K);
    std::string err;
    if(!read_matrix_f32(ck,r,f32.data(),err)){
        std::printf("\n  read failed for %s: %s\n",what,err.c_str());
        *ok=false;return d;
    }
    std::vector<sycl::half> h(f32.size());
    for(size_t i=0;i<f32.size();++i)h[i]=sycl::half(f32[i]);
    d.fp16=dev_copy<sycl::half>(q,h.data(),h.size()*sizeof(sycl::half));
    d.w=QuantWeight{Fmt::BF16,N,K,nullptr,nullptr,nullptr,
                    int64_t(K*sizeof(sycl::half)),0};
    if(!d.fp16)*ok=false;
    return d;
}

sycl::half* upload_f16_vector_t(sycl::queue& q,const Qwen35Model& ck,
                                const TensorRef& r,const char* what,bool* ok){
    if(!r.ok()){
        std::printf("\n  invalid FP16 tensor: %s\n",what);*ok=false;return nullptr;
    }
    const size_t count=size_t(r.t.numel());
    std::vector<float> f32(count);
    std::string err;
    if(!read_matrix_f32(ck,r,f32.data(),err)){
        std::printf("\n  conversion failed for %s: %s\n",what,err.c_str());
        *ok=false;return nullptr;
    }
    std::vector<sycl::half> h(count);
    for(size_t i=0;i<count;++i)h[i]=sycl::half(f32[i]);
    sycl::half* d=dev_copy<sycl::half>(q,h.data(),h.size()*sizeof(sycl::half));
    if(!d)*ok=false;
    return d;
}

DevQuant concat_upload_many_f16_t(sycl::queue& q,const Qwen35Model& ck,
                                  const std::vector<TensorRef>& refs,
                                  const char* what,bool* ok){
    DevQuant d;
    if(refs.empty()){*ok=false;return d;}
    int K=-1,N=0;
    for(const auto&r:refs){
        if(!r.ok()||r.t.shape.size()!=2||(K>=0&&int(r.t.shape[1])!=K)){
            std::printf("\n  cannot concatenate %s (shape mismatch)\n",what);
            *ok=false;return d;
        }
        K=int(r.t.shape[1]);N+=int(r.t.shape[0]);
    }
    std::vector<float> f32(size_t(N)*K);size_t off=0;std::string err;
    for(const auto&r:refs){
        const size_t count=size_t(r.t.shape[0])*K;
        if(!read_matrix_f32(ck,r,f32.data()+off,err)){
            std::printf("\n  read failed for %s: %s\n",what,err.c_str());
            *ok=false;return d;
        }
        off+=count;
    }
    std::vector<sycl::half> h(f32.size());
    for(size_t i=0;i<f32.size();++i)h[i]=sycl::half(f32[i]);
    d.fp16=dev_copy<sycl::half>(q,h.data(),h.size()*sizeof(sycl::half));
    d.w=QuantWeight{Fmt::BF16,N,K,nullptr,nullptr,nullptr,
                    int64_t(K*sizeof(sycl::half)),0};
    if(!d.fp16)*ok=false;
    return d;
}

DevQuant concat4_native_mxfp4_t(sycl::queue& q,const Qwen35Model& ck,
 const TensorRef&a,const TensorRef&b,const TensorRef&c,const TensorRef&e,
 const char*what,bool*ok){
 DevQuant d;const TensorRef*r[4]={&a,&b,&c,&e};
 if(!a.ok()||!b.ok()||!c.ok()||!e.ok()||a.t.shape.size()!=2||b.t.shape.size()!=2||
    c.t.shape.size()!=2||e.t.shape.size()!=2||a.t.shape[1]!=b.t.shape[1]||
    a.t.shape[1]!=c.t.shape[1]||a.t.shape[1]!=e.t.shape[1]){
  std::printf("\n  cannot concatenate %s (shape mismatch)\n",what);*ok=false;return d;}
 const int K=int(a.t.shape[1]);int N=0;size_t pb=0,sb=0;
 for(auto*x:r){if(!x->native||x->native->encoding!=uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2)){
   std::printf("\n  %s requires native MXFP4 inputs\n",what);*ok=false;return d;}
  int n=int(x->t.shape[0]);N+=n;pb+=size_t(n)*K/2;sb+=size_t(n)*K/kMXBlock;}
 std::vector<uint8_t>hp(pb),hs(sb);size_t po=0,so=0;
 for(auto*x:r){int n=int(x->t.shape[0]);size_t np=size_t(n)*K/2,ns=size_t(n)*K/kMXBlock;
  std::memcpy(hp.data()+po,static_cast<const uint8_t*>(ck.native_model->payload(*x->native))+x->native_payload_offset,np);
  std::memcpy(hs.data()+so,static_cast<const uint8_t*>(ck.native_model->scales(*x->native))+x->native_scale_offset,ns);
  po+=np;so+=ns;}
 d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());d.scales=dev_copy<uint8_t>(q,hs.data(),hs.size());
 d.w=QuantWeight{Fmt::MXFP4,N,K,d.payload,d.scales,nullptr,int64_t(K/2),K/kMXBlock};
 if(!d.payload||!d.scales)*ok=false;return d;
}

bool repack_gptq_ref(const Qwen35Model& ck, const TensorRef& r,
                     PackedWeight& p, std::string& err) {
    if (!r.gptq) return false;
    std::vector<uint8_t> qw(size_t(r.t.end - r.t.begin));
    std::vector<uint8_t> qz(size_t(r.qzeros_t.end - r.qzeros_t.begin));
    std::vector<uint8_t> sc(size_t(r.scales_t.end - r.scales_t.begin));
    if (!ck.shards[r.shard]->read_raw(r.t, qw.data(), err) ||
        !ck.shards[r.qzeros_shard]->read_raw(r.qzeros_t, qz.data(), err) ||
        !ck.shards[r.scales_shard]->read_raw(r.scales_t, sc.data(), err))
        return false;
    GptqTensor g{reinterpret_cast<const int32_t*>(qw.data()),
                  reinterpret_cast<const int32_t*>(qz.data()),
                  reinterpret_cast<const uint16_t*>(sc.data()),
                  int(r.t.shape[1]), int(r.t.shape[0]), r.gptq_group};
    p = gptq_repack_int4(g);
    if (p.payload.empty()) { err = "GPTQ direct repack failed"; return false; }
    return true;
}

bool read_compressed_int4_ref(const Qwen35Model& ck, const TensorRef& r,
                              PackedWeight& p, std::string& err) {
    if (!r.compressed_int4 || r.t.shape.size() != 2 || r.gptq_group <= 0)
        return false;
    const int N = int(r.t.shape[0]);
    const int K = int(r.t.shape[1]);

    if (K % r.gptq_group) { err = "compressed INT4 group mismatch"; return false; }
    const int groups = K / r.gptq_group;
    const size_t payload_bytes = size_t(N) * K / 2;
    const size_t scale_bytes = size_t(N) * groups * sizeof(bf16_t);
    if (size_t(r.t.end-r.t.begin) != payload_bytes ||
        size_t(r.scales_t.end-r.scales_t.begin) != scale_bytes) {
        err = "compressed INT4 physical size mismatch";
        return false;
    }
    p.fmt = Fmt::INT4;
    p.N = N; p.K = K;
    p.row_bytes = K / 2;
    p.row_scales = groups;
    p.payload.resize(payload_bytes);
    p.scales_raw.resize(scale_bytes);
    p.zeros.assign(size_t(N) * groups, uint8_t(8));
    TensorRef sr; sr.shard = r.scales_shard; sr.t = r.scales_t;
    if (!ck.read_raw(r,p.payload.data(),err) ||
        !ck.read_raw(sr,p.scales_raw.data(),err)) return false;
    return true;
}

static DevQuant upload_packed(sycl::queue& q, const PackedWeight& p);

DevQuant quantize_upload_t(sycl::queue& q, const Qwen35Model& ck,
                           const TensorRef& r, Fmt fmt, const char* what,
                           bool* ok) {
    DevQuant d;
    if (!r.ok()) { std::printf("\n  missing tensor: %s\n", what); *ok = false; return d; }
    if (r.t.shape.size() != 2) {
        std::printf("\n  %s: expected 2-D, got %zu dims\n", what, r.t.shape.size());
        *ok = false; return d;
    }
    const int N = int(r.t.shape[0]);
    const int K = int(r.t.shape[1]);
    const bool direct_fp8=r.row_scaled&&r.scales_t.numel()==N&&
        ((fmt==Fmt::FP8_E4M3&&r.t.dtype==STDtype::F8_E4M3)||
         (fmt==Fmt::FP8_E5M2&&r.t.dtype==STDtype::F8_E5M2));
    if(direct_fp8){
        std::vector<uint8_t> hp(size_t(N)*K);
        std::vector<float> hs(size_t(N),0.0f);std::string rr;
        if(!ck.read_raw(r,hp.data(),rr)||
           !ck.shards[r.scales_shard]->read_f32(r.scales_t,hs.data(),rr)){
            std::printf("\n  direct FP8 read failed for %s: %s\n",what,rr.c_str());*ok=false;return d;}
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.scales=dev_copy<float>(q,hs.data(),hs.size()*sizeof(float));
        d.w=QuantWeight{fmt,N,K,d.payload,d.scales,nullptr,int64_t(K),1};
        if(!d.payload||!d.scales)*ok=false;return d;
    }
    // !r.nvfp4: this tensor's LOGICAL shape is [N][K] while its payload
    // is K/2 bytes per row, so a straight BF16 read would ask for 4x the
    // bytes that exist.  The dtype test alone should already exclude it;
    // one more term costs nothing and removes the whole class.
    if(fmt==Fmt::BF16&&!r.row_scaled&&!r.native&&!r.nvfp4&&r.t.dtype==STDtype::BF16){
        std::vector<uint8_t> hp(size_t(N)*K*sizeof(bf16_t));std::string rr;
        if(!ck.read_raw(r,hp.data(),rr)){std::printf("\n  direct BF16 read failed for %s: %s\n",what,rr.c_str());*ok=false;return d;}
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.w=QuantWeight{Fmt::BF16,N,K,d.payload,nullptr,nullptr,int64_t(K*2),0};
        if(!d.payload)*ok=false;return d;
    }

    // compressed-tensors weight_packed -> direct MXFP4 upload (no re-quant).
    // r carries the weight_scale in scales_shard/scales_t; both are already
    // in GRIMOIRE's MXFP4 layout (E2M1 payload + E8M0 group-32 scales).
    // !r.nvfp4 is load-bearing.  An NVFP4 tensor is ALSO called
    // weight_packed and ALSO holds E2M1 nibbles; copying it straight to
    // VRAM would hand the kernel E4M3 per-16 scales to read as E8M0
    // per-32, and under-read the scale buffer by half on top.  The
    // result is a model that loads, runs and is wrong.  b70/nvfp4.hpp.
    if (r.t.name.find("weight_packed") != std::string::npos && !r.native &&
        !r.nvfp4 && (fmt == Fmt::MXFP4)) {
        DevQuant d;
        const size_t pb = size_t(N) * K / 2;
        const size_t sb = size_t(N) * (K / kMXBlock);
        std::vector<uint8_t> hp(pb), hs(sb);
        std::string rr;
        TensorRef scr; scr.shard = r.scales_shard; scr.t = r.scales_t;
        if (!ck.read_raw(r, hp.data(), rr) || !ck.read_raw(scr, hs.data(), rr)) {
            std::printf("\n  packed MXFP4 read failed for %s: %s\n", what, rr.c_str());
            *ok = false; return d;
        }
        d.payload = dev_copy<uint8_t>(q, hp.data(), hp.size());
        d.scales  = dev_copy<uint8_t>(q, hs.data(), hs.size());
        d.w = QuantWeight{Fmt::MXFP4, N, K, d.payload, d.scales, nullptr,
                          int64_t(K / 2), K / kMXBlock};
        if (!d.payload || !d.scales) *ok = false;
        return d;
    }

    // INT4 groups are 128 wide and MX blocks 32; a row that is not a
    // whole number of groups would silently mis-scale its tail.
    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    Fmt use = fmt;
    if (blk > 1 && (K % blk) != 0) {
        std::printf("\n  %s: K=%d not a multiple of %d, keeping bf16\n", what, K, blk);
        use = Fmt::BF16;
    }

    std::string rerr;
    PackedWeight p;
    if (r.native && (r.native->encoding!=uint32_t(NativeEncoding::RAW) ||
                     ck.native_model->header().version>=3)) {
        QuantWeight w;
        if (!ck.native_view(r,w,rerr) || (keep_qwen_bf16(r.t.name) && w.fmt!=Fmt::BF16 &&
                                       ck.native_model->header().version>=3)) {
            std::printf("\n  native upload refused for %s: %s (critical tensors require BF16)\n",what,rerr.c_str());
            *ok=false; return d;
        }
        p=copy_packed(w); // Saved precision is authoritative; no second quantization.
    } else if (r.compressed_int4 && use == Fmt::INT4) {
        if (!read_compressed_int4_ref(ck, r, p, rerr)) {
            std::printf("\n  direct compressed INT4 read failed for %s: %s\n",
                        what, rerr.c_str());
            *ok = false; return d;
        }
    } else if (r.gptq && use == Fmt::INT4) {
        if (!repack_gptq_ref(ck, r, p, rerr)) {
            std::printf("\n  direct GPTQ read failed for %s: %s\n", what, rerr.c_str());
            *ok = false; return d;
        }
    } else {
        std::vector<float> f32(size_t(N) * K);
        if (!read_matrix_f32(ck, r, f32.data(), rerr)) {
        std::printf("\n  read failed for %s: %s\n", what, rerr.c_str());
        std::fflush(stdout);
        *ok = false;
        return d;
        }
        p = quantize(f32.data(), N, K, use);
    }
    return upload_packed(q, p);
}

// Move a host-packed weight onto the device.  Split out of
// quantize_upload_t so the Agnes FFN fold can build its merged matrix in
// f32, quantize it with the ordinary packer, and upload it exactly the
// way every other weight is uploaded -- including the int4 scale
// transpose below, which the W4A8 tiles depend on.
static DevQuant upload_packed(sycl::queue& q, const PackedWeight& p) {
    DevQuant d;
    const int N = p.view().N;
    d.payload = dev_copy<uint8_t>(q, p.payload.data(), p.payload.size());
    if (!p.scales_raw.empty())
        d.scales = dev_copy<uint8_t>(q, p.scales_raw.data(), p.scales_raw.size());
    if (!p.zeros.empty())
        d.zeros = dev_copy<uint8_t>(q, p.zeros.data(), p.zeros.size());
    d.w         = p.view();
    d.w.payload = d.payload;
    d.w.scales  = d.scales;
    d.w.zeros   = d.zeros;
    if (p.fmt == Fmt::INT4 && p.row_scales > 0 && !p.scales_raw.empty() &&
        p.zeros.size() == size_t(N) * p.row_scales &&
        std::all_of(p.zeros.begin(), p.zeros.end(), [](uint8_t z){ return z == 8; })) {
        std::vector<bf16_t> tr(size_t(N) * p.row_scales);
        const auto* src = reinterpret_cast<const bf16_t*>(p.scales_raw.data());
        for (int n=0;n<N;++n) for(int g=0;g<p.row_scales;++g)
            tr[size_t(g)*N+n]=src[size_t(n)*p.row_scales+g];
        d.od_scales=dev_copy<bf16_t>(q,tr.data(),tr.size()*sizeof(bf16_t));
        std::vector<sycl::half> trh(tr.size());
        for(size_t i=0;i<tr.size();++i)trh[i]=sycl::half(bf16_to_f32(tr[i]));
        d.od_scales_fp16=dev_copy<sycl::half>(q,trh.data(),trh.size()*sizeof(sycl::half));
        d.od_w4=d.od_scales!=nullptr;
    }
    return d;
}

// ---------------------------------------------------------------------
//  Agnes parallel_ffn, folded into the main FFN.
//
//  Agnes puts a second, narrow SwiGLU inside every mlp:
//
//      mlp(x) = down(silu(gate(x)) * up(x))
//             + pdown(silu(pgate(x)) * pup(x))
//
//  (mlp.parallel_ffn.{gate,up,down}_proj is a submodule of mlp, sibling
//  to gate/up/down, with no norm of its own -- so it consumes the same
//  post_attention_layernorm output, and both results land in the same
//  residual.)
//
//  Two SwiGLUs summed are ONE SwiGLU over the concatenated intermediate:
//  a matmul over a concatenated contraction dimension IS the sum of the
//  two partial products.  So no kernel changes and no second FFN pass --
//  the whole feature is a load-time concatenation:
//
//      gate' = [gate ; pgate]      rows,  I' = I + pI
//      up'   = [up   ; pup  ]      rows
//      down' = [down | pdown]      COLUMNS -- along K, per row
//
//  gate_up' keeps the [all gate rows][all up rows] order the SwiGLU
//  kernel expects, so it is a four-way row concatenation in the order
//  gate, pgate, up, pup.
//
//  down' is the one that is not an append: for row-major [N][K], row n of
//  the result is row n of down followed by row n of pdown, so the two
//  sources interleave per row.  Appending the bytes would silently
//  produce a different matrix.
//
//  Both are built in f32 and handed to the ordinary quantizer, so every
//  projection format works with no per-format special case.
static DevQuant concat_rows_f32_t(sycl::queue& q, const Qwen35Model& ck,
                                  const std::vector<TensorRef>& refs,
                                  Fmt fmt, const char* what, bool* ok) {
    DevQuant d;
    if (refs.empty()) { *ok = false; return d; }
    const int K = int(refs[0].t.shape.size() == 2 ? refs[0].t.shape[1] : 0);
    int N = 0;
    for (const auto& r : refs) {
        if (!r.ok() || r.t.shape.size() != 2 || int(r.t.shape[1]) != K) {
            std::printf("\n  cannot row-concatenate %s (shape mismatch)\n", what);
            *ok = false; return d;
        }
        N += int(r.t.shape[0]);
    }
    std::vector<float> m(size_t(N) * K);
    size_t at = 0;
    std::string err;
    for (const auto& r : refs) {
        if (!read_matrix_f32(ck, r, m.data() + at, err)) {
            std::printf("\n  %s: %s\n", what, err.c_str()); *ok = false; return d;
        }
        at += size_t(r.t.shape[0]) * K;
    }
    Fmt use = fmt;
    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (blk > 1 && (K % blk) != 0) use = Fmt::BF16;
    return upload_packed(q, quantize(m.data(), N, K, use));
}

// Pack many same-width tensors along N by quantizing each one SEPARATELY
// and appending, so the host never holds more than ONE of them as f32.
//
// concat_rows_f32_t materialises the whole concatenation first, which is
// right for a 4-way FFN fold and wrong for MoVA: the real K2 checkpoint
// has 64 value experts per sparse layer, so that is a 64x jump in
// load-time peak host memory on a box that is being pipelined precisely
// because the model does not fit.  It would show up as an OOM on the
// Tower and nowhere else.
//
// Appending is exact because EVERY format in this engine is row-local:
// BF16 is elementwise, FP8/INT8 scale per output channel, INT4 per group
// within a row, MX per 32-element block within a row.  No format's
// encoding of row n depends on any other row, so quantize-then-append
// equals append-then-quantize, bit for bit.  bin/test_k2_kernels asserts
// that directly rather than leaving it as an argument.
static DevQuant concat_rows_quantized_t(sycl::queue& q, const Qwen35Model& ck,
                                        const std::vector<TensorRef>& refs,
                                        Fmt fmt, const char* what, bool* ok) {
    DevQuant d;
    if (refs.empty()) { *ok = false; return d; }
    const int K = int(refs[0].t.shape.size() == 2 ? refs[0].t.shape[1] : 0);
    int N = 0;
    for (const auto& r : refs) {
        if (!r.ok() || r.t.shape.size() != 2 || int(r.t.shape[1]) != K) {
            std::printf("\n  cannot row-concatenate %s (shape mismatch)\n", what);
            *ok = false; return d;
        }
        N += int(r.t.shape[0]);
    }
    Fmt use = fmt;
    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (blk > 1 && (K % blk) != 0) use = Fmt::BF16;

    PackedWeight out;
    out.fmt = use; out.N = N; out.K = K;
    std::vector<float> m;
    std::string err;
    bool first = true;
    for (const auto& r : refs) {
        const int n = int(r.t.shape[0]);
        m.assign(size_t(n) * K, 0.0f);
        if (!read_matrix_f32(ck, r, m.data(), err)) {
            std::printf("\n  %s: %s\n", what, err.c_str()); *ok = false; return d;
        }
        PackedWeight pw = quantize(m.data(), n, K, use);
        if (first) {
            out.row_bytes = pw.row_bytes; out.row_scales = pw.row_scales;
            first = false;
        } else if (pw.row_bytes != out.row_bytes ||
                   pw.row_scales != out.row_scales) {
            // Same format and same K, so this cannot differ -- if it ever
            // does, every row index past this point is wrong and the
            // values stay finite.  Refuse instead.
            std::printf("\n  %s: inconsistent packed row stride\n", what);
            *ok = false; return d;
        }
        out.payload.insert(out.payload.end(), pw.payload.begin(), pw.payload.end());
        out.scales_raw.insert(out.scales_raw.end(),
                              pw.scales_raw.begin(), pw.scales_raw.end());
        out.zeros.insert(out.zeros.end(), pw.zeros.begin(), pw.zeros.end());
    }
    return upload_packed(q, out);
}

// down' = [a | b] along K: row n is a's row n followed by b's row n.
static DevQuant concat_cols_f32_t(sycl::queue& q, const Qwen35Model& ck,
                                  const TensorRef& ra, const TensorRef& rb,
                                  Fmt fmt, const char* what, bool* ok) {
    DevQuant d;
    if (!ra.ok() || !rb.ok() || ra.t.shape.size() != 2 || rb.t.shape.size() != 2 ||
        ra.t.shape[0] != rb.t.shape[0]) {
        std::printf("\n  cannot column-concatenate %s (shape mismatch)\n", what);
        *ok = false; return d;
    }
    const int N  = int(ra.t.shape[0]);
    const int Ka = int(ra.t.shape[1]), Kb = int(rb.t.shape[1]);
    const int K  = Ka + Kb;
    std::vector<float> a(size_t(N) * Ka), b(size_t(N) * Kb);
    std::string err;
    if (!read_matrix_f32(ck, ra, a.data(), err) ||
        !read_matrix_f32(ck, rb, b.data(), err)) {
        std::printf("\n  %s: %s\n", what, err.c_str()); *ok = false; return d;
    }
    std::vector<float> m(size_t(N) * K);
    for (int n = 0; n < N; ++n) {
        std::memcpy(m.data() + size_t(n) * K,      a.data() + size_t(n) * Ka,
                    size_t(Ka) * sizeof(float));
        std::memcpy(m.data() + size_t(n) * K + Ka, b.data() + size_t(n) * Kb,
                    size_t(Kb) * sizeof(float));
    }
    Fmt use = fmt;
    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    if (blk > 1 && (K % blk) != 0) use = Fmt::BF16;
    return upload_packed(q, quantize(m.data(), N, K, use));
}

DevQuant concat_upload_t(sycl::queue& q, const Qwen35Model& ck,
                         const TensorRef& ra, const TensorRef& rb,
                         Fmt fmt, const char* what, bool* ok) {
    DevQuant d;
    if (!ra.ok() || !rb.ok() ||
        ra.t.shape.size() != 2 || rb.t.shape.size() != 2 ||
        ra.t.shape[1] != rb.t.shape[1]) {
        std::printf("\n  cannot concatenate %s (shape mismatch)\n", what);
        std::fflush(stdout);
        *ok = false;
        return d;
    }
    const int Na = int(ra.t.shape[0]);
    const int Nb = int(rb.t.shape[0]);
    const int K  = int(ra.t.shape[1]);
    const int N  = Na + Nb;

    if(fmt==Fmt::BF16&&!ra.row_scaled&&!rb.row_scaled&&!ra.native&&!rb.native&&
       ra.t.dtype==STDtype::BF16&&rb.t.dtype==STDtype::BF16){
        std::vector<uint8_t> hp(size_t(N)*K*sizeof(bf16_t));std::string rr;
        if(!ck.read_raw(ra,hp.data(),rr)||
           !ck.read_raw(rb,hp.data()+size_t(Na)*K*sizeof(bf16_t),rr)){
            std::printf("\n  direct BF16 concatenate failed for %s: %s\n",what,rr.c_str());*ok=false;return d;}
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.w=QuantWeight{Fmt::BF16,N,K,d.payload,nullptr,nullptr,int64_t(K*2),0};
        if(!d.payload)*ok=false;return d;
    }
    const bool direct_fp8=
       ((fmt==Fmt::FP8_E4M3&&ra.t.dtype==STDtype::F8_E4M3&&rb.t.dtype==STDtype::F8_E4M3)||
        (fmt==Fmt::FP8_E5M2&&ra.t.dtype==STDtype::F8_E5M2&&rb.t.dtype==STDtype::F8_E5M2))&&
       ra.row_scaled&&rb.row_scaled&&ra.scales_t.numel()==Na&&rb.scales_t.numel()==Nb;
    if(direct_fp8){
        std::vector<uint8_t> hp(size_t(N)*K);std::vector<float> hs(size_t(N),0.0f);std::string rr;
        if(!ck.read_raw(ra,hp.data(),rr)||!ck.read_raw(rb,hp.data()+size_t(Na)*K,rr)||
           !ck.shards[ra.scales_shard]->read_f32(ra.scales_t,hs.data(),rr)||
           !ck.shards[rb.scales_shard]->read_f32(rb.scales_t,hs.data()+Na,rr)){
            std::printf("\n  direct FP8 concatenate failed for %s: %s\n",what,rr.c_str());*ok=false;return d;}
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.scales=dev_copy<float>(q,hs.data(),hs.size()*sizeof(float));
        d.w=QuantWeight{fmt,N,K,d.payload,d.scales,nullptr,int64_t(K),1};
        if(!d.payload||!d.scales)*ok=false;return d;
    }

    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    Fmt use = fmt;
    if (blk > 1 && (K % blk) != 0) use = Fmt::BF16;

    // compressed-tensors weight_packed (HF safetensors): same MXFP4 layout,
    // read via read_raw and concatenate gate|up along N.
    // !nvfp4 on BOTH sides, and this is the site where getting it wrong
    // costs most.  A merged linear is where NVFP4's per-tensor global
    // scale actually bites: gate and up each carry THEIR OWN, and the
    // reference implementations call out collapsing them (vLLM's stock
    // path takes .max()) as a real accuracy loss.  Falling through to
    // the generic path below dequantizes each side through
    // read_matrix_f32 -- with its own global scale -- before they are
    // concatenated, which is what makes that correct by construction
    // rather than by care.  b70/nvfp4.hpp.
    const bool hf_packed = !ra.native && !rb.native &&
        !ra.nvfp4 && !rb.nvfp4 &&
        ra.t.name.find("weight_packed") != std::string::npos &&
        rb.t.name.find("weight_packed") != std::string::npos && fmt == Fmt::MXFP4;
    if (hf_packed) {
        const size_t pba=size_t(Na)*K/2, pbb=size_t(Nb)*K/2;
        const size_t sba=size_t(Na)*(K/kMXBlock), sbb=size_t(Nb)*(K/kMXBlock);
        std::vector<uint8_t> hp(pba+pbb), hs(sba+sbb);
        std::string rr;
        TensorRef sa; sa.shard=ra.scales_shard; sa.t=ra.scales_t;
        TensorRef sb; sb.shard=rb.scales_shard; sb.t=rb.scales_t;
        if(!ck.read_raw(ra,hp.data(),rr)||!ck.read_raw(rb,hp.data()+pba,rr)||
           !ck.read_raw(sa,hs.data(),rr)||!ck.read_raw(sb,hs.data()+sba,rr)){
            std::printf("\n  packed MXFP4 concat read failed for %s: %s\n", what, rr.c_str());
            *ok=false; return d;
        }
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.scales=dev_copy<uint8_t>(q,hs.data(),hs.size());
        d.w=QuantWeight{Fmt::MXFP4,N,K,d.payload,d.scales,nullptr,
                        int64_t(K/2),K/kMXBlock};
        if(!d.payload||!d.scales)*ok=false;
        return d;
    }

    std::string rerr;
    PackedWeight p;
    if (ra.native && rb.native && (ck.native_model->header().version>=3 ||
        (ra.native->encoding!=0 && rb.native->encoding!=0))) {
        QuantWeight a,b;
        if (!ck.native_view(ra,a,rerr) || !ck.native_view(rb,b,rerr) || a.fmt!=b.fmt ||
            ((keep_qwen_bf16(ra.t.name) || keep_qwen_bf16(rb.t.name)) && a.fmt!=Fmt::BF16 && ck.native_model->header().version>=3)) {
            std::printf("\n  native concat refused for %s: %s (formats must match; critical tensors require BF16)\n",what,rerr.c_str());
            *ok=false; return d;
        }
        p=copy_packed(a); auto tail=copy_packed(b); p.N=N;
        p.payload.insert(p.payload.end(),tail.payload.begin(),tail.payload.end());
        p.scales_raw.insert(p.scales_raw.end(),tail.scales_raw.begin(),tail.scales_raw.end());
        p.zeros.insert(p.zeros.end(),tail.zeros.begin(),tail.zeros.end());
    } else if (ra.compressed_int4 && rb.compressed_int4 && use == Fmt::INT4) {
        PackedWeight a,b;
        if(!read_compressed_int4_ref(ck,ra,a,rerr)||
           !read_compressed_int4_ref(ck,rb,b,rerr)){
            std::printf("\n  direct compressed INT4 concatenate failed for %s: %s\n",
                        what,rerr.c_str());
            *ok=false;return d;
        }
        if(a.K!=b.K||a.row_scales!=b.row_scales){
            std::printf("\n  compressed INT4 concatenate layout mismatch for %s\n",what);
            *ok=false;return d;
        }
        p=a;p.N=N;
        p.payload.insert(p.payload.end(),b.payload.begin(),b.payload.end());
        p.scales_raw.insert(p.scales_raw.end(),b.scales_raw.begin(),b.scales_raw.end());
        p.zeros.insert(p.zeros.end(),b.zeros.begin(),b.zeros.end());
    } else if (ra.gptq && rb.gptq && use == Fmt::INT4) {
        PackedWeight a,b;
        if(!repack_gptq_ref(ck,ra,a,rerr)||!repack_gptq_ref(ck,rb,b,rerr)){
            std::printf("\n  direct GPTQ concatenate failed for %s: %s\n",what,rerr.c_str());
            *ok=false;return d;
        }
        p=a;p.N=N;
        p.payload.insert(p.payload.end(),b.payload.begin(),b.payload.end());
        p.scales_raw.insert(p.scales_raw.end(),b.scales_raw.begin(),b.scales_raw.end());
        p.zeros.insert(p.zeros.end(),b.zeros.begin(),b.zeros.end());
    } else {
        std::vector<float> f32(size_t(N) * K);
        if (!read_matrix_f32(ck, ra, f32.data(), rerr) ||
            !read_matrix_f32(ck, rb, f32.data() + size_t(Na) * K, rerr)) {
            std::printf("\n  read failed for %s: %s\n", what, rerr.c_str());
            std::fflush(stdout); *ok = false; return d;
        }
        p = quantize(f32.data(), N, K, use);
    }
    d.payload = dev_copy<uint8_t>(q, p.payload.data(), p.payload.size());
    if (!p.scales_raw.empty())
        d.scales = dev_copy<uint8_t>(q, p.scales_raw.data(), p.scales_raw.size());
    if (!p.zeros.empty())
        d.zeros = dev_copy<uint8_t>(q, p.zeros.data(), p.zeros.size());
    d.w         = p.view();
    d.w.payload = d.payload;
    d.w.scales  = d.scales;
    d.w.zeros   = d.zeros;
    if (p.fmt == Fmt::INT4 && p.row_scales > 0 && !p.scales_raw.empty() &&
        p.zeros.size() == size_t(N) * p.row_scales &&
        std::all_of(p.zeros.begin(), p.zeros.end(), [](uint8_t z){ return z == 8; })) {
        std::vector<bf16_t> tr(size_t(N) * p.row_scales);
        const auto* src=reinterpret_cast<const bf16_t*>(p.scales_raw.data());
        for(int n=0;n<N;++n)for(int g=0;g<p.row_scales;++g)
            tr[size_t(g)*N+n]=src[size_t(n)*p.row_scales+g];
        d.od_scales=dev_copy<bf16_t>(q,tr.data(),tr.size()*sizeof(bf16_t));
        std::vector<sycl::half> trh(tr.size());
        for(size_t i=0;i<tr.size();++i)trh[i]=sycl::half(bf16_to_f32(tr[i]));
        d.od_scales_fp16=dev_copy<sycl::half>(q,trh.data(),trh.size()*sizeof(sycl::half));
        d.od_w4=d.od_scales!=nullptr;
    }
    return d;
}

DevQuant concat_upload_many_int4_t(sycl::queue& q,const Qwen35Model& ck,
                                    const std::vector<TensorRef>& refs,
                                    const char* what,bool* ok){
    DevQuant d;
    if(refs.empty()){*ok=false;return d;}
    PackedWeight p;
    int N=0,K=-1;
    std::string err;
    for(size_t i=0;i<refs.size();++i){
        PackedWeight part;
        if(!refs[i].ok()||refs[i].t.shape.size()!=2||
           !read_compressed_int4_ref(ck,refs[i],part,err)){
            std::printf("\n  direct compressed INT4 concatenate failed for %s: %s\n",
                what,err.c_str());*ok=false;return d;
        }
        if(K>=0&&(part.K!=K||part.row_scales!=p.row_scales)){
            std::printf("\n  compressed INT4 concatenate layout mismatch for %s\n",what);
            *ok=false;return d;
        }
        if(i==0){p=std::move(part);K=p.K;N=p.N;}
        else{
            N+=part.N;
            p.payload.insert(p.payload.end(),part.payload.begin(),part.payload.end());
            p.scales_raw.insert(p.scales_raw.end(),part.scales_raw.begin(),
                                part.scales_raw.end());
            p.zeros.insert(p.zeros.end(),part.zeros.begin(),part.zeros.end());
        }
    }
    p.N=N;
    d.payload=dev_copy<uint8_t>(q,p.payload.data(),p.payload.size());
    d.scales=dev_copy<uint8_t>(q,p.scales_raw.data(),p.scales_raw.size());
    d.zeros=dev_copy<uint8_t>(q,p.zeros.data(),p.zeros.size());
    d.w=p.view();d.w.payload=d.payload;d.w.scales=d.scales;d.w.zeros=d.zeros;
    if(!d.payload||!d.scales||!d.zeros){*ok=false;return d;}
    std::vector<bf16_t> tr(size_t(N)*p.row_scales);
    const auto* src=reinterpret_cast<const bf16_t*>(p.scales_raw.data());
    for(int n=0;n<N;++n)for(int g=0;g<p.row_scales;++g)
        tr[size_t(g)*N+n]=src[size_t(n)*p.row_scales+g];
    d.od_scales=dev_copy<bf16_t>(q,tr.data(),tr.size()*sizeof(bf16_t));
    std::vector<sycl::half> trh(tr.size());
    for(size_t i=0;i<tr.size();++i)trh[i]=sycl::half(bf16_to_f32(tr[i]));
    d.od_scales_fp16=dev_copy<sycl::half>(q,trh.data(),
        trh.size()*sizeof(sycl::half));
    d.od_w4=d.od_scales&&d.od_scales_fp16;
    if(!d.od_w4)*ok=false;
    return d;
}

} // namespace

// ---------------------------------------------------------------------
// Engine construction
// ---------------------------------------------------------------------
struct Grimoire {
    // The queue MUST be in-order. forward() launches ~44 kernels per layer
    // pass and passes an EMPTY dependency list to every one of them, so the
    // only thing sequencing a 40-layer chain is the queue itself. A default-
    // constructed sycl::queue is OUT OF ORDER: every kernel becomes free to
    // run concurrently with the one whose output it reads, which races on the
    // residual stream and emits fluent-looking garbage. Also pin the GPU
    // explicitly -- the default selector is not required to pick a GPU at all.
    // GRIMOIRE_TIMELINE=1 adds profiling so forward() can report where a
    // token's device time actually goes. Profiling is NOT free, so the
    // property is only requested when the timeline is asked for.
    static sycl::property_list queue_props() {
        if (std::getenv("GRIMOIRE_TIMELINE") ||
            std::getenv("GRIMOIRE_PROFILE_PREFILL"))
            return {sycl::property::queue::in_order(),
                    sycl::property::queue::enable_profiling()};
        return {sycl::property::queue::in_order()};
    }
    // Match vLLM's XPU worker model: ZE_AFFINITY_MASK keeps the full 0,1
    // visibility list, while each child process selects device local_rank.
    static sycl::device rank_device() {
        // This used to select devices whose NAME contains "B70", which
        // made every other Battlemage card invisible: a B580, a B60 or a
        // B50 simply was not in the list, so on a mixed box the third
        // rank fell through to the default selector and on a B580-only
        // box nothing was selectable at all.  GRIMOIRE has to run on
        // every Battlemage part, so take all GPUs and merely PREFER the
        // discrete Arc ones -- which keeps an integrated display GPU from
        // being picked when a real card is present, without hardcoding
        // one product name.
        const auto all = sycl::device::get_devices(sycl::info::device_type::gpu);
        std::vector<sycl::device> pick;
        for (const auto& d : all) {
            const std::string n = d.get_info<sycl::info::device::name>();
            // Arc discrete parts: "Intel(R) Arc(TM) B580 Graphics",
            // "Intel(R) Arc(TM) Pro B70 Graphics", and the Battlemage
            // codename as the driver sometimes reports it.
            if (n.find("Arc") != std::string::npos ||
                n.find("Battlemage") != std::string::npos ||
                n.find("BMG") != std::string::npos)
                pick.push_back(d);
        }
        if (pick.empty()) pick = all;
        // GRIMOIRE_DEVICES remaps rank -> device index, for a box whose
        // cards are not all the same: with two B70s and a B580 the small
        // card should be a specific pipeline stage, not whichever one the
        // driver happens to enumerate third.
        if (const char* order = std::getenv("GRIMOIRE_DEVICES");
            order && *order && !pick.empty()) {
            std::vector<sycl::device> remap;
            for (const char* p = order; *p; ) {
                char* end = nullptr;
                const long v = std::strtol(p, &end, 10);
                if (end == p) break;
                if (v >= 0 && v < long(pick.size())) remap.push_back(pick[size_t(v)]);
                p = end;
                while (*p == ',' || *p == ' ') ++p;
            }
            if (!remap.empty()) pick.swap(remap);
        }
        const char* e = std::getenv("GRIMOIRE_PP_RANK");
        if (!e || !*e) e = std::getenv("GRIMOIRE_TP_RANK");
        const bool rank_requested = e && *e;
        const int rank = rank_requested ? std::atoi(e) : 0;
        if (rank >= 0 && rank < int(pick.size())) {
            // Say which card this rank got, with a PHYSICAL identity where
            // the driver exposes one.  On a mixed box the layer split
            // depends on which rank landed where, and the vector index
            // alone is not an identity -- it moves with ZE_AFFINITY_MASK
            // and with whatever the driver happened to enumerate.
            static bool said = false;
            if (!said) {
                said = true;
                const auto& dev = pick[size_t(rank)];
                std::string id;
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
                if (dev.has(sycl::aspect::ext_intel_pci_address))
                    id = " pci " + dev.get_info<
                        sycl::ext::intel::info::device::pci_address>();
#endif
                std::fprintf(stderr, "  rank %d -> GPU %d/%zu: %s%s\n", rank, rank,
                             pick.size(),
                             dev.get_info<sycl::info::device::name>().c_str(),
                             id.c_str());
            }
            return pick[size_t(rank)];
        }
        // An explicitly ranked process whose device does not exist must
        // NOT fall through to the default selector: every rank would then
        // pick the same card, the collectives would still connect, and the
        // run would look like it worked while two ranks fought over one
        // GPU.  Fail where the cause is still visible.
        if (rank_requested && !pick.empty()) {
            throw std::runtime_error(
                "GRIMOIRE rank " + std::to_string(rank) + " has no device: only " +
                std::to_string(pick.size()) + " GPU(s) are visible. Check "
                "ZE_AFFINITY_MASK and GRIMOIRE_DEVICES.");
        }
        // GRIMOIRE_DEVICE_ANY: run on whatever SYCL device exists when
        // there is no GPU at all.  This is a CORRECTNESS harness, not a
        // fallback -- it is what lets tools/test_k2_e2e_device.cpp drive
        // the whole engine on a host with an OpenCL CPU device, so a path
        // like K2's can be shown to run before it ever reaches the card.
        // It is deliberately opt-in, it never fires when a GPU is present,
        // and it says so loudly, because a timing taken here would be
        // meaningless and rule 8 already says what happens when a number
        // gets quoted from a config nobody looked at.
        if (const char* any = std::getenv("GRIMOIRE_DEVICE_ANY");
            any && *any && *any != '0') {
            const auto gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
            if (gpus.empty()) {
                sycl::device d{sycl::default_selector_v};
                std::fprintf(stderr,
                    "\n  *** GRIMOIRE_DEVICE_ANY: no GPU -- running on '%s'.\n"
                    "  *** CORRECTNESS ONLY.  Do not benchmark this.\n\n",
                    d.get_info<sycl::info::device::name>().c_str());
                return d;
            }
        }
        return sycl::device{sycl::gpu_selector_v};
    }
    sycl::queue   q{rank_device(), queue_props()};
    sycl::queue   q_aux{q.get_context(), q.get_device(), queue_props()};

    // ---- pipeline parallel across 2 B70s -------------------------------
    // Layers [0,pipe_split) live on device 0 (the queue `q` above); layers
    // [pipe_split, n_layers) live on device 1 (`q1`).  Only the hidden state
    // (H floats = 8 KB) crosses the link, once per token, host-staged -- so
    // unlike tensor-parallel there is no per-layer all-reduce and the slow
    // USB4 link is not on the critical path.  The point is CAPACITY: each
    // card holds ~half the weights, so an FP8/BF16 model that does not fit
    // one 32 GB B70 fits across two.  GRIMOIRE_PIPELINE=1 turns it on and
    // picks the second B70 by PCI order; default is single-device, unchanged.
    bool pipeline = false;
    int  pipe_split = 0;
    std::unique_ptr<sycl::queue> q1;      // device-1 queue when pipeline
    float* pipe_host = nullptr;           // pinned staging for the boundary
    size_t pipe_host_elems = 0;
    static bool pipeline_enabled() {
        const char* e = std::getenv("GRIMOIRE_PIPELINE");
        return e && *e && std::atoi(e) != 0;
    }
    // GRIMOIRE_PIPELINE="a,b" -> a layers on GPU0, b on GPU1 (must sum to
    // n_layers).  A bare non-zero value means an even split.  vLLM-style
    // explicit per-GPU layer counts.
    int pipeline_split_env(int n_layers) const {
        const char* e = std::getenv("GRIMOIRE_PIPELINE");
        if (!e) return n_layers / 2;
        const char* comma = std::strchr(e, ',');
        if (!comma) return n_layers / 2;
        int a = std::atoi(e);
        if (a < 1 || a >= n_layers) return n_layers / 2;
        return a;
    }
    // The queue that owns layer i's weights and runs its kernels.
    sycl::queue& qL(int i) { return (pipeline && i >= pipe_split) ? *q1 : q; }
    bool on_dev1(int i) const { return pipeline && i >= pipe_split; }

    // ---- external-dock pipeline: one process owns one GPU -------------
    // A USB-C/USB4 eGPU needs an independent Level Zero/IOMMU context per
    // card. Rank 0 runs the early layers, rank 1 runs the late layers, and a
    // Unix socket carries the materialized hidden stream once per stage.
    int pp_rank = []{ const char* e=std::getenv("GRIMOIRE_PP_RANK");
        return e&&*e?std::atoi(e):-1; }();
    int pp_world = []{ const char* e=std::getenv("GRIMOIRE_PP_WORLD_SIZE");
        return e&&*e?std::atoi(e):2; }();
    int tp_rank = []{ const char* e=std::getenv("GRIMOIRE_TP_RANK");
        return e&&*e?std::atoi(e):-1; }();
    int tp_world = []{ const char* e=std::getenv("GRIMOIRE_TP_WORLD_SIZE");
        return e&&*e?std::atoi(e):2; }();
    int pp_begin = 0, pp_end = 0;
    int pp_prev_fd = -1, pp_next_fd = -1;
    std::vector<int> tp_peer_fd;
    std::string pp_socket;
    bool pp_enabled() const { return pp_rank >= 0; }
    bool tp_enabled() const { return tp_rank >= 0; }
    int comm_rank() const { return pp_enabled()?pp_rank:tp_rank; }
    static bool fd_write_all(int fd,const void* data,size_t bytes) {
        const uint8_t* p=static_cast<const uint8_t*>(data);
        while(bytes){const ssize_t n=::send(fd,p,bytes,MSG_NOSIGNAL);
            if(n>0){p+=n;bytes-=size_t(n);continue;}
            if(n<0&&errno==EINTR)continue;return false;} return true;
    }
    static bool fd_read_all(int fd,void* data,size_t bytes) {
        uint8_t* p=static_cast<uint8_t*>(data);
        while(bytes){const ssize_t n=::recv(fd,p,bytes,0);
            if(n>0){p+=n;bytes-=size_t(n);continue;}
            if(n<0&&errno==EINTR)continue;return false;} return true;
    }
    bool pp_connect(std::string& err);
    // A whole REQUEST, forwarded down the pipeline.
    //
    // Under PP every stage runs the SAME generation loop and they stay in
    // step by exchanging one message per token.  The CLI gets away with
    // never sending the request itself, because every rank was launched
    // with the same -p and reads the same prompt off its own command
    // line.  A SERVER has no such luck: the prompt arrives on rank 0's
    // HTTP socket and the other stages have never seen it.  That, and
    // nothing deeper, is why there has never been a two-card server.
    struct PPRequest {
        std::vector<int32_t> prompt;
        int budget = 0, eos = -1, eot = -1;
        bool shutdown = false;      // the last message a worker gets
        int kind = 0; // 0: generate, 1: shutdown, 2: admit slot, 3: batch step
        std::vector<int32_t> slots, positions;
    };
    bool serving_control = false; // rank 0 scheduler owns the control stream
    bool pp_send_request(const PPRequest& r);
    bool pp_recv_request(PPRequest& r);
    // Whether speculation is live for the WHOLE pipeline.  Under PP only
    // the last stage owns the MTP head, so the ranks cannot each decide
    // for themselves: they would take different branches of the decode
    // loop and deadlock on the next collective.  The last stage publishes
    // one answer at connect time and everyone uses it.
    int  pp_spec = 0;
    // Taps forwarded along the pipeline for a DFlash drafter, or 0.
    //
    // DFlash's drafter consumes the residual stream tapped at several
    // TARGET layers, concatenated into one row per token.  Under PP those
    // layers live on different stages, so no stage can build that row from
    // what it computes alone.  Each stage therefore captures the taps it
    // owns and forwards the whole block to the next one, which overwrites
    // its copy, fills in its own taps, and forwards again -- by the last
    // stage every tap is present.  The last stage is the one that hosts
    // the drafter, same as the MTP head.
    //
    // Zero unless a drafter is configured, and every send below is guarded
    // on it, so a pipeline without DFlash moves exactly the bytes it
    // always did.
    int  pp_taps = 0;
    // Draft block width the LAST stage will propose, or 0 when no DFlash
    // drafter is loaded there.  Earlier stages have no drafter and cannot
    // ask their own copy, so they take the width from the handshake and
    // use it to size the backward hop that carries the drafted block.
    int  pp_dflash = 0;
    bool spec_active() const {
        if (pp_enabled()) return pp_spec != 0;
        return mtp_enabled() && mtp.ok;
    }
    // Does a rejected draft have recurrent state to restore?  Keyed on
    // cfg, not on this rank's layers: under PP one stage can own every
    // linear layer and another none, and the ranks must agree.
    bool has_recurrent_state() const {
        return cfg.lin_v_heads > 0 && cfg.lin_v_dim > 0;
    }
    // Can a rejected draft be rolled back EXACTLY?
    //
    // commit_spec_prefix rebuilds the DeltaNet state and conv ring from
    // spec_dn_steps / spec_conv_inputs, and only the BATCHED verify writes
    // those.  With a sequential verify it restores state that was never
    // saved: measured on a hybrid model as ten identical tokens and then
    // permanent divergence -- output that stays fluent and stops being the
    // model's.
    //
    // Restore-and-replay was tried as a way to make the sequential path
    // exact and does not work: restore_recurrent alone is clean, but
    // re-running forward() at an already-processed position corrupts
    // memory on a DeltaNet model.  See generation.hpp.  So when the
    // batched verify is unavailable AND the model is recurrent, the answer
    // is to not speculate -- never to speculate approximately.
    //
    // On a B70 the batched verify is available, so single-GPU and PP are
    // unaffected.  TP is the case this disables, because TP always
    // declines the batched prefill.
    bool spec_verify_available() const {
        if (!has_recurrent_state()) return true;
        if (tp_enabled()) return false;
        // device_can_matrix, not the raw aspect: a CPU that advertises
        // ext_intel_matrix and cannot compile joint_matrix would claim a
        // batched verify it then fails to run.
        return device_can_matrix(q);
    }

    bool pp_sync_tokens(std::vector<int32_t>& toks);
    bool pp_send_hidden(const float* dev,size_t elems);
    bool pp_recv_hidden(float* dev,size_t elems);
    bool prefix_cache_usable() const;
    // Single source of the refusal reason, shared by prefix_cache_usable()
    // and the startup banner (external audit finding 2, 2026-09-21) --
    // the banner used to reconstruct this list on its own and drifted:
    // it still named MTP after MTP was deliberately allowed, and never
    // learned about the Qwen4-Exp exclusion at all.  Empty means usable.
    std::string prefix_cache_unusable_reason() const;
    bool pp_send_taps(int first,int rows);
    bool pp_recv_taps(int first,int rows);
    int  pp_sync_token(int token);
    bool tp_allgather(float* dev, int elems, int begin, int count);
    bool tp_allreduce_sum(float* dev, int elems);
    bool tp_shard_rows(DevQuant& dq, sycl::queue& owner, std::string& err);
    Qwen35Model   ck;              // mmapped checkpoint, host side
    Qwen35Config  cfg;

    // Quantized-at-load projections, per layer.
    struct LayerDev {
        LayerKind kind;
        bf16_t *in_norm = nullptr, *post_norm = nullptr;

        DevQuant la_qkv, la_z, la_out, la_ab, la_all; // quantized
        DevQuant sh_gu;                              // gate|up concatenated
        bf16_t  *la_conv = nullptr, *la_Alog = nullptr, *la_dtb = nullptr, *la_norm = nullptr;

        DevQuant q_proj, k_proj, v_proj, qkv_proj, o_proj;
        bf16_t  *q_norm = nullptr, *k_norm = nullptr;
        bf16_t  *pre_ff_norm = nullptr, *post_ff_norm = nullptr;  // Muse sandwich
        sycl::half *in_norm_f16=nullptr, *post_norm_f16=nullptr;
        sycl::half *pre_ff_norm_f16=nullptr, *post_ff_norm_f16=nullptr;
        DevQuant o_gate;                                          // Muse attn output gate
        DevQuant sh_gate, sh_up, sh_down;
        DevQuant sh_gate_q;                          // shared_expert_gate [1][H]
        bool     has_sh_gate = false;
        DevQuant router;

        // experts: zero-copy MXFP4, expert-major
        MoeLayer moe;
        int expert_begin = 0, expert_count = 0;
        uint8_t *gu_pack = nullptr, *gu_scale = nullptr, *gu_zero = nullptr;
        uint8_t *dn_pack = nullptr, *dn_scale = nullptr, *dn_zero = nullptr;
        bool xe2_signed_int4 = false;

        // W4A8 prefill copies of the FFN weights: symmetric int4 g128 plus
        // one f32 scale per (row, group).  Additive -- the MXFP4 originals
        // stay resident because decode's GEMV is far faster at M=1.
        uint8_t *sh_gu_i4 = nullptr, *sh_dn_i4 = nullptr;
        float   *sh_gu_ws = nullptr, *sh_dn_ws = nullptr;

        // ---- K2-Horizon MoVA -------------------------------------
        // A sparse layer has NO v_proj: the value is a routed mixture over
        // v_experts[], each [kv_heads*head_dim, hidden].  v_router is
        // [mova_experts, hidden] -- N=64 does not divide 256, so it stays
        // BF16 and must never reach a W4A8 tile (rule 4).
        // This layer's attention geometry.  Equal to cfg.head_dim /
        // cfg.n_kv_heads for every architecture except gemma-4, which
        // gives sliding and full-attention layers different shapes.
        // Resolved once at upload so no forward path has to re-derive it.
        int  kv_heads = 0;
        int  head_dim = 0;
        // This layer's RoPE.  Gemma-4 keys rope_parameters by layer type;
        // everywhere else these are cfg.rope_theta / cfg.partial_rope and
        // rope_proportional is false.
        float rope_theta = 0.0f;
        float partial_rope = 1.0f;
        // proportional RoPE divides every inverse frequency by this; 1.0
        // everywhere else, which is the identity.
        float rope_factor = 1.0f;
        bool  rope_proportional = false;
        // gemma-4: hidden_states *= layer_scalar as the LAST act of the
        // layer, after the residual add.  1.0 is the identity everywhere
        // else, so the multiply is unconditional and costs nothing.
        float layer_scalar = 1.0f;
        bool k2_sparse = false;
        // Routed FFN is decided PER LAYER, not per model: a K2 layer in
        // mlp_only_layers has a plain MLP while the model as a whole is
        // MoE.  Every forward path branches on this, never on cfg.is_moe().
        bool moe_layer = false;
        DevQuant v_router;
        bf16_t*  v_router_bias = nullptr;
        // Exactly ONE of these is populated, never both -- they hold the
        // same weights and holding both would double a K2 layer's value
        // parameters on the card.
        //   v_experts_packed : the E experts concatenated along N as one
        //     [E*N][K] weight, so a single kernel can index them from the
        //     device-resident routing table.  The default.
        //   v_experts        : the separate per-expert weights, kept for
        //     TENSOR PARALLEL, which shards each expert's rows across
        //     ranks -- a shard of the concatenated matrix would cut across
        //     expert boundaries instead.
        DevQuant v_experts_packed;
        int      v_experts_n = 0;          // rows per expert in the packed form
        std::vector<DevQuant> v_experts;
        bf16_t*  router_bias = nullptr;      // mlp.gate.bias

        // Views into dn_base/conv_base, which hold n_seq_slots copies --
        // exactly as k_cache is a view into k_base.  A hybrid model keeps
        // its whole history in these two buffers rather than in a KV
        // cache, so ONE copy per engine is what stopped several
        // conversations being stepped together: they would advance the
        // same state and each would read the others' history.
        float *dn_state = nullptr, *conv_ring = nullptr;
        float *dn_base = nullptr, *conv_base = nullptr;
        size_t dn_slot = 0, conv_slot = 0;      // elements per slot
        // k_cache/v_cache are a VIEW into k_base/v_base, which hold
        // n_seq_slots copies of this layer's cache back to back.
        // bind_seq_slot() moves the view; every one of the forty-odd
        // readers below still says d.k_cache and none of them had to
        // change.  The OWNER is k_base -- release() frees that, never the
        // view, which points at whichever slot happened to be live.
        uint8_t *k_cache = nullptr, *v_cache = nullptr;
        uint8_t *k_base  = nullptr, *v_base  = nullptr;
        size_t   kv_slot = 0;          // elements per slot
        sycl::half *k_cache_f16 = nullptr, *v_cache_f16 = nullptr;
        bool muse_sliding = false;

        // ---- Qwen4-Exp ------------------------------------------------
        // TWO hyper-connections per layer.  There is no input_layernorm
        // and no post_attention_layernorm on this architecture: hc_attn's
        // norm stands where the first would be and hc_mlp's where the
        // second would.
        struct HCDev {
            bf16_t*  norm = nullptr;    // [hc*H], applied as (1 + w)
            DevQuant down;              // [lowrank][hc*H]
            DevQuant inject;            // [hc][hc*H]  -- absent on the mixer
            DevQuant up;                // [hc*H][lowrank]
        };
        HCDev hc_attn, hc_mlp;

        // QSA.  The indexer keeps TWO caches: the raw index keys exactly
        // as projected (no norm, no RoPE -- the pooling averages those),
        // and the pooled-then-normed-then-roped key of each COMPLETE
        // block, which is what the block scores are computed against.
        bool     qsa = false;
        DevQuant ix_qk;                 // index_qk_proj [(ih+ikv)*ihd][H]
        bf16_t  *ix_qn = nullptr, *ix_kn = nullptr;
        float   *ix_raw_base = nullptr, *ix_cmp_base = nullptr;
        float   *ix_kraw = nullptr;     // [max_seq][ihd]
        float   *ix_kcmp = nullptr;     // [max_seq/ratio + 1][ihd]

        // PLE.  Only on the layers named by ple_layer_ids.  The table is
        // HOST memory by design (rule: this is the one exception to
        // VRAM-only, and it is the architecture's own choice).
        bool     ple = false;
        int      ple_dense_id = 0;
        // key and value stay SEPARATE.  Merged into one [hc*H + H][E]
        // GEMM their rows interleave per token, and the PLE gate reads
        // key with a stride of hc*H and value with a stride of H -- which
        // is true at M == 1 and false for every batch.
        DevQuant ple_key, ple_value;
        bf16_t  *ple_nk=nullptr, *ple_nq=nullptr, *ple_nc=nullptr, *ple_cw=nullptr;
        const void* ple_table = nullptr;
        bool     ple_fp8 = false;
        float    ple_scale = 1.0f;
        int64_t  ple_rows = 0;
        int64_t *ple_mul=nullptr, *ple_size=nullptr, *ple_off=nullptr;
        // The dilated conv's carried history: (kernel-1)*dilation rows of
        // conv_in, oldest first.  Only this window is needed, which is
        // what keeps a PLE layer's state constant in context length.
        float   *ple_hist = nullptr, *ple_hist_base = nullptr;
    };
    std::vector<LayerDev> L;

    // ---- Qwen4-Exp model-level state ---------------------------------
    // The tail mixer.  It replaces the final norm entirely (this model has
    // no model.norm), and it is the one hyper-connection built with
    // use_combine = false, so it has no injection projection.
    LayerDev::HCDev hc_final;
    // The multi-stream residual and the buffers the two hyper-connections
    // pass between them.  q4_pend / q4_pinj carry a DEFERRED combine into
    // the next layer -- the reference returns the pair rather than
    // combining at once, and the tail consumes whatever is still pending.
    float   *q4_hyper=nullptr, *q4_normed=nullptr, *q4_gate=nullptr;
    float   *q4_lora=nullptr,  *q4_inj=nullptr,   *q4_pinj=nullptr;
    float   *q4_pend=nullptr;
    // QSA scratch
    float   *q4_ixqk=nullptr, *q4_pool=nullptr, *q4_lg=nullptr;
    int32_t *q4_blk=nullptr, *q4_idx=nullptr, *q4_vis=nullptr;
    int32_t *q4_seq=nullptr, *q4_qpos=nullptr;
    // PLE scratch
    float   *q4_emb=nullptr, *q4_kv=nullptr, *q4_gated=nullptr, *q4_conv=nullptr;
    int64_t *q4_ids=nullptr;
    // The token history the n-gram hash walks backwards through.  The
    // engine never kept one; PLE needs ngram_size-1 predecessors and the
    // EOS walk needs them in order, so the whole request's tokens live
    // here.  int32 * max_seq is nothing next to a KV cache.
    int32_t *q4_tok=nullptr, *q4_tok_base=nullptr;
    int      q4_blocks_cap = 0;    // compressed key rows a QSA layer holds
    int      q4_ix_width = 0;      // index_qk_proj output rows
    int      q4_expand_w = 0;      // token_topk + compress_ratio - 1

    // MoVA scratch, one set reused by every layer: router logits, the
    // top-k table, and one expert output row.  Tiny -- 64 + 4 + 4 + 1024
    // floats -- and allocated once in build().
    float*   mova_logits = nullptr;
    int32_t* mova_rex = nullptr;
    float*   mova_rwt = nullptr;
    float*   mova_expert_out = nullptr;

    // Cached raw oneDNN W4A16 primitives used by Muse prompt prefill and
    // DFlash verification. vLLM's XPUwNa16LinearKernel caches the primitive
    // by shape; rebuilding it for every speculative step discards that path.
    struct OneDnnPlan {
        int m=0, n=0, k=0;
        void* plan=nullptr;
        void* scratch=nullptr;
    };
    std::vector<OneDnnPlan> muse_od_plans;
    std::vector<OneDnnPlan> dflash_f16_plans;
    int8_t* muse_od_zp=nullptr;

    // Single-entry exact prompt-prefix cache. State stays device-resident so
    // a cache hit restores KV + recurrent state with device-to-device copies.
    // What a slot has to COPY.  The KV cache is not in here: each slot
    // owns its own rows inside the layer's allocation and the live
    // pointers are a view onto them, so switching conversations moves a
    // pointer.  The recurrent state is different and does need a copy --
    // see save_prefix().
    struct PrefixLayerCache {
        float *dn = nullptr, *conv = nullptr;
    };
    struct PrefixCache {
        bool valid = false;
        bool output_valid = false; // batched snapshots have state, not scalar logits
        std::vector<int32_t> tokens;
        std::vector<PrefixLayerCache> layers;
        float *hidden = nullptr, *logits = nullptr;
        uint64_t used = 0;                 // LRU stamp
    };
    // N conversations, not one.
    //
    // With a single snapshot two agents wipe each other's on every turn
    // and both fall back to re-reading their whole history -- the exact
    // cost prefix reuse exists to remove, reintroduced by the second
    // caller.  A slot per conversation is what makes reuse survive
    // interleaving, and it is worth having before batched decode exists:
    // agents still take turns on the card, but each one resumes.
    //
    // Each slot is a FULL copy of the KV and recurrent state, so N slots
    // cost N times the cache.  Sized by GRIMOIRE_PREFIX_SLOTS and 1 by
    // default, which is byte-for-byte today's behaviour.
    std::vector<PrefixCache> prefix_slots;
    uint64_t prefix_clock = 0;
    mutable int prefix_hit = -1;          // slot prefix_reuse() matched
    // How many conversations the KV cache has room for, and which one is
    // live.  Fixed at build() because the cache is allocated that deep:
    // reading the environment later would describe a cache that does not
    // exist.
    int n_seq_slots = 1;
    int seq_slot = 0;
    void bind_seq_slot(int j);
    // Empty a named slot and leave it live.  See the definition for why
    // the order matters.
    void clear_seq_slot(int j);
    void clear_drafter_cache();
    static bool prefix_cache_enabled() {
        const char* e = std::getenv("GRIMOIRE_PREFIX_CACHE");
        return e && *e && std::atoi(e) != 0;
    }
    // Read EVERY time, not captured in a static.  The count is consumed
    // once per engine (prefix_slots is sized on first use and never
    // resized), so caching it buys nothing -- and a static captures
    // whatever the environment happened to be at the first call in the
    // process, which is a different engine's answer.  That cost a gate
    // run: a test that sets the variable for its second arm got the
    // first arm's value and both agents shared one slot.
    static int prefix_cache_slots() {
        const char* e = std::getenv("GRIMOIRE_PREFIX_SLOTS");
        const int v = e && *e ? std::atoi(e) : 1;
        return v < 1 ? 1 : (v > 64 ? 64 : v);
    }
    // How deep the KV cache is allocated.  Two callers want the same
    // thing for different reasons -- the prefix cache wants a slot per
    // resident conversation, a batched decode wants a slot per row in
    // flight -- so they share one number and take the larger request.
    // GRIMOIRE_SEQ_SLOTS is the name that means both; the older
    // GRIMOIRE_PREFIX_SLOTS still works and only counts when the cache
    // is on, which is what it always meant.
    static int seq_slots_requested() {
        const char* e = std::getenv("GRIMOIRE_SEQ_SLOTS");
        const int v = e && *e ? std::atoi(e) : 1;
        const int a = v < 1 ? 1 : (v > 64 ? 64 : v);
        const int b = prefix_cache_enabled() ? prefix_cache_slots() : 1;
        return a > b ? a : b;
    }
    // The slot to write this request's snapshot into: the one it resumed
    // from (extend it in place -- a conversation should not consume a
    // second slot every turn), otherwise a free one, otherwise the
    // least recently used.
    int prefix_slot_for_write();
    bool restore_prefix(const std::vector<int32_t>& tokens);
    // How many leading tokens the snapshot covers (0 == no reuse), and
    // the restore that leaves the cursor there so the rest can be
    // prefilled on top.
    int  prefix_reuse(const std::vector<int32_t>& tokens,
                      const std::vector<bool>* busy = nullptr) const;
    bool restore_prefix_upto(int n);
    bool save_prefix(const std::vector<int32_t>& tokens, bool output_valid = true);
    int admit_sequence(const std::vector<int32_t>& prompt, const std::vector<bool>& busy);
    void cache_sequence(int slot, int position, const std::vector<int32_t>& prompt,
                        const std::vector<int32_t>& reply);
    // What generate_tokens() calls at the end of a request, covering the
    // prompt AND the reply.  Named apart from save_prefix() so the
    // generation template does not depend on the private one's contract.
    void save_prefix_now(const std::vector<int32_t>& tokens) {
        (void)save_prefix(tokens);
    }

    bf16_t*  embed = nullptr;
    int embed_begin = 0, embed_count = 0;
    bf16_t*  fnorm = nullptr;
    sycl::half* fnorm_f16 = nullptr;
    DevQuant lm_head;
    // lm_head aliases the embedding table (tie_word_embeddings).  It must
    // not be freed twice, and it must not be quantized in place.
    bool     tied_lm_head = false;

    // ---- MTP (multi-token prediction) head ------------------------
    // The checkpoint already carries it: mtp.fc [H][2H], three norms, and a
    // single decoder layer whose tensor set and shapes are IDENTICAL to a
    // normal full-attention layer (verified against layers.3), in the same
    // MXFP4 encoding.  So it loads through the existing machinery.
    //
    //   x   = fc @ [ rmsnorm(h_t, pre_h) ; rmsnorm(embed(t+1), pre_e) ]
    //   x   = decoder_layer(x)
    //   h'  = rmsnorm(x, norm)
    //   t+2 = argmax(lm_head(h'))
    struct MtpHead {
        bool     ok    = false;
        DevQuant fc;
        bf16_t  *pre_h = nullptr, *pre_e = nullptr, *norm = nullptr;
        LayerDev L;
        float   *cat = nullptr;    // [2H] concat fed to fc
        float   *x   = nullptr;    // [H]
        float   *h2  = nullptr;    // [H] normed
        float   *resid = nullptr;  // [H]
    } mtp;

    // ---- DFlash masked block drafter ------------------------------
    // GRIMOIRE_DFLASH_MODEL selects the original 0.4B DFlashDraftModel used
    // by the proven Ornith SGLang result (6-8 committed tokens/step).  Keep
    // GRIMOIRE_DFLASH2_MODEL as a compatibility alias for the newer
    // DFlash2DraftModel, whose grouped-conv and selector tensors are optional
    // extensions of the same six-layer Qwen3 draft core.
    //
    // Both variants share the target embed/lm_head and consume eight target
    // residual taps through fc.weight.  Each draft layer owns an independent
    // KV cache: target-derived context K/V is inserted before the 16-query
    // non-causal block is evaluated.
    struct DFlash2Head {
        struct Layer {
            DevQuant q, k, v, qkv, o, gate_up, down;
            DevQuant attn_conv_proj, mlp_conv_proj;
            bf16_t *in_norm=nullptr, *post_norm=nullptr;
            bf16_t *q_norm=nullptr, *k_norm=nullptr;
            sycl::half *in_norm_f16=nullptr, *post_norm_f16=nullptr;
            sycl::half *q_norm_f16=nullptr, *k_norm_f16=nullptr;
            bf16_t *attn_conv_base=nullptr, *mlp_conv_base=nullptr;
            uint8_t *k_cache=nullptr, *v_cache=nullptr;
            sycl::half *k_cache_f16=nullptr, *v_cache_f16=nullptr;
            // Attention shape, resolved per layer the way the reference
            // does (see include/b70/dflash_config.hpp).  window == 0 is
            // full attention; causal is a property of the LAYER, not of
            // the drafter.
            int  window=0;
            bool causal=false;
        };
        bool ok=false, v2=false;
        DevQuant fc, selector_hidden;
        DevQuant fused_context_kv;
        DevQuant shared_embed_f16, shared_lm_head_f16, draft_lm_head_i4;
        bf16_t *hidden_norm=nullptr, *norm=nullptr;
        sycl::half *hidden_norm_f16=nullptr, *norm_f16=nullptr;
        bf16_t *predecessor=nullptr, *successor=nullptr;
        // DFlash2 candidate selector.  rank comes from the hidden_projection
        // weight itself (Linear(hidden -> rank)); top_k must come from the
        // draft config, because launch_topk16_rows emits exactly 16 and a
        // checkpoint asking for a different width must NOT be silently run
        // at 16.
        int selector_top_k=0, selector_rank=0;
        bool selector_ok=false;
        int32_t* sel_ids=nullptr;     // [steps*16] candidate token ids
        float*   sel_unary=nullptr;   // [steps*16] their logits
        float*   sel_hidden=nullptr;  // [steps*rank] hidden_projection(h)
        float*   sel_scores=nullptr;  // [steps*K*K] edge scores
        // Token-major [max_seq,n_taps,H]. The draft fc consumes one
        // contiguous concatenated target-feature row per verified token.
        float *target_aux=nullptr;
        // Persistent original-DFlash scratch. Fixed addresses are also the
        // foundation for capturing the 16-query draft in a reusable graph.
        float *ctx=nullptr, *h=nullptr, *resid=nullptr, *normed=nullptr;
        float *context_kv_all=nullptr;
        float *q=nullptr, *k=nullptr, *v=nullptr, *attn=nullptr;
        float *proj=nullptr, *gate_up=nullptr, *mlp=nullptr, *logits=nullptr;
        sycl::half *q_f16=nullptr, *k_f16=nullptr, *v_f16=nullptr;
        sycl::half *attn_f16=nullptr;
        sycl::half *linear_in_f16=nullptr, *linear_out_f16=nullptr;
        sycl::half *context_k_all_f16=nullptr, *context_v_all_f16=nullptr;
        bf16_t *k_norm_all=nullptr;
        sycl::half *k_norm_all_f16=nullptr;
        sycl_bf16 *bf=nullptr;
        int8_t *a8=nullptr;
        float *a8s=nullptr;
        int32_t *tokens=nullptr, *draft_ids=nullptr;
        int draft_logits_stride=0;
        // NInfer Build-2 proposal vocabulary.  The Q4G64 byte plane is
        // already native signed-s4 DPAS layout; scales are widened from FP16
        // once at load, and argmax rows are remapped through token_ids.
        uint8_t *draft_head_i4=nullptr;
        float *draft_head_i4s=nullptr;
        int32_t *draft_head_token_ids=nullptr;
        int draft_head_rows=0;
        // The DRAFT's own output head and embedding table, when the
        // checkpoint ships them.  vLLM shares the TARGET's lm_head and
        // embed_tokens ONLY with a drafter that has none of its own
        // (_should_share, ref/dflash_speculator.py).  A head trained over a
        // reduced draft vocabulary, decoded instead through the target's
        // head, is a different model -- and, like everything else about a
        // drafter, it fails only as an acceptance rate: the tokens still
        // look like words and the verified output is still correct.
        DevQuant draft_lm_head;
        bf16_t  *draft_embed=nullptr;
        int32_t *draft_vocab_map=nullptr;  // draft id -> target id (from d2t)
        int      draft_vocab_rows=0;
        int32_t *block_table=nullptr, *cu_q=nullptr, *cu_k=nullptr;
        int32_t *seqused_k=nullptr;
        // DFlash2 dynamic grouped convolution.  Geometry is derived from the
        // artifact, not the config: base_kernel is [2,taps,hidden] and
        // kernel_projection is [2*taps*groups, hidden].
        float *conv_delta=nullptr, *conv_scratch=nullptr;
        bool fp16_draft=true;   // drafter weights uploaded as FP16
        // Context-K norm source for the Muse drafter.  false (default) =
        // each layer's own checkpoint weight; true = layer 0's weight for
        // every layer, which is what the Fusion reference was measured to
        // apply.  See the load site for the numbers.
        bool knorm_layer0=false;
        int ctx_chunk=16;       // rows per draft-context ingest iteration
        int conv_taps=0, conv_groups=0, conv_block=16;
        // Muse speculative verifier scratch, reused after the draft pass.
        float *verify_logits=nullptr;
        sycl_bf16 *verify_bf=nullptr, *verify_bf_out=nullptr;
        int8_t *verify_a8=nullptr;
        float *verify_a8s=nullptr;
        int32_t *verify_ids=nullptr;
        void *fc_plan=nullptr, *fc_scratch=nullptr;
        int context_pos=0;
        int hidden=0, inter=0, q_heads=0, kv_heads=0, head_dim=0;
        int mask_token=0, sliding_window=0;
        // The DRAFT's rms_norm_eps, not the target's.  Hardcoding 1e-6
        // here silently changed every norm in the draft forward for any
        // checkpoint that ships a different one.
        float rms_eps=1.0e-6f;
        // Fusion pages the DRAFT KV cache at 16, not at the target's 64.
        // Verified against the running reference: its context slots for
        // positions 0..63 are 368..431 and its query slots for 64..79 are
        // 432..447, i.e. base 368 = block 23 * 16, which is not a multiple of
        // 64. The drafter's own config.json also declares "block_size": 16.
        // At 64 the 80-key draft sequence is one whole page plus a 16-key
        // partial page, and the paged kernel returns zeros for that trailing
        // partial page: the appends land correctly (cache[64:80] is
        // bit-identical to the source K/V) but attention reads them as zero,
        // so the 16 draft rows see only the 64 context keys and never the
        // bonus token. At 16 the same 80 keys are exactly 5 whole pages.
        //
        // That investigation concluded 16 and the default stayed at 64, so
        // the finding was written down and never applied: the Muse draft
        // ran with its anchor invisible.  It is resolved from the draft
        // config now (dflash_config.block_size), 16 when the config names
        // none, and printed at load.  block_table is shared with the
        // target's own paged attention, which pages at 64 and reads only
        // the first (max_seq+63)/64 entries -- a SMALLER draft page makes
        // that table longer, never shorter, so this stays in bounds.
        int block_size=16, num_blocks=0;
        float rope_theta=0.0f;
        std::vector<Layer> layers;
        std::vector<int> target_layers;
    } dflash2;

    // Pointer views switch the drafter with the target sequence. Only slot
    // zero uses the original build allocations; later slots own new caches.
    struct DraftSlot {
        uint8_t *mk=nullptr,*mv=nullptr;
        float *aux=nullptr,*hidden=nullptr;
        std::vector<uint8_t*> k,v;
        std::vector<sycl::half*> k16,v16;
        std::vector<void*> owned;
        int context=0;
    };
    std::vector<DraftSlot> draft_slots;
    float* batch_conv_steps=nullptr;
    void init_draft_slots();
    bool speculative_batch() const {
        return !pp_enabled() && !tp_enabled() && (mtp.ok || dflash2.ok);
    }
    bool decode_spec_batch(const std::vector<int32_t>& tokens,
        const std::vector<int>& slots, const std::vector<int>& positions,
        const std::vector<int>& remaining, std::vector<std::vector<int32_t>>& replies,
        std::vector<int>& consumed);

    // Speculative verification advances every recurrent layer optimistically.
    // Keep one device-side checkpoint so a rejection can restore the exact
    // pre-verify state and replay only the accepted prefix. Attention KV
    // entries do not need copying: replay overwrites the speculative slots.
    float* spec_dn_state = nullptr;
    float* spec_conv_ring = nullptr;
    float* spec_dn_steps = nullptr;
    float* spec_conv_inputs = nullptr;
    float* spec_hidden_steps = nullptr;
    // Whether spec_hidden_steps actually HOLDS this round's hidden states.
    // Only the BATCHED verify writes it.  The pointer is allocated as soon
    // as MTP loads, so testing the pointer says nothing about the contents:
    // when prefill() declines and generation falls back to a sequential
    // verify, commit_spec_prefix would copy an allocated-but-never-written
    // buffer into s.h, and the next mtp_draft would draft from uninitialised
    // device memory.  Non-finite logits there make argmax return INT_MAX --
    // a correct reduction over garbage -- which surfaces as "MTP draft
    // failed", intermittently, because it depends on what the allocator
    // last left behind.
    bool   spec_hidden_valid = false;
    size_t spec_dn_elems = 0, spec_conv_elems = 0;
    size_t spec_conv_input_elems = 0;
    static constexpr int kSpecBatch = 16;
    // Read EVERY time, not captured in a static.  A static answers with
    // whatever the environment was at the FIRST call in the process,
    // which is a different engine's answer -- and the call sites are all
    // per-build or per-request, never per-token, so there is nothing to
    // save by caching it.
    //
    // This is the second time the same shape has bitten here.  It cost a
    // gate run when prefix_cache_slots() did it, and then it silently
    // emptied a gate: a test that set GRIMOIRE_MTP for its last arm got
    // the first arm's value, no drafter loaded, and the arm checked a
    // configuration it was not in.  The negative control is what caught
    // it -- restoring the refusal the arm was supposed to detect changed
    // nothing, which can only mean the arm was never in that case.
    static bool mtp_enabled() {
        const char* e = std::getenv("GRIMOIRE_MTP");
        return e && *e && std::atoi(e) != 0;
    }

    Scratch s{};
    int32_t* tp_expert = nullptr;
    float* tp_weight = nullptr;
    int max_seq = 8192;
    int pos = 0;

    double load_seconds = 0;
    double vram_gb = 0;

    // ---- command graph -------------------------------------------
    // The whole 40-layer sequence is recorded ONCE and replayed per
    // token. Every per-token value (position, sequence length) lives in
    // device memory so the recording stays valid; nothing inside the
    // graph captures a host variable that changes.
    bool recording = false;
    bool graph_ok  = false;
    std::unique_ptr<sycl_ext::command_graph<sycl_ext::graph_state::executable>> gexec;

    struct LayerDevRef { const DevQuant* qkv; bool ok; };
    LayerDevRef first_linear_layer() const {
        for (const auto& d : L)
            if (d.kind == LayerKind::LINEAR_ATTN) return { &d.la_qkv, true };
        return { nullptr, false };
    }

    // Stage-by-stage numeric probe. Off unless GRIMOIRE_DEBUG is set,
    // because it serialises the queue.
    bool  debug = false;
    // ---- device timeline ------------------------------------------
    // A marker is a 1-thread empty kernel. On an in-order queue the gap
    // between marker i's end and marker i+1's start IS the device time of
    // everything submitted between them, gaps included. That measures the
    // real cost of a region without instrumenting 40 launch sites.
    bool  timeline = std::getenv("GRIMOIRE_TIMELINE") != nullptr;
    bool  dag = false; // All model paths require in-order submission.
    int   dag_mask = 0; // 1 linear-attn, 2 full-attn, 4 MoE/shared overlap
    // Exhaustive B70 sweep: all four fusions preserve the token hash and,
    // together with GEMV 16/1, are the fastest coherent configuration.
    int   fusion_mask = 15; // 1 DN norm+gate, 2 QK norm+rope, 4 MoE join, 8 pos
    std::vector<sycl::event> dag_tail;
    sycl::event dag_logits;
    bool  tl_done = false;
    int   tl_tok = 0;          // dump a STEADY token, not the warm-up one
    std::vector<std::pair<sycl::event, std::string>> tl;
    static constexpr int kTlToken = 3;
    void mark(const char* tag) {
        if (!timeline || tl_done || tl_tok != kTlToken) return;
        tl.emplace_back(q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
        }), tag);
    }
    void dump_timeline();
    int   probe_layer = 0;
    float* probe_buf = nullptr;
    void probe(const char* tag, const float* p, int n);
    void sync() { q.wait_and_throw(); }     // forward() no longer drains; callers that
                                  // time a region must end it with this.

    // Any decode GEMV.  When a weight has been converted its MXFP4 payload is
    // gone, so every decode call site must come through here.
    // ---- K2-Horizon MoVA value projection (M=1 decode) ------------
    // A sparse layer has no v_proj.  The value is a routed mixture:
    //   logits = v_router @ x                       [mova_experts]
    //   top-k on sigmoid(logits) + bias, and the weight that scales an
    //   expert is the UNBIASED sigmoid -- the bias steers SELECTION only
    //   value  = sum_j w_j * silu(v_experts[idx_j] @ x)
    // The SiLU sits on the EXPERT OUTPUT, before the router weight.
    //
    // BRING-UP PATH.  The routing table comes back to the host once per
    // layer so the chosen experts can go through the proven gemv_any.
    // That is one sync per layer and it will NOT be fast; it is correct,
    // it reuses kernels that already work, and it produces text to read,
    // which is the bar before any speed claim.  The device-resident form
    // wants the 64 experts packed expert-major into a single weight so a
    // grouped GEMV can index them with no sync at all.
    sycl::event mova_value_m1(LayerDev& d, const float* x, float* y,
                              const std::vector<sycl::event>& deps) {
        // Packed form: no readback at all, and one kernel instead of
        // 1 + top_k.  Decode pays the same two stalls per sparse layer
        // that prefill did, just fewer of them, so it takes this path too.
        if (d.v_experts_packed.w.N > 0 && d.v_experts_n > 0) {
            const int E = cfg.mova_experts;
            const int K = std::min(cfg.mova_top_k, E);
            if (E <= 0 || K <= 0) return q.submit([&](sycl::handler& h){
                h.depends_on(deps); h.single_task([=](){}); });
            sycl::event ev = gemv_any(d.v_router, x, mova_logits, deps);
            ev = launch_router_topk_k2(q, mova_logits, d.v_router_bias,
                                       1, E, K, mova_rex, mova_rwt,
                                       /*normalize=*/K > 1, cfg.router_scale, {ev});
            return launch_mova_value_packed(q, d.v_experts_packed.w, x,
                                            mova_rex, mova_rwt, y,
                                            1, d.v_experts_n, E, K, {ev});
        }
        const int E = int(d.v_experts.size());
        const int K = std::min(cfg.mova_top_k, E);
        // output_rows(), NOT w.N.  Under TP w.N is this rank's SHARD, while
        // gemv_any all-gathers the full row into mova_expert_out -- so
        // sizing the clear and the accumulate from w.N left the top
        // (world-1)/world of every MoVA value vector holding whatever the
        // previous layer wrote.  Silent: the shapes all still line up.
        const int N = d.v_experts.empty() ? 0 : d.v_experts[0].output_rows();
        if (E <= 0 || K <= 0 || N <= 0) return q.submit([&](sycl::handler& h){
            h.depends_on(deps); h.single_task([=](){}); });

        sycl::event ev = gemv_any(d.v_router, x, mova_logits, deps);
        ev = launch_router_topk_k2(q, mova_logits, d.v_router_bias,
                                   1, E, K, mova_rex, mova_rwt,
                                   /*normalize=*/K > 1, cfg.router_scale, {ev});
        // one readback per layer -- see the note above
        int32_t idx[16]; float wts[16];
        q.memcpy(idx, mova_rex, size_t(K) * sizeof(int32_t), {ev});
        q.memcpy(wts, mova_rwt, size_t(K) * sizeof(float)).wait();

        q.memset(y, 0, size_t(N) * sizeof(float)).wait();
        sycl::event last;
        for (int j = 0; j < K; ++j) {
            const int e = idx[j];
            if (e < 0 || e >= E) continue;          // router produced no route
            last = gemv_any(d.v_experts[size_t(e)], x, mova_expert_out, {});
            last = launch_silu_scale_accum(q, mova_expert_out, y, wts[j], N, {last});
        }
        return last;
    }

    // Batched MoVA value projection.
    //
    // mova_value_m1 costs TWO host stalls (the routing readback and the
    // zeroing wait).  Calling it once per token in prefill is therefore
    // 2*M stalls per sparse layer: at 4096 tokens over 45 sparse layers
    // that is ~368,000 synchronous round trips, which is not "slow", it
    // is unusable.
    //
    // This does the same arithmetic with ONE readback for the whole
    // batch: route every token in a single top-k launch, bring the
    // [M][K] table back once, then issue the expert GEMVs with no
    // further synchronisation.  The kernel COUNT is unchanged -- what
    // goes away is the stalling, which is the part that dominated.
    //
    // The remaining work, and it is the bigger win, is to pack the
    // experts expert-major into one weight so a grouped GEMV indexes
    // them on device and the readback disappears entirely.  That needs a
    // new format-templated kernel (the MoE one does not fit: it is
    // gate|up SwiGLU shaped, MoVA is a single matrix per expert), which
    // is why it is not done here.
    //
    // Buffers: `logits`, `rex` and `rwt` are the caller's MoE routing
    // scratch.  That is safe and deliberate -- MoVA runs in the ATTENTION
    // half of the layer and the MoE stage recomputes its routing before
    // using it -- but it is an ordering dependency, so if the FFN ever
    // moves before attention this needs its own buffers.
    sycl::event mova_value_batched(LayerDev& d, const float* x, float* y, int M,
                                   float* logits, int32_t* rex, float* rwt,
                                   std::vector<int32_t>& idx_host,
                                   std::vector<float>& wt_host) {
        const int H = cfg.hidden;
        // Packed form: the routing table never leaves the device, so the
        // one remaining stall goes away with it.
        if (d.v_experts_packed.w.N > 0 && d.v_experts_n > 0) {
            const int E = cfg.mova_experts;
            const int K = std::min(cfg.mova_top_k, E);
            if (E <= 0 || K <= 0 || M <= 0)
                return q.submit([&](sycl::handler& h){ h.single_task([=](){}); });
            for (int m = 0; m < M; ++m)
                gemv_any(d.v_router, x + size_t(m) * H,
                         logits + size_t(m) * E, {});
            sycl::event ev = launch_router_topk_k2(q, logits, d.v_router_bias,
                                                   M, E, K, rex, rwt,
                                                   /*normalize=*/K > 1,
                                                   cfg.router_scale, {});
            return launch_mova_value_packed(q, d.v_experts_packed.w, x,
                                            rex, rwt, y, M, d.v_experts_n,
                                            E, K, {ev});
        }
        const int E = int(d.v_experts.size());
        const int K = std::min(cfg.mova_top_k, E);
        const int N = d.v_experts.empty() ? 0 : d.v_experts[0].output_rows();
        if (E <= 0 || K <= 0 || N <= 0 || M <= 0)
            return q.submit([&](sycl::handler& h){ h.single_task([=](){}); });

        // Router: one GEMV per token, no stall between them.  NOT the
        // batched GEMM -- v_router is N = mova_experts (64 on the real
        // checkpoint), and rule 4 says a weight whose N is not a multiple
        // of 256 must not reach a W4A8/XMX tile.
        for (int m = 0; m < M; ++m)
            gemv_any(d.v_router, x + size_t(m) * H, logits + size_t(m) * E, {});

        sycl::event ev = launch_router_topk_k2(q, logits, d.v_router_bias,
                                               M, E, K, rex, rwt,
                                               /*normalize=*/K > 1,
                                               cfg.router_scale, {});
        idx_host.resize(size_t(M) * K);
        wt_host.resize(size_t(M) * K);
        q.memcpy(idx_host.data(), rex, size_t(M) * K * sizeof(int32_t), {ev});
        q.memcpy(wt_host.data(), rwt, size_t(M) * K * sizeof(float));
        q.memset(y, 0, size_t(M) * N * sizeof(float));
        q.wait();                                   // the ONE stall

        sycl::event last;
        for (int m = 0; m < M; ++m)
            for (int j = 0; j < K; ++j) {
                const int e = idx_host[size_t(m) * K + j];
                if (e < 0 || e >= E) continue;      // router produced no route
                last = gemv_any(d.v_experts[size_t(e)], x + size_t(m) * H,
                                mova_expert_out, {});
                last = launch_silu_scale_accum(q, mova_expert_out,
                                               y + size_t(m) * N,
                                               wt_host[size_t(m) * K + j], N, {last});
            }
        return last;
    }

    void gemm_tp(const DevQuant& w, const float* x, float* y, int rows);

    sycl::event gemv_any(const DevQuant& dq, const float* x, float* y,
                         const std::vector<sycl::event>& deps) {
        if (tp_enabled() && dq.tp_sharded()) {
            const int begin = dq.row_begin;
            const int local = dq.w.N;
            sycl::event ev;
            if (dq.has_i4()) {
                ev = launch_gemv_int4sym(q,dq.i4,dq.i4s,x,y+begin,
                                         local,dq.w.K,deps);
            } else {
                ev = launch_gemv(q,dq.w,x,y+begin,deps);
            }
            ev.wait();
            if (!tp_allgather(y,dq.output_rows(),begin,local))
                throw std::runtime_error("TP projection all-gather failed");
            return ev;
        }
        if (dq.has_i4()) {
            if (onednn_i4_gemv(dq, x, y, deps)) return mx_last;
            return launch_gemv_int4sym(q, dq.i4, dq.i4s, x, y, dq.w.N, dq.w.K, deps);
        }
        if (onednn_mx_gemv(dq, x, y, deps)) return mx_last;
        return launch_gemv(q, dq.w, x, y, deps);
    }

    // Route a single-token MXFP4 projection through oneDNN. Opt-in while it is
    // being measured; the plan is cached per (N,K) because building a oneDNN
    // primitive_desc per call would dwarf the 48 us it takes to run.
    struct MxPlanEntry { void* plan; void* scratch; };
    std::map<std::pair<int,int>, MxPlanEntry> mx_plans;
    sycl::event mx_last{};
    sycl_bf16* mx_a = nullptr; sycl_bf16* mx_o = nullptr;
    size_t mx_a_cap = 0, mx_o_cap = 0;

    // The W4A8-converted int4 weights are the bulk of decode: 21.8 ms of the
    // 43.9 ms token. The oneDNN int4 matmul needs no weight repacking (its
    // wei desc {k,n} with stride {1,k} reads GRIMOIRE's [N,K] row-major u4
    // directly, same trick as the MXFP4 plan), but its scales are group-major
    // bf16 [K/gs, N] while GRIMOIRE keeps f32 [N, K/gs]. Transpose+convert
    // once per weight and cache it -- ~2 bytes per 128 weights, ~120 MB total.
    struct I4Entry { void* plan; void* scratch; sycl_bf16* scales; };
    std::map<const float*, I4Entry> i4_plans;
    int8_t* i4_zp = nullptr;

    bool onednn_i4_gemv(const DevQuant& dq, const float* x, float* y,
                        const std::vector<sycl::event>& deps) {
        static const bool on = std::getenv("GRIMOIRE_ONEDNN_I4") != nullptr;
        if (!on) return false;
        const int N = dq.w.N, K = dq.w.K;
        constexpr int GS = 128;
        if ((K % GS) != 0) return false;
        static OneDnnS4Api api = load_onednn_s4();
        if (!api) return false;

        auto it = i4_plans.find(dq.i4s);
        if (it == i4_plans.end()) {
            void* pl = api.create(&q, 1, N, K, GS, 1);  // bf16 activations, s4 weights
            if (!pl) { i4_plans[dq.i4s] = {nullptr,nullptr,nullptr}; return false; }
            const size_t sb = api.scratch_size(pl);
            void* sc = sb ? sycl::malloc_device<uint8_t>(sb, q) : nullptr;
            const int G = K / GS;
            sycl_bf16* ts = sycl::malloc_device<sycl_bf16>(size_t(G) * N, q);
            if (!ts) { api.destroy(pl); i4_plans[dq.i4s] = {nullptr,nullptr,nullptr}; return false; }
            const float* src = dq.i4s;
            q.parallel_for(sycl::range<2>(size_t(G), size_t(N)),
                [=](sycl::id<2> id) {
                    const size_t g = id[0], n = id[1];
                    ts[g * size_t(N) + n] =
                        sycl_bf16(src[n * size_t(G) + g]);
                }).wait();
            it = i4_plans.emplace(dq.i4s, I4Entry{pl, sc, ts}).first;
        }
        if (!it->second.plan) return false;

        if (mx_a_cap < size_t(K)) {
            if (mx_a) sycl::free(mx_a, q);
            mx_a = sycl::malloc_device<sycl_bf16>(size_t(K), q); mx_a_cap = size_t(K);
        }
        if (mx_o_cap < size_t(N)) {
            if (mx_o) sycl::free(mx_o, q);
            mx_o = sycl::malloc_device<sycl_bf16>(size_t(N), q); mx_o_cap = size_t(N);
        }
        if (!mx_a || !mx_o) return false;

        launch_f32_to_bf16(q, x, mx_a, K, deps);
        api.execute(it->second.plan, mx_a, dq.i4, it->second.scales,
                    mx_o, it->second.scratch);
        mx_last = launch_bf16_to_f32(q, mx_o, y, N, {});
        return true;
    }

    bool onednn_mx_gemv(const DevQuant& dq, const float* x, float* y,
                        const std::vector<sycl::event>& deps) {
        static const bool on = std::getenv("GRIMOIRE_ONEDNN_GEMV") != nullptr;
        if (!on) return false;
        if (dq.w.fmt != Fmt::MXFP4 || !dq.w.payload || !dq.w.scales) return false;
        const int N = dq.w.N, K = dq.w.K;
        if ((K % 32) != 0) return false;
        static OneDnnMXApi api = load_onednn_mx();
        if (!api) return false;

        auto key = std::make_pair(N, K);
        auto it = mx_plans.find(key);
        if (it == mx_plans.end()) {
            void* pl = api.create(&q, 1, N, K);
            if (!pl) { mx_plans[key] = {nullptr, nullptr}; return false; }
            const size_t sb = api.scratch_size(pl);
            void* sc = sb ? sycl::malloc_device<uint8_t>(sb, q) : nullptr;
            it = mx_plans.emplace(key, MxPlanEntry{pl, sc}).first;
        }
        if (!it->second.plan) return false;

        if (mx_a_cap < size_t(K)) {
            if (mx_a) sycl::free(mx_a, q);
            mx_a = sycl::malloc_device<sycl_bf16>(size_t(K), q); mx_a_cap = size_t(K);
        }
        if (mx_o_cap < size_t(N)) {
            if (mx_o) sycl::free(mx_o, q);
            mx_o = sycl::malloc_device<sycl_bf16>(size_t(N), q); mx_o_cap = size_t(N);
        }
        if (!mx_a || !mx_o) return false;

        // NO waits: q is in-order, so the convert -> matmul -> convert chain
        // is already ordered. An earlier version waited on both conversions,
        // which turned every one of the ~200 projections per token into a full
        // pipeline sync and cost more than the kernel saved (TG 17.2 -> 10.0).
        launch_f32_to_bf16(q, x, mx_a, K, deps);
        api.execute(it->second.plan, mx_a, dq.w.payload, dq.w.scales,
                    mx_o, it->second.scratch);
        mx_last = launch_bf16_to_f32(q, mx_o, y, N, {});
        return true;
    }

    // Decode FFN GEMV.  When the W4A8 path converted this layer's weights the
    // MXFP4 originals are gone, so read the symmetric int4 copies instead --
    // same bytes, one copy, and no SLM table lookups per nibble.
    sycl::event ffn_gemv(const LayerDev& d, bool gate_up, const float* x,
                         float* y, const std::vector<sycl::event>& deps) {
        const DevQuant& dq  = gate_up ? d.sh_gu : d.sh_down;
        return gemv_any(dq, x, y, deps);
    }

    bool embed_one(int token,float* out) {
        const std::vector<sycl::event> none{};
        if(!tp_enabled()){launch_embed(q,embed,token,out,cfg.hidden,none);return true;}
        if(token>=embed_begin&&token<embed_begin+embed_count)
            launch_embed(q,embed,token-embed_begin,out,cfg.hidden,none);
        else q.memset(out,0,size_t(cfg.hidden)*sizeof(float));
        return tp_allreduce_sum(out,cfg.hidden);
    }

    // Batched embed that is correct under tensor parallel.  The table is
    // row-sharded over the VOCABULARY, so launch_embed_batched -- which
    // indexes it by absolute token id -- reads whatever row happens to sit
    // at that offset in this rank's slice.  Nothing errors: every rank
    // produces a plausible embedding of the WRONG token, and the only
    // symptom is that the drafter proposes badly.  That is precisely the
    // failure mode this project keeps paying for, so route every batched
    // embed through here rather than calling the kernel directly.
    bool embed_rows(const bf16_t* table,const int32_t* tokens,float* out,
                    int count) {
        const std::vector<sycl::event> none{};
        if(count<=0)return true;
        // A null table is a LOAD bug, not a runtime one -- under PP the
        // embedding table is uploaded only on the stages that need it.
        // Say so instead of faulting: on a GPU this dereference is a
        // DEVICE_LOST and a power cycle, not a SIGSEGV.
        if(!table){
            std::fprintf(stderr,"  embed_rows: no embedding table on this "
                "rank -- a drafter asked to embed %d token(s) where the "
                "table was never uploaded\n",count);
            return false;
        }
        // Only the TARGET's table is sharded.  A drafter that ships its own
        // embed_tokens loaded it whole, so it must NOT be offset.
        if(!tp_enabled()||table!=embed){
            launch_embed_batched(q,table,tokens,out,count,cfg.hidden,none);
            return true;
        }
        launch_embed_batched_shard(q,table,tokens,out,count,cfg.hidden,
                                   embed_begin,embed_count,none);
        return tp_allreduce_sum(out,count*cfg.hidden);
    }

    bool build_graph();
    const float* step();          // one token: graph replay if available

    bool build(const std::string& dir, const UploadOptions& opt, std::string& err);
    // Empty when every feature this checkpoint declares has a forward path
    // in this engine; otherwise the reason, named, for build() to refuse
    // with.  Kept apart from the loader on purpose: the loader describes
    // the file, this describes the engine, and the two go out of date
    // independently.
    std::string unsupported_reason() const;
    void reset();
    // At the next decode step K/V is appended at p and attention reads [0,p+1).
    // Capture values into a device command: never copy from a temporary host int.
    void set_cursor(int p) {
        int32_t* dp=s.d_pos; int32_t* ds=s.d_seq_len;
        q.single_task([=] { *dp=p; *ds=p+1; });
    }
    void check_token(int t) const {
        if(t<0 || t>=cfg.vocab)throw std::invalid_argument("token outside model vocabulary");
        if(pos<0 || pos>=max_seq)throw std::out_of_range("context capacity exhausted");
    }
    const float* forward(int token);      // returns device logits
    const float* forward_muse(int token); // Muse Glimmer dense sandwich path
    const float* forward_gemma4(int token); // gemma-4 sandwich + per-layer-type
    const float* forward_qwen4_exp(int token); // Qwen4-Exp: HC + QSA + PLE
    bf16_t* muse_zero = nullptr;           // zeroed weight -> scaleless (1+0) norm
    // gemma-4's v_norm is Gemma4RMSNorm(head_dim, eps, with_scale=False):
    // a norm with NO weight parameter, i.e. scale 1.  gemma-4 runs on the
    // plain-weight convention (offset 0), so reproducing "no weight" takes
    // a vector of ONES -- a zero vector would multiply the value by zero
    // and produce silence, not an identity.  Sized from max_head_dim so
    // either layer type can use it.
    bf16_t* gemma_vnorm = nullptr;
    sycl::half* muse_zero_f16 = nullptr;   // same, for the FP16 activation path
    const float* forward_dag(int token);  // out-of-order queue + true dependencies
    // ONE TOKEN PER SEQUENCE, several sequences at once.
    //
    // Batched decode is a prefill whose M rows belong to M DIFFERENT
    // conversations rather than to consecutive positions of one.  That
    // is worth saying plainly, because it is why this is a parameter and
    // not a second copy of prefill(): every weight in the model is read
    // once for the whole batch either way, and the only thing that
    // differs is which cache each row reads and at what position.  Decode
    // is weight-bound, so eight conversations stepped together cost
    // about what one costs -- that is the entire point.
    //
    // slot[r] names the sequence's KV rows (see bind_seq_slot) and
    // pos[r] its position in that sequence.  Both are HOST arrays: this
    // path never records a command graph, so there is nothing to bake.
    struct SeqBatch { const int* slot; const int* pos; bool verify=false; };
    // allow_exact_restore: whether an exact-match cache hit may REBIND
    // the live slot out from under the caller (external audit,
    // 2026-09-21).  The serial single-conversation path
    // (generate_tokens()) wants this on: there is only ever one active
    // conversation, so "some other slot already has this exact prompt"
    // can only mean an earlier turn of the SAME conversation, and taking
    // it is correct. The scheduler is different -- it has ALREADY chosen
    // and cleared a specific slot for THIS request via clear_seq_slot()
    // before calling prefill(), as one entry in a table of several
    // SIMULTANEOUSLY active requests. If two admitted requests carry an
    // identical prompt (the ordinary case for agents sharing a system
    // prompt), an exact-match hit would silently rebind this request
    // onto WHICHEVER slot holds that prompt -- possibly a different,
    // currently-live request's slot -- while the scheduler's own
    // bookkeeping (j->slot, slot_busy) still names the slot it started
    // with. Two live requests then read and write the same physical KV
    // rows, and restoring an in-flight request's own snapshot back onto
    // itself can rewind its recurrent state to an earlier turn. The
    // scheduler passes false and lives with a fresh ingestion on every
    // admission; nothing else changes for it.
    bool prefill(const std::vector<int32_t>& tokens,
                 std::vector<int32_t>* next_tokens = nullptr,
                 const SeqBatch* batch = nullptr,
                 bool allow_exact_restore = true);
    // Why a batch cannot be run: a sentence naming the feature, or empty
    // if it can.  Kept apart from prefill() so the caller can refuse
    // BEFORE it commits sequences to slots, and so the reason reaches a
    // human instead of a false return.
    std::string batch_unsupported_reason() const;
    // Step M resident sequences by one token each.  Returns the token
    // each row produced.
    bool decode_batch(const std::vector<int32_t>& toks,
                      const std::vector<int>& slots,
                      const std::vector<int>& poss,
                      std::vector<int32_t>& out);
    static constexpr int kMaxBatchRows = kSpecBatch;
    bool prefill_muse(const std::vector<int32_t>& tokens,
                      std::vector<int32_t>* next_tokens = nullptr,
                      bool allow_exact_restore = true);
    bool prefill_sandwich(const std::vector<int32_t>& tokens,
                        std::vector<int32_t>* next_tokens = nullptr,
                        bool allow_exact_restore = true,
                        const SeqBatch* seqb = nullptr);
    bool prefill_qwen4_exp(const std::vector<int32_t>& tokens,
                           std::vector<int32_t>* next_tokens, const SeqBatch* seqb=nullptr);
    void snapshot_recurrent();
    void restore_recurrent(int saved_pos);
    void commit_spec_prefix(int saved_pos, int accepted);
    // MTP draft: given the hidden state of the token just processed and the
    // token the main model chose next, predict the token AFTER that.
    // from_mtp_hidden: chained drafts feed the head its OWN previous hidden
    // state instead of the main model's h_t, which is how depth > 1 works.
    int  mtp_draft(int next_token, int position, bool from_mtp_hidden = false);
    // Populate the MTP head's KV cache over prompt positions so the drafter
    // does not start each request blind. Writes K/V only -- no attention,
    // no FFN, no lm_head. See mtp_warm() for why this matters.
    void mtp_warm(const float* hidden, int next_token, int position);
    bool dflash_draft(int bonus_token, int position,
                      std::vector<int32_t>& draft_tokens,
                      bool context_only = false);
    // Rows in the DFlash query block, anchor included, so M-1 tokens are
    // drafted per step.  The caller reports it: an A/B of two block widths
    // is read as accepted-per-step, and labelling every run "depth 15"
    // when GRIMOIRE_DFLASH_M=4 drafts three makes that unreadable.
    int          dflash_block_rows() const;
    int          argmax_token();
    void release();
};

// ---------------------------------------------------------------------
bool Grimoire::pp_connect(std::string& err) {
    const char* env = tp_enabled() ? std::getenv("GRIMOIRE_TP_SOCKET")
                                   : std::getenv("GRIMOIRE_PP_SOCKET");
    pp_socket = env && *env ? env :
        (tp_enabled() ? "/tmp/grimoire-tp.sock" : "/tmp/grimoire-pp.sock");
    auto make_addr = [&](const std::string& path, sockaddr_un& addr) {
        if (path.size() >= sizeof(addr.sun_path)) return false;
        addr = {}; addr.sun_family = AF_UNIX;
        std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path)-1);
        return true;
    };
    auto connect_to = [&](const std::string& path, int& fd) {
        sockaddr_un addr{};
        if (!make_addr(path, addr)) { err="parallel socket path is too long"; return false; }
        fd=::socket(AF_UNIX,SOCK_STREAM,0);
        if(fd<0){err="parallel socket() failed";return false;}
        for(int attempt=0;attempt<6000;++attempt){
            if(::connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==0)return true;
            if(errno!=ENOENT&&errno!=ECONNREFUSED)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        const int se=errno;::close(fd);fd=-1;
        err=std::string("parallel connection failed: ")+std::strerror(se);return false;
    };
    auto listen_one = [&](const std::string& path, int& fd) {
        sockaddr_un addr{};
        if (!make_addr(path, addr)) { err="parallel socket path is too long"; return false; }
        const int ls=::socket(AF_UNIX,SOCK_STREAM,0);
        if(ls<0){err="parallel socket() failed";return false;}
        ::unlink(path.c_str());
        if(::bind(ls,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0||::listen(ls,16)<0){
            err=std::string("parallel bind/listen failed: ")+std::strerror(errno);
            ::close(ls);return false;
        }
        do{fd=::accept(ls,nullptr,nullptr);}while(fd<0&&errno==EINTR);
        ::close(ls);return fd>=0;
    };

    if (pp_enabled()) {
        // A PP chain has one socket between every adjacent pair.  Stage r
        // listens on "base-r" and connects forward to "base-(r+1)".
        if(pp_rank>0){
            const std::string path=pp_socket+"-"+std::to_string(pp_rank);
            std::printf("  PP rank %d: waiting on %s\n",pp_rank,path.c_str());
            std::fflush(stdout);
            if(!listen_one(path,pp_prev_fd))return false;
        }
        if(pp_rank+1<pp_world){
            const std::string path=pp_socket+"-"+std::to_string(pp_rank+1);
            if(!connect_to(path,pp_next_fd))return false;
        }
    } else {
        // TP uses rank 0 as the collective hub.  Every peer identifies its
        // rank immediately after connecting, so launch order is irrelevant.
        tp_peer_fd.assign(size_t(tp_world),-1);
        if(tp_rank==0){
            sockaddr_un addr{};
            if(!make_addr(pp_socket,addr)){err="parallel socket path is too long";return false;}
            const int ls=::socket(AF_UNIX,SOCK_STREAM,0);
            if(ls<0){err="TP socket() failed";return false;}
            ::unlink(pp_socket.c_str());
            if(::bind(ls,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))<0||
               ::listen(ls,tp_world-1)<0){err=std::string("TP bind/listen failed: ")+std::strerror(errno);::close(ls);return false;}
            while(std::count_if(tp_peer_fd.begin()+1,tp_peer_fd.end(),[](int x){return x>=0;})<tp_world-1){
                int fd=-1;do{fd=::accept(ls,nullptr,nullptr);}while(fd<0&&errno==EINTR);
                int32_t rank=-1;
                if(fd<0||!fd_read_all(fd,&rank,sizeof(rank))||rank<=0||rank>=tp_world||tp_peer_fd[size_t(rank)]>=0){
                    if(fd>=0)::close(fd);err="TP peer handshake failed";::close(ls);return false;
                }
                tp_peer_fd[size_t(rank)]=fd;
            }
            ::close(ls);
        }else{
            int fd=-1;if(!connect_to(pp_socket,fd))return false;
            const int32_t rank=tp_rank;
            if(!fd_write_all(fd,&rank,sizeof(rank))){::close(fd);err="TP handshake failed";return false;}
            tp_peer_fd[0]=fd;
        }
    }
    if (pp_enabled()) {
        // One backward hop, same shape as pp_sync_token: the last stage
        // says whether it loaded an MTP head and every earlier stage
        // adopts that answer.  Deciding locally would let rank 0 draft
        // while rank 1 did not.
        pp_spec = pp_sync_token(mtp.ok ? 1 : 0);
        if (pp_spec < 0) { err = "PP speculation handshake failed"; return false; }
        // Second hop, same shape, for the DFlash tap block.  Every stage
        // resolved the tap count from the drafter's own config, so they
        // SHOULD already agree -- but a stage that disagreed would read a
        // different number of floats off the socket than the previous one
        // wrote, desynchronising the stream for every later message
        // (hidden states included) with no error at the point of failure.
        // Check it once here, where it is one comparison.
        // Third hop: the DFlash draft width.  Same reason pp_spec exists --
        // if one stage ran the speculative loop and another the plain one
        // they would deadlock on the next collective, so the LAST stage
        // (the only one with a drafter) decides for everybody.
        pp_dflash = pp_sync_token(dflash2.ok ? dflash_block_rows() : 0);
        if (pp_dflash < 0) { err = "PP DFlash handshake failed"; return false; }
        const int want_taps = pp_sync_token(pp_taps);
        if (want_taps < 0) { err = "PP tap handshake failed"; return false; }
        if (want_taps != pp_taps) {
            err = "PP stages disagree about the DFlash tap count (this "
                  "stage " + std::to_string(pp_taps) + ", the last stage " +
                  std::to_string(want_taps) + ") -- every stage must be "
                  "given the same GRIMOIRE_DFLASH_MODEL";
            return false;
        }
    }
    std::printf("  %s rank %d/%d: connected%s\n",tp_enabled()?"TP":"PP",
                comm_rank(),(tp_enabled()?tp_world:pp_world),
                pp_enabled() ? (pp_spec ? ", speculation ON" : ", speculation off") : "");
    std::fflush(stdout);return true;
}

// Broadcast the verified tokens backward along the pipeline.  Only the
// last stage runs the head, so only it knows what the target chose; every
// earlier stage needs the same answer to compute the same acceptance
// count and roll back to the same position.  Size is known to all ranks
// (the candidate count), so only the payload travels.
bool Grimoire::pp_sync_tokens(std::vector<int32_t>& toks) {
    if (toks.empty()) return true;
    const size_t bytes = toks.size() * sizeof(int32_t);
    if (pp_rank == pp_world - 1) {
        if (pp_rank > 0 && !fd_write_all(pp_prev_fd, toks.data(), bytes)) return false;
    } else {
        if (!fd_read_all(pp_next_fd, toks.data(), bytes)) return false;
        if (pp_rank > 0 && !fd_write_all(pp_prev_fd, toks.data(), bytes)) return false;
    }
    return true;
}

// Forward this stage's view of the DFlash tap block for `rows` positions
// starting at `first`.  The receiver overwrites its own copy with it and
// then fills in the taps it owns, so the union grows along the pipe and
// the last stage ends up with every tap.
//
// Sent as ONE contiguous message because target_aux is token-major
// [max_seq][taps][hidden]: the rows for a span of positions are already
// adjacent, so no gather is needed.
bool Grimoire::pp_send_taps(int first, int rows) {
    if (!pp_taps || rows <= 0) return true;
    const size_t elems = size_t(rows) * size_t(pp_taps) * size_t(cfg.hidden);
    return pp_send_hidden(dflash2.target_aux +
                          int64_t(first) * pp_taps * cfg.hidden, elems);
}

bool Grimoire::pp_recv_taps(int first, int rows) {
    if (!pp_taps || rows <= 0) return true;
    const size_t elems = size_t(rows) * size_t(pp_taps) * size_t(cfg.hidden);
    return pp_recv_hidden(dflash2.target_aux +
                          int64_t(first) * pp_taps * cfg.hidden, elems);
}

// Header first, then the tokens.  Fixed-width int32 throughout: these
// stages are separate PROCESSES and the only thing keeping them in step
// is that both sides agree, byte for byte, on every message.
bool Grimoire::pp_send_request(const PPRequest& r) {
    std::vector<int> peers;
    if(tp_enabled()) {
        if(tp_rank!=0) return true;
        for(int i=1;i<tp_world;++i) peers.push_back(tp_peer_fd[size_t(i)]);
    } else if(pp_next_fd>=0) peers.push_back(pp_next_fd);
    if(r.kind==3 && (r.slots.size()!=r.prompt.size() || r.positions.size()!=r.prompt.size()))
        return false;
    const int32_t head[5]={r.shutdown?1:r.kind,int32_t(r.prompt.size()),
                          int32_t(r.budget),int32_t(r.eos),int32_t(r.eot)};
    const size_t bytes=r.prompt.size()*sizeof(int32_t);
    for(int fd:peers) {
        if(!fd_write_all(fd,head,sizeof head)) return false;
        if(bytes && !fd_write_all(fd,r.prompt.data(),bytes)) return false;
        if(r.kind==3 && (!fd_write_all(fd,r.slots.data(),bytes) ||
                       !fd_write_all(fd,r.positions.data(),bytes))) return false;
    }
    return true;
}

bool Grimoire::pp_recv_request(PPRequest& r) {
    const int upstream=tp_enabled()?(tp_rank>0?tp_peer_fd[0]:-1):pp_prev_fd;
    if (upstream < 0) return false;             // rank 0 has no upstream
    int32_t head[5] = {0,0,0,0,0};
    if (!fd_read_all(upstream, head, sizeof head)) return false;
    if (head[0]<0 || head[0]>4) return false;
    r.kind=head[0];
    r.shutdown = r.kind == 1;
    const int32_t n = head[1];
    // A length off the wire sizes an allocation, so it is checked before
    // it is trusted: a desynchronised pipe would otherwise turn into a
    // multi-gigabyte resize rather than a clean failure.
    if (n < 0 || n > (r.kind==3?kMaxBatchRows:max_seq)) return false;
    if ((r.kind==2 || r.kind==3) && n==0) return false;
    if (r.kind==2 && (head[2]<0 || head[2]>=n_seq_slots)) return false;
    if(r.kind==4 && (head[2]<0 || head[2]>=n_seq_slots || head[3]!=n)) return false;
    r.budget = head[2]; r.eos = head[3]; r.eot = head[4];
    r.prompt.assign(size_t(n), 0);
    if (n && !fd_read_all(upstream, r.prompt.data(),
                          size_t(n) * sizeof(int32_t))) return false;
    if (r.kind==3) {
        r.slots.resize(size_t(n)); r.positions.resize(size_t(n));
        if (!fd_read_all(upstream,r.slots.data(),size_t(n)*sizeof(int32_t)) ||
            !fd_read_all(upstream,r.positions.data(),size_t(n)*sizeof(int32_t)))
            return false;
        for (int i=0;i<n;++i) {
            if (r.slots[size_t(i)]<0 || r.slots[size_t(i)]>=n_seq_slots ||
                r.positions[size_t(i)]<0 || r.positions[size_t(i)]>=max_seq)
                return false;
            for(int j=0;j<i;++j) if(r.slots[size_t(i)]==r.slots[size_t(j)]) return false;
        }
    }
    // Pass it on BEFORE running it, so every stage downstream starts its
    // own prefill while this one is still working.  Forwarding after
    // would serialise the pipeline on its own control messages.
    return pp_send_request(r);
}

bool Grimoire::pp_send_hidden(const float* dev, size_t elems) {
    if (elems > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(elems, q);
        pipe_host_elems = pipe_host ? elems : 0;
    }
    if (!pipe_host) return false;
    q.memcpy(pipe_host, dev, elems * sizeof(float)).wait();
    return pp_next_fd>=0 && fd_write_all(pp_next_fd,pipe_host,elems*sizeof(float));
}

bool Grimoire::pp_recv_hidden(float* dev, size_t elems) {
    if (elems > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(elems, q);
        pipe_host_elems = pipe_host ? elems : 0;
    }
    if (!pipe_host) return false;
    if (pp_prev_fd<0 || !fd_read_all(pp_prev_fd,pipe_host,elems*sizeof(float))) return false;
    q.memcpy(dev, pipe_host, elems * sizeof(float)).wait();
    return true;
}

int Grimoire::pp_sync_token(int token) {
    int32_t wire = int32_t(token);
    if(pp_rank==pp_world-1){if(pp_rank>0&&!fd_write_all(pp_prev_fd,&wire,sizeof(wire)))return -1;}
    else{
        if(!fd_read_all(pp_next_fd,&wire,sizeof(wire)))return -1;
        if(pp_rank>0&&!fd_write_all(pp_prev_fd,&wire,sizeof(wire)))return -1;
    }
    return int(wire);
}

// Each rank projects all active rows with its local weight shard before
// gathering each full output row. Weights remain matrix-batched under TP.
void Grimoire::gemm_tp(const DevQuant& w, const float* x, float* y, int rows) {
    float* local=sycl::malloc_device<float>(size_t(rows)*std::max(1,w.w.N),q);
    sycl_bf16* activation=sycl::malloc_device<sycl_bf16>(size_t(rows)*w.w.K,q);
    if(!local || !activation) {
        if(local) sycl::free(local,q);
        if(activation) sycl::free(activation,q);
        throw std::bad_alloc();
    }
    try {
        if(w.w.N>0) {
            if(w.has_i4())
                launch_gemv_int4sym_batch(q,w.i4,w.i4s,x,local,w.w.N,w.w.K,rows,{});
            else if(device_can_matrix(q)) {
                launch_f32_to_bf16(q,x,activation,size_t(rows)*w.w.K,{});
                launch_gemm_xmx(q,w.w,activation,local,rows);
            } else launch_gemm_batched(q,w.w,x,local,rows);
        }
        const int width=w.output_rows();
        for(int r=0;r<rows;++r) {
            if(w.w.N>0)
                q.memcpy(y+int64_t(r)*width+w.row_begin,local+int64_t(r)*w.w.N,
                         size_t(w.w.N)*sizeof(float));
            if(!tp_allgather(y+int64_t(r)*width,width,w.row_begin,w.w.N))
                throw std::runtime_error("TP batched projection all-gather failed");
        }
        q.wait_and_throw();
    } catch(...) {
        q.wait(); sycl::free(local,q); sycl::free(activation,q); throw;
    }
    sycl::free(local,q); sycl::free(activation,q);
}

bool Grimoire::tp_allgather(float* dev, int elems, int begin, int count) {
    if (!tp_enabled() || tp_peer_fd.empty()) return false;
    if (size_t(elems) > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(size_t(elems), q);
        pipe_host_elems = pipe_host ? size_t(elems) : 0;
    }
    if (!pipe_host) return false;
    q.memcpy(pipe_host + begin, dev + begin, size_t(count) * sizeof(float)).wait();
    if (tp_rank == 0) {
        for(int r=1;r<tp_world;++r){
            const int b=(elems*r)/tp_world,e=(elems*(r+1))/tp_world;
            if(!fd_read_all(tp_peer_fd[size_t(r)],pipe_host+b,size_t(e-b)*sizeof(float)))return false;
        }
        for(int r=1;r<tp_world;++r)
            if(!fd_write_all(tp_peer_fd[size_t(r)],pipe_host,size_t(elems)*sizeof(float)))return false;
    } else {
        const int fd=tp_peer_fd[0];
        if(!fd_write_all(fd,pipe_host+begin,size_t(count)*sizeof(float))||
           !fd_read_all(fd,pipe_host,size_t(elems)*sizeof(float)))return false;
    }
    q.memcpy(dev,pipe_host,size_t(elems)*sizeof(float)).wait();
    return true;
}

bool Grimoire::tp_allreduce_sum(float* dev,int elems){
    if(!tp_enabled()||tp_peer_fd.empty())return false;
    const size_t n=size_t(elems);
    if(n>pipe_host_elems){if(pipe_host)sycl::free(pipe_host,q);pipe_host=sycl::malloc_host<float>(n,q);pipe_host_elems=pipe_host?n:0;}
    if(!pipe_host)return false;
    q.memcpy(pipe_host,dev,n*sizeof(float)).wait();
    if(tp_rank==0){
        std::vector<float> peer(n);
        for(int r=1;r<tp_world;++r){if(!fd_read_all(tp_peer_fd[size_t(r)],peer.data(),n*sizeof(float)))return false;
            for(size_t i=0;i<n;++i)pipe_host[i]+=peer[i];}
        for(int r=1;r<tp_world;++r)if(!fd_write_all(tp_peer_fd[size_t(r)],pipe_host,n*sizeof(float)))return false;
    }else{const int fd=tp_peer_fd[0];if(!fd_write_all(fd,pipe_host,n*sizeof(float))||!fd_read_all(fd,pipe_host,n*sizeof(float)))return false;}
    q.memcpy(dev,pipe_host,n*sizeof(float)).wait();return true;
}

bool Grimoire::tp_shard_rows(DevQuant& d,sycl::queue& owner,std::string& err){
    if(!tp_enabled()||d.tp_sharded()||d.w.N<=0)return true;
    const int total=d.w.N;
    // Tiny vectors such as the one-row shared-expert gate are cheaper and
    // safer to replicate than to create empty ranks.
    if(total<tp_world)return true;
    const int begin=(total*tp_rank)/tp_world;
    const int end=(total*(tp_rank+1))/tp_world;
    const int local=end-begin;
    if(local<=0)return true;
    const size_t pbytes=size_t(local)*size_t(d.w.row_bytes);
    const size_t scale_elem=scale_value_bytes(d.w.fmt);
    const size_t srow=size_t(d.w.row_scales)*scale_elem;
    uint8_t* np=d.payload?sycl::malloc_device<uint8_t>(pbytes,owner):nullptr;
    void* ns=d.scales&&srow?sycl::malloc_device<uint8_t>(size_t(local)*srow,owner):nullptr;
    uint8_t* nz=d.zeros&&d.w.row_scales?sycl::malloc_device<uint8_t>(size_t(local)*d.w.row_scales,owner):nullptr;
    if((d.payload&&!np)||(d.scales&&srow&&!ns)||(d.zeros&&d.w.row_scales&&!nz)){
        if(np)sycl::free(np,owner);if(ns)sycl::free(ns,owner);if(nz)sycl::free(nz,owner);
        err="TP row-shard allocation failed";return false;
    }
    if(np)owner.memcpy(np,d.payload+size_t(begin)*d.w.row_bytes,pbytes);
    if(ns)owner.memcpy(ns,static_cast<uint8_t*>(d.scales)+size_t(begin)*srow,size_t(local)*srow);
    if(nz)owner.memcpy(nz,d.zeros+size_t(begin)*d.w.row_scales,size_t(local)*d.w.row_scales);
    owner.wait();
    if(d.payload)sycl::free(d.payload,owner);if(d.scales)sycl::free(d.scales,owner);if(d.zeros)sycl::free(d.zeros,owner);
    // oneDNN-specific companions describe the old full-row layout; TP uses
    // the exact GEMV path and does not retain duplicate full matrices.
    if(d.od_scales){sycl::free(d.od_scales,owner);d.od_scales=nullptr;}
    if(d.od_scales_fp16){sycl::free(d.od_scales_fp16,owner);d.od_scales_fp16=nullptr;}
    d.payload=np;d.scales=ns;d.zeros=nz;
    d.w.payload=np;d.w.scales=ns;d.w.zeros=nz;
    d.full_N=total;d.row_begin=begin;d.w.N=local;
    return true;
}

// ---------------------------------------------------------------------
std::string Grimoire::unsupported_reason() const {
    // Qwen4-Exp / Qwen3.8-Flash-Next.  The loader resolves this config
    // (rule 10: the loader says what the file IS), and the engine says
    // here what it cannot RUN.  Its Qwen3-Next base -- Gated DeltaNet
    // layers interleaved with MoE -- is a shape this engine already
    // executes, which is exactly why the refusal has to be specific: the
    // parts that are missing are NOT the parts that look familiar.
    //
    // Refuse before a single byte is uploaded, and name the mechanism,
    // because each of the three is silent rather than loud if faked:
    // running QSA as dense attention, dropping the n-gram embedding, or
    // collapsing the hyper-connection streams to one residual all produce
    // fluent text that is not the model's output.
    if (cfg.is_qwen4_exp) {
        // All three mechanisms are implemented now (forward_qwen4_exp,
        // and their operators in ops.cpp / attention.cpp against the host
        // references in b70/qwen4_exp.hpp).  What is refused here is what
        // is NOT built, named individually -- each of these is a wrong
        // number or a hang rather than an error if it runs anyway.
        if (tp_enabled() || pp_enabled())
            return "qwen4_exp under TP or PP.  The multi-stream residual is "
                   "hc_count * hidden wide and a stage boundary transports ONE "
                   "tensor, so a pending hyper-connection combine has to be "
                   "materialised before it is sent and the stages have to agree "
                   "on hc_count; neither is wired.  Single process works.";
        if (cfg.hc_count <= 0 || cfg.hc_lowrank <= 0)
            return "qwen4_exp with no hc_count / hc_lowrank: the residual "
                   "stream's width and the mix's rank are not optional, and "
                   "zero would collapse the streams silently.";
        {
            bool any_qsa = false;
            for (bool b : cfg.qsa_attention) any_qsa = any_qsa || b;
            if (any_qsa) {
                if (cfg.indexer_n_heads <= 0 || cfg.indexer_head_dim <= 0 ||
                    cfg.indexer_budget <= 0 || cfg.indexer_compress_ratio <= 0)
                    return "qwen4_exp has qwen_sparse_attention layers but the "
                           "indexer_* fields are incomplete.  The reference "
                           "requires all five together (_validate_qsa_config); "
                           "running QSA as dense attention is fluent and wrong.";
                if (cfg.indexer_kv_heads != 1)
                    return "qwen_sparse_attention with indexer_kv_heads != 1.  "
                           "The indexer is MQA: one key stream shared by every "
                           "indexer head, which the reference validates and the "
                           "scoring assumes.";
                if (cfg.indexer_budget % cfg.indexer_compress_ratio)
                    return "qwen_sparse_attention with indexer_budget not a "
                           "multiple of indexer_compress_ratio: the block top-k "
                           "is their quotient and the remainder would silently "
                           "drop tokens off the end of the expansion.";
            }
        }
        if (!cfg.ple_layer_ids.empty()) {
            const int nh = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
            if (cfg.ngram_size < 2 || cfg.heads_per_ngram <= 0)
                return "qwen4_exp PLE with ngram_size < 2 or heads_per_ngram "
                       "<= 0: there would be no n-gram order to hash.";
            if (nh <= 0 || cfg.ple_embed_dim % nh)
                return "qwen4_exp PLE: ple_embed_dim is not divisible by the "
                       "(ngram_size - 1) * heads_per_ngram n-gram heads, so the "
                       "table rows do not tile the embedding.";
            if (cfg.ple_conv_kernel <= 0)
                return "qwen4_exp PLE with ple_conv_kernel_size <= 0: the short "
                       "convolution is part of the layer's output, not an "
                       "optional extra.";
        }
        // mtp_enabled(), NOT mtp.ok: this runs before a single byte is
        // uploaded, which is the whole point of asking here, and the MTP
        // head is loaded hundreds of lines later.  Reading mtp.ok would
        // be a refusal that can never fire -- a claim that decays the
        // moment anyone relies on it (rule 13).
        if (mtp_enabled() || std::getenv("GRIMOIRE_DFLASH_MODEL"))
            return "qwen4_exp with speculation requested (GRIMOIRE_MTP or a "
                   "DFlash drafter).  A drafter here has to be fed the "
                   "PRE-mixer multi-stream hidden state -- the reference "
                   "keeps a separate _mtp_hidden_buffer for exactly that -- "
                   "and not the single stream the output head reads.  Handing "
                   "it the mixed row drafts from the wrong tensor and shows up "
                   "only as poor acceptance, never as an error.  Unset it and "
                   "the model runs.";
    }

    // head_dim per lane.  Every flash kernel in attention.cpp and
    // prefill.cpp accumulates its head into a PRIVATE array of MAXD
    // floats, with dpl = head_dim / SG_SIZE and SG_SIZE 16 -- so a head
    // wider than the array writes past the end of device stack memory.
    // That is a DEVICE_LOST, not a wrong number.
    //
    // Each of those kernels is now a TEMPLATE on that width and is
    // instantiated twice (kernels.hpp): head_dim <= MAX_HEAD_DIM keeps the
    // 16-slot kernel it has always used, and a wider head dispatches to
    // the 32-slot one, so gemma-4's 512-wide full-attention layers run
    // without changing the private array of any narrower model.  The bound
    // that remains is the wide instantiation's own.
    //
    // NOT VERIFIED ON THE CARD: the 32-slot kernel doubles the private
    // array for the models that reach it, and register pressure at dpl 32
    // is a B70 measurement (rule 8).  Correctness is gated off-card by
    // bin/test_model_matrix, which runs a 512-wide head and diffs it
    // against a host reference; throughput at that width is not.
    const int max_hd = cfg.max_head_dim();
    if (max_hd > MAX_HEAD_DIM_WIDE)
        return "head_dim " + std::to_string(max_hd) +
               " exceeds what the flash-attention kernels can hold "
               "(MAX_DPL_WIDE " + std::to_string(MAX_DPL_WIDE) +
               " x SG_SIZE " + std::to_string(SG_SIZE) + " = " +
               std::to_string(MAX_HEAD_DIM_WIDE) +
               ").  Every head accumulates "
               "into a private array of that size, so a wider head writes "
               "past the end of device stack memory -- a DEVICE_LOST, not a "
               "wrong answer.  Refusing.";
    // The same kernels compute dims-per-lane as head_dim / SG_SIZE with
    // INTEGER division, while the merge pass tiles ceil(head_dim/SG_SIZE).
    // At head_dim 24 the split pass writes 16 dimensions and the merge
    // reads 24, so the last 8 are whatever the previous token left there.
    // No supported model has a ragged head, and nothing would report it.
    // Check EVERY width a layer can have, not the maximum and the
    // model-wide one.  head_dim 32 with global_head_dim 24 passes both of
    // those -- the max is aligned and the local value is aligned -- while
    // the full-attention layers run ragged at 24.
    for (int hd : {cfg.head_dim, cfg.global_head_dim})
        if (hd > 0 && hd % SG_SIZE != 0)
            return "head_dim must be a multiple of " + std::to_string(SG_SIZE) +
                   " (got " + std::to_string(hd) +
                   "; this model uses " + std::to_string(cfg.head_dim) +
                   (cfg.global_head_dim > 0
                        ? " and " + std::to_string(cfg.global_head_dim) : "") +
                   ").  The flash kernels split by integer division and merge "
                   "by ceiling, so a ragged head leaves its last dimensions "
                   "holding the previous token's values.  Refusing.";

    // GeGLU.  forward_gemma4() is the ONLY path that dispatches it; every
    // other FFN dispatch here -- decode GEMV, batched prefill, the MoE
    // path, speculative verify -- is swiglu.  A gelu checkpoint of any
    // other architecture would run silu in gelu's place and emit fluent
    // text that is not the model's output.
    if (cfg.geglu && !cfg.is_gemma4)
        return "this checkpoint's hidden activation is gelu, and the only "
               "path that dispatches GeGLU is gemma-4's.  silu would "
               "silently run in its place, so refuse.";
    return {};
}

bool Grimoire::build(const std::string& dir, const UploadOptions& opt, std::string& err) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    // ---- pipeline setup FIRST: both queues on ONE shared context -----
    // A single process using two B70s must keep BOTH queues in the SAME sycl
    // context, or device-1 allocation throws error 39 once device 0 is loaded
    // (verified: tools/bar2.cpp allocates 10 GB on each card simultaneously
    // in a shared context; two separate contexts fail).  So before ANY weight
    // is uploaded, rebuild q/q_aux on a context that spans both cards and make
    // q1 the device-1 queue on that same context.
    if (pipeline_enabled()) {
        std::vector<sycl::device> b70;
        for (auto& dv : sycl::device::get_devices(sycl::info::device_type::gpu))
            if (dv.get_info<sycl::info::device::name>().find("B70") != std::string::npos)
                b70.push_back(dv);
        sycl::device d0 = q.get_device(), d1;
        bool found = false;
        for (auto& dv : b70) if (!(dv == d0)) { d1 = dv; found = true; break; }
        if (!found) {
            std::printf("  pipeline: only one B70 visible -- single-GPU\n");
        } else {
            sycl::context shared{{d0, d1}};
            q     = sycl::queue{shared, d0, queue_props()};
            q_aux = sycl::queue{shared, d0, queue_props()};
            q1    = std::make_unique<sycl::queue>(shared, d1, queue_props());
            pipeline = true;
            // BE HONEST ABOUT WHAT THIS DOES TODAY.  qL(i) places layer i's
            // WEIGHTS on device 1 past the split, but forward()/prefill()
            // submit every kernel to `q`, which is device 0's queue -- the
            // engine is built around that single member queue and every
            // launcher (gemv_any, mm, mmb, the ops.cpp wrappers) takes it
            // implicitly.  The shared context makes the device-1 pointers
            // legal to dereference, so this RUNS and is NUMERICALLY CORRECT;
            // it is not a fault.  But those weights are then read across the
            // link on every token instead of out of local VRAM, which is the
            // opposite of what pipeline parallelism is for.
            //
            // It is therefore only worth using to fit a model that does not
            // fit on one card, and even then the multiprocess launchers
            // (tools/pp2run.sh) are the faster path because each rank runs
            // its own layers on its own device.
            //
            // Making this real means threading a per-layer queue through the
            // execution layer so layer i's kernels are submitted to qL(i),
            // with per-device scratch and a boundary transfer of the hidden
            // state.  That is an architectural change, not a patch.
            std::printf("  pipeline: 2x B70 on one shared context\n");
            std::printf("  WARNING: GRIMOIRE_PIPELINE splits WEIGHT PLACEMENT only.\n"
                        "           All kernels run on device 0; device-1 weights are\n"
                        "           read over the link every token. Correct but slow.\n"
                        "           Use tools/pp2run.sh for real pipeline execution.\n");
        }
    }
    {   // Say which device we actually got. There are two GPUs on the target
        // box; benchmarking the wrong one silently is worse than being slow.
        const auto& dev = q.get_device();
        std::printf("  device: %s | driver %s | %u EUs | %.1f GiB\n",
                    dev.get_info<sycl::info::device::name>().c_str(),
                    dev.get_info<sycl::info::device::driver_version>().c_str(),
                    dev.get_info<sycl::info::device::max_compute_units>(),
                    double(dev.get_info<sycl::info::device::global_mem_size>()) / (1 << 30));
    }
    if (!ck.load(dir, err)) return false;
    cfg = ck.cfg;
    // The loader is a pure reader: it says what the checkpoint IS.  Whether
    // a forward path in this engine can execute it is a separate question,
    // and answering it wrongly is the silent-output class this project
    // keeps paying for.  Ask here, once, before a single byte is uploaded.
    if (const std::string why = unsupported_reason(); !why.empty()) {
        err = why;
        return false;
    }
    max_seq = opt.max_seq;
    if(max_seq<2 || max_seq==INT32_MAX){err="invalid context capacity";return false;}
    // Resolved HERE, once, because the KV cache is allocated this deep a
    // few hundred lines below.  Without the cache enabled there is one
    // slot and the allocation is byte-for-byte what it always was.
    n_seq_slots = seq_slots_requested();
    seq_slot = 0;

    if (pp_enabled() && tp_enabled()) {
        err = "GRIMOIRE_PP_RANK and GRIMOIRE_TP_RANK are mutually exclusive";
        return false;
    }
    if(tp_enabled()){
        if(tp_world<2||tp_rank>=tp_world){err="TP rank must be 0..TP_WORLD_SIZE-1 and world size must be at least 2";return false;}
        std::printf("  multiprocess TP rank %d/%d: stores 1/%d of every weight\n",
                    tp_rank,tp_world,tp_world);
    }

    if ((pp_enabled() || tp_enabled()) && pipeline_enabled()) {
        err = "multiprocess rank mode and single-process GRIMOIRE_PIPELINE are mutually exclusive";
        return false;
    }
    if (pp_enabled()) {
        std::vector<int> counts;
        const char* layers=std::getenv("GRIMOIRE_PP_LAYERS");
        if(layers&&*layers){
            const char* p=layers;
            while(*p){char* end=nullptr;long v=std::strtol(p,&end,10);
                if(end==p||v<1){err="GRIMOIRE_PP_LAYERS must be x,x,... with positive layer counts";return false;}
                counts.push_back(int(v));p=end;if(!*p)break;if(*p!=','){err="GRIMOIRE_PP_LAYERS must be x,x,...";return false;}++p;}
            const char* world_env=std::getenv("GRIMOIRE_PP_WORLD_SIZE");
            if(world_env&&*world_env&&int(counts.size())!=pp_world){err="PP_LAYERS must contain one number per GPU";return false;}
            pp_world=int(counts.size());
        }else if(pp_world==2){
            const char* split_env=std::getenv("GRIMOIRE_PP_SPLIT");
            const int split=split_env&&*split_env?std::atoi(split_env):(cfg.n_layers*3)/5;
            counts={split,cfg.n_layers-split};
        }else{
            err="GRIMOIRE_PP_LAYERS is required for PP with more than two GPUs";return false;
        }
        if(pp_world<2||pp_rank>=pp_world||int(counts.size())!=pp_world||
           std::accumulate(counts.begin(),counts.end(),0)!=cfg.n_layers){
            err="PP requires one positive layer count per GPU and the counts must sum to model layers";return false;
        }
        pp_begin=std::accumulate(counts.begin(),counts.begin()+pp_rank,0);
        pp_end=pp_begin+counts[size_t(pp_rank)];
        std::printf("  multiprocess PP rank %d/%d: layers [%d,%d)\n",
                    pp_rank,pp_world,pp_begin,pp_end);
    }

    const int H = cfg.hidden;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;

    size_t bytes = 0;
    bool ok = true;
    auto acct = [&](size_t b) { bytes += b; };
    auto shard = [&](DevQuant& d,sycl::queue& owner) {
        if(ok&&!tp_shard_rows(d,owner,err))ok=false;
    };

    // Drop the file mappings now. Every subsequent read is a pread, and
    // holding 16 mappings (~21 GB of VA) while the Level Zero driver
    // reserves its own address space is asking for trouble.
    ck.unmap_all();

    std::printf("  checkpoint resolved: %d layers, vocab %d, hidden %d (mappings released)\n",
                cfg.n_layers, cfg.vocab, cfg.hidden);
    std::fflush(stdout);

    // ---- embeddings ---------------------------------------------------
    std::printf("  embed_tokens  %.2f GiB ... ", double(ck.bytes(ck.embed)) / 1073741824.0);
    std::fflush(stdout);
    // Rank 0 embeds the input tokens.  The LAST stage needs the table too
    // when it hosts a DRAFTER, because every drafter embeds the tokens it
    // proposes:
    //   - the MTP head is fc([norm(embed(next_token)); norm(hidden)]);
    //   - a DFlash drafter embeds its whole query block, and one that
    //     ships no embed_tokens of its own does it from the TARGET's
    //     table.
    // Without this the last stage reads a null pointer on its first
    // draft -- MEASURED, as a SIGSEGV on the last stage and a socket
    // exception on every earlier one, and only under PP, which is the
    // configuration hardest to debug.
    //
    // Whether a DFlash drafter ships its own table is not known here --
    // it is read several hundred lines below -- so the table is loaded
    // whenever a drafter is CONFIGURED.  That can be one unnecessary copy
    // on one card for a drafter that turns out to have its own; the other
    // way round is a crash, so pay the copy.
    const char* dflash_env = std::getenv("GRIMOIRE_DFLASH_MODEL");
    if (!dflash_env || !*dflash_env) dflash_env = std::getenv("GRIMOIRE_DFLASH2_MODEL");
    const bool last_stage_drafts =
        (mtp_enabled() || (dflash_env && *dflash_env)) && pp_rank == pp_world - 1;
    // A tied model has no lm_head tensor: its output projection IS the
    // embedding table.  The LAST stage runs that projection, so it needs
    // the table even when it embeds no token of its own -- otherwise it
    // loads with nothing to produce logits with.
    const bool tied_head = cfg.tie_embeddings && !ck.lm_head.ok();
    const bool need_embed = !pp_enabled() || pp_rank == 0 || last_stage_drafts
                         || (tied_head && pp_rank == pp_world - 1);
    if(need_embed)
        embed=dev_copy_t<bf16_t>(q,ck,ck.embed,"embed_tokens",&ok);
    if (!ok) { err = "embed upload failed"; return false; }
    embed_begin=0;embed_count=cfg.vocab;
    if(tp_enabled()){
        embed_begin=(cfg.vocab*tp_rank)/tp_world;
        const int end=(cfg.vocab*(tp_rank+1))/tp_world;
        embed_count=end-embed_begin;
        bf16_t* local=sycl::malloc_device<bf16_t>(size_t(embed_count)*H,q);
        if(!local){err="TP embedding shard allocation failed";return false;}
        q.memcpy(local,embed+size_t(embed_begin)*H,size_t(embed_count)*H*sizeof(bf16_t)).wait();
        sycl::free(embed,q);embed=local;
    }
    std::printf("ok\n");
    if(embed)acct(size_t(embed_count)*H*sizeof(bf16_t));

    std::printf("  final_norm    ... ");
    std::fflush(stdout);
    // Qwen4-Exp has NO model.norm.  Its tail hyper-connection mixer's
    // grouped norm plus gated mean is the final norm and the thing the
    // head reads, so asking for model.norm.weight here would report a
    // missing tensor for something the checkpoint correctly does not
    // have.  The mixer is use_combine = false: no injection projection.
    if (cfg.is_qwen4_exp) {
        if (!pp_enabled() || pp_rank == pp_world-1) {
            hc_final.norm = dev_copy_t<bf16_t>(q, ck, ck.hc_final.norm,
                                "hyper_connection_mixer.hc_norm", &ok);
            hc_final.down = quantize_upload_t(q, ck, ck.hc_final.down, Fmt::BF16,
                                "hyper_connection_mixer.down", &ok);
            hc_final.up   = quantize_upload_t(q, ck, ck.hc_final.up, Fmt::BF16,
                                "hyper_connection_mixer.up", &ok);
        }
    } else
    if(!pp_enabled()||pp_rank==pp_world-1)
        fnorm=dev_copy_t<bf16_t>(q,ck,ck.final_norm,"model.norm.weight",&ok);
    if(cfg.is_muse&&(!pp_enabled()||pp_rank==pp_world-1))
        fnorm_f16=upload_f16_vector_t(q,ck,ck.final_norm,
                                      "model.norm.weight.fp16",&ok);
    if (!ok) { err = "final_norm upload failed"; return false; }
    std::printf("ok\n");

    // ---- lm_head: the single biggest bf16 tensor ----------------------
    // compressed-tensors explicitly excludes Muse's untied lm_head.  vLLM
    // therefore executes the checkpoint BF16 tensor even though the decoder
    // Linear layers are W4A16.  Do not let --proj int4 requantize this head.
    const bool preserve_muse_lm_head = cfg.is_muse;
    std::printf("  lm_head       %s ... ", preserve_muse_lm_head
        ? "preserving checkpoint bf16" : fmt_name(opt.lm_head_fmt));
    std::fflush(stdout);
    if((!pp_enabled()||pp_rank==pp_world-1)&&ck.lm_head.ok()&&ck.lm_head.t.shape.size()==2) {
        const int V = int(ck.lm_head.t.shape[0]);
        if (!preserve_muse_lm_head && opt.quantize_lm_head &&
            opt.lm_head_fmt != Fmt::BF16) {
            lm_head = quantize_upload_t(q, ck, ck.lm_head, opt.lm_head_fmt, "lm_head", &ok);
        } else {
            lm_head.payload = dev_copy_t<uint8_t>(q, ck, ck.lm_head, "lm_head", &ok);
            lm_head.w = QuantWeight{Fmt::BF16, V, H, lm_head.payload, nullptr, nullptr,
                                    int64_t(H) * 2, 0};
        }
        shard(lm_head,q);
        if(!ok){if(err.empty())err="lm_head TP sharding failed";return false;}
        acct(size_t(lm_head.w.bytes()));
    }
    // tie_word_embeddings.  cfg.tie_embeddings was PARSED and never used:
    // a tied checkpoint ships no lm_head tensor at all, so the block above
    // was skipped, "ok" was printed anyway, and lm_head stayed a default
    // DevQuant with N == 0.  gemv_any then wrote NOTHING into s.logits, so
    // argmax read stale device memory -- the same token every step, for
    // every prompt, in vocabulary and perfectly reproducible.  Nothing in
    // the engine said a word.
    //
    // No architecture here had needed it before (Qwen, Ornith, K2 and Muse
    // all ship an explicit lm_head); gemma-4 is the first that is tied.
    //
    // The output projection IS the embedding table, so alias it rather
    // than copy a second gigabyte onto the card.  Under TP that is already
    // the right sharding: the table is row-sharded over the VOCABULARY and
    // an lm_head shards over its output rows, which are the vocabulary.
    // tied_lm_head records the alias so release() frees the buffer once.
    if (!lm_head.payload && !ck.lm_head.ok() && cfg.tie_embeddings && embed &&
        (!pp_enabled() || pp_rank == pp_world - 1)) {
        lm_head.payload = reinterpret_cast<uint8_t*>(embed);
        lm_head.w = QuantWeight{Fmt::BF16, embed_count, H, lm_head.payload,
                                nullptr, nullptr, int64_t(H) * 2, 0};
        if (tp_enabled()) { lm_head.full_N = cfg.vocab; lm_head.row_begin = embed_begin; }
        tied_lm_head = true;
    }
    if (!lm_head.payload && (!pp_enabled() || pp_rank == pp_world - 1)) {
        err = tied_head
            ? "tie_word_embeddings is set but the embedding table is not on "
              "this rank, so there is nothing to project through"
            : "no lm_head: the checkpoint ships none and tie_word_embeddings "
              "is not set, so there is no output projection to run";
        return false;
    }

    std::printf("%s\n", tied_lm_head ? "tied to embed_tokens" : "ok");
    std::fflush(stdout);

    // ---- layers
    if (cfg.is_gemma4) {
        const size_t hd = size_t(cfg.max_head_dim());
        gemma_vnorm = sycl::malloc_device<bf16_t>(hd, q);
        if (!gemma_vnorm) { err = "gemma-4 v_norm weight allocation failed"; return false; }
        std::vector<bf16_t> ones(hd, f32_to_bf16(1.0f));
        q.memcpy(gemma_vnorm, ones.data(), hd * sizeof(bf16_t)).wait();
    }
    if (cfg.is_muse) {
        muse_zero = sycl::malloc_device<bf16_t>(cfg.hidden, q);
        q.memset(muse_zero, 0, size_t(cfg.hidden) * sizeof(bf16_t)).wait();
        muse_zero_f16 = sycl::malloc_device<sycl::half>(cfg.hidden, q);
        q.memset(muse_zero_f16, 0, size_t(cfg.hidden) * sizeof(sycl::half)).wait();
    }

    if (pipeline) {
        pipe_split = pipeline_split_env(cfg.n_layers);
        pipe_host  = sycl::malloc_host<float>(size_t(cfg.hidden), q);
        std::printf("  pipeline: layers [0,%d) on GPU0, [%d,%d) on GPU1\n",
                    pipe_split, pipe_split, cfg.n_layers);
    }
    L.resize(cfg.n_layers);
    // Resolve every layer's geometry up front, for ALL n_layers -- not only
    // the ones this rank uploads.  Under PP a stage owns [pp_begin, pp_end)
    // and leaves the rest untouched; anything that walks the whole vector
    // afterwards (the prefix cache, prefill's staging sizing) would then
    // read a zero head_dim as a fact rather than as "not mine".
    for (int i = 0; i < cfg.n_layers; ++i) {
        LayerDev& g = L[size_t(i)];
        g.kv_heads          = cfg.layer_kv_heads(i);
        g.head_dim          = cfg.layer_head_dim(i);
        g.rope_theta        = cfg.layer_rope_theta(i);
        g.partial_rope      = cfg.layer_partial_rope(i);
        g.rope_proportional = cfg.layer_rope_proportional(i);
        g.rope_factor       = cfg.layer_rope_factor(i);
    }
    // Muse decoder projections remain checkpoint INT4-W4A16 while its
    // excluded lm_head stays BF16; these formats are intentionally distinct.
    const Fmt PF = cfg.is_muse ? Fmt::INT4 : opt.lm_head_fmt;

    for (int i = 0; i < cfg.n_layers; ++i) {
        const Qwen35Layer& src = ck.layers[i];
        sycl::queue& lq = qL(i);   // device that owns this layer (pipeline)
        if (pipeline && i == pipe_split && ck.native_model) {
            q.wait();
            ck.native_model->drop_resident();
        }
        LayerDev& d = L[i];
        d.kind = src.kind;
        d.muse_sliding = cfg.is_muse &&
            i < int(cfg.muse_sliding_attention.size()) &&
            cfg.muse_sliding_attention[size_t(i)];
        if (pp_enabled() && (i < pp_begin || i >= pp_end)) continue;

        // Qwen4-Exp has NEITHER of these: the two hyper-connections'
        // grouped norms stand where they would be.  Asking for them
        // reports a missing tensor for something the file correctly does
        // not contain.
        if (!cfg.is_qwen4_exp) {
            d.in_norm   = dev_copy_t<bf16_t>(lq, ck, src.input_norm, "input_layernorm", &ok);
            d.post_norm = dev_copy_t<bf16_t>(lq, ck, src.post_attn_norm, "post_attention_layernorm", &ok);
        }
        if(cfg.is_muse){
            d.in_norm_f16=upload_f16_vector_t(lq,ck,src.input_norm,
                                               "input_layernorm.fp16",&ok);
            d.post_norm_f16=upload_f16_vector_t(lq,ck,src.post_attn_norm,
                                                 "post_attention_layernorm.fp16",&ok);
        }

        // ---- Qwen4-Exp: the residual structure, QSA and PLE ----------
        // The hyper-connection projections stay BF16 because the
        // reference builds them with quant_config=None -- they are the
        // gate that decides how much of every block reaches every stream,
        // they are tiny next to a single expert, and quantizing them
        // would be this engine's choice rather than the model's.
        if (cfg.is_qwen4_exp) {
            const int HC = cfg.hc_count, WIDE = HC * H;
            auto up_hc = [&](LayerDev::HCDev& dst,
                             const Qwen35Layer::HCRef& sref, const char* tag) {
                dst.norm = dev_copy_t<bf16_t>(lq, ck, sref.norm,
                                              (std::string(tag)+".hc_norm").c_str(), &ok);
                dst.down = quantize_upload_t(lq, ck, sref.down, Fmt::BF16,
                                             (std::string(tag)+".down").c_str(), &ok);
                if (sref.inject.ok())
                    dst.inject = quantize_upload_t(lq, ck, sref.inject, Fmt::BF16,
                                              (std::string(tag)+".inject").c_str(), &ok);
                dst.up   = quantize_upload_t(lq, ck, sref.up, Fmt::BF16,
                                             (std::string(tag)+".up").c_str(), &ok);
            };
            up_hc(d.hc_attn, src.hc_attn, "attn_hyper_connection");
            up_hc(d.hc_mlp,  src.hc_mlp,  "mlp_hyper_connection");
            acct(size_t(d.hc_attn.down.w.bytes() + d.hc_attn.up.w.bytes() +
                        d.hc_attn.inject.w.bytes() + d.hc_mlp.down.w.bytes() +
                        d.hc_mlp.up.w.bytes() + d.hc_mlp.inject.w.bytes()));

            d.qsa = src.qsa;
            if (d.qsa) {
                const int IHD = cfg.indexer_head_dim;
                d.ix_qk = quantize_upload_t(lq, ck, src.ix_qk_proj, Fmt::BF16,
                                            "indexer.index_qk_proj", &ok);
                d.ix_qn = dev_copy_t<bf16_t>(lq, ck, src.ix_q_norm,
                                             "indexer.q_layernorm", &ok);
                d.ix_kn = dev_copy_t<bf16_t>(lq, ck, src.ix_k_norm,
                                             "indexer.k_layernorm", &ok);
                d.ix_raw_base = d.ix_kraw = sycl::malloc_device<float>(size_t(n_seq_slots)*max_seq*IHD, lq);
                const int nb = max_seq / cfg.indexer_compress_ratio + 1;
                d.ix_cmp_base = d.ix_kcmp = sycl::malloc_device<float>(size_t(n_seq_slots)*nb*IHD, lq);
                if (!d.ix_kraw || !d.ix_kcmp) {
                    err = "QSA indexer key cache allocation failed"; return false;
                }
                acct(size_t(max_seq + nb) * IHD * 4);
            }

            d.ple = src.ple;
            d.ple_dense_id = src.ple_dense_id;
            if (d.ple) {
                const int NH = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
                d.ple_key   = quantize_upload_t(lq, ck, src.ple_key, PF,
                                                "ple.key_proj", &ok);
                d.ple_value = quantize_upload_t(lq, ck, src.ple_value, PF,
                                                "ple.value_proj", &ok);
                d.ple_nk = dev_copy_t<bf16_t>(lq, ck, src.ple_norm_key,
                                              "ple.norm_key", &ok);
                d.ple_nq = dev_copy_t<bf16_t>(lq, ck, src.ple_norm_query,
                                              "ple.norm_query", &ok);
                d.ple_nc = dev_copy_t<bf16_t>(lq, ck, src.ple_norm_conv,
                                              "ple.norm_conv", &ok);
                d.ple_cw = dev_copy_t<bf16_t>(lq, ck, src.ple_conv1d,
                                              "ple.conv1d", &ok);
                const int state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size;
                d.ple_hist_base = d.ple_hist = sycl::malloc_device<float>(
                    size_t(n_seq_slots) * (state_len > 0 ? state_len : 1) * WIDE, lq);
                if (!d.ple_hist) { err="PLE history allocation failed"; return false; }

                // The n-gram table is the ONE weight in this engine that
                // does NOT go to the card.  20,000,000 rows per head slice
                // is the architecture's design and vLLM pins it in host
                // memory for exactly this reason; the gather touches only
                // the rows of the current tokens.
                if (!src.ple_table.ok() || src.ple_table.t.shape.size() != 2) {
                    err = "ple_embedding.ngram_embedding has an unexpected shape";
                    return false;
                }
                // BF16, or FP8-E4M3 with one global scale.  That second
                // case is not an optimisation: the published table is
                // ~51B parameters, which is 51 GB of host memory at FP8
                // and 102 GB at BF16.
                const bool tfp8 = src.ple_table.t.dtype == STDtype::F8_E4M3;
                if (!tfp8 && src.ple_table.t.dtype != STDtype::BF16) {
                    err = "the PLE n-gram table is neither BF16 nor FP8-E4M3; "
                          "this engine reads that table straight out of host "
                          "memory and has a dequantiser for those two only";
                    return false;
                }
                float tscale = 1.0f;
                if (tfp8) {
                    // The reference REFUSES an FP8 table with no scale
                    // (process_weights_after_loading) rather than
                    // defaulting to 1.0, and so does this: an unscaled
                    // table is a differently-weighted embedding, which is
                    // fluent.
                    if (!src.ple_table_scale.ok()) {
                        err = "the PLE n-gram table is FP8 but ships no "
                              "weight_scale; reading it unscaled would be a "
                              "different embedding, not an error";
                        return false;
                    }
                    if (!read_scalar_f32(ck, src.ple_table_scale, tscale, err)) {
                        err = "reading the PLE table's global scale failed: " + err;
                        return false;
                    }
                }
                const int64_t rows = int64_t(src.ple_table.t.shape[0]);
                const int64_t wid  = int64_t(src.ple_table.t.shape[1]);
                const size_t  esz  = tfp8 ? 1 : sizeof(bf16_t);
                void* host = sycl::malloc_host(size_t(rows*wid) * esz, lq);
                if (!host) { err = "PLE host table allocation failed"; return false; }
                // read_raw, NOT ck.data(): the file mappings were dropped
                // above (unmap_all), so every read after that point is a
                // pread and dereferencing the mapping is a segfault.
                if (!ck.read_raw(src.ple_table, host, err)) {
                    err = "reading the PLE n-gram table failed: " + err;
                    return false;
                }
                d.ple_table = host;
                d.ple_fp8   = tfp8;
                d.ple_scale = tscale;
                d.ple_rows  = rows;

                // Layout and multipliers are DERIVED, never stored.
                std::vector<int64_t> mul(size_t(cfg.ngram_size), 0),
                                     sz(size_t(NH), 0), of(size_t(NH), 0);
                qwen4_exp::ngram_multipliers(cfg.ngram_size, cfg.vocab,
                                             cfg.ngram_seed, d.ple_dense_id,
                                             mul.data());
                const int64_t total = qwen4_exp::ngram_vocab_layout(
                    cfg.ngram_vocab_base, NH, d.ple_dense_id, sz.data(), of.data());
                if (total > rows) {
                    err = "the PLE table is smaller than the derived n-gram "
                          "layout needs; ngram_vocab_size_base or the head "
                          "count does not match this checkpoint";
                    return false;
                }
                d.ple_mul  = sycl::malloc_device<int64_t>(mul.size(), lq);
                d.ple_size = sycl::malloc_device<int64_t>(sz.size(), lq);
                d.ple_off  = sycl::malloc_device<int64_t>(of.size(), lq);
                lq.memcpy(d.ple_mul,  mul.data(), mul.size()*sizeof(int64_t));
                lq.memcpy(d.ple_size, sz.data(),  sz.size()*sizeof(int64_t));
                lq.memcpy(d.ple_off,  of.data(),  of.size()*sizeof(int64_t)).wait();
                acct(size_t(d.ple_key.w.bytes() + d.ple_value.w.bytes()));
            }
        }

        if (d.kind == LayerKind::LINEAR_ATTN) {
            // The three big ones. 67 MB/layer in bf16 -> ~18 MB at int4.
            d.la_qkv = quantize_upload_t(lq, ck, src.la_in_qkv, PF, "la.in_proj_qkv", &ok);
            d.la_z   = quantize_upload_t(lq, ck, src.la_in_z,   PF, "la.in_proj_z",   &ok);
            d.la_out = quantize_upload_t(lq, ck, src.la_out,    PF, "la.out_proj",    &ok);
            shard(d.la_qkv,lq);shard(d.la_z,lq);shard(d.la_out,lq);
            acct(size_t(d.la_qkv.w.bytes() + d.la_z.w.bytes() + d.la_out.w.bytes()
                      + d.la_ab.w.bytes()));

            // small tensors stay bf16
            // a and b are both [Hv][H] and feed the same gate kernel;
            // concatenating them into [2*Hv][H] at load turns two tiny
            // GEMV launches into one for every linear layer.
            d.la_ab = concat_upload_t(lq, ck, src.la_in_a, src.la_in_b,
                                      Fmt::BF16, "la.in_proj_ab", &ok);
            shard(d.la_ab,lq);
            // Experimental only: the single wider GEMM measured slower on B70
            // than the three specialized shapes.  Do not spend VRAM on the
            // concatenated copy in the production path.
            if(!tp_enabled()&&PF==Fmt::MXFP4&&std::getenv("GRIMOIRE_FUSE_DN_PROJECTIONS"))
                d.la_all=concat4_native_mxfp4_t(lq,ck,src.la_in_qkv,src.la_in_z,
                    src.la_in_a,src.la_in_b,"la.in_proj_qkv_z_ab",&ok);
            d.la_conv = dev_copy_t<bf16_t>(lq, ck, src.la_conv1d, "la.conv1d", &ok);
            d.la_Alog = dev_copy_t<bf16_t>(lq, ck, src.la_A_log, "la.A_log", &ok);
            d.la_dtb  = dev_copy_t<bf16_t>(lq, ck, src.la_dt_bias, "la.dt_bias", &ok);
            d.la_norm = dev_copy_t<bf16_t>(lq, ck, src.la_norm, "la.norm", &ok);

            // recurrent state: constant in context length
            const int conv_ch = src.la_conv1d.ok() && !src.la_conv1d.t.shape.empty()
                              ? int(src.la_conv1d.t.shape[0]) : qkv_ch;
            const int conv_k  = src.la_conv1d.ok() && src.la_conv1d.t.shape.size() >= 3
                              ? int(src.la_conv1d.t.shape[2]) : cfg.conv_kernel;
            d.dn_slot   = size_t(Hv) * Dv * Dk;
            d.conv_slot = size_t(conv_ch) * (conv_k - 1);
            // One copy per sequence slot.  Constant in context length, so
            // N of them is cheap next to N KV caches -- and without them a
            // hybrid model cannot have two conversations resident at all.
            d.dn_base   = sycl::malloc_device<float>(d.dn_slot * size_t(n_seq_slots), lq);
            d.conv_base = sycl::malloc_device<float>(d.conv_slot * size_t(n_seq_slots), lq);
            d.dn_state  = d.dn_base;
            d.conv_ring = d.conv_base;
            acct((d.dn_slot + d.conv_slot) * 4 * size_t(n_seq_slots));
        } else {
            d.q_proj = quantize_upload_t(lq, ck, src.q_proj, PF, "self_attn.q_proj", &ok);
            d.k_proj = quantize_upload_t(lq, ck, src.k_proj, PF, "self_attn.k_proj", &ok);
            // K2: only a DENSE layer has v_proj.  On a sparse layer MoVA
            // replaces it entirely, so asking for it would report a missing
            // tensor for something the checkpoint correctly does not have.
            const bool k2_mova = cfg.is_k2 && src.k2_sparse && cfg.mova_experts > 0;
            // gemma-4 attention_k_eq_v: a FULL-attention layer ships no
            // v_proj either, and V is the k_proj output.  The loader has
            // already checked that the checkpoint agrees with the config
            // in both directions, so trust src here -- asking for a tensor
            // the model correctly does not have fails the whole upload and
            // the mixed-layer checkpoint could never load.
            const bool g4_k_eq_v = cfg.is_gemma4 && !src.v_proj.ok();
            const bool no_v = k2_mova || g4_k_eq_v;
            if (!no_v)
                d.v_proj = quantize_upload_t(lq, ck, src.v_proj, PF, "self_attn.v_proj", &ok);
            shard(d.q_proj,lq);shard(d.k_proj,lq);
            if (!no_v) shard(d.v_proj,lq);
            if (cfg.is_k2) {
                d.k2_sparse = src.k2_sparse;
                // Softplus output gate, present on EVERY K2 layer.
                if (cfg.attn_gate && src.attn_gate.ok())
                    d.o_gate = quantize_upload_t(lq, ck, src.attn_gate, PF,
                        "self_attn.gate_proj", &ok);
                if (k2_mova) {
                    // BF16 and unsharded: 0.16 MB, and N=64 cannot go on a
                    // 256-wide tile.
                    d.v_router = quantize_upload_t(lq, ck, src.v_router,
                        Fmt::BF16, "self_attn.v_router", &ok);
                    if (src.v_router_bias.ok())
                        d.v_router_bias = dev_copy_t<bf16_t>(lq, ck, src.v_router_bias,
                            "self_attn.v_router.bias", &ok);
                    // GRIMOIRE_MOVA_PER_EXPERT=1 keeps the old per-expert
                    // path so the two can be run against each other -- on
                    // output first (they must be identical; bin/test_k2_e2e
                    // checks that) and then, on the card, on time.  Read
                    // through getenv on every load, not a function-local
                    // static, so a single process can load both ways.
                    const char* per_ex = std::getenv("GRIMOIRE_MOVA_PER_EXPERT");
                    const bool want_pack =
                        !tp_enabled() && !(per_ex && *per_ex && *per_ex != '0');
                    if (want_pack) {
                        // Expert-major: one [E*N][K] weight, so the routed
                        // value projection reads the routing table on the
                        // DEVICE instead of stalling on it once per layer.
                        // Same bytes as the E separate weights -- this is a
                        // different layout, not a second copy.
                        std::vector<TensorRef> refs;
                        refs.reserve(size_t(cfg.mova_experts));
                        for (int e = 0; e < cfg.mova_experts; ++e)
                            refs.push_back(src.v_experts[size_t(e)]);
                        // The packer goes through read_matrix_f32 +
                        // quantize.  quantize_upload_t does NOT for a
                        // checkpoint that already ships quantized weights:
                        // it imports a native .b70, a compressed-INT4, a
                        // GPTQ or a row-scaled FP8 tensor directly,
                        // because saved precision is authoritative.
                        // Packing such a tensor would DECODE AND RE-QUANTIZE
                        // it -- a different model, not a different layout,
                        // and since TP keeps the per-expert path the two
                        // would load different effective models from one
                        // checkpoint.  Refuse to pack those and say so;
                        // a silent fall back to the slower path is its own
                        // bug.  (BF16 is not in this list: bf16 -> f32 ->
                        // bf16 is exact, so packing it changes nothing.)
                        bool saved_precision = false;
                        for (const auto& r : refs)
                            if (r.native || r.compressed_int4 || r.gptq ||
                                r.row_scaled) { saved_precision = true; break; }
                        // Quantized per expert and appended: 64 experts
                        // held as f32 at once would be a 64x load-time
                        // spike on exactly the box that has no room.
                        if (saved_precision) {
                            static bool said = false;
                            if (!said) {
                                said = true;
                                std::printf("\n  MoVA: experts ship saved "
                                    "precision (native/int4/gptq/fp8) -- keeping "
                                    "the per-expert path so the checkpoint's own "
                                    "weights are used verbatim; the packed path "
                                    "would re-quantize them\n");
                            }
                        } else
                        d.v_experts_packed = concat_rows_quantized_t(
                            lq, ck, refs, PF, "self_attn.v_experts.packed", &ok);
                        if (ok && d.v_experts_packed.w.N > 0) {
                            const int rows = d.v_experts_packed.w.N;
                            if (cfg.mova_experts <= 0 || rows % cfg.mova_experts) {
                                // Refuse rather than divide wrongly: every
                                // index below is expert*N + row, so a bad N
                                // reads another expert's weights and the
                                // output stays plausible.
                                err = "MoVA packed experts have " +
                                      std::to_string(rows) + " rows, not a "
                                      "multiple of " +
                                      std::to_string(cfg.mova_experts) +
                                      " experts";
                                return false;
                            }
                            d.v_experts_n = rows / cfg.mova_experts;
                            acct(d.v_experts_packed.w.bytes());
                        }
                    }
                    if (!want_pack || d.v_experts_packed.w.N == 0) {
                        d.v_experts.resize(size_t(cfg.mova_experts));
                        for (int e = 0; e < cfg.mova_experts && ok; ++e) {
                            const std::string vn =
                                "self_attn.v_experts." + std::to_string(e);
                            d.v_experts[size_t(e)] = quantize_upload_t(
                                lq, ck, src.v_experts[size_t(e)], PF, vn.c_str(), &ok);
                            shard(d.v_experts[size_t(e)], lq);
                            acct(d.v_experts[size_t(e)].w.bytes());
                        }
                    }
                }
            }
            // Fused QKV for prefill_muse's W4A16 path.  The condition was
            // the REQUESTED format (PF == INT4) while
            // concat_upload_many_int4_t requires the SOURCE tensors to
            // already be compressed INT4 -- so asking for int4 on a Muse
            // checkpoint that is not compressed failed the whole upload
            // with "direct compressed INT4 concatenate failed", and the
            // model would not load at all.  Third format assumption in the
            // Muse path, after the FFN resolve and the FFN fallback.
            //
            // Test the SOURCE.  When it is not compressed the fusion is
            // simply skipped: q/k/v stay separate, which is what every
            // other architecture uses, and prefill_muse's W4A16 path
            // already declines and falls back when qkv_proj is absent.
            if(cfg.is_muse&&PF==Fmt::INT4&&!tp_enabled()&&
               src.q_proj.compressed_int4&&src.k_proj.compressed_int4&&
               src.v_proj.compressed_int4){
                d.qkv_proj=concat_upload_many_int4_t(lq,ck,
                    {src.q_proj,src.k_proj,src.v_proj},"self_attn.qkv_proj",&ok);
            }
            d.o_proj = quantize_upload_t(lq, ck, src.o_proj, PF, "self_attn.o_proj", &ok);
            shard(d.o_proj,lq);
            // Per-head q/k RMSNorm, applied before RoPE. These were
            // resolved from the checkpoint but never uploaded or used.
            if (src.q_norm.ok())
                d.q_norm = dev_copy_t<bf16_t>(lq, ck, src.q_norm, "self_attn.q_norm", &ok);
            if (src.k_norm.ok())
                d.k_norm = dev_copy_t<bf16_t>(lq, ck, src.k_norm, "self_attn.k_norm", &ok);
            if (cfg.is_gemma4) {
                // Same sandwich shape as Muse, without the f16 companions:
                // gemma-4 has no Fusion FlashAttention path, so nothing
                // reads the half-precision copies.
                d.pre_ff_norm  = dev_copy_t<bf16_t>(lq, ck, src.pre_ff_norm,  "mlp.pre_ff_norm",  &ok);
                d.post_ff_norm = dev_copy_t<bf16_t>(lq, ck, src.post_ff_norm, "mlp.post_ff_norm", &ok);
                // hidden_states *= layer_scalar, AFTER the residual add, as
                // the last act of the layer.  A [1] tensor, read to the host
                // once here: it is a per-layer constant, so keeping it on the
                // device would cost a kernel launch per layer to apply.
                float ls = 1.0f;
                if (!read_matrix_f32(ck, src.layer_scalar, &ls, err)) ok = false;
                d.layer_scalar = ls;
            }
            if (cfg.is_muse) {
                d.pre_ff_norm  = dev_copy_t<bf16_t>(lq, ck, src.pre_ff_norm,  "mlp.pre_ff_norm",  &ok);
                d.post_ff_norm = dev_copy_t<bf16_t>(lq, ck, src.post_ff_norm, "mlp.post_ff_norm", &ok);
                d.pre_ff_norm_f16=upload_f16_vector_t(lq,ck,src.pre_ff_norm,
                                                       "mlp.pre_ff_norm.fp16",&ok);
                d.post_ff_norm_f16=upload_f16_vector_t(lq,ck,src.post_ff_norm,
                                                        "mlp.post_ff_norm.fp16",&ok);
                if (src.attn_gate.ok())
                    d.o_gate = quantize_upload_t(lq, ck, src.attn_gate, PF, "self_attn.gate_proj", &ok);
                shard(d.o_gate,lq);
            }
            acct(size_t(d.q_proj.w.bytes() + d.k_proj.w.bytes()
                      + d.v_proj.w.bytes() + d.qkv_proj.w.bytes()
                      + d.o_proj.w.bytes()));

            // FP8 E4M3 KV: K is D-major for coalesced scoring, V is D-minor.
            //
            // Sized from THIS LAYER's geometry, not the model's.  On
            // gemma-4 a full-attention layer is 4 KV heads of 512 where a
            // sliding one is 16 of 256, so a single model-wide size is
            // either too small for one kind or wasteful for the other.
            // For every other architecture layer_* returns the model-wide
            // value and this expression is exactly what it was.
            // Geometry was resolved for every layer right after L.resize().
            const size_t kv_elems = size_t(d.kv_heads) * d.head_dim * max_seq;
            // n_seq_slots copies, so several conversations can be resident
            // at once and switching between them is a pointer move rather
            // than a copy of the whole cache.  At the default of one slot
            // this allocates exactly what it always did.
            d.kv_slot = kv_elems;
            d.k_base = sycl::malloc_device<uint8_t>(kv_elems * size_t(n_seq_slots), lq);
            d.v_base = sycl::malloc_device<uint8_t>(kv_elems * size_t(n_seq_slots), lq);
            d.k_cache = d.k_base;
            d.v_cache = d.v_base;
            acct(2 * kv_elems * size_t(n_seq_slots));
            if(cfg.is_muse){
                // Match Fusion's XPU FlashAttention KV-cache group.
                constexpr int block_size=64;
                const int blocks=(max_seq+block_size-1)/block_size;
                const size_t elems=size_t(blocks)*block_size*d.kv_heads*
                    d.head_dim;
                d.k_cache_f16=sycl::malloc_device<sycl::half>(elems,lq);
                d.v_cache_f16=sycl::malloc_device<sycl::half>(elems,lq);
                if(!d.k_cache_f16||!d.v_cache_f16)ok=false;
                acct(2*elems*sizeof(sycl::half));
            }
        }

        // ---- FFN ------------------------------------------------------
        // cfg.is_moe() is a property of the MODEL; being routed is a
        // property of the LAYER.  K2 is the first architecture where those
        // differ: layers in mlp_only_layers have no router and no experts,
        // just one intermediate_size MLP, which the loader already maps
        // into sh_gate/sh_up/sh_down.  Asking such a layer for mlp.gate
        // fails the upload -- and the engine then walked on and
        // dereferenced the router it never got.
        d.moe_layer = cfg.is_moe() && !(cfg.is_k2 && !src.k2_sparse);
        if (d.moe_layer) {
            const int I=cfg.moe_inter,Efull=cfg.n_experts;
            d.expert_begin=tp_enabled()?(Efull*tp_rank)/tp_world:0;
            const int expert_end=tp_enabled()?(Efull*(tp_rank+1))/tp_world:Efull;
            const int E=expert_end-d.expert_begin;
            d.expert_count=E;
            if(E<1){err="TP world size exceeds expert count";return false;}
            d.router = quantize_upload_t(lq, ck, src.router, Fmt::BF16, "mlp.gate", &ok);
            // K2's router bias steers SELECTION only; without it the top-k
            // is taken on unbiased scores and the routes are wrong.
            if (cfg.is_k2 && src.router_bias.ok())
                d.router_bias = dev_copy_t<bf16_t>(lq, ck, src.router_bias,
                    "mlp.gate.bias", &ok);
            shard(d.router,lq);

            // Experts: concatenate gate and up into one [E][2I][H] block
            // and copy the packed bytes verbatim. No dequantize, no
            // requantize -- the on-disk layout IS the kernel layout.
            Fmt EF = (!src.e_gate_p.empty() && src.e_gate_p[0].gptq) ? Fmt::INT4 : Fmt::MXFP4;
            if (!src.e_gate_p.empty() && src.e_gate_p[0].native) {
                QuantWeight view;
                if (!ck.native_view(src.e_gate_p[0],view,err)) return false;
                EF=view.fmt;
            }
            const size_t gu_row  = bytes_per_row(EF, H);
            const size_t gu_srow = scales_per_row(EF, H) * scale_element_bytes(EF);
            const size_t dn_row  = bytes_per_row(EF, I);
            const size_t dn_srow = scales_per_row(EF, I) * scale_element_bytes(EF);
            const size_t gu_zrow = EF == Fmt::INT4 ? scales_per_row(EF, H) : 0;
            const size_t dn_zrow = EF == Fmt::INT4 ? scales_per_row(EF, I) : 0;

            d.gu_pack  = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_row, lq);
            if (gu_srow) d.gu_scale = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_srow, lq);
            d.dn_pack  = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_row, lq);
            if (dn_srow) d.dn_scale = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_srow, lq);
            if (gu_zrow) d.gu_zero = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_zrow, lq);
            if (dn_zrow) d.dn_zero = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_zrow, lq);
            if(!d.gu_pack||!d.dn_pack||(gu_srow&&!d.gu_scale)||(dn_srow&&!d.dn_scale)||
               (gu_zrow&&!d.gu_zero)||(dn_zrow&&!d.dn_zero)){
                err="expert shard allocation failed";return false;
            }

            if (!d.gu_pack || !d.dn_pack || (gu_srow && !d.gu_scale) || (dn_srow && !d.dn_scale) ||
                (gu_zrow && !d.gu_zero) || (dn_zrow && !d.dn_zero)) {
                err="expert device allocation failed"; return false;
            }
            // Same mmap hazard: stage each expert through host memory.
            // Build the whole layer's block on the host, then do four
            // large device copies instead of 1536 small ones.
            {
                std::vector<uint8_t> h_gu(size_t(E) * 2 * I * gu_row);
                std::vector<uint8_t> h_gs(size_t(E) * 2 * I * gu_srow);
                std::vector<uint8_t> h_dn(size_t(E) * H * dn_row);
                std::vector<uint8_t> h_ds(size_t(E) * H * dn_srow);
                std::vector<uint8_t> h_gz(size_t(E) * 2 * I * gu_zrow);
                std::vector<uint8_t> h_dz(size_t(E) * H * dn_zrow);
                auto stage_expert = [&](const TensorRef& r, uint8_t* payload,
                                        uint8_t* scales, uint8_t* zeros, int N, int K,
                                        size_t row_bytes, size_t scale_bytes, size_t zero_bytes,
                                        std::string& rr) {
                    if(r.row_scaled&&r.scales_t.numel()==N&&
                       ((EF==Fmt::FP8_E4M3&&r.t.dtype==STDtype::F8_E4M3)||
                        (EF==Fmt::FP8_E5M2&&r.t.dtype==STDtype::F8_E5M2))){
                        std::vector<float> hs(size_t(N),0.0f);
                        if(!ck.read_raw(r,payload,rr)||
                           !ck.shards[r.scales_shard]->read_f32(r.scales_t,hs.data(),rr))return false;
                        std::memcpy(scales,hs.data(),size_t(N)*sizeof(float));return true;
                    }
                    if(EF==Fmt::BF16&&!r.row_scaled&&!r.native&&!r.nvfp4&&r.t.dtype==STDtype::BF16)
                        return ck.read_raw(r,payload,rr);
                    // !r.nvfp4: an NVFP4 expert is also called
                    // weight_packed and also holds E2M1 nibbles; reading
                    // it straight would hand E4M3 per-16 scales to a
                    // kernel expecting E8M0 per-32.  b70/nvfp4.hpp.
                    const bool packed = EF==Fmt::MXFP4&&!r.gptq&&!r.row_scaled&&!r.nvfp4&&
                                        r.t.name.find("weight_packed") != std::string::npos;
                    if (packed) return ck.read_raw(r, payload, rr);
                    PackedWeight pw;
                    if (r.native) {
                        QuantWeight view;
                        if (!ck.native_view(r,view,rr)) return false;
                        if (view.fmt!=EF || view.N!=N || view.K!=K) {
                            rr="native expert format or shape differs within layer"; return false;
                        }
                        pw=copy_packed(view);
                    } else if (r.gptq) {
                        if (!repack_gptq_ref(ck, r, pw, rr)) return false;
                    } else {
                        std::vector<float> f32(size_t(N) * K);
                        if (!read_matrix_f32(ck, r, f32.data(), rr)) return false;
                        pw = quantize(f32.data(), N, K, EF);
                    }
                    if (pw.payload.size() != size_t(N) * row_bytes ||
                        pw.scales_raw.size() != size_t(N) * scale_bytes) {
                        rr = "expert requantized layout mismatch"; return false;
                    }
                    std::memcpy(payload, pw.payload.data(), pw.payload.size());
                    if (!pw.scales_raw.empty()) std::memcpy(scales, pw.scales_raw.data(), pw.scales_raw.size());
                    if (zero_bytes) {
                        if (pw.zeros.size() != size_t(N) * zero_bytes) {
                            rr = "expert zero layout mismatch"; return false;
                        }
                        std::memcpy(zeros, pw.zeros.data(), pw.zeros.size());
                    }
                    return true;
                };
                for (int le=0;le<E;++le) {
                    const int e=d.expert_begin+le;
                    const size_t goff=size_t(le)*2*I;
                    if (!src.e_gate_p[e].ok() || !src.e_up_p[e].ok() ||
                        !src.e_down_p[e].ok()) {
                        std::printf("\n  missing expert %d in layer %d\n", e, i);
                        err = "expert tensors incomplete";
                        return false;
                    }
                    std::string rr;
                    const bool direct=EF==Fmt::MXFP4&&!src.e_gate_p[e].gptq&&!src.e_gate_p[e].row_scaled&&
                                        !src.e_gate_p[e].nvfp4&&!src.e_up_p[e].nvfp4&&
                                        src.e_gate_p[e].t.name.find("weight_packed") != std::string::npos;
                    bool rok;
                    if (direct) {
                        rok = ck.read_raw(src.e_gate_p[e], h_gu.data() + goff * gu_row, rr)
                           && ck.read_raw(src.e_up_p[e], h_gu.data() + (goff + I) * gu_row, rr)
                           && ck.read_raw(src.e_gate_s[e], h_gs.data() + goff * gu_srow, rr)
                           && ck.read_raw(src.e_up_s[e], h_gs.data() + (goff + I) * gu_srow, rr)
                           && ck.read_raw(src.e_down_p[e],h_dn.data()+size_t(le)*H*dn_row,rr)
                           && ck.read_raw(src.e_down_s[e],h_ds.data()+size_t(le)*H*dn_srow,rr);
                    } else {
                        rok = stage_expert(src.e_gate_p[e], h_gu.data() + goff * gu_row,
                                           gu_srow ? h_gs.data() + goff * gu_srow : nullptr, gu_zrow ? h_gz.data() + goff * gu_zrow : nullptr,
                                           I, H, gu_row, gu_srow, gu_zrow, rr)
                           && stage_expert(src.e_up_p[e], h_gu.data() + (goff + I) * gu_row,
                                           gu_srow ? h_gs.data() + (goff + I) * gu_srow : nullptr, gu_zrow ? h_gz.data() + (goff + I) * gu_zrow : nullptr,
                                           I, H, gu_row, gu_srow, gu_zrow, rr)
                           && stage_expert(src.e_down_p[e], h_dn.data() + size_t(le) * H * dn_row,
                                           dn_srow ? h_ds.data() + size_t(le) * H * dn_srow : nullptr, dn_zrow ? h_dz.data() + size_t(le) * H * dn_zrow : nullptr,
                                           H, I, dn_row, dn_srow, dn_zrow, rr);
                    }
                    if (!rok) { err = "expert read failed: " + rr; return false; }
                }
                // Intel's persistent grouped GEMM consumes signed s4. GPTQ
                // stores affine u4, and AutoGPTQ encodes the zero point minus
                // one. A zero point of exactly 8 makes the conversion lossless:
                // subtracting 8 in two's-complement is one XOR per nibble.
                // Mark the zero arrays with 0xff so the decode GEMV path reads
                // this same in-place payload as signed s4 (no multi-GiB copy).
                if (EF == Fmt::INT4) {
                    const bool all_zp8 =
                        std::all_of(h_gz.begin(), h_gz.end(), [](uint8_t z){return z==8;}) &&
                        std::all_of(h_dz.begin(), h_dz.end(), [](uint8_t z){return z==8;});
                    if (all_zp8) {
                        for (uint8_t& v : h_gu) v ^= 0x88;
                        for (uint8_t& v : h_dn) v ^= 0x88;
                        std::fill(h_gz.begin(), h_gz.end(), uint8_t(0xff));
                        std::fill(h_dz.begin(), h_dz.end(), uint8_t(0xff));
                        d.xe2_signed_int4 = true;
                    }
                }
                lq.memcpy(d.gu_pack,  h_gu.data(), h_gu.size());
                if (d.gu_scale) lq.memcpy(d.gu_scale, h_gs.data(), h_gs.size());
                lq.memcpy(d.dn_pack,  h_dn.data(), h_dn.size());
                if (d.dn_scale) lq.memcpy(d.dn_scale, h_ds.data(), h_ds.size());
                if (d.gu_zero) lq.memcpy(d.gu_zero, h_gz.data(), h_gz.size());
                if (d.dn_zero) lq.memcpy(d.dn_zero, h_dz.data(), h_dz.size());
                lq.wait_and_throw();
            }
            acct(size_t(E) * (2 * I * (gu_row + gu_srow + gu_zrow)
                            + H * (dn_row + dn_srow + dn_zrow)));

            d.moe.cfg.hidden = H; d.moe.cfg.inter = I;
            d.moe.cfg.num_experts = E; d.moe.cfg.top_k = cfg.top_k;
            d.moe.cfg.shared_inter = cfg.shared_inter;
            d.moe.gate_up = QuantWeight{EF, E * 2 * I, H, d.gu_pack, d.gu_scale,
                                        d.gu_zero, int64_t(gu_row), scales_per_row(EF, H)};
            d.moe.down    = QuantWeight{EF, E * H, I, d.dn_pack, d.dn_scale,
                                        d.dn_zero, int64_t(dn_row), scales_per_row(EF, I)};

            // gate and up share an input and are consumed together by
            // SwiGLU: one [2*I][H] matrix, one launch.
            d.sh_gu   = concat_upload_t(lq, ck, src.sh_gate, src.sh_up, PF,
                                        "shared.gate_up", &ok);
            d.sh_down = quantize_upload_t(lq, ck, src.sh_down, PF, "shared.down_proj", &ok);
            shard(d.sh_gu,lq);shard(d.sh_down,lq);
            if (src.sh_gate_w.ok()) {
                d.sh_gate_q = quantize_upload_t(lq, ck, src.sh_gate_w, Fmt::BF16, "shared_expert_gate", &ok);
                shard(d.sh_gate_q,lq);
                d.has_sh_gate = true;
            }
            acct(size_t(d.sh_gu.w.bytes() + d.sh_down.w.bytes()));
        } else if (src.pf_gate.ok() && src.pf_up.ok() && src.pf_down.ok()) {
            // Agnes: fold the parallel SwiGLU into the main one.  The
            // SwiGLU kernel reads gate_up as [all gate rows][all up rows]
            // and takes the intermediate width from d.sh_gu, so the row
            // order is gate, pgate, up, pup -- NOT gate, up, pgate, pup.
            // down is concatenated along K, per row, because row-major
            // rows interleave; appending the bytes would build a
            // different matrix that still has the right shape.
            d.sh_gu   = concat_rows_f32_t(lq, ck,
                {src.sh_gate, src.pf_gate, src.sh_up, src.pf_up}, PF,
                "mlp.gate_up+parallel_ffn", &ok);
            d.sh_down = concat_cols_f32_t(lq, ck, src.sh_down, src.pf_down, PF,
                                          "mlp.down_proj+parallel_ffn", &ok);
            shard(d.sh_gu,lq);shard(d.sh_down,lq);
            acct(size_t(d.sh_gu.w.bytes() + d.sh_down.w.bytes()));
        } else {
            d.sh_gu   = concat_upload_t(lq, ck, src.sh_gate, src.sh_up, PF,
                                        "mlp.gate_up", &ok);
            d.sh_down = quantize_upload_t(lq, ck, src.sh_down, PF, "mlp.down_proj", &ok);
            shard(d.sh_gu,lq);shard(d.sh_down,lq);
            acct(size_t(d.sh_gu.w.bytes() + d.sh_down.w.bytes()));
        }

        // ---- W4A8 weights (opt-in) -----------------------------------
        // Symmetric int4 g128 copies of the two FFN matrices, converted on
        // device from the MXFP4 originals.  Opt-in because it costs ~9 GB of
        // VRAM on top of the model: the MXFP4 weights stay because decode's
        // GEMV beats any GEMM at M=1.
        if (w4a8_enabled() && w4a8_tiles_available() && ok) {
            // Convert EVERY projection, not just the FFN.  Measured with the
            // M16 tile: la_qkv 2.78 ms at M=4 against the decode GEMV's 3.44,
            // q+gate 1.03 vs 1.20 -- and flat all the way to M=16, which is
            // what makes a speculative verify batch affordable.
            auto conv = [&](DevQuant& dq, const char* what) {
                if (dq.w.fmt != Fmt::MXFP4 || !dq.payload) return true;
                const int N = dq.w.N, K = dq.w.K;
                if (K % 128) return true;
                // Every W4A8 tile is 256 wide in N, and the B 2-D block loads
                // do NOT clamp to the tensor: a shape whose N is not a
                // multiple of 256 reads past the end and faults the device.
                // la_ab is N=96 -- 96 rows of a 245 KB tensor, read as 256 --
                // which is a DEVICE_LOST, and it is only 0.25 MB, so leave it
                // on the MXFP4 GEMV where it is already fast.
                if (N % 256) return true;
                uint8_t* pack = sycl::malloc_device<uint8_t>(size_t(N) * (K / 2), lq);
                float*   ws   = sycl::malloc_device<float>(size_t(N) * (K / 128), lq);
                if (!pack || !ws) {
                    err = std::string("W4A8 allocation failed for ") + what;
                    return false;
                }
                launch_mxfp4_to_int4sym(lq, dq.w.payload,
                    static_cast<const uint8_t*>(dq.w.scales),
                    dq.w.row_bytes, dq.w.row_scales, pack, ws, N, K, {}).wait();
                const size_t freed = dq.w.bytes();
                if (dq.payload) { sycl::free(dq.payload, lq); dq.payload = nullptr; }
                if (dq.scales)  { sycl::free(dq.scales, lq);  dq.scales  = nullptr; }
                if (dq.zeros)   { sycl::free(dq.zeros, lq);   dq.zeros   = nullptr; }
                dq.w.payload = nullptr; dq.w.scales = nullptr;
                dq.i4 = pack; dq.i4s = ws;
                acct(size_t(N) * (K / 2) + size_t(N) * (K / 128) * 4);
                bytes -= std::min(bytes, freed);
                return true;
            };
            DevQuant* all[] = { &d.sh_gu, &d.sh_down, &d.la_qkv, &d.la_z,
                                &d.la_out, &d.la_ab, &d.q_proj, &d.k_proj,
                                &d.v_proj, &d.o_proj };
            const char* names[] = { "sh_gu", "sh_down", "la_qkv", "la_z",
                                    "la_out", "la_ab", "q_proj", "k_proj",
                                    "v_proj", "o_proj" };
            for (size_t t = 0; t < sizeof(all)/sizeof(all[0]); ++t)
                if (!conv(*all[t], names[t])) return false;
            d.sh_gu_i4 = d.sh_gu.i4;   d.sh_gu_ws = d.sh_gu.i4s;
            d.sh_dn_i4 = d.sh_down.i4; d.sh_dn_ws = d.sh_down.i4s;
        }

        if (!ok) { err = "upload failed; see the tensor named above"; return false; }
        std::printf("\r  layers        %d/%d  %.2f GiB   ",
                    i + 1, cfg.n_layers, double(bytes) / 1073741824.0);
        std::fflush(stdout);
    }
    std::printf("\n");

    // Fix the norm convention for this model once, before any forward
    // pass.  Two independent choices: how many groups a row is split into,
    // and what is added to the stored weight.
    //
    //   Qwen / Muse   whole row, (1 + w)   -- zero-centered
    //   K2            grouped,    w
    //   gemma-4       whole row,  w
    //
    // gemma-4 is the trap: Gemma-2 and Gemma-3 DID use (1 + w), so the
    // convention is the one a reader would assume from the family name.
    // ref/gemma4.py:211 is unambiguous -- `normed_output * self.weight`,
    // no offset.  Applying (1 + w) instead leaves the model fluent and
    // shifts every normalised activation.
    const float norm_offset = (cfg.is_k2 || cfg.is_gemma4) ? 0.0f : 1.0f;
    set_norm_convention(cfg.is_k2 ? cfg.norm_groups : 1, norm_offset);

    // MoVA scratch: router logits, the top-k table and one expert output
    // row, reused by every sparse layer.
    if (cfg.is_k2 && cfg.mova_experts > 0) {
        const int NV = cfg.max_kv_heads() * cfg.max_head_dim();
        mova_logits     = sycl::malloc_device<float>(size_t(cfg.mova_experts), q);
        mova_rex        = sycl::malloc_device<int32_t>(size_t(cfg.mova_top_k), q);
        mova_rwt        = sycl::malloc_device<float>(size_t(cfg.mova_top_k), q);
        mova_expert_out = sycl::malloc_device<float>(size_t(NV), q);
        if (!mova_logits || !mova_rex || !mova_rwt || !mova_expert_out) {
            err = "MoVA scratch allocation failed"; return false;
        }
    }

    // Verification needs one vocabulary projection per candidate.  Convert
    // the head to the same symmetric-int4 representation as the W4A8 model
    // projections so the verifier can stream it once for all rows.  Decode
    // also consumes this representation through gemv_any(), so there is no
    // second copy and no decode/verify weight mismatch.
    if (w4a8_enabled() && w4a8_tiles_available() &&
        lm_head.w.fmt == Fmt::MXFP4 && lm_head.payload &&
        lm_head.w.K % 128 == 0 && lm_head.w.N % 256 == 0) {
        const int N = lm_head.w.N, K = lm_head.w.K;
        uint8_t* pack = sycl::malloc_device<uint8_t>(size_t(N) * (K / 2), q);
        float* scales = sycl::malloc_device<float>(size_t(N) * (K / 128), q);
        if (!pack || !scales) {
            err = "W4A8 allocation failed for lm_head";
            return false;
        }
        launch_mxfp4_to_int4sym(q, lm_head.w.payload,
            static_cast<const uint8_t*>(lm_head.w.scales),
            lm_head.w.row_bytes, lm_head.w.row_scales, pack, scales, N, K, {}).wait();
        const size_t freed = lm_head.w.bytes();
        sycl::free(lm_head.payload, q); lm_head.payload = nullptr;
        sycl::free(lm_head.scales, q);  lm_head.scales = nullptr;
        if (lm_head.zeros) { sycl::free(lm_head.zeros, q); lm_head.zeros = nullptr; }
        lm_head.w.payload = nullptr; lm_head.w.scales = nullptr;
        lm_head.i4 = pack; lm_head.i4s = scales;
        acct(size_t(N) * (K / 2) + size_t(N) * (K / 128) * sizeof(float));
        bytes -= std::min(bytes, freed);
    }

    // ---- MTP head -----------------------------------------------------
    // TP: the MTP head is small and is loaded REPLICATED on every rank --
    // no shard() call below -- so each rank drafts identically with no
    // extra collective.  Its two TP-sensitive spots (the sharded
    // embedding table, and the reduced-draft-vocab lm_head shortcut) are
    // handled in mtp_draft/mtp_warm.  PP is still excluded: the head
    // needs the final hidden state, which only the last stage has.
    if(mtp_enabled()&&(!pp_enabled()||pp_rank==pp_world-1)) {
        std::printf("\n  mtp head      ");
        std::fflush(stdout);
        auto ref = [&](const char* n) -> TensorRef {
            auto it = ck.index.find(n);
            return it == ck.index.end() ? TensorRef{} : it->second;
        };
        const TensorRef t_fc   = ref("mtp.fc.weight");
        const TensorRef t_preh = ref("mtp.pre_fc_norm_hidden.weight");
        const TensorRef t_pree = ref("mtp.pre_fc_norm_embedding.weight");
        const TensorRef t_nrm  = ref("mtp.norm.weight");
        const TensorRef t_in   = ref("mtp.layers.0.input_layernorm.weight");
        const TensorRef t_pon  = ref("mtp.layers.0.post_attention_layernorm.weight");
        const TensorRef t_q    = ref("mtp.layers.0.self_attn.q_proj.weight");
        const TensorRef t_k    = ref("mtp.layers.0.self_attn.k_proj.weight");
        const TensorRef t_v    = ref("mtp.layers.0.self_attn.v_proj.weight");
        const TensorRef t_o    = ref("mtp.layers.0.self_attn.o_proj.weight");
        const TensorRef t_qn   = ref("mtp.layers.0.self_attn.q_norm.weight");
        const TensorRef t_kn   = ref("mtp.layers.0.self_attn.k_norm.weight");
        const TensorRef t_g    = ref("mtp.layers.0.mlp.gate_proj.weight");
        const TensorRef t_u    = ref("mtp.layers.0.mlp.up_proj.weight");
        const TensorRef t_d    = ref("mtp.layers.0.mlp.down_proj.weight");
        const TensorRef t_router = ref("mtp.layers.0.mlp.gate.weight");
        const TensorRef t_e0 = resolve_expert(ck,"mtp.layers.0.mlp.experts.",0,
            cfg.n_experts,cfg.moe_inter,cfg.hidden)[0];
        const bool mtp_moe = cfg.is_moe() && t_router.ok() && t_e0.ok();
        if (!t_fc.ok() || !t_q.ok() || (!t_g.ok() && !mtp_moe)) {
            err="MTP requested but required head tensors are missing or incompatible";return false;
        } else {
            bool mok = true;
            // Preserve raw BF16 heads. Native packed heads retain their own
            // encoding; the target --proj format cannot silently lower precision.
            auto mtp_fmt = [&](const TensorRef& r) {
                if(r.native && r.native->encoding!=uint32_t(NativeEncoding::RAW)) {
                    QuantWeight w;std::string why;
                    if(!ck.native_view(r,w,why))throw std::invalid_argument(why);
                    return w.fmt;
                }
                return Fmt::BF16;
            };
            auto shape_ok=[](const TensorRef& r,std::initializer_list<int64_t> shape) {
                return r.ok() && r.t.shape==std::vector<int64_t>(shape);
            };
            const int H=cfg.hidden,HD=cfg.head_dim,Q=cfg.n_heads*HD,KV=cfg.n_kv_heads*HD;
            if(!shape_ok(t_fc,{H,2LL*H}) || !shape_ok(t_preh,{H}) || !shape_ok(t_pree,{H}) ||
               !shape_ok(t_nrm,{H}) || !shape_ok(t_in,{H}) || !shape_ok(t_pon,{H}) ||
               (!shape_ok(t_q,{Q,H})&&!shape_ok(t_q,{2LL*Q,H})) ||
               !shape_ok(t_k,{KV,H}) || !shape_ok(t_v,{KV,H}) || !shape_ok(t_o,{H,Q}) ||
               (t_qn.ok()&&!shape_ok(t_qn,{HD})) || (t_kn.ok()&&!shape_ok(t_kn,{HD}))) {
                err="MTP tensor shapes do not match target geometry";return false;
            }
            mtp.fc      = quantize_upload_t(q, ck, t_fc, mtp_fmt(t_fc), "mtp.fc", &mok);
            mtp.pre_h   = dev_copy_t<bf16_t>(q, ck, t_preh, "mtp.pre_h", &mok);
            mtp.pre_e   = dev_copy_t<bf16_t>(q, ck, t_pree, "mtp.pre_e", &mok);
            mtp.norm    = dev_copy_t<bf16_t>(q, ck, t_nrm,  "mtp.norm",  &mok);
            LayerDev& m = mtp.L;
            m.kind      = LayerKind::FULL_ATTN;
            // The MTP head is one extra layer appended to the model, so it
            // carries the model-wide attention geometry, not a layer type's.
            // Set explicitly: every forward path reads LayerDev now, and a
            // zero here would be a zero-width head rather than a wrong one.
            m.kv_heads    = cfg.n_kv_heads;
            m.head_dim    = cfg.head_dim;
            m.rope_theta  = cfg.rope_theta;
            m.partial_rope = cfg.partial_rope;
            m.in_norm   = dev_copy_t<bf16_t>(q, ck, t_in,  "mtp.in_norm",   &mok);
            m.post_norm = dev_copy_t<bf16_t>(q, ck, t_pon, "mtp.post_norm", &mok);
            m.q_proj    = quantize_upload_t(q, ck, t_q, mtp_fmt(t_q), "mtp.q_proj", &mok);
            m.k_proj    = quantize_upload_t(q, ck, t_k, mtp_fmt(t_k), "mtp.k_proj", &mok);
            m.v_proj    = quantize_upload_t(q, ck, t_v, mtp_fmt(t_v), "mtp.v_proj", &mok);
            m.o_proj    = quantize_upload_t(q, ck, t_o, mtp_fmt(t_o), "mtp.o_proj", &mok);
            if (t_qn.ok()) m.q_norm = dev_copy_t<bf16_t>(q, ck, t_qn, "mtp.q_norm", &mok);
            if (t_kn.ok()) m.k_norm = dev_copy_t<bf16_t>(q, ck, t_kn, "mtp.k_norm", &mok);

            size_t mtp_ffn_bytes = 0;
            if (mtp_moe) {
                const int H=cfg.hidden,I=cfg.moe_inter,E=cfg.n_experts;
                const Fmt EF=mtp_fmt(t_e0);
                const size_t gu_row=bytes_per_row(EF,H),dn_row=bytes_per_row(EF,I);
                const size_t gu_srow=scales_per_row(EF,H)*scale_element_bytes(EF);
                const size_t dn_srow=scales_per_row(EF,I)*scale_element_bytes(EF);
                const size_t gu_zrow=EF==Fmt::INT4?scales_per_row(EF,H):0;
                const size_t dn_zrow=EF==Fmt::INT4?scales_per_row(EF,I):0;
                const size_t gu_rows=size_t(E)*2*I,dn_rows=size_t(E)*H;
                m.router=quantize_upload_t(q,ck,t_router,Fmt::BF16,"mtp.router",&mok);
                auto alloc=[&](size_t n){return n?sycl::malloc_device<uint8_t>(n,q):nullptr;};
                m.gu_pack=alloc(gu_rows*gu_row);m.dn_pack=alloc(dn_rows*dn_row);
                m.gu_scale=alloc(gu_rows*gu_srow);m.dn_scale=alloc(dn_rows*dn_srow);
                m.gu_zero=alloc(gu_rows*gu_zrow);m.dn_zero=alloc(dn_rows*dn_zrow);
                if(!m.gu_pack||!m.dn_pack||(gu_srow&&!m.gu_scale)||(dn_srow&&!m.dn_scale)||
                   (gu_zrow&&!m.gu_zero)||(dn_zrow&&!m.dn_zero)){err="MTP expert allocation failed";return false;}
                auto copy_expert=[&](const TensorRef& r,size_t row,int n,int k,bool down) {
                    if(!mok)return;
                    if(r.t.shape!=std::vector<int64_t>{n,k}){mok=false;return;}
                    DevQuant t=quantize_upload_t(q,ck,r,EF,"mtp.expert",&mok);
                    if(!mok){t.release(q);return;}
                    if(t.w.fmt!=EF || !t.w.payload){t.release(q);mok=false;return;}
                    const size_t rb=down?dn_row:gu_row,sb=down?dn_srow:gu_srow,zb=down?dn_zrow:gu_zrow;
                    q.memcpy((down?m.dn_pack:m.gu_pack)+row*rb,t.w.payload,size_t(n)*rb);
                    if(sb)q.memcpy((down?m.dn_scale:m.gu_scale)+row*sb,t.w.scales,size_t(n)*sb);
                    if(zb)q.memcpy((down?m.dn_zero:m.gu_zero)+row*zb,t.w.zeros,size_t(n)*zb);
                    q.wait_and_throw();t.release(q);
                };
                for(int e=0;e<E&&mok;++e) {
                    const auto r=resolve_expert(ck,"mtp.layers.0.mlp.experts.",e,E,I,H);
                    copy_expert(r[0],size_t(e)*2*I,I,H,false);
                    copy_expert(r[1],(size_t(e)*2+1)*I,I,H,false);
                    copy_expert(r[2],size_t(e)*H,H,I,true);
                }
                m.moe.cfg.hidden=H;m.moe.cfg.inter=I;m.moe.cfg.num_experts=E;
                m.moe.cfg.top_k=cfg.top_k;m.moe.cfg.shared_inter=cfg.shared_inter;
                m.moe.gate_up=QuantWeight{EF,E*2*I,H,m.gu_pack,m.gu_scale,m.gu_zero,int64_t(gu_row),scales_per_row(EF,H)};
                m.moe.down=QuantWeight{EF,E*H,I,m.dn_pack,m.dn_scale,m.dn_zero,int64_t(dn_row),scales_per_row(EF,I)};
                const TensorRef t_sg = ref(
                    "mtp.layers.0.mlp.shared_expert.gate_proj.weight");
                const TensorRef t_su = ref(
                    "mtp.layers.0.mlp.shared_expert.up_proj.weight");
                const TensorRef t_sd = ref(
                    "mtp.layers.0.mlp.shared_expert.down_proj.weight");
                const TensorRef t_sgw = ref(
                    "mtp.layers.0.mlp.shared_expert_gate.weight");
                m.sh_gu = concat_upload_t(q, ck, t_sg, t_su, mtp_fmt(t_sg),
                                          "mtp.shared.gate_up", &mok);
                m.sh_down = quantize_upload_t(q, ck, t_sd, mtp_fmt(t_sd),
                                              "mtp.shared.down", &mok);
                // shared_expert_gate is OPTIONAL -- the very next line
                // says so.  Uploading it unconditionally set ok=false
                // through the "missing tensor" path, so any MoE
                // checkpoint whose MTP head has no shared_expert_gate
                // failed to load at all, with speculation reported as a
                // load failure rather than as an absent sub-weight.
                if (t_sgw.ok()) {
                    m.sh_gate_q = quantize_upload_t(q, ck, t_sgw, Fmt::BF16,
                                                    "mtp.shared.gate", &mok);
                    m.has_sh_gate = true;
                }
                mtp_ffn_bytes = gu_rows * (gu_row + gu_srow + gu_zrow)
                              + dn_rows * (dn_row + dn_srow + dn_zrow)
                              + m.router.w.bytes() + m.sh_gu.w.bytes()
                              + m.sh_down.w.bytes() + m.sh_gate_q.w.bytes();
            } else {
                m.sh_gu = concat_upload_t(q, ck, t_g, t_u, mtp_fmt(t_g),
                                          "mtp.gate_up", &mok);
                m.sh_down = quantize_upload_t(q, ck, t_d, mtp_fmt(t_d),
                                              "mtp.down", &mok);
                mtp_ffn_bytes = m.sh_gu.w.bytes() + m.sh_down.w.bytes();
            }
            m.k_cache   = sycl::malloc_device<uint8_t>(
                              size_t(m.kv_heads) * m.head_dim * max_seq, q);
            m.v_cache   = sycl::malloc_device<uint8_t>(
                              size_t(m.kv_heads) * m.head_dim * max_seq, q);
            const size_t mb = mtp.fc.w.bytes() + m.q_proj.w.bytes() + m.k_proj.w.bytes()
                            + m.v_proj.w.bytes() + m.o_proj.w.bytes()
                            + mtp_ffn_bytes
                            + size_t(m.kv_heads) * m.head_dim * max_seq * 2;
            acct(mb);
            if(!mok || !m.k_cache || !m.v_cache){err="MTP head loading failed";return false;}
            mtp.ok = true;
            std::printf("%s (%.2f GiB)\n", mok ? "ok" : "FAILED",
                        double(mb) / 1073741824.0);
        }
    }
    // The last pipeline stage loaded the embedding table only so a drafter
    // could embed the tokens it proposes.  If no drafter needs it, give it
    // back -- on a big model that is over a gigabyte on the card that is
    // short of VRAM, which is the reason PP is being used at all.
    //
    // A DFlash drafter has not been loaded yet at this point, so the
    // release for that case happens AFTER its block below, where whether
    // it ships its own embed_tokens is finally known.  Releasing here on
    // !mtp.ok alone is what took the table away from a DFlash drafter
    // that shares the target's -- MEASURED, as a failed draft on the last
    // stage under PP and nowhere else.
    const bool dflash_configured = dflash_env && *dflash_env;
    auto release_last_stage_embed = [&]() {
        // Never give back a table the lm_head is aliasing: on a tied model
        // freeing it here leaves the output projection pointing at freed
        // device memory, which is rule 1's failure mode exactly.
        if (!(pp_enabled() && pp_rank == pp_world - 1 && pp_rank != 0) ||
            mtp.ok || !embed || tied_lm_head) return;
        const size_t freed = size_t(embed_count) * H * sizeof(bf16_t);
        sycl::free(embed, q);
        embed = nullptr;
        bytes -= std::min(bytes, freed);
    };
    if (!dflash_configured) release_last_stage_embed();

    // ---- DFlash sidecar weights -------------------------------------
    const char* dpath=std::getenv("GRIMOIRE_DFLASH_MODEL");
    if(!dpath||!*dpath)dpath=std::getenv("GRIMOIRE_DFLASH2_MODEL");
    // TENSOR parallel is fine: the drafter is small and is loaded WHOLE on
    // every rank (nothing in this block calls the `shard` lambda), so each
    // rank drafts identically with no extra collective -- the same argument
    // that makes the MTP head work under TP.  The two places that did read
    // sharded target state are fixed: the block embed goes through
    // embed_rows(), and a shared lm_head goes through gemv_any().
    //
    // PIPELINE parallel: the LAST stage hosts the drafter, exactly as it
    // hosts the MTP head, because it is the stage that owns the final
    // hidden state.  The taps the drafter consumes are captured on
    // whichever stage owns the layer they name and forwarded along the
    // pipe (see pp_taps), so by the time the last stage drafts, the
    // concatenated tap row is complete.
    //
    // An EARLIER stage takes the short branch below instead: it resolves
    // the same tap set from the same config and allocates the same tap
    // buffer, but loads no draft weights -- it never drafts, it only
    // captures and forwards.  Resolving the tap set from the drafter's own
    // config on every stage is what keeps the stages from disagreeing
    // about which layers to tap, which would be silent.
    if(dpath&&*dpath&&pp_enabled()&&pp_rank!=pp_world-1) {
        DFlashSettings dcfg;
        std::ifstream dcfg_in(std::string(dpath)+"/config.json");
        if(!dcfg_in){
            err="DFlash: no config.json in "+std::string(dpath);return false;
        }
        const std::string text((std::istreambuf_iterator<char>(dcfg_in)),
                                std::istreambuf_iterator<char>());
        dcfg=parse_dflash_config(text);
        if(!dcfg.error.empty()){
            err="DFlash draft config: "+dcfg.error; return false;
        }
        dflash2.target_layers=dcfg.target_layers;
        if(dflash2.target_layers.empty())
            dflash2.target_layers={1,6,11,16,22,27,32,37};
        // Same refusal the owning stage makes: a tap naming the last layer
        // has no capture point, and fc would then read whatever that slice
        // of target_aux was allocated with.
        for(int id:dflash2.target_layers){
            if(id>=0&&id+1<cfg.n_layers)continue;
            err="DFlash target layer id "+std::to_string(id)+
                " has no capture point in a "+std::to_string(cfg.n_layers)+
                "-layer target";
            return false;
        }
        const size_t NTAP=dflash2.target_layers.size();
        dflash2.target_aux=sycl::malloc_device<float>(
            size_t(max_seq)*NTAP*cfg.hidden,q);
        if(!dflash2.target_aux){
            err="DFlash tap buffer allocation failed on PP stage "+
                std::to_string(pp_rank);
            return false;
        }
        // Uninitialised device memory would otherwise reach fc for any
        // position a stage forwards before capturing anything into it.
        q.memset(dflash2.target_aux,0,
                 size_t(max_seq)*NTAP*cfg.hidden*sizeof(float)).wait();
        pp_taps=int(NTAP);
        std::printf("  dflash taps: %d, captured here and forwarded "
                    "(stage %d of %d hosts no drafter)\n",
                    pp_taps,pp_rank,pp_world);
    }
    if(dpath&&*dpath&&(!pp_enabled()||pp_rank==pp_world-1)) {
        std::printf("\n  dflash       ");
        std::fflush(stdout);
        Qwen35Model dc;
        dc.dir=dpath;
        auto st=std::make_unique<SafeTensors>();
        std::string derr;
        if(!st->open(dc.dir+"/model.safetensors",derr)){
            err="DFlash2: "+derr; return false;
        }
        for(const auto& kv:st->tensors()){
            TensorRef r;r.shard=0;r.t=kv.second;dc.index[kv.first]=r;
        }
        dc.shards.push_back(std::move(st));
        dc.unmap_all();
        auto dr=[&](const std::string& n)->TensorRef{
            auto it=dc.index.find(n);return it==dc.index.end()?TensorRef{}:it->second;
        };
        bool dok=true;
        size_t db=0;
        dflash2.v2=dr("candidate_selector.hidden_projection.weight").ok();
        dflash2.hidden=cfg.hidden;
        dflash2.head_dim=128;
        // ---- draft config, resolved like the reference ---------------
        // Read it BEFORE the geometry below, because head_dim is one of
        // the things it decides.  Every field it carries is silent when
        // wrong -- the drafter runs, the output stays correct, and only
        // the acceptance rate moves -- so the rules live in one place
        // transcribed from ref/qwen3_dflash.py rather than guessed here.
        // Muse takes only the draft page size from it; the rest of its
        // table below was measured against the running Fusion reference
        // and stays as measured.
        DFlashSettings dcfg;
        {
            std::ifstream dcfg_in(std::string(dpath)+"/config.json");
            if(dcfg_in){
                const std::string text((std::istreambuf_iterator<char>(dcfg_in)),
                                        std::istreambuf_iterator<char>());
                dcfg=parse_dflash_config(text);
            }else dcfg.error="no config.json in the draft directory "+
                             std::string(dpath);
        }
        if(!cfg.is_muse){
            if(!dcfg.error.empty()){
                err="DFlash draft config: "+dcfg.error; return false;
            }
            // A draft whose hidden width differs from the target's is a
            // feature (target_hidden_size), not a number to plug in: fc,
            // the shared embedding and the shared lm_head all assume the
            // two are equal here.  Refuse rather than run a drafter whose
            // every tensor is the wrong width.
            if(dcfg.hidden>0&&dcfg.hidden!=cfg.hidden){
                err="DFlash draft hidden_size "+std::to_string(dcfg.hidden)+
                    " != target hidden_size "+std::to_string(cfg.hidden)+
                    "; a draft of a different width is not implemented";
                return false;
            }
            if(dcfg.head_dim>0)dflash2.head_dim=dcfg.head_dim;
            // With use_aux_hidden_state off the reference feeds the draft
            // the target's LAST hidden state and has no fc at all; this
            // engine's context path is fc-over-taps only.  Refuse rather
            // than run the taps through an fc the drafter never had.
            if(!dcfg.use_aux_hidden_state){
                err="this DFlash drafter sets use_aux_hidden_state=false "
                    "(it consumes the target's last hidden state, not the "
                    "layer taps); that path is not implemented";
                return false;
            }
        }
        // ---- draft architecture validation ---------------------------
        // dr() returns an EMPTY TensorRef for a name the checkpoint does not
        // have, and .t.shape is then empty -- so reading shape[0] straight
        // off it was out-of-bounds host indexing on any drafter whose layout
        // differs from the dense one.  A MoE draft (e.g. the DaoCloud
        // Ornith-1.5 2.6B-A0.3B variant) has no layers.0.mlp.gate_proj.weight
        // at all, so it hit that path rather than being told it is
        // unsupported.  Validate every tensor this geometry is derived from,
        // and name what is missing.
        std::string arch_err;
        auto draft_rows=[&](const char* name)->int{
            const TensorRef r=dr(name);
            if(!r.ok()){
                if(arch_err.empty())arch_err=std::string("no ")+name;
                return 0;
            }
            if(r.t.shape.size()!=2){
                if(arch_err.empty())arch_err=std::string(name)+" is rank "+
                    std::to_string(r.t.shape.size())+", expected a 2-D matrix";
                return 0;
            }
            return int(r.t.shape[0]);
        };
        const int q_rows =draft_rows("layers.0.self_attn.q_proj.weight");
        const int kv_rows=draft_rows("layers.0.self_attn.k_proj.weight");
        const int ff_rows=draft_rows("layers.0.mlp.gate_proj.weight");
        if(arch_err.empty()&&dflash2.head_dim>0&&
           (q_rows%dflash2.head_dim||kv_rows%dflash2.head_dim))
            arch_err="q/k projection rows ("+std::to_string(q_rows)+"/"+
                 std::to_string(kv_rows)+") are not a multiple of head_dim "+
                 std::to_string(dflash2.head_dim);
        if(!arch_err.empty()){
            // Name the capability, not just the symptom: a MoE draft needs a
            // routed draft FFN path that does not exist here, and DaoCloud's
            // layout announces itself with aux_hidden_state_layer_ids where
            // the dense drafts use target_layer_ids.
            const bool moe_draft=dr("layers.0.mlp.experts.0.gate_proj.weight").ok()||
                                 dr("layers.0.mlp.gate.weight").ok();
            err="unsupported DFlash draft architecture: "+arch_err+
                (moe_draft?". This looks like a MoE draft; only the DENSE "
                           "draft layout is implemented."
                         :". Only the dense draft layout is implemented.");
            return false;
        }
        dflash2.q_heads =q_rows/dflash2.head_dim;
        dflash2.kv_heads=kv_rows/dflash2.head_dim;
        dflash2.inter   =ff_rows;
        if(cfg.is_muse){
            dflash2.layers.resize(5);
            dflash2.target_layers={1,13,25,37,49};
            dflash2.mask_token=201818;
            dflash2.rope_theta=500000.0f;
            dflash2.sliding_window=2048;
            dflash2.rms_eps=1.0e-5f;
            if(dcfg.error.empty())dflash2.block_size=dcfg.block_size;
            dflash2.shared_embed_f16=upload_f16_t(
                q,ck,ck.embed,"dflash2.shared_embed",&dok);
            dflash2.shared_lm_head_f16=upload_f16_t(
                q,ck,ck.lm_head,"dflash2.shared_lm_head",&dok);
            db+=dflash2.shared_embed_f16.w.bytes();
            db+=dflash2.shared_lm_head_f16.w.bytes();
        }else{
            // Non-Muse DFlash/DFlash2 geometry (layer count, taps, mask
            // token, rope theta, per-layer attention shape, norm epsilon)
            // is specific to the DRAFT checkpoint, not the target model,
            // and dcfg above resolved all of it the way the reference
            // does.  What is left here is only the fallbacks for keys a
            // checkpoint may genuinely omit -- each one printed, because a
            // guessed value in this list costs acceptance and nothing else
            // ever says so.
            dflash2.layers.resize(size_t(dcfg.n_layers));
            dflash2.rope_theta=dcfg.rope_theta;
            dflash2.rms_eps=dcfg.rms_eps;
            dflash2.selector_top_k=dcfg.selector_top_k;
            dflash2.selector_rank=dcfg.selector_rank;
            dflash2.block_size=dcfg.block_size;
            if(dcfg.mask_token>=0)dflash2.mask_token=dcfg.mask_token;
            else{
                dflash2.mask_token=248077;
                dcfg.notes.push_back("falling back to mask token 248077 "
                    "(z-lab Qwen3.5 DFlash); the mask rows carry the whole "
                    "draft, so a wrong id here is most of the acceptance");
            }
            if(!dcfg.target_layers.empty())dflash2.target_layers=dcfg.target_layers;
            else{
                dflash2.target_layers={1,6,11,16,22,27,32,37};
                dcfg.notes.push_back("falling back to the z-lab Qwen3.5 tap "
                    "set {1,6,11,16,22,27,32,37}; checked against fc below");
            }
            // The window is per layer now; keep the scalar for the Muse
            // paged kernel's sake and report the widest one in use.
            dflash2.sliding_window=0;
            for(const DFlashLayerAttn& la:dcfg.layers)
                dflash2.sliding_window=std::max(dflash2.sliding_window,la.window);
        }
        // The draft page size decides whether the paged draft attention
        // can see its own anchor row, and the value that was in this tree
        // (64) contradicted the measurement recorded beside it (16).  It
        // is resolved from the config now; this overrides it, so the two
        // can be A/B'd on accepted-per-step rather than argued about.
        if(const char* bs=std::getenv("GRIMOIRE_DFLASH_BLOCK")){
            const int v=std::atoi(bs);
            if(v>0)dflash2.block_size=v;
        }
        if(dflash2.q_heads<=0||dflash2.kv_heads<=0||dflash2.inter<=0)dok=false;
        // The Muse assistant checkpoint is BF16 and vLLM casts it to FP16 at
        // load.  Quantizing these weights to MXFP4 is not an equivalent
        // implementation and changes draft agreement, so Muse never inherits
        // the target projection format here.
        // The drafter ships BF16 and is uploaded as FP16 to mirror vLLM's
        // numerics. That makes every draft stream 4x the bytes the INT4 target
        // does: measured 7.35 GB per draft at 223 GB/s = 33 ms, against the
        // verify's 15 GB at 536 GB/s. GRIMOIRE_MUSE_DRAFT_MXFP4=1 quantizes the
        // drafter instead, cutting draft traffic ~4x and 3.5 GiB of device
        // memory. It changes draft agreement, so judge it on measured
        // acceptance, not on whether it loads.
        const bool muse_draft_q =
            cfg.is_muse && std::getenv("GRIMOIRE_MUSE_DRAFT_MXFP4") != nullptr;
        const bool muse_fp16_draft = cfg.is_muse && !muse_draft_q;
        dflash2.fp16_draft = muse_fp16_draft;
        // The v2 drafter inherits the TARGET's projection format, so a BF16
        // draft checkpoint is quantized automatically at load.  That may be
        // fine, but it has never been measured here: acceptance is a
        // property of the draft's numerics, and loading successfully proves
        // nothing about it.  GRIMOIRE_DFLASH_DRAFT_BF16=1 keeps the draft in
        // BF16 so the first acceptance comparison is against the reference
        // precision rather than against a requantized copy of it.
        static const bool draft_keep_bf16 =
            std::getenv("GRIMOIRE_DFLASH_DRAFT_BF16") != nullptr;
        const Fmt dflash_fmt=muse_draft_q?Fmt::MXFP4
            :(cfg.is_muse?Fmt::BF16
              :(dflash2.v2&&!draft_keep_bf16?PF:Fmt::BF16));
        std::printf("\n  dflash2 draft format: %s%s\n",
            dflash_fmt==Fmt::BF16?"bf16":
            dflash_fmt==Fmt::MXFP4?"mxfp4":
            dflash_fmt==Fmt::INT4?"int4":"other",
            (dflash2.v2&&!draft_keep_bf16&&dflash_fmt!=Fmt::BF16)
                ? " (inherited from the target; GRIMOIRE_DFLASH_DRAFT_BF16=1 to keep bf16)"
                : "");
        auto qload=[&](const std::string& n,const char* what){
            DevQuant d=muse_fp16_draft?upload_f16_t(q,dc,dr(n),what,&dok):
                quantize_upload_t(q,dc,dr(n),dflash_fmt,what,&dok);
            db+=d.w.bytes();return d;
        };
        // The draft consumes the shared vocabulary projection too. Keeping
        // that 2.50 GiB matrix in FP16 left it as the largest steady draft
        // region after quantizing the five layers. A symmetric INT4 copy is
        // used only for proposals; target verification remains exact FP16.
        if(cfg.is_muse&&muse_draft_q&&
           !std::getenv("GRIMOIRE_MUSE_DRAFT_FP16_HEAD")){
            auto& head=dflash2.draft_lm_head_i4;
            const int N=dflash2.shared_lm_head_f16.w.N;
            const int K=dflash2.shared_lm_head_f16.w.K;
            const int NP=(N+255)&~255;
            uint8_t* pack=sycl::malloc_device<uint8_t>(size_t(NP)*(K/2),q);
            float* scales=sycl::malloc_device<float>(size_t(NP)*(K/128),q);
            if(!pack||!scales||K%128){
                if(pack)sycl::free(pack,q);
                if(scales)sycl::free(scales,q);
                dok=false;
            }else{
                q.memset(pack,0,size_t(NP)*(K/2));
                q.memset(scales,0,size_t(NP)*(K/128)*sizeof(float));
                launch_f16_to_int4sym(q,dflash2.shared_lm_head_f16.fp16,
                    pack,scales,N,K,{}).wait();
                head.w.N=NP;head.w.K=K;head.i4=pack;head.i4s=scales;
                db+=size_t(NP)*(K/2)+size_t(NP)*(K/128)*sizeof(float);
            }
        }
        auto bload=[&](const std::string& n,const char* what){
            TensorRef r=dr(n);bf16_t* p=dev_copy_t<bf16_t>(q,dc,r,what,&dok);
            if(r.ok())db+=size_t(r.t.numel())*sizeof(bf16_t);return p;
        };
        auto hload=[&](const std::string& n,const char* what){
            TensorRef r=dr(n);
            sycl::half* p=upload_f16_vector_t(q,dc,r,what,&dok);
            if(r.ok())db+=size_t(r.t.numel())*sizeof(sycl::half);
            return p;
        };
        const char* fc_name=dr("encoder.fc.weight").ok()?"encoder.fc.weight":"fc.weight";
        const char* hn_name=dr("encoder.output_norm_enc.weight").ok()?
                            "encoder.output_norm_enc.weight":"hidden_norm.weight";
        dflash2.fc=qload(fc_name,"dflash.fc");
        // A tap is captured at ENTRY to layer id+1, so a tap naming the
        // last layer is never written at all and fc then reads whatever
        // that slice of target_aux was allocated with -- uninitialised
        // DEVICE memory, straight into the context projection.
        for(int id:dflash2.target_layers){
            if(id>=0&&id+1<cfg.n_layers)continue;
            err="DFlash target layer id "+std::to_string(id)+
                " has no capture point in a "+std::to_string(cfg.n_layers)+
                "-layer target (the tap is taken at entry to layer id+1); "
                "this draft/target pair does not match";
            return false;
        }
        // fc consumes exactly n_taps * target_hidden floats per row, and the
        // engine hands it target_layers.size() * cfg.hidden.  The reference
        // raises on this mismatch (combine_hidden_states) because it means
        // the tap set does not belong to this drafter -- here it is worse
        // than wrong numbers: too many taps reads past target_aux on DEVICE,
        // which is a DEVICE_LOST and a power cycle, not an exception.
        if(dflash2.fc.w.K>0&&cfg.hidden>0){
            const int want=dflash2.fc.w.K/cfg.hidden;
            if(dflash2.fc.w.K%cfg.hidden||
               want!=int(dflash2.target_layers.size())){
                err="DFlash fc expects "+std::to_string(dflash2.fc.w.K)+
                    " concatenated target features ("+std::to_string(want)+
                    " x hidden "+std::to_string(cfg.hidden)+") but the tap set "
                    "has "+std::to_string(dflash2.target_layers.size())+
                    " entries -- this draft/target pair does not match";
                return false;
            }
        }
        if(cfg.is_muse){
            dflash2.hidden_norm_f16=hload(hn_name,"dflash.hidden_norm");
            dflash2.norm_f16=hload("norm.weight","dflash2.norm");
        }else{
            dflash2.hidden_norm=bload(hn_name,"dflash.hidden_norm");
            dflash2.norm=bload("norm.weight","dflash2.norm");
            // A drafter that ships its own lm_head / embed_tokens must be
            // run with them.  Until now every non-Muse drafter was decoded
            // through the TARGET's lm_head and embedded from the TARGET's
            // table, which is right only for a checkpoint that ships
            // neither -- and wrong invisibly for one that does.
            const char* head_name=dr("lm_head.weight").ok()?"lm_head.weight":
                (dr("model.lm_head.weight").ok()?"model.lm_head.weight":nullptr);
            if(head_name){
                dflash2.draft_lm_head=qload(head_name,"dflash.lm_head");
                dflash2.draft_vocab_rows=dflash2.draft_lm_head.w.N;
                TensorRef d2t=dr("d2t");
                if(!d2t.ok())d2t=dr("model.d2t");
                if(!d2t.ok())d2t=dr("draft_id_to_target_id");
                if(d2t.ok()){
                    // d2t is a DELTA per draft id, not an absolute id:
                    // the reference computes arange(draft_vocab) + d2t.
                    const int64_t n=d2t.t.numel();
                    if(n!=int64_t(dflash2.draft_vocab_rows)){
                        err="DFlash d2t has "+std::to_string(n)+" entries for "+
                            std::to_string(dflash2.draft_vocab_rows)+
                            " draft lm_head rows";
                        return false;
                    }
                    std::vector<float> delta(size_t(n),0.0f);
                    std::string rerr;
                    if(!dc.shards[d2t.shard]->read_f32(d2t.t,delta.data(),rerr)){
                        err="DFlash d2t: "+rerr; return false;
                    }
                    std::vector<int32_t> ids(size_t(n),0);
                    for(int64_t i=0;i<n;++i){
                        const long v=long(i)+long(delta[size_t(i)]);
                        if(v<0||v>=long(cfg.vocab)){
                            err="DFlash d2t maps draft id "+std::to_string(i)+
                                " to "+std::to_string(v)+", outside the "
                                "target vocabulary";
                            return false;
                        }
                        ids[size_t(i)]=int32_t(v);
                    }
                    dflash2.draft_vocab_map=
                        sycl::malloc_device<int32_t>(size_t(n),q);
                    if(!dflash2.draft_vocab_map)dok=false;
                    else{
                        q.memcpy(dflash2.draft_vocab_map,ids.data(),
                                 size_t(n)*sizeof(int32_t)).wait();
                        db+=size_t(n)*sizeof(int32_t);
                    }
                }else if(dflash2.draft_vocab_rows!=cfg.vocab){
                    err="the DFlash drafter has an lm_head of "+
                        std::to_string(dflash2.draft_vocab_rows)+" rows over a "
                        "target vocabulary of "+std::to_string(cfg.vocab)+
                        " but ships no d2t mapping; its token ids cannot be "
                        "interpreted";
                    return false;
                }
            }
            const char* emb_name=dr("embed_tokens.weight").ok()?
                "embed_tokens.weight":
                (dr("model.embed_tokens.weight").ok()?
                     "model.embed_tokens.weight":nullptr);
            if(emb_name)dflash2.draft_embed=bload(emb_name,"dflash.embed_tokens");
        }
        if(dflash2.v2){
            dflash2.selector_hidden=qload(
                "candidate_selector.hidden_projection.weight","dflash2.selector_hidden");
            dflash2.predecessor=bload(
                "candidate_selector.predecessor_codebook","dflash2.predecessor");
            dflash2.successor=bload(
                "candidate_selector.successor_codebook","dflash2.successor");
            // The selector runs only when every piece is present AND the
            // checkpoint's top_k matches what topk16_rows can emit.  Loading
            // the weights is not the same as being able to use them: the old
            // path took a bare argmax per position and never scored an edge,
            // which is a different algorithm, not a faster one.
            {
                const int rank=dflash2.selector_hidden.w.N;
                if(dflash2.selector_rank&&dflash2.selector_rank!=rank)
                    std::printf("\n  dflash2 selector: config rank %d != "
                        "hidden_projection rows %d -- selector disabled\n",
                        dflash2.selector_rank,rank);
                else if(dflash2.selector_top_k>16)
                    std::printf("\n  dflash2 selector: selector_top_k %d > 16 "
                        "(topk16_rows limit) -- selector disabled\n",
                        dflash2.selector_top_k);
                else if(dflash2.predecessor&&dflash2.successor&&rank>0){
                    if(!dflash2.selector_top_k)dflash2.selector_top_k=16;
                    dflash2.selector_rank=rank;
                    const int steps=15;            // MMAX-1
                    const int K=16;                // topk16_rows width
                    dflash2.sel_ids=sycl::malloc_device<int32_t>(size_t(steps)*K,q);
                    dflash2.sel_unary=sycl::malloc_device<float>(size_t(steps)*K,q);
                    dflash2.sel_hidden=sycl::malloc_device<float>(size_t(steps)*rank,q);
                    dflash2.sel_scores=sycl::malloc_device<float>(size_t(steps)*K*K,q);
                    dflash2.selector_ok=dflash2.sel_ids&&dflash2.sel_unary&&
                                        dflash2.sel_hidden&&dflash2.sel_scores;
                    if(!dflash2.selector_ok)dok=false;
                    else db+=size_t(steps)*(K*sizeof(int32_t)+K*sizeof(float)
                             +size_t(rank)*sizeof(float)+size_t(K)*K*sizeof(float));
                }
            }
        }
        const size_t draft_kv_elems=size_t(dflash2.kv_heads)*dflash2.head_dim*max_seq;
        if(cfg.is_muse)dflash2.num_blocks=(max_seq+dflash2.block_size-1)/dflash2.block_size;
        for(size_t i=0;i<dflash2.layers.size();++i){
            auto& d=dflash2.layers[size_t(i)];
            const std::string p="layers."+std::to_string(i)+".";
            if(cfg.is_muse){
                d.in_norm_f16=hload(p+"input_layernorm.weight","dflash2.input_norm");
                d.post_norm_f16=hload(p+"post_attention_layernorm.weight","dflash2.post_norm");
                d.q_norm_f16=hload(p+"self_attn.q_norm.weight","dflash2.q_norm");
                d.k_norm_f16=hload(p+"self_attn.k_norm.weight","dflash2.k_norm");
            }else{
                d.in_norm=bload(p+"input_layernorm.weight","dflash2.input_norm");
                d.post_norm=bload(p+"post_attention_layernorm.weight","dflash2.post_norm");
                d.q_norm=bload(p+"self_attn.q_norm.weight","dflash2.q_norm");
                d.k_norm=bload(p+"self_attn.k_norm.weight","dflash2.k_norm");
            }
            if(cfg.is_muse){
                std::vector<TensorRef> qkv={dr(p+"self_attn.q_proj.weight"),
                    dr(p+"self_attn.k_proj.weight"),
                    dr(p+"self_attn.v_proj.weight")};
                d.qkv=concat_upload_many_f16_t(
                    q,dc,qkv,"dflash2.qkv_proj",&dok);
                db+=d.qkv.w.bytes();
            }else{
                d.q=qload(p+"self_attn.q_proj.weight","dflash2.q_proj");
                d.k=qload(p+"self_attn.k_proj.weight","dflash2.k_proj");
                d.v=qload(p+"self_attn.v_proj.weight","dflash2.v_proj");
            }
            d.o=qload(p+"self_attn.o_proj.weight","dflash2.o_proj");
            if(cfg.is_muse&&muse_draft_q){
                d.gate_up=concat_upload_t(q,dc,
                    dr(p+"mlp.gate_proj.weight"),dr(p+"mlp.up_proj.weight"),
                    dflash_fmt,"dflash2.gate_up",&dok);
            }else if(cfg.is_muse){
                std::vector<TensorRef> gu={dr(p+"mlp.gate_proj.weight"),
                                           dr(p+"mlp.up_proj.weight")};
                d.gate_up=concat_upload_many_f16_t(
                    q,dc,gu,"dflash2.gate_up",&dok);
            }else d.gate_up=concat_upload_t(q,dc,
                dr(p+"mlp.gate_proj.weight"),dr(p+"mlp.up_proj.weight"),
                dflash_fmt,"dflash2.gate_up",&dok);
            db+=d.gate_up.w.bytes();
            d.down=qload(p+"mlp.down_proj.weight","dflash2.down_proj");
            if(dflash2.v2){
                if(!dflash2.conv_taps){
                    const TensorRef kb=dr(p+"attention_conv.base_kernel");
                    const TensorRef kp=dr(p+"attention_conv.kernel_projection.weight");
                    if(kb.ok()&&kb.t.shape.size()>=2&&kp.ok()){
                        dflash2.conv_taps=int(kb.t.shape[1]);
                        const int outw=int(kp.t.shape[0]);
                        if(dflash2.conv_taps>0)
                            dflash2.conv_groups=outw/(2*dflash2.conv_taps);
                    }
                    if(dflash2.conv_taps<=0||dflash2.conv_groups<=0||
                       dflash2.hidden%dflash2.conv_groups){
                        std::fprintf(stderr,"\n  DFlash2 conv geometry unusable "
                            "(taps %d groups %d hidden %d)\n",dflash2.conv_taps,
                            dflash2.conv_groups,dflash2.hidden);
                        dok=false;
                    }
                }
                d.attn_conv_proj=qload(p+"attention_conv.kernel_projection.weight",
                                       "dflash2.attn_conv_proj");
                d.mlp_conv_proj=qload(p+"mlp_conv.kernel_projection.weight",
                                      "dflash2.mlp_conv_proj");
                d.attn_conv_base=bload(p+"attention_conv.base_kernel",
                                       "dflash2.attn_conv_base");
                d.mlp_conv_base=bload(p+"mlp_conv.base_kernel",
                                      "dflash2.mlp_conv_base");
            }
            // Attention shape.  Muse keeps the table measured against the
            // running Fusion reference.  Everything else takes what the
            // draft config says, resolved exactly as the reference resolves
            // it -- which for a config that describes no attention shape is
            // FULL, NON-CAUSAL attention on every layer.
            //
            // This replaces "layers 0..4 slide at 4096, DFlash2 slides
            // everywhere", which was an assumption, not a checkpoint fact,
            // and which no self-check could ever have contradicted: a
            // wrongly-windowed draft layer still produces fluent tokens and
            // still leaves the output correct.  GRIMOIRE_DFLASH_LEGACY_SLIDING=1
            // restores it so the two can be A/B'd on accepted-tokens-per-step
            // on the card, which is the only measurement that settles it.
            static const bool legacy_sliding=
                std::getenv("GRIMOIRE_DFLASH_LEGACY_SLIDING")!=nullptr;
            if(cfg.is_muse||legacy_sliding){
                const bool sliding=cfg.is_muse||dflash2.v2||i<5;
                // The old code's own window, not the resolved one.  It
                // read a top-level "sliding_window" and fell back to 4096;
                // the resolver leaves the per-layer window at 0 for a
                // config that names none, and an escape hatch that
                // silently means "full attention" would A/B nothing.
                const int legacy_window=cfg.is_muse?2048:
                    (dcfg.config_window?dcfg.config_window:4096);
                d.window=sliding?legacy_window:0;
                d.causal=false;
            }else{
                if(size_t(i)>=dcfg.layers.size()){
                    err="DFlash draft config resolved "+
                        std::to_string(dcfg.layers.size())+" layers but the "
                        "checkpoint has more"; return false;
                }
                const DFlashLayerAttn la=dcfg.layers[size_t(i)];
                d.window=la.window;
                d.causal=la.causal;
            }
            if(cfg.is_muse){
                const size_t cache_elems=size_t(dflash2.num_blocks)*dflash2.block_size*
                    dflash2.kv_heads*dflash2.head_dim;
                d.k_cache_f16=sycl::malloc_device<sycl::half>(cache_elems,q);
                d.v_cache_f16=sycl::malloc_device<sycl::half>(cache_elems,q);
                if(!d.k_cache_f16||!d.v_cache_f16)dok=false;
                db+=2*cache_elems*sizeof(sycl::half);
            }else{
                d.k_cache=sycl::malloc_device<uint8_t>(draft_kv_elems,q);
                d.v_cache=sycl::malloc_device<uint8_t>(draft_kv_elems,q);
                if(!d.k_cache||!d.v_cache)dok=false;
                db+=2*draft_kv_elems;
            }
        }
        if(cfg.is_muse){
            std::vector<TensorRef> fused_refs;
            fused_refs.reserve(dflash2.layers.size()*2);
            for(size_t i=0;i<dflash2.layers.size();++i){
                const std::string p="layers."+std::to_string(i)+".self_attn.";
                fused_refs.push_back(dr(p+"k_proj.weight"));
                fused_refs.push_back(dr(p+"v_proj.weight"));
            }
            dflash2.fused_context_kv=concat_upload_many_f16_t(
                q,dc,fused_refs,"dflash2.fused_context_kv",&dok);
            db+=dflash2.fused_context_kv.w.bytes();
            dflash2.k_norm_all_f16=sycl::malloc_device<sycl::half>(
                dflash2.layers.size()*dflash2.head_dim,q);
            if(!dflash2.k_norm_all_f16)dok=false;
            else{
                // Context-K norm source.  DEFAULT is each layer's own
                // checkpoint weight -- the model-faithful thing, and the
                // only defensible default: a loader that substitutes one
                // layer's weight for another has redefined the model, and
                // every acceptance number taken under it describes a
                // different drafter than the one on disk.
                //
                // The compatibility mode below is kept because it is not a
                // guess.  Measured against the running Fusion reference:
                // the effective context-K weight recovered from Fusion's own
                // pre-RoPE context K is IDENTICAL across all five draft
                // layers (pairwise cos 1.0000, rms 1.08547) and equals
                // layers.0.self_attn.k_norm.weight, while the checkpoint's
                // five k_norm tensors genuinely differ (rms 1.085, 1.349,
                // 0.895, 1.381, 0.955).  Under per-layer weights Grimoire's
                // context K sits at cos 0.93-0.97 against Fusion; layer 0's
                // weight everywhere raises every layer to ~0.99.
                //
                // So one of two things is true, and the measurement that
                // separates them has not been run: either Fusion collapses
                // the norm (and matching it buys acceptance against a
                // reference that is itself wrong), or Grimoire feeds its
                // context-K path something the per-layer weights then
                // expose.  Until that A/B exists -- accepted tokens per
                // step, same prompt, same seed, both modes -- the faithful
                // weights are the default and the reference-matching mode
                // is opt-in and named in the capability matrix, so no
                // number can be quoted without saying which one produced
                // it.
                //
                //   GRIMOIRE_MUSE_KNORM_LAYER0=1   Fusion-matching (cos ~0.99)
                //   unset                          checkpoint per-layer
                const char* kn0=std::getenv("GRIMOIRE_MUSE_KNORM_LAYER0");
                dflash2.knorm_layer0=kn0&&*kn0&&*kn0!='0';
                for(size_t i=0;i<dflash2.layers.size();++i)
                    q.memcpy(dflash2.k_norm_all_f16+i*dflash2.head_dim,
                        dflash2.layers[dflash2.knorm_layer0?0:i].k_norm_f16,
                        size_t(dflash2.head_dim)*sizeof(sycl::half));
                db+=dflash2.layers.size()*dflash2.head_dim*sizeof(sycl::half);
            }
        }
        // Optional exact NInfer Build-2 proposal head.  Files are headerless
        // extracts of text/draft_head (Q4G64_F16S row-split: base plane then
        // FP16 scale plane) and text/draft_head_token_ids (I32).  Keeping the
        // loader explicit prevents silently pairing this model-specific head
        // with an unrelated target.
        const char* dh_path=std::getenv("GRIMOIRE_DFLASH_HEAD_Q4G64");
        const char* di_path=std::getenv("GRIMOIRE_DFLASH_HEAD_TOKEN_IDS");
        if(dh_path&&*dh_path){
            constexpr int DN=131072,DK=2048,DG=64;
            constexpr size_t code_bytes=size_t(DN)*DK/2;
            constexpr size_t scale_count=size_t(DN)*DK/DG;
            constexpr size_t head_bytes=code_bytes+scale_count*sizeof(uint16_t);
            constexpr size_t ids_bytes=size_t(DN)*sizeof(int32_t);
            if(!di_path||!*di_path){
                std::fprintf(stderr,"\n  DFlash NInfer head requires "
                    "GRIMOIRE_DFLASH_HEAD_TOKEN_IDS\n");
                dok=false;
            }else{
                std::vector<uint8_t> head(head_bytes);
                std::vector<int32_t> ids(DN);
                auto read_exact=[](const char* path,void* dst,size_t bytes){
                    std::FILE* f=std::fopen(path,"rb");
                    if(!f)return false;
                    const size_t got=std::fread(dst,1,bytes,f);
                    const int extra=std::fgetc(f);
                    std::fclose(f);
                    return got==bytes&&extra==EOF;
                };
                if(!read_exact(dh_path,head.data(),head.size())||
                   !read_exact(di_path,ids.data(),ids_bytes)){
                    std::fprintf(stderr,"\n  DFlash NInfer head extract has "
                        "the wrong size or cannot be read\n");
                    dok=false;
                }else{
                    std::vector<float> scales(scale_count);
                    const auto* hs=reinterpret_cast<const uint16_t*>(
                        head.data()+code_bytes);
                    for(size_t i=0;i<scale_count;++i)scales[i]=f16_to_f32(hs[i]);
                    dflash2.draft_head_i4=sycl::malloc_device<uint8_t>(code_bytes,q);
                    dflash2.draft_head_i4s=sycl::malloc_device<float>(scale_count,q);
                    dflash2.draft_head_token_ids=
                        sycl::malloc_device<int32_t>(DN,q);
                    if(!dflash2.draft_head_i4||!dflash2.draft_head_i4s||
                       !dflash2.draft_head_token_ids)dok=false;
                    else{
                        q.memcpy(dflash2.draft_head_i4,head.data(),code_bytes);
                        q.memcpy(dflash2.draft_head_i4s,scales.data(),
                                 scale_count*sizeof(float));
                        q.memcpy(dflash2.draft_head_token_ids,ids.data(),ids_bytes).wait();
                        dflash2.draft_head_rows=DN;
                        db+=code_bytes+scale_count*sizeof(float)+ids_bytes;
                    }
                }
            }
        }
        dflash2.target_aux=sycl::malloc_device<float>(
            size_t(max_seq)*dflash2.target_layers.size()*cfg.hidden,q);
        if(!dflash2.target_aux)dok=false;
        db+=size_t(max_seq)*dflash2.target_layers.size()*cfg.hidden*sizeof(float);
        // v2 was skipped here entirely, so a DFlash2 checkpoint loaded its
        // weights and then had no scratch to run in. The draft geometry is the
        // same for both versions; v2 only adds the convolution buffers below.
        if(!dflash2.v2||dflash2.conv_taps>0){
            constexpr int DM=16;
            const int DH=dflash2.hidden;
            const int DQ=dflash2.q_heads*dflash2.head_dim;
            const int DKV=dflash2.kv_heads*dflash2.head_dim;
            const int DI2=2*dflash2.inter;
            auto dfd=[&](size_t n){db+=n*sizeof(float);return sycl::malloc_device<float>(n,q);};
            // Muse's context projection is independent of the 16-token draft
            // query block. A wider batch avoids thousands of tiny launches.
            // Non-Muse keeps DM because it also uses fixed-size q/k/v scratch.
            dflash2.ctx_chunk=DM;
            if(cfg.is_muse){
                const char* v=std::getenv("GRIMOIRE_DFLASH_CTX_CHUNK");
                int c=v&&*v?std::atoi(v):256;
                if(c<DM)c=DM;
                if(c>4096)c=4096;
                dflash2.ctx_chunk=c;
            }
            const int DMC=std::max(DM,dflash2.ctx_chunk);
            dflash2.ctx=dfd(size_t(DMC)*DH);
            if(cfg.is_muse){
                const size_t all_kv=size_t(dflash2.layers.size())*DMC*DKV;
                dflash2.context_kv_all=dfd(2*all_kv);
                dflash2.context_k_all_f16=
                    sycl::malloc_device<sycl::half>(all_kv,q);
                dflash2.context_v_all_f16=
                    sycl::malloc_device<sycl::half>(all_kv,q);
                db+=2*all_kv*sizeof(sycl::half);
                if(!dflash2.context_kv_all||!dflash2.context_k_all_f16||
                   !dflash2.context_v_all_f16)dok=false;
            }
            // MLP activation has intermediate width, not hidden width.  Keep
            // it separate from the down-projection output: an in-place GEMM
            // races its own input and also changes the row stride I -> H.
            dflash2.h=dfd(size_t(DM)*dflash2.inter);
            dflash2.resid=dfd(size_t(DM)*DH);
            dflash2.normed=dfd(size_t(DMC)*DH);
            dflash2.q=dfd(size_t(DM)*DQ);
            dflash2.k=dfd(size_t(DM)*DKV);
            dflash2.v=dfd(size_t(DM)*DKV);
            dflash2.attn=dfd(size_t(DM)*DQ);
            dflash2.proj=dfd(size_t(DM)*DH);
            dflash2.gate_up=dfd(size_t(DM)*DI2);
            dflash2.mlp=dfd(size_t(DM)*DH);
            if(dflash2.v2&&dflash2.conv_taps>0){
                dflash2.conv_delta=dfd(size_t(DM)*2*dflash2.conv_taps*
                                       dflash2.conv_groups);
                dflash2.conv_scratch=dfd(size_t(DM)*DH);
            }
            const int draft_vocab=dflash2.draft_head_rows?
                dflash2.draft_head_rows:
                (dflash2.draft_vocab_rows?dflash2.draft_vocab_rows:
                 (dflash2.draft_lm_head_i4.has_i4()?
                     dflash2.draft_lm_head_i4.w.N:cfg.vocab));
            dflash2.draft_logits_stride=draft_vocab;
            dflash2.logits=dfd(size_t(DM-1)*dflash2.draft_logits_stride);
            if(cfg.is_muse){
                auto df16=[&](size_t n){
                    db+=n*sizeof(sycl::half);
                    return sycl::malloc_device<sycl::half>(n,q);
                };
                dflash2.q_f16=df16(size_t(DM)*DQ);
                dflash2.k_f16=df16(size_t(DM)*DKV);
                dflash2.v_f16=df16(size_t(DM)*DKV);
                dflash2.attn_f16=df16(size_t(DM)*DQ);
                const int linear_in_width=std::max(DH,
                    int(dflash2.target_layers.size())*DH);
                // The shared lm_head runs through this same FP16 linear, so
                // the destination must hold a full vocab-wide row. Sizing this
                // from the drafter's own projections alone overflows the
                // buffer by ~5x at the draft-logits step: the first rows land
                // in valid memory and every later row reads back inf/nan,
                // which collapses draft acceptance to zero with no error.
                const int linear_out_width=std::max({DH,DQ,DKV,DI2,
                    int(dflash2.layers.size())*2*DKV,cfg.vocab});
                dflash2.linear_in_f16=df16(size_t(DMC)*linear_in_width);
                dflash2.linear_out_f16=df16(size_t(DMC)*linear_out_width);
                // The block table is SHARED with the target's own paged
                // attention, which pages at 64 and indexes
                // ceil(max_seq/64) entries.  Sizing it from the draft page
                // alone is only safe while the draft page is <= 64: a
                // configured or GRIMOIRE_DEFAULT-overridden 128 gives the
                // table ceil(max_seq/128) entries and the target then
                // reads twice that many.  Size it for BOTH consumers.
                const int table_entries = std::max(
                    dflash2.num_blocks, (max_seq + 63) / 64);
                dflash2.block_table=sycl::malloc_device<int32_t>(size_t(table_entries),q);
                dflash2.cu_q=sycl::malloc_device<int32_t>(2,q);
                dflash2.cu_k=sycl::malloc_device<int32_t>(2,q);
                dflash2.seqused_k=sycl::malloc_device<int32_t>(1,q);
                db+=size_t(table_entries+5)*sizeof(int32_t);
                if(!dflash2.q_f16||!dflash2.k_f16||!dflash2.v_f16||
                   !dflash2.attn_f16||!dflash2.linear_in_f16||
                   !dflash2.linear_out_f16||!dflash2.block_table||!dflash2.cu_q||
                   !dflash2.cu_k||!dflash2.seqused_k)dok=false;
                if(dok){
                    std::vector<int32_t> blocks(size_t(table_entries), 0);
                    std::iota(blocks.begin(),blocks.end(),0);
                    const int32_t cuq[2]={0,DM};
                    const int32_t cuk[2]={0,0};
                    q.memcpy(dflash2.block_table,blocks.data(),
                        blocks.size()*sizeof(int32_t)).wait();
                    q.memcpy(dflash2.cu_q,cuq,sizeof(cuq)).wait();
                    q.memcpy(dflash2.cu_k,cuk,sizeof(cuk)).wait();
                }
            }
            dflash2.bf=sycl::malloc_device<sycl_bf16>(
                size_t(DMC)*dflash2.target_layers.size()*DH,q);
            dflash2.a8=sycl::malloc_device<int8_t>(size_t(DM)*DH,q);
            dflash2.a8s=sycl::malloc_device<float>(DM,q);
            dflash2.tokens=sycl::malloc_device<int32_t>(DM,q);
            dflash2.draft_ids=sycl::malloc_device<int32_t>(DM-1,q);
            if(cfg.is_muse){
                // The verify batch projects the shared lm_head through the
                // same xb/yb staging buffers, so they must hold a vocab-wide
                // row. Sizing from the layer weights alone overflows yb by
                // ~5x (16*202048 halves into 16*39936) and smashes megabytes
                // of device memory past it, which corrupts the target's own
                // decode whenever the drafter is loaded.
                const int VW=std::max({cfg.hidden,cfg.n_heads*cfg.max_head_dim(),
                    cfg.max_kv_heads()*cfg.max_head_dim(),2*cfg.dense_inter,
                    cfg.vocab});
                dflash2.verify_logits=dfd(size_t(DM)*cfg.vocab);
                dflash2.verify_bf=sycl::malloc_device<sycl_bf16>(size_t(DM)*VW,q);
                dflash2.verify_bf_out=sycl::malloc_device<sycl_bf16>(size_t(DM)*VW,q);
                dflash2.verify_a8=sycl::malloc_device<int8_t>(size_t(DM)*VW,q);
                dflash2.verify_a8s=sycl::malloc_device<float>(DM,q);
                dflash2.verify_ids=sycl::malloc_device<int32_t>(DM,q);
                db+=size_t(DM)*VW*(2*sizeof(sycl_bf16)+sizeof(int8_t))
                    +size_t(DM)*(sizeof(float)+sizeof(int32_t));
                if(!dflash2.verify_logits||!dflash2.verify_bf||!dflash2.verify_bf_out||
                   !dflash2.verify_a8||!dflash2.verify_a8s||
                   !dflash2.verify_ids)dok=false;
                if(std::getenv("GRIMOIRE_DFLASH_ONEDNN_BF16")&&
                   dflash2.fc.w.fmt==Fmt::BF16){
                    auto od=load_onednn_bf16();
                    if(od){
                        dflash2.fc_plan=od.create(&q,DM,dflash2.fc.w.N,dflash2.fc.w.K);
                        if(dflash2.fc_plan){
                            const size_t sb=od.scratch_size(dflash2.fc_plan);
                            dflash2.fc_scratch=sycl::malloc_device<uint8_t>(std::max<size_t>(1,sb),q);
                            db+=std::max<size_t>(1,sb);
                            if(!dflash2.fc_scratch)dok=false;
                        }
                    }
                }
            }
            db+=size_t(DMC)*dflash2.target_layers.size()*DH*sizeof(bf16_t)
                +size_t(DM)*DH+size_t(DM)*sizeof(float)
                +size_t(2*DM-1)*sizeof(int32_t);
            if(!dflash2.ctx||!dflash2.h||!dflash2.resid||!dflash2.normed||
               !dflash2.q||!dflash2.k||!dflash2.v||!dflash2.attn||
               !dflash2.proj||!dflash2.gate_up||!dflash2.mlp||!dflash2.logits||
               !dflash2.bf||!dflash2.a8||!dflash2.a8s||!dflash2.tokens||
               !dflash2.draft_ids)dok=false;
        }
        dflash2.ok=dok;
        if(!dok){err="DFlash weight upload failed";return false;}
        // The last stage receives taps from every stage before it, so it
        // must expect the same block width they send.  Every stage derived
        // this from the SAME draft config; pp_connect checks that they
        // agree rather than trusting it.
        if(pp_enabled())pp_taps=int(dflash2.target_layers.size());
        acct(db);
        std::printf("ok (%s, %.2f GiB device, target taps + draft KV ready)\n",
                    dflash2.v2?"DFlash2DraftModel":"DFlashDraftModel",
                    double(db)/1073741824.0);
        // Print what the drafter actually resolved to.  None of these values
        // can be inferred from the output -- a drafter running on the wrong
        // rope theta, mask token or attention shape still writes fluent text
        // and still leaves the answer correct; only acceptance moves.  So the
        // one place they can be checked is here, before the first token.
        {
            int sliding=0,causal=0;
            for(const auto& d:dflash2.layers){
                if(d.window)++sliding;
                if(d.causal)++causal;
            }
            std::string taps;
            for(size_t i=0;i<dflash2.target_layers.size();++i)
                taps+=(i?",":"")+std::to_string(dflash2.target_layers[i]);
            std::printf("  dflash config: %zu layers, taps [%s]%s%s, "
                        "mask %d, rope_theta %g, eps %g, head_dim %d, "
                        "draft page %d\n",
                        dflash2.layers.size(), taps.c_str(),
                        dcfg.tap_key.empty()?"":" from ", dcfg.tap_key.c_str(),
                        dflash2.mask_token, double(dflash2.rope_theta),
                        double(dflash2.rms_eps), dflash2.head_dim,
                        dflash2.block_size);
            std::printf("  dflash attention: %d of %zu layers sliding"
                        " (window %d), %d causal\n",
                        sliding, dflash2.layers.size(),
                        dflash2.sliding_window, causal);
            // Which head and which embedding table the draft actually used.
            // A drafter silently decoded through the TARGET's head is the
            // single largest divergence this loader can have, and nothing
            // downstream of it ever says so.
            std::printf("  dflash head: %s", dflash2.draft_head_rows
                ? "NInfer reduced extract"
                : dflash2.draft_lm_head.w.N ? "the drafter's own lm_head"
                : "the TARGET's lm_head (the drafter ships none)");
            if(dflash2.draft_vocab_rows)
                std::printf(", %d rows%s", dflash2.draft_vocab_rows,
                    dflash2.draft_vocab_map ? " + d2t mapping" : "");
            std::printf("\n  dflash embed: %s\n", dflash2.draft_embed
                ? "the drafter's own embed_tokens"
                : "the TARGET's embed_tokens (the drafter ships none)");
            for(const std::string& n:dcfg.notes)
                std::printf("  dflash note: %s\n", n.c_str());
        }
    }

    if (mtp.ok) {
        mtp.cat   = sycl::malloc_device<float>(size_t(cfg.hidden) * 2, q);
        mtp.x     = sycl::malloc_device<float>(size_t(cfg.hidden), q);
        mtp.h2    = sycl::malloc_device<float>(size_t(cfg.hidden), q);
        mtp.resid = sycl::malloc_device<float>(size_t(cfg.hidden), q);
    }

    // EVERY pipeline stage needs these, not just the one holding the head:
    // each stage rolls back its OWN layers when a draft is rejected.  The
    // predicate is the request (an env var every rank reads) rather than
    // mtp.ok, because mtp.ok is false on the early stages by design -- and
    // pp_connect, which publishes the pipeline's answer, has not run yet.
    // If the last stage then fails to load a head, these go unused.
    if (mtp.ok || dflash2.ok || (pp_enabled() && mtp_enabled())) {
        // One exact rollback image for the hybrid recurrent state. This is
        // about 157 MiB on Qwen3.8-27B and is copied device-to-device.
        for (const auto& d : L) {
            if (d.dn_state)
                spec_dn_elems += size_t(cfg.lin_v_heads) * cfg.lin_v_dim * cfg.lin_k_dim;
            if (d.conv_ring)
                spec_conv_elems += size_t(qkv_ch) * (cfg.conv_kernel - 1);
            if (d.conv_ring)
                spec_conv_input_elems += size_t(qkv_ch);
        }
        if (spec_dn_elems)
            spec_dn_state = sycl::malloc_device<float>(spec_dn_elems, q);
        if (spec_conv_elems)
            spec_conv_ring = sycl::malloc_device<float>(spec_conv_elems, q);
        if (spec_dn_elems)
            spec_dn_steps = sycl::malloc_device<float>(kSpecBatch * spec_dn_elems, q);
        if (spec_conv_input_elems)
            spec_conv_inputs = sycl::malloc_device<float>(kSpecBatch * spec_conv_input_elems, q);
        spec_hidden_steps = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.hidden, q);
        // POISON it.  This buffer is written only by the batched verify,
        // and reading it unwritten is a heisenbug: whatever the allocator
        // last left there is usually finite, so the drafter produces
        // plausible-but-wrong tokens most of the time and nonsense
        // occasionally.  Filling it with NaN turns that whole class into a
        // deterministic, loud failure the first time anyone reads it --
        // non-finite logits, argmax returns INT_MAX, and generation
        // rejects the token by name instead of once in three runs.
        if (spec_hidden_steps) {
            const float nan = std::numeric_limits<float>::quiet_NaN();
            q.fill(spec_hidden_steps, nan, size_t(kSpecBatch) * cfg.hidden).wait();
        }
        if ((spec_dn_elems && !spec_dn_state) ||
            (spec_conv_elems && !spec_conv_ring) ||
            (spec_dn_elems && !spec_dn_steps) ||
            (spec_conv_input_elems && !spec_conv_inputs) || !spec_hidden_steps) {
            err = "speculative rollback-state allocation failed";
            return false;
        }
        acct((spec_dn_elems * (1 + kSpecBatch) + spec_conv_elems +
              spec_conv_input_elems * kSpecBatch + size_t(kSpecBatch) * cfg.hidden) *
             sizeof(float));
    }

    // Now the drafter is loaded, so it is known whether it brought its own
    // embedding table.  One that did never touches the target's, so the
    // copy this stage holds can go back.  One that did NOT embeds every
    // token it drafts from it, and must keep it.
    if (dflash_configured && (!dflash2.ok || dflash2.draft_embed))
        release_last_stage_embed();

    std::printf("\n  scratch buffers ... ");
    std::fflush(stdout);

    // ---- scratch ------------------------------------------------------
    const int TK = cfg.is_moe() ? cfg.top_k : 1;
    // Size the FFN scratch from the layers that were ACTUALLY uploaded, the
    // way qkv_max and aux_max below already do -- not from one model-wide
    // config field.  K2 is the case that breaks the assumption: its three
    // dense layers carry a full intermediate_size MLP (6144) in sh_gu while
    // the sparse layers' shared expert is moe_intermediate_size (768), so
    // `cfg.is_moe() ? shared_inter : dense_inter` picks 768 and every dense
    // layer then writes 2*6144 floats into a 2*768 buffer.  That is a 46 KB
    // device heap overrun three times per token: DEVICE_LOST on a B70, and
    // nothing in a self-check would have named it.
    //
    // output_rows(), NOT w.N: under TP w.N is this rank's SHARD while
    // gemv_any all-gathers the FULL row into this buffer, so sizing from
    // the shard under-allocates by (world-1)/world and the gather then
    // writes past the end.
    int SI = cfg.is_moe() ? cfg.shared_inter : cfg.dense_inter;
    for (const auto& dl : L) {
        const int half = dl.sh_gu.output_rows() / 2;   // sh_gu is gate|up
        if (half > SI) SI = half;
    }
    // The MTP head is a layer too and shares this scratch, but it is not
    // in L.  Its FFN is usually the same width as the target's; "usually"
    // is not a guarantee worth a device heap overrun.
    if (mtp.ok) SI = std::max(SI, mtp.L.sh_gu.output_rows() / 2);
    if (SI <= 0) SI = 1;
    s.h       = sycl::malloc_device<float>(H, q);
    s.h2      = sycl::malloc_device<float>(H, q);
    s.resid   = sycl::malloc_device<float>(H, q);
    // qkv scratch serves both paths; the full-attn q_proj is larger
    // when attn_output_gate doubles its rows.
    int qkv_max = qkv_ch;
    for (const auto& dl : L) {
        if (dl.la_qkv.output_rows() > qkv_max) qkv_max = dl.la_qkv.output_rows();
        if (dl.q_proj.output_rows() > qkv_max) qkv_max = dl.q_proj.output_rows();
    }
    s.qkv     = sycl::malloc_device<float>(qkv_max, q);
    int aux_max = Hv * Dv;
    for (const auto& dl : L) {
        if (dl.la_z.output_rows()   > aux_max) aux_max = dl.la_z.output_rows();
        if (dl.k_proj.output_rows() > aux_max) aux_max = dl.k_proj.output_rows();
        if (dl.v_proj.output_rows() > aux_max) aux_max = dl.v_proj.output_rows();
    }
    s.zbuf    = sycl::malloc_device<float>(aux_max, q);
    s.abuf    = sycl::malloc_device<float>(aux_max * 2, q);  // a|b concatenated
    s.bbuf    = sycl::malloc_device<float>(aux_max, q);
    // attn_out is shared by the DeltaNet path (Hv*Dv) and the full
    // attention path (n_heads*head_dim). With head_dim 256 the latter is
    // 4096, equal to the former only by coincidence -- size it from both.
    {
        int ao = std::max(H, Hv * Dv);
        ao = std::max(ao, cfg.n_heads * cfg.max_head_dim());
        s.attn_out = sycl::malloc_device<float>(ao, q);
    }
    s.moe_h   = sycl::malloc_device<float>(size_t(TK) * SI + SI, q);
    s.moe_y   = sycl::malloc_device<float>(H, q);
    s.logits  = sycl::malloc_device<float>(cfg.vocab, q);
    s.rlogits = sycl::malloc_device<float>(cfg.is_moe() ? cfg.n_experts : 1, q);
    s.d_expert= sycl::malloc_device<int32_t>(TK, q);
    s.d_weight= sycl::malloc_device<float>(TK, q);
    if(tp_enabled()&&cfg.is_moe()){
        tp_expert=sycl::malloc_device<int32_t>(TK,q);
        tp_weight=sycl::malloc_device<float>(TK,q);
        if(!tp_expert||!tp_weight){err="TP MoE route allocation failed";return false;}
    }
    s.qsplit  = sycl::malloc_device<float>(size_t(cfg.n_heads) * cfg.max_head_dim(), q);
    s.gsplit  = sycl::malloc_device<float>(size_t(cfg.n_heads) * cfg.max_head_dim(), q);
    probe_buf   = sycl::malloc_device<float>(4, q);
    debug       = std::getenv("GRIMOIRE_DEBUG") != nullptr;
    // Which layer to instrument. Layer 0 is linear_attention; the
    // full_attention layers are 3, 7, 11, ... and had never been looked
    // at until now.
    { const char* e = std::getenv("GRIMOIRE_PROBE_LAYER");
      probe_layer = e ? std::atoi(e) : 0; }
    s.d_pos     = sycl::malloc_device<int32_t>(1, q);
    s.d_seq_len = sycl::malloc_device<int32_t>(1, q);
    s.d_tok   = sycl::malloc_device<int32_t>(1, q);
    // Two-stage argmax partials, allocated once (see launch_argmax).
    g_argmax_pv = sycl::malloc_device<float>(kArgmaxGroups, q);
    g_argmax_pi = sycl::malloc_device<int32_t>(kArgmaxGroups, q);
    s.d_val   = sycl::malloc_device<float>(1, q);
    s.alpha   = sycl::malloc_device<float>(Hv > 0 ? Hv : 1, q);
    s.beta    = sycl::malloc_device<float>(Hv > 0 ? Hv : 1, q);
    // holds gate|up concatenated, so twice the intermediate width
    s.sh_g    = sycl::malloc_device<float>(size_t(SI) * 2, q);
    s.sh_u    = sycl::malloc_device<float>(SI, q);
    s.sh_out  = sycl::malloc_device<float>(H, q);
    s.sh_gate_val = sycl::malloc_device<float>(1, q);
    // Decode uses row 0.  Speculative verification needs one split-K result
    // per row so its queries can share a single streamed K/V tile.
    // MAX_SPLITS, not GRAPH_SPLITS: single-token decode now sizes its split-K
    // width from the live sequence length (see decode_splits in attention.cpp),
    // so the workspace must cover the widest split it can choose. The batched
    // verify path still uses GRAPH_SPLITS and simply occupies a prefix.
    s.part    = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS * cfg.max_head_dim(), q);
    s.pm      = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS, q);
    s.pl      = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS, q);

    // ---- Qwen4-Exp scratch -------------------------------------------
    // Everything here is sized for ONE row; the batched prefill allocates
    // its own wider copies.  q4_pend / q4_pinj are the DEFERRED combine
    // the reference carries from one layer into the next.
    if (cfg.is_qwen4_exp) {
        const int HC = cfg.hc_count, WIDE = HC * H, LR = cfg.hc_lowrank;
        q4_hyper  = sycl::malloc_device<float>(size_t(WIDE), q);
        q4_normed = sycl::malloc_device<float>(size_t(WIDE), q);
        q4_gate   = sycl::malloc_device<float>(size_t(WIDE), q);
        q4_lora   = sycl::malloc_device<float>(size_t(LR > 0 ? LR : 1), q);
        q4_inj    = sycl::malloc_device<float>(size_t(HC), q);
        q4_pinj   = sycl::malloc_device<float>(size_t(HC), q);
        q4_pend   = sycl::malloc_device<float>(size_t(H), q);
        q4_tok_base = q4_tok = sycl::malloc_device<int32_t>(size_t(n_seq_slots)*max_seq, q);
        if (!q4_hyper||!q4_normed||!q4_gate||!q4_lora||!q4_inj||!q4_pinj||
            !q4_pend||!q4_tok) {
            err = "Qwen4-Exp hyper-connection scratch allocation failed";
            return false;
        }
        bool any_qsa = false, any_ple = false;
        for (const auto& dl : L) { any_qsa |= dl.qsa; any_ple |= dl.ple; }
        if (any_qsa) {
            const int IHD = cfg.indexer_head_dim;
            const int IH  = cfg.indexer_n_heads, IKV = cfg.indexer_kv_heads;
            q4_ix_width  = (IH + IKV) * IHD;
            q4_blocks_cap= max_seq / cfg.indexer_compress_ratio + 1;
            const int btopk = cfg.indexer_budget / cfg.indexer_compress_ratio;
            q4_expand_w  = cfg.indexer_budget + cfg.indexer_compress_ratio - 1;
            q4_ixqk = sycl::malloc_device<float>(size_t(q4_ix_width), q);
            q4_pool = sycl::malloc_device<float>(size_t(IHD), q);
            q4_lg   = sycl::malloc_device<float>(size_t(q4_blocks_cap), q);
            q4_blk  = sycl::malloc_device<int32_t>(size_t(btopk > 0 ? btopk : 1), q);
            q4_idx  = sycl::malloc_device<int32_t>(size_t(q4_expand_w), q);
            q4_vis  = sycl::malloc_device<int32_t>(1, q);
            q4_seq  = sycl::malloc_device<int32_t>(1, q);
            q4_qpos = sycl::malloc_device<int32_t>(1, q);
            if (!q4_ixqk||!q4_pool||!q4_lg||!q4_blk||!q4_idx||!q4_vis||
                !q4_seq||!q4_qpos) { err = "QSA scratch allocation failed"; return false; }
        }
        if (any_ple) {
            const int NH = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
            q4_ids   = sycl::malloc_device<int64_t>(size_t(NH), q);
            q4_emb   = sycl::malloc_device<float>(size_t(cfg.ple_embed_dim), q);
            q4_kv    = sycl::malloc_device<float>(size_t(WIDE + H), q);
            q4_gated = sycl::malloc_device<float>(size_t(WIDE), q);
            // state_len history rows plus this token's row, contiguous,
            // so the conv reads one array and a tap that falls into the
            // history needs no special case.
            const int state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size;
            q4_conv  = sycl::malloc_device<float>(size_t(state_len + 1) * WIDE, q);
            if (!q4_ids||!q4_emb||!q4_kv||!q4_gated||!q4_conv) {
                err = "PLE scratch allocation failed"; return false;
            }
        }
    }
    q.wait();

    if (pp_enabled() || tp_enabled()) {
        pipe_host = sycl::malloc_host<float>(size_t(H), q);
        if (!pipe_host) { err = "multiprocess host staging allocation failed"; return false; }
        pipe_host_elems = size_t(H);
        if (!pp_connect(err)) return false;
    }

    std::printf("ok\n  zeroing recurrent state ... ");
    std::fflush(stdout);
    // EVERY slot, not just the bound one.  reset() clears the sequence
    // that is live; a slot nobody has used yet holds whatever the
    // allocator left, and the first conversation admitted into it would
    // start from that.  Uninitialised recurrent state is not a crash, it
    // is a fluent answer conditioned on noise.
    for (auto& d : L) {
        if (d.dn_base)
            q.memset(d.dn_base, 0, d.dn_slot * size_t(n_seq_slots) * sizeof(float));
        if (d.conv_base)
            q.memset(d.conv_base, 0, d.conv_slot * size_t(n_seq_slots) * sizeof(float));
    }
    q.wait_and_throw();
    reset();
    std::printf("ok\n");
    vram_gb = double(bytes) / 1073741824.0;
    load_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();

    // ---- capability matrix ----------------------------------------
    // Several features degrade SILENTLY: speculation simply does not load
    // under TP or PP, and TP prompt processing falls back to a
    // token-at-a-time path.  A run that quietly lost speculation looks
    // like a slow model rather than a disabled feature, and benchmarking
    // it as though it were the full configuration is how a fallback ends
    // up being reported as a result.  State what is actually active.
    {
        std::printf("  capabilities:\n");
        if (tp_enabled())
            std::printf("    parallel      TENSOR, rank %d of %d\n", tp_rank, tp_world);
        else if (pp_enabled())
            std::printf("    parallel      PIPELINE, rank %d of %d, layers [%d,%d)\n",
                        pp_rank, pp_world, pp_begin, pp_end);
        else if (pipeline)
            std::printf("    parallel      single-process weight split "
                        "(kernels all on device 0 -- see the warning above)\n");
        else
            std::printf("    parallel      none (single device)\n");

        const bool par = tp_enabled() || pp_enabled();
        if (!spec_verify_available() && (mtp.ok || dflash2.ok ||
                                         (pp_enabled() && (pp_spec || pp_dflash))))
            std::printf("    speculation   DISABLED -- recurrent "
                        "(linear-attention) model and no batched verify here, "
                        "so a rejected draft could not be rolled back exactly\n");
        else if (mtp.ok)       std::printf("    speculation   MTP%s\n",
                                   pp_enabled() ? " (head on this, the last stage)" : "");
        else if (pp_enabled() && pp_spec)
            std::printf("    speculation   MTP (head on the last stage; this "
                        "stage drafts in lockstep with it)\n");
        else if (dflash2.ok)   std::printf("    speculation   DFlash%s%s%s\n",
                                   dflash2.v2 ? "2" : "",
                                   dflash2.selector_ok ? " + candidate selector"
                                                       : " (argmax draft, no selector)",
                                   tp_enabled() ? " (drafter replicated on every rank)"
                                 : pp_enabled() ? " (drafter on this, the last stage;"
                                                  " taps forwarded from every stage)"
                                                : "");
        else if (pp_enabled() && pp_dflash > 0)
            std::printf("    speculation   DFlash (drafter on the last stage; "
                        "this stage captures %d target tap%s and forwards "
                        "them, and drafts in lockstep)\n",
                        pp_taps, pp_taps == 1 ? "" : "s");
        else if (par && std::getenv("GRIMOIRE_DFLASH_MODEL"))
            std::printf("    speculation   DISABLED -- a DFlash drafter was "
                        "given but did not load\n");
        else if (par && mtp_enabled())
            std::printf("    speculation   DISABLED -- MTP requested but the "
                        "head did not load\n");
        else                   std::printf("    speculation   none\n");

        // Say what the cache DECIDED, not what was asked for.  The
        // depth is fixed at allocation from the environment, but
        // prefix_cache_usable() can refuse afterwards -- a drafter loads
        // later in build() than the cache is sized, so asking for eight
        // slots and a DFlash model reserves eight copies of the KV cache
        // and then never uses one.  That is worth a line rather than a
        // silent multiple of VRAM.
        if (n_seq_slots > 1 || prefix_cache_enabled()) {
            // Name the REAL reason -- from prefix_cache_usable()'s OWN
            // list, not a second copy of it (external audit finding 2,
            // 2026-09-21).  This used to reconstruct the reasons here
            // independently and drifted: it kept naming MTP after MTP
            // was deliberately allowed, and never learned about the
            // Qwen4-Exp exclusion at all.  One function, read here and
            // by prefix_cache_usable() itself, cannot drift from itself.
            const std::string why = prefix_cache_unusable_reason();
            if (!why.empty())
                std::printf("    prefix cache  off (%d sequence slot%s "
                            "reserved) -- %s\n", n_seq_slots,
                            n_seq_slots == 1 ? "" : "s", why.c_str());
            else
                std::printf("    prefix cache  %d conversation%s resident\n",
                            n_seq_slots, n_seq_slots == 1 ? "" : "s");
        }
        // And say whether requests will be batched, which is a separate
        // question from whether their contexts are cached.
        {
            const std::string bw = batch_unsupported_reason();
            if (bw.empty())
                std::printf("    batching      up to %d sequences per step\n",
                            std::min(n_seq_slots, kMaxBatchRows));
            else
                std::printf("    batching      one at a time -- %s\n", bw.c_str());
        }
        // A device with no XMX cannot run the batched path at all (see
        // Grimoire::prefill).  Saying "batched" there would be the exact
        // silent-fallback problem this matrix exists to remove.
        // Same predicate Grimoire::prefill uses, for the same reason.
        const bool batched_ok = device_can_matrix(q);
        std::printf("    prefill       %s\n",
            tp_enabled()  ? "SEQUENTIAL fallback (no batched TP prefill) -- "
                            "do not benchmark this as prompt-processing throughput"
          : !batched_ok   ? "SEQUENTIAL fallback (not a GPU, no matrix hardware) -- "
                            "do not benchmark this as prompt-processing throughput"
                          : "batched");
        {
            // Report what set_norm_convention actually installed, not a
            // second copy of the decision.  The old line was a ternary on
            // cfg.is_k2 alone, so gemma-4 -- which runs plain w like K2
            // but ungrouped -- printed "(1 + w)" while the kernels ran w.
            int ng = 1; float noff = 1.0f;
            get_norm_convention(&ng, &noff);
            char nb[96];
            std::snprintf(nb, sizeof nb, "%s, %s",
                ng > 1 ? "grouped" : "whole-row",
                noff == 0.0f ? "weight applied directly (w)"
                             : "zero-centered (1 + w)");
            std::printf("    norms         %s\n", nb);
        }
        // An acceptance rate measured under the compatibility norm is not
        // comparable to one measured under the checkpoint's own weights.
        // Say which is live so the two never get averaged together.
        if (cfg.is_muse && dflash2.ok)
            std::printf("    draft ctx-K   %s\n", dflash2.knorm_layer0
                ? "layer 0's k_norm for ALL layers "
                  "(GRIMOIRE_MUSE_KNORM_LAYER0, Fusion-matching -- NOT the "
                  "checkpoint)"
                : "per-layer checkpoint k_norm");
    }
    return true;
}

// ---------------------------------------------------------------------
// A new sequence must start from a zeroed recurrent state. The DeltaNet
// state carries the ENTIRE history in a fixed buffer, so forgetting this
// silently conditions the next conversation on the previous one -- with
// no error and no obvious symptom.
// ---------------------------------------------------------------------
void Grimoire::reset() {
    // MOVE BEFORE WIPING.  The live pointers are a view into one slot,
    // so zeroing "the state" zeroes THAT CONVERSATION.  Staying put
    // would evict the most recently used slot, which is the worst
    // possible choice; taking a free one, or the least recently used, is
    // the eviction this cache is supposed to do.
    if (prefix_cache_usable() && n_seq_slots > 1) {
        prefix_hit = -1;                       // not resuming anything
        bind_seq_slot(prefix_slot_for_write());
    }
    clear_seq_slot(seq_slot);
}

// Empty a NAMED slot and leave it live.
//
// Binding first is the whole point, and getting it backwards is a bug
// that only one kind of model can show.  A batch driver that reset() and
// then bound would clear whatever slot happened to be live -- somebody
// else's conversation -- and leave this one starting on whatever its
// slot still held.  On a dense or MoE model that is invisible: a stale
// KV row past `pos` is masked out and the answers are exactly right.  On
// a hybrid model the ENTIRE history is the conv ring and the DeltaNet
// state, so the sequence starts mid-thought and stays fluent.  It took
// prompts long enough for that state to accumulate before any gate could
// see it.
void Grimoire::clear_seq_slot(int j) {
    bind_seq_slot(j);
    // A new sequence inherits nothing, this flag included: the buffer
    // still holds the previous request's states, and they describe a
    // different sequence.
    spec_hidden_valid = false;
    if (!prefix_slots.empty() && seq_slot < int(prefix_slots.size()))
        prefix_slots[size_t(seq_slot)].valid = false;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const int qkv_ch = 2 * cfg.lin_k_heads * cfg.lin_k_dim + Hv * Dv;
    for (auto& d : L) {
        if (d.dn_state)
            q.memset(d.dn_state, 0, size_t(Hv) * Dv * Dk * sizeof(float));
        if (d.conv_ring)
            q.memset(d.conv_ring, 0, size_t(qkv_ch) * (cfg.conv_kernel - 1) * sizeof(float));
        // A PLE layer's dilated conv reads (kernel-1)*dilation rows of
        // history.  Zero is what "before the start of the sequence"
        // means, and a stale window from the previous request is a wrong
        // number rather than an error.
        if (d.ple_hist) {
            const int state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size;
            q.memset(d.ple_hist, 0,
                     size_t(state_len > 0 ? state_len : 1) *
                     size_t(cfg.hc_count) * cfg.hidden * sizeof(float));
        }
    }
    // MTP head KV cache. Nothing outside mtp_draft ever writes it: prefill
    // does not touch it and reset() did not clear it, so a long-lived server
    // carried draft K/V from EARLIER REQUESTS into the next one and the head
    // attended over stale context. See 3b5b510, which localised the whole TG
    // gap to acceptance and left this as its untested leading hypothesis:
    // a1a92eb's 92% acceptance / 37.6 TG was measured in a FRESH CLI PROCESS
    // whose MTP cache was newly allocated and effectively zero, i.e. with the
    // head's attention contributing nothing -- while the server accumulates
    // real K/V and so attends to WRONG context rather than NO context.
    // The arrival-order signature it predicted is exactly what the container
    // logs show on 2026-09-05: the first requests after a restart give
    // 33% / 0% / 53% acceptance and every request after settles at 11-26%.
    // Zeroing here reproduces the fresh-process condition per request.
    //
    // This is a DIAGNOSTIC, not the real fix. If it lifts acceptance, the
    // actual fix is to POPULATE this cache during prefill (batched forward of
    // the MTP layer over the prompt using the hidden states prefill already
    // has in `bh`), which is what vLLM does and why it holds ~52 TG at 4.4k
    // context. No batched path exists for that layer today.
    if (mtp.ok && mtp.L.k_cache && mtp.L.v_cache) {
        const size_t kv_bytes = size_t(mtp.L.kv_heads) * mtp.L.head_dim * max_seq;
        q.memset(mtp.L.k_cache, 0, kv_bytes);
        q.memset(mtp.L.v_cache, 0, kv_bytes);
    }
    if(s.d_pos && s.d_seq_len)set_cursor(0);
    q.wait();
    dag_tail.clear();
    dflash2.context_pos = 0;
    pos = 0;
}

// Prefix caching has to be a PIPELINE-WIDE decision.
//
// mtp.ok and dflash2.ok are LOCAL: under PP only the last stage holds
// either, so asking them lets an earlier stage cache a prompt the last
// stage declined to cache.  On the repeat that earlier stage restores and
// returns from prefill WITHOUT sending the hidden state, while the last
// stage blocks reading it -- and the two wait on each other.  Nothing
// times out and nothing logs.
//
// pp_spec and pp_dflash are the handshake answers every stage shares, so
// ask those instead.  Plain PP without a drafter is unaffected: every
// stage then makes the same decision, which is what made caching safe
// there in the first place.
bool Grimoire::prefix_cache_usable() const {
    return prefix_cache_unusable_reason().empty();
}

std::string Grimoire::prefix_cache_unusable_reason() const {
    if (!prefix_cache_enabled()) return "GRIMOIRE_PREFIX_CACHE is not set";
    if (cfg.is_muse) return "not usable with Muse";
    // Qwen4-Exp (external audit, 2026-09-21).  prefill_qwen4_exp()'s own
    // comment already explains why it never calls save_prefix()/
    // restore_prefix() itself: the snapshot covers ordinary KV and
    // DeltaNet state, and knows nothing about the QSA indexer's two key
    // caches, a PLE layer's conv history, or the q4_tok n-gram history --
    // so a restored snapshot would be half this conversation and half
    // whichever one saved it, silently.  That exclusion was deliberate
    // and specific to prefill_qwen4_exp()'s own call path.  Lifting the
    // MTP refusal above opened a GENERIC path -- the end-of-request
    // snapshot in generation.hpp and the growing-prefix restore in
    // prefill() -- that never routes through prefill_qwen4_exp() at all
    // and so never saw that exclusion.  Nothing here re-checked it for
    // the new path; this does.
    if (cfg.is_qwen4_exp) return "not usable with Qwen4-Exp";
    // MTP IS ALLOWED, AND WAS REFUSED FOR A REASON THAT WAS NOT ABOUT IT
    // (rule 13).  The refusal arrived with the F2 fix at 4cdcba8, whose
    // subject was a PP DEADLOCK: an earlier stage cached a prompt the
    // last stage declined to cache, returned from prefill without
    // sending the hidden state, and the last stage blocked forever.  PP
    // is refused outright three lines down, so that case is covered
    // twice and this half only ever cost single-process runs the cache.
    //
    // What a drafter and a restored prefix actually interact through is
    // the drafter's OWN KV cache, which the snapshot does not cover.  On
    // a resume the head would attend to whatever conversation ran last.
    // That cannot make the output wrong -- speculation is exact, every
    // draft is verified and a bad one is rejected (test_spec_e2e) -- it
    // can only lower acceptance.  restore zeroes that cache, which is
    // the same condition every request already starts from today
    // (reset() clears it per request, deliberately; see mtp_warm).
    //
    // DFlash is still refused, and NOT by elimination: its drafter
    // ingests target context through taps captured during prefill, and
    // dflash2.context_pos counts from the last reset.  A resume would
    // leave it pointing into a span whose taps were never captured.
    // Lifting that needs the tap buffer in the snapshot; nobody has
    // done it, so it says so rather than being quietly allowed.
    if (dflash2.ok) return "not usable with a DFlash drafter";
    // Under PP, refuse outright.
    //
    // A cache HIT makes prefill() return before it sends the hidden state,
    // so the stages only stay in step if every one of them hits.  Nothing
    // guarantees that: GRIMOIRE_PREFIX_CACHE is read per process, each
    // stage tests its OWN valid flag and token list, and save_prefix() can
    // fail its allocation on one stage and succeed on another (its return
    // value is not checked).  One stage skipping the transfer while
    // another waits for it is the deadlock this predicate was written to
    // remove, and the drafter was only the loudest way to reach it.
    //
    // Making this work needs a per-request hit/miss agreement -- a hop
    // like pp_spec, but on every prefill rather than once at connect.
    // Until that exists, single-process and TP keep the cache and PP does
    // not.  Losing a prefix cache costs one prefill; a deadlocked pipeline
    // costs the request and the process.
    if (pp_enabled()) return "not usable under pipeline parallel";
    return {};
}

bool Grimoire::save_prefix(const std::vector<int32_t>& tokens, bool output_valid) {
    if (!prefix_cache_usable() || tokens.empty()) return false;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const size_t dn_bytes = size_t(Hv) * Dv * Dk * sizeof(float);
    const size_t conv_bytes = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
        Hv * Dv) * (cfg.conv_kernel - 1) * sizeof(float);
    if (prefix_slots.empty()) prefix_slots.resize(size_t(n_seq_slots));
    // Remember which slot this conversation owns.  A request can save
    // TWICE -- prefill() snapshots the prompt at start_pos 0, and the
    // caller snapshots prompt-plus-reply at the end -- and without this
    // the second save sees prefix_hit == -1, takes a DIFFERENT free
    // slot, and every cold request costs two slots: one holding a
    // prompt-only snapshot that the longer one supersedes immediately.
    // With four slots and two agents that is the whole cache spent on
    // dead duplicates, and each duplicate is a full copy of the KV.
    // The slot this conversation ran IN is the slot it saves to: its KV
    // rows are already there, because the live pointers have been a view
    // onto them since the request started.  Writing anywhere else would
    // mean copying the cache, which is the thing this avoids.
    prefix_hit = seq_slot;
    PrefixCache& prefix_cache = prefix_slots[size_t(prefix_hit)];
    if (prefix_cache.layers.empty()) {
        // A non-empty `layers` is what says "allocated" to the next call,
        // so a failure must leave it EMPTY again.  Returning with it
        // half-filled made the next save memcpy into the null entries --
        // on the card a device write through null, not a clean error.
        auto unwind = [&] {
            for (auto& c : prefix_cache.layers)
                for (float* p : {c.dn, c.conv}) if (p) sycl::free(p, q);
            prefix_cache.layers.clear();
            if (prefix_cache.hidden) sycl::free(prefix_cache.hidden, q);
            if (prefix_cache.logits) sycl::free(prefix_cache.logits, q);
            prefix_cache.hidden = prefix_cache.logits = nullptr;
            std::fprintf(stderr, "  prefix cache allocation failed\n");
            return false;
        };
        prefix_cache.layers.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            auto& c = prefix_cache.layers[i];
            const auto& d = L[i];
            if (d.dn_state) c.dn = sycl::malloc_device<float>(dn_bytes / sizeof(float), q);
            if (d.conv_ring) c.conv = sycl::malloc_device<float>(conv_bytes / sizeof(float), q);
            if ((d.dn_state && !c.dn) || (d.conv_ring && !c.conv)) return unwind();
        }
        prefix_cache.hidden = sycl::malloc_device<float>(cfg.hidden, q);
        prefix_cache.logits = sycl::malloc_device<float>(cfg.vocab, q);
        if (!prefix_cache.hidden || !prefix_cache.logits) return unwind();
    }
    // The RECURRENT state is still copied, and the asymmetry is the
    // point.  A KV row beyond `pos` is dead -- the next write overwrites
    // it -- so a slot's cache is correct just by being left alone.  The
    // conv ring and the DeltaNet state are order-dependent: a prefill
    // that submitted work and then failed has already advanced them, and
    // the retry has to put them back.  They are also small: constant in
    // context length, where the cache is linear in it.
    for (size_t i = 0; i < L.size(); ++i) {
        const auto& d = L[i]; auto& c = prefix_cache.layers[i];
        if (d.dn_state) { q.memcpy(c.dn, d.dn_state, dn_bytes); g_prefix_bytes_copied += long(dn_bytes); }
        if (d.conv_ring) { q.memcpy(c.conv, d.conv_ring, conv_bytes); g_prefix_bytes_copied += long(conv_bytes); }
    }
    if (output_valid) {
        q.memcpy(prefix_cache.hidden, s.h, size_t(cfg.hidden) * sizeof(float));
        q.memcpy(prefix_cache.logits, s.logits, size_t(cfg.vocab) * sizeof(float));
        g_prefix_bytes_copied += long(size_t(cfg.hidden) * sizeof(float)) +
                                 long(size_t(cfg.vocab) * sizeof(float));
    }
    q.wait_and_throw();
    prefix_cache.output_valid = output_valid;
    prefix_cache.tokens = tokens;
    prefix_cache.valid = true;
    prefix_cache.used = ++prefix_clock;
    return true;
}

// How many leading tokens of `tokens` the snapshot already covers.
//
// The cache used to demand EQUALITY, which is the one thing a chat never
// gives you: every turn is the previous prompt plus a reply plus a new
// message, so the snapshot is always a strict PREFIX of the next request
// and the cache missed every single time.  A ten-turn conversation
// re-read its first nine turns on turn ten, at concurrency one, by
// design.
//
// A prefix match is the whole fix.  The snapshot is taken at the end of
// a request, so it describes the state after exactly those tokens --
// which is precisely the state a longer prompt beginning with them wants
// to start from.  Returning 0 means no reuse and the caller prefills
// everything, exactly as before.
//
// It has to be a PREFIX and not a longest-common-prefix: the recurrent
// state of a hybrid model is not indexed by position, so it cannot be
// rewound to an arbitrary point.  The only position it can resume from
// is the one it was snapshotted at.
int Grimoire::prefix_reuse(const std::vector<int32_t>& tokens,
                          const std::vector<bool>* busy) const {
    prefix_hit = -1;
    if (!prefix_cache_usable()) return 0;
    int best = 0;
    for (size_t sl = 0; sl < prefix_slots.size(); ++sl) {
        const PrefixCache& c = prefix_slots[sl];
        if (!c.valid || (busy && (*busy)[sl])) continue;
        const size_t n = c.tokens.size();
        // Leave at least one token to process: prefill() owes s.logits for
        // the last row, and a zero-token prefill would leave the caller
        // reading the PREVIOUS request's logits -- in vocabulary, right
        // length, and the wrong continuation (rule 12's failure mode).
        if (n == 0 || n >= tokens.size()) continue;
        if (!std::equal(c.tokens.begin(), c.tokens.end(), tokens.begin()))
            continue;
        // Longest wins.  Two conversations can share an opening -- the
        // same system prompt and tool list is the NORMAL case for agents
        // -- so the first match is not the best one.
        if (int(n) > best) { best = int(n); prefix_hit = int(sl); }
    }
    return best;
}

// Move the live KV view onto slot j.  This is the whole of "switch
// conversations" for the cache: the rows never move, only the pointer
// that names them.  It is O(layers) pointer writes where the copy it
// replaced was linear in max_seq -- and, more to the point, it is what
// lets several sequences be resident at once, which is what a batched
// decode across conversations will need.
//
// graph_ok goes false because a recorded command graph BAKES the
// pointers it was captured with.  Replaying one after a rebind would
// write this conversation's keys into the previous one's rows: correct
// arithmetic, wrong sequence, and nothing to see in the output but a
// reply that drifts.
// What stops a batch, named.  Rule 10: the engine says what it can RUN,
// and it says which feature is missing rather than returning false.
//
// Every entry here is a real coupling, not caution:
//
//  * a recurrent layer used to be here: one conv ring and one DeltaNet
//    state for the whole engine meant two conversations stepped together
//    would advance the same state and each would read the other's
//    history.  Both are now allocated per sequence slot, like the KV
//    cache, so hybrid models -- Ornith among them -- batch.
//  * Muse, gemma-4 and Qwen4-Exp each have their OWN batched path with
//    their own residual graph.  The loop this parameter modifies is the
//    Qwen graph; running them through it would contradict their decode
//    token for token and still read as English.
//  * TP and PP add collectives whose shape every rank must agree on, and
//    nothing here negotiates a batch width across ranks.
//  * a drafter and a batch both want the verify path; combining them is
//    a scheduling question nobody has answered yet.
std::string Grimoire::batch_unsupported_reason() const {
    if ((pp_enabled() || tp_enabled()) && (pp_spec || pp_dflash || mtp.ok || dflash2.ok))
        return "distributed speculative batching is not implemented";
    if (n_seq_slots < 2)
        return "only one sequence slot -- set GRIMOIRE_SEQ_SLOTS";
    if (!device_can_matrix(q) && !std::getenv("GRIMOIRE_BATCHED_PREFILL_NOXMX"))
        return "this device has no matrix hardware "
               "(GRIMOIRE_BATCHED_PREFILL_NOXMX=1 runs it slowly, for checking)";
    return {};
}

// Step M resident sequences by one token each.  The tokens come back in
// row order; the caller owns each sequence's position and advances it.
bool Grimoire::decode_batch(const std::vector<int32_t>& toks,
                            const std::vector<int>& slots,
                            const std::vector<int>& poss,
                            std::vector<int32_t>& out) {
    if (toks.size() != slots.size() || toks.size() != poss.size())
        throw std::invalid_argument("batch row arrays disagree in length");
    SeqBatch b{slots.data(), poss.data()};
    return prefill(toks, &out, &b);
}

// The drafter's KV cache is NOT in the snapshot, so a resumed request
// would find it holding whatever conversation ran last.  That cannot
// make the answer wrong -- every draft is verified and a bad one is
// rejected -- but it would have the head attending to another chat, and
// acceptance is the one thing a drafter is for.  Zeroing is exactly the
// condition reset() already gives every request today.
long g_spec_batch_steps=0, g_spec_batch_proposals=0, g_spec_batch_accepted=0, g_spec_batch_sequences=0;

bool Grimoire::decode_spec_batch(const std::vector<int32_t>& tokens,
        const std::vector<int>& slots, const std::vector<int>& positions,
        const std::vector<int>& remaining, std::vector<std::vector<int32_t>>& replies,
        std::vector<int>& consumed) {
    const int n=int(tokens.size());
    if(n==0 || n>kSpecBatch || slots.size()!=tokens.size() ||
       positions.size()!=tokens.size() || remaining.size()!=tokens.size())
        throw std::invalid_argument("invalid speculative batch shape");
    init_draft_slots();
    std::vector<int32_t> candidates;
    std::vector<int> candidate_slots, candidate_positions, begin, lengths;
    const char* configured=std::getenv("GRIMOIRE_MTP_K");
    const int requested=dflash2.ok?dflash_block_rows()-1:
        std::clamp(configured?std::atoi(configured):3,0,15);
    const int max_depth=std::max(0,std::min(requested,kSpecBatch/n-1));
    for(int row=0;row<n;++row) {
        sync(); bind_seq_slot(slots[size_t(row)]);
        pos=positions[size_t(row)]; set_cursor(pos);
        q.memcpy(s.h,draft_slots[size_t(seq_slot)].hidden,size_t(cfg.hidden)*sizeof(float));
        int depth=std::max(0,std::min({max_depth,max_seq-pos-1,remaining[size_t(row)]-1}));
        // DFlash always executes its full block, even when only a shorter
        // prefix will be verified. At the context boundary verify the anchor.
        if(dflash2.ok && pos+dflash_block_rows()>max_seq) depth=0;
        std::vector<int32_t> block{tokens[size_t(row)]};
        if(depth>0) {
            if(dflash2.ok) {
                std::vector<int32_t> draft;
                if(!dflash_draft(block[0],pos,draft) || draft.empty())
                    throw std::runtime_error("batched DFlash proposal failed");
                block.insert(block.end(),draft.begin(),draft.begin()+std::min(depth,int(draft.size())));
            } else {
                int token=block[0];
                for(int k=0;k<depth;++k) {
                    token=mtp_draft(token,pos+k,k>0);
                    if(token<0 || token>=cfg.vocab)
                        throw std::runtime_error("batched MTP proposal failed");
                    block.push_back(token);
                }
            }
        }
        begin.push_back(int(candidates.size()));
        lengths.push_back(int(block.size()));
        for(size_t k=0;k<block.size();++k) {
            candidates.push_back(block[k]); candidate_slots.push_back(seq_slot);
            candidate_positions.push_back(pos+int(k));
        }
    }
    SeqBatch batch{candidate_slots.data(),candidate_positions.data(),true};
    std::vector<int32_t> verified;
    spec_hidden_valid=false;
    if(!prefill(candidates,&verified,&batch,false) ||
       verified.size()!=candidates.size() || !spec_hidden_valid)
        throw std::runtime_error("batched speculative target verification failed");
    replies.assign(size_t(n),{}); consumed.assign(size_t(n),0);
    const size_t dn_n=size_t(cfg.lin_v_heads)*cfg.lin_v_dim*cfg.lin_k_dim;
    ++g_spec_batch_steps; g_spec_batch_sequences+=n;
    for(int row=0;row<n;++row) {
        const int first=begin[size_t(row)], count=lengths[size_t(row)];
        int accepted=1;
        while(accepted<count && candidates[size_t(first+accepted)]==verified[size_t(first+accepted-1)])
            ++accepted;
        g_spec_batch_proposals+=count-1; g_spec_batch_accepted+=accepted-1;
        bind_seq_slot(slots[size_t(row)]);
        const int last=first+accepted-1;
        size_t doff=0,coff=0;
        for(auto& layer:L) {
            if(layer.dn_state) {
                q.memcpy(layer.dn_state,spec_dn_steps+size_t(last)*spec_dn_elems+doff,
                         dn_n*sizeof(float)); doff+=dn_n;
            }
            if(layer.conv_ring) {
                q.memcpy(layer.conv_ring,batch_conv_steps+size_t(last)*spec_conv_elems+coff,
                         layer.conv_slot*sizeof(float)); coff+=layer.conv_slot;
            }
        }
        if(mtp.ok && !dflash2.ok) for(int k=0;k<accepted;++k)
            mtp_warm(spec_hidden_steps+int64_t(first+k)*cfg.hidden,
                     verified[size_t(first+k)],positions[size_t(row)]+k);
        q.memcpy(draft_slots[size_t(seq_slot)].hidden,
                 spec_hidden_steps+int64_t(last)*cfg.hidden,size_t(cfg.hidden)*sizeof(float));
        q.memcpy(s.h,spec_hidden_steps+int64_t(last)*cfg.hidden,size_t(cfg.hidden)*sizeof(float));
        pos=positions[size_t(row)]+accepted; set_cursor(pos);
        // Accepted candidate outputs followed by the verifier's bonus.
        for(int k=1;k<accepted;++k) replies[size_t(row)].push_back(candidates[size_t(first+k)]);
        replies[size_t(row)].push_back(verified[size_t(last)]);
        consumed[size_t(row)]=accepted;
    }
    sync();
    return true;
}

void Grimoire::clear_drafter_cache() {
    if (!mtp.ok || !mtp.L.k_cache || !mtp.L.v_cache) return;
    const size_t kv_bytes = size_t(mtp.L.kv_heads) * mtp.L.head_dim * max_seq;
    q.memset(mtp.L.k_cache, 0, kv_bytes);
    q.memset(mtp.L.v_cache, 0, kv_bytes);
}

void Grimoire::init_draft_slots() {
    if(!draft_slots.empty() || !(mtp.ok || dflash2.ok)) return;
    // ALL OR NOTHING.  A non-empty draft_slots is what says "done" above,
    // so a throw halfway through used to leave later slots holding null
    // cache pointers that the next bind_seq_slot() installed as the live
    // drafter caches -- a device write through null, which on the card is
    // a DEVICE_LOST (rule 1), not a clean error.  Today the first call is
    // build()'s closing reset(), where a throw ends the load anyway, so
    // this is a guard for any caller that survives the throw and keeps
    // the engine.  Build into locals; publish only once all succeeded.
    std::vector<DraftSlot> built(static_cast<size_t>(n_seq_slots));
    float* conv_steps=nullptr;
    size_t extra=0;
    try {
        const size_t mkbytes=size_t(mtp.L.kv_heads)*mtp.L.head_dim*max_seq;
        const size_t dkbytes=size_t(dflash2.kv_heads)*dflash2.head_dim*max_seq;
        const size_t halfbytes=size_t(dflash2.num_blocks)*dflash2.block_size*
                               dflash2.kv_heads*dflash2.head_dim*sizeof(sycl::half);
        for(int j=0;j<n_seq_slots;++j) {
            auto& slot=built[size_t(j)];
            auto allocate=[&](size_t bytes)->void* {
                auto* ptr=sycl::malloc_device<uint8_t>(std::max<size_t>(1,bytes),q);
                if(!ptr) throw std::bad_alloc();
                slot.owned.push_back(ptr); q.memset(ptr,0,bytes);
                extra+=bytes;
                return ptr;
            };
            slot.hidden=static_cast<float*>(allocate(size_t(cfg.hidden)*sizeof(float)));
            if(mtp.ok) {
                slot.mk=j?static_cast<uint8_t*>(allocate(mkbytes)):mtp.L.k_cache;
                slot.mv=j?static_cast<uint8_t*>(allocate(mkbytes)):mtp.L.v_cache;
            }
            if(dflash2.target_aux) slot.aux=j?static_cast<float*>(allocate(
                size_t(max_seq)*dflash2.target_layers.size()*cfg.hidden*sizeof(float))):dflash2.target_aux;
            for(auto& layer:dflash2.layers) {
                slot.k.push_back(layer.k_cache?(j?static_cast<uint8_t*>(allocate(dkbytes)):layer.k_cache):nullptr);
                slot.v.push_back(layer.v_cache?(j?static_cast<uint8_t*>(allocate(dkbytes)):layer.v_cache):nullptr);
                slot.k16.push_back(layer.k_cache_f16?(j?static_cast<sycl::half*>(allocate(halfbytes)):layer.k_cache_f16):nullptr);
                slot.v16.push_back(layer.v_cache_f16?(j?static_cast<sycl::half*>(allocate(halfbytes)):layer.v_cache_f16):nullptr);
            }
        }
        if(spec_conv_elems) {
            conv_steps=sycl::malloc_device<float>(size_t(kSpecBatch)*spec_conv_elems,q);
            if(!conv_steps) throw std::bad_alloc();
            extra+=size_t(kSpecBatch)*spec_conv_elems*sizeof(float);
        }
        q.wait_and_throw();
    } catch(...) {
        q.wait();
        for(auto& slot:built) for(void* ptr:slot.owned) sycl::free(ptr,q);
        if(conv_steps) sycl::free(conv_steps,q);
        std::fprintf(stderr,"  drafter: per-sequence caches for %d slots could not be "
                     "allocated -- lower GRIMOIRE_SEQ_SLOTS or --ctx\n",n_seq_slots);
        throw;
    }
    draft_slots.swap(built);
    batch_conv_steps=conv_steps;
    // Allocated by build()'s closing reset() and never passed to acct(),
    // so the "GiB resident" figure printed at load does not include it.
    // For a DFlash drafter it is dominated by the tap buffer: max_seq *
    // taps * hidden * 4 bytes for EVERY slot past the first (Ornith's 8
    // taps x 2048 is 64 KiB per token of --ctx, per slot).  Say it, so
    // GRIMOIRE_SEQ_SLOTS can be sized against the card.
    if(n_seq_slots>1)
        std::fprintf(stderr,"  drafter: per-sequence caches for %d slots, %.2f GiB "
                     "beyond the load-time budget\n",n_seq_slots,double(extra)/double(1ull<<30));
}

void Grimoire::bind_seq_slot(int j) {
    init_draft_slots();
    if (j < 0 || j >= n_seq_slots || j == seq_slot) return;
    for (auto& d : L) {
        if (d.k_base) d.k_cache = d.k_base + size_t(j) * d.kv_slot;
        if (d.v_base) d.v_cache = d.v_base + size_t(j) * d.kv_slot;
        if (d.dn_base) d.dn_state = d.dn_base + size_t(j) * d.dn_slot;
        if (d.conv_base) d.conv_ring = d.conv_base + size_t(j) * d.conv_slot;
        if (d.ix_raw_base) d.ix_kraw=d.ix_raw_base+size_t(j)*max_seq*cfg.indexer_head_dim;
        if (d.ix_cmp_base) d.ix_kcmp=d.ix_cmp_base+size_t(j)*
            (max_seq/std::max(1,cfg.indexer_compress_ratio)+1)*cfg.indexer_head_dim;
        if (d.ple_hist_base) d.ple_hist=d.ple_hist_base+size_t(j)*
            std::max(1,(cfg.ple_conv_kernel-1)*cfg.ngram_size)*cfg.hc_count*cfg.hidden;
    }
    if(q4_tok_base) q4_tok=q4_tok_base+size_t(j)*max_seq;
    if(!draft_slots.empty()) {
        draft_slots[size_t(seq_slot)].context=dflash2.context_pos;
        auto& slot=draft_slots[size_t(j)];
        mtp.L.k_cache=slot.mk; mtp.L.v_cache=slot.mv;
        dflash2.target_aux=slot.aux; dflash2.context_pos=slot.context;
        for(size_t i=0;i<dflash2.layers.size();++i) {
            auto& layer=dflash2.layers[i];
            layer.k_cache=slot.k[i]; layer.v_cache=slot.v[i];
            layer.k_cache_f16=slot.k16[i]; layer.v_cache_f16=slot.v16[i];
        }
    }
    seq_slot = j;
    graph_ok = false;
}

// Which slot a request that is NOT resuming should run in.  Since the
// cache stopped copying, this is an eviction choice and not a
// bookkeeping one: whatever slot this returns is about to be cleared by
// reset() and the conversation in it is gone.
//
// A resuming request never comes here -- it runs in the slot it matched,
// which is what keeps a conversation in one slot as it grows.
int Grimoire::prefix_slot_for_write() {
    if (prefix_slots.empty()) prefix_slots.resize(size_t(n_seq_slots));
    if (prefix_hit >= 0 && size_t(prefix_hit) < prefix_slots.size())
        return prefix_hit;
    for (size_t i = 0; i < prefix_slots.size(); ++i)
        if (!prefix_slots[i].valid) return int(i);
    size_t lru = 0;
    for (size_t i = 1; i < prefix_slots.size(); ++i)
        if (prefix_slots[i].used < prefix_slots[lru].used) lru = i;
    return int(lru);
}

bool Grimoire::restore_prefix(const std::vector<int32_t>& tokens) {
    if (!prefix_cache_usable()) return false;
    int hit = -1;
    for (size_t sl = 0; sl < prefix_slots.size(); ++sl)
        if (prefix_slots[sl].valid && prefix_slots[sl].output_valid &&
            prefix_slots[sl].tokens == tokens) hit = int(sl);
    if (hit < 0) return false;
    // Claim the slot, for the same reason save_prefix() does: whatever
    // this request saves next belongs to the conversation it just
    // restored, not to a fresh slot.
    prefix_hit = hit;
    PrefixCache& prefix_cache = prefix_slots[size_t(hit)];
    prefix_cache.used = ++prefix_clock;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const size_t dn_bytes = size_t(Hv) * Dv * Dk * sizeof(float);
    const size_t conv_bytes = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
        Hv * Dv) * (cfg.conv_kernel - 1) * sizeof(float);
    // The cache does not move: bind the slot and the live pointers ARE
    // its rows.  Only the recurrent state is copied -- see save_prefix()
    // for why those two are treated differently.
    bind_seq_slot(hit);
    clear_drafter_cache();
    for (size_t i = 0; i < L.size(); ++i) {
        auto& d = L[i]; const auto& c = prefix_cache.layers[i];
        if (d.dn_state) { q.memcpy(d.dn_state, c.dn, dn_bytes); g_prefix_bytes_copied += long(dn_bytes); }
        if (d.conv_ring) { q.memcpy(d.conv_ring, c.conv, conv_bytes); g_prefix_bytes_copied += long(conv_bytes); }
    }
    q.memcpy(s.h, prefix_cache.hidden, size_t(cfg.hidden) * sizeof(float));
    q.memcpy(s.logits, prefix_cache.logits, size_t(cfg.vocab) * sizeof(float));
    g_prefix_bytes_copied += long(size_t(cfg.hidden) * sizeof(float)) +
                             long(size_t(cfg.vocab) * sizeof(float));
    pos = int(tokens.size());
    set_cursor(pos); q.wait_and_throw();
    std::printf("  prefix cache HIT: %zu tokens\n", tokens.size());
    return true;
}

// Restore the snapshot and leave the cursor AT ITS LENGTH, so the caller
// can prefill the remaining tokens on top.  Shares its body with
// restore_prefix(); the only difference is that the caller is not
// claiming the request is finished after it.
bool Grimoire::restore_prefix_upto(int n) {
    if (n <= 0 || !prefix_cache_usable()) return false;
    if (prefix_hit < 0 || size_t(prefix_hit) >= prefix_slots.size()) return false;
    PrefixCache& prefix_cache = prefix_slots[size_t(prefix_hit)];
    if (!prefix_cache.valid || size_t(n) != prefix_cache.tokens.size()) return false;
    prefix_cache.used = ++prefix_clock;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const size_t dn_bytes = size_t(Hv) * Dv * Dk * sizeof(float);
    const size_t conv_bytes = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
        Hv * Dv) * (cfg.conv_kernel - 1) * sizeof(float);
    // Same move as restore_prefix(): the KV rows are already in this
    // slot, so binding it is the restore.
    bind_seq_slot(prefix_hit);
    clear_drafter_cache();
    for (size_t i = 0; i < L.size(); ++i) {
        auto& d = L[i]; const auto& c = prefix_cache.layers[i];
        if (d.dn_state) { q.memcpy(d.dn_state, c.dn, dn_bytes); g_prefix_bytes_copied += long(dn_bytes); }
        if (d.conv_ring) { q.memcpy(d.conv_ring, c.conv, conv_bytes); g_prefix_bytes_copied += long(conv_bytes); }
    }
    if (prefix_cache.output_valid) {
        q.memcpy(s.h, prefix_cache.hidden, size_t(cfg.hidden) * sizeof(float));
        g_prefix_bytes_copied += long(size_t(cfg.hidden) * sizeof(float));
    }
    pos = n;
    set_cursor(pos);
    q.wait_and_throw();
    ++g_prefix_tokens_reused_calls;
    g_prefix_tokens_reused += n;
    return true;
}

// Admission owns the choice of physical slot. Cache lookup excludes every
// live request, including live requests with an identical system prompt.
int Grimoire::admit_sequence(const std::vector<int32_t>& prompt,
                            const std::vector<bool>& busy) {
    if (busy.size() != size_t(n_seq_slots))
        throw std::invalid_argument("sequence ownership size mismatch");
    sync();
    const int reused = prefix_reuse(prompt, &busy);
    int slot = reused ? prefix_hit : -1;
    if (slot < 0) {
        for (int i=0; i<n_seq_slots; ++i) {
            if (busy[size_t(i)]) continue;
            if (slot < 0) slot = i;
            if (size_t(i) >= prefix_slots.size() || !prefix_slots[size_t(i)].valid) {
                slot = i; break;
            }
            if (prefix_slots[size_t(i)].used < prefix_slots[size_t(slot)].used)
                slot = i;
        }
    }
    if (slot < 0) throw std::runtime_error("no idle sequence slot");
    if(serving_control && (pp_enabled() || tp_enabled()) && comm_rank()==0) {
        PPRequest request;
        request.kind=2; request.budget=slot; request.prompt=prompt;
        if(!pp_send_request(request))
            throw std::runtime_error("parallel admission broadcast failed");
    }
    try {
        auto restore = [&] {
            if (reused) {
                prefix_hit = slot;
                if (!restore_prefix_upto(reused))
                    throw std::runtime_error("sequence prefix restore failed");
            } else clear_seq_slot(slot);
        };
        restore();
        const std::vector<int32_t> tail(prompt.begin()+reused, prompt.end());
        if (!prefill(tail, nullptr, nullptr, false)) {
            // A failed prefill may already have advanced recurrent state.
            sync(); restore();
            for (int32_t t : tail)
                if (!forward(t)) throw std::runtime_error("prompt ingestion failed");
        }
        sync();
        if(!draft_slots.empty()) q.memcpy(draft_slots[size_t(slot)].hidden,s.h,size_t(cfg.hidden)*sizeof(float)).wait();
        return slot;
    } catch (...) {
        if (size_t(slot) < prefix_slots.size()) prefix_slots[size_t(slot)].valid=false;
        throw;
    }
}

void Grimoire::cache_sequence(int slot, int position,
                             const std::vector<int32_t>& prompt,
                             const std::vector<int32_t>& reply) {
    if (!prefix_cache_usable()) return;
    std::vector<int32_t> processed = prompt;
    processed.insert(processed.end(), reply.begin(), reply.end());
    // Tokens may have been delivered before being fed back to the target.
    // Never label the snapshot with state the engine has not processed.
    if (position <= 0 || size_t(position) > processed.size()) {
        if (size_t(slot) < prefix_slots.size()) prefix_slots[size_t(slot)].valid=false;
        return;
    }
    processed.resize(size_t(position));
    if(serving_control && tp_enabled() && tp_rank==0) {
        PPRequest request;
        request.kind=4; request.budget=slot; request.eos=position; request.prompt=processed;
        if(!pp_send_request(request))
            throw std::runtime_error("TP prefix commit broadcast failed");
    }
    sync(); bind_seq_slot(slot); pos=position;
    if (!save_prefix(processed, false) && size_t(slot)<prefix_slots.size())
        prefix_slots[size_t(slot)].valid=false;
}

void Grimoire::snapshot_recurrent() {
    // Opens every speculative round, so it is where the hidden-step
    // capture is invalidated: whatever the last round left in the buffer
    // does not describe this one, and only a batched verify will refill it.
    spec_hidden_valid = false;
    const size_t dn_n = size_t(cfg.lin_v_heads) * cfg.lin_v_dim * cfg.lin_k_dim;
    const size_t cv_n = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
                               cfg.lin_v_heads * cfg.lin_v_dim) *
                        (cfg.conv_kernel - 1);
    size_t doff = 0, coff = 0;
    for (const auto& d : L) {
        if (d.dn_state) {
            q.memcpy(spec_dn_state + doff, d.dn_state, dn_n * sizeof(float));
            doff += dn_n;
        }
        if (d.conv_ring) {
            q.memcpy(spec_conv_ring + coff, d.conv_ring, cv_n * sizeof(float));
            coff += cv_n;
        }
    }
}

void Grimoire::restore_recurrent(int saved_pos) {
    const size_t dn_n = size_t(cfg.lin_v_heads) * cfg.lin_v_dim * cfg.lin_k_dim;
    const size_t cv_n = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
                               cfg.lin_v_heads * cfg.lin_v_dim) *
                        (cfg.conv_kernel - 1);
    size_t doff = 0, coff = 0;
    for (auto& d : L) {
        if (d.dn_state) {
            q.memcpy(d.dn_state, spec_dn_state + doff, dn_n * sizeof(float));
            doff += dn_n;
        }
        if (d.conv_ring) {
            q.memcpy(d.conv_ring, spec_conv_ring + coff, cv_n * sizeof(float));
            coff += cv_n;
        }
    }
    pos = saved_pos;
    set_cursor(pos);
}

void Grimoire::commit_spec_prefix(int saved_pos, int accepted) {
    const size_t dn_n = size_t(cfg.lin_v_heads) * cfg.lin_v_dim * cfg.lin_k_dim;
    const int channels = 2 * cfg.lin_k_heads * cfg.lin_k_dim +
                         cfg.lin_v_heads * cfg.lin_v_dim;
    const int hist = cfg.conv_kernel - 1;
    const size_t cv_n = size_t(channels) * hist;
    const int step = accepted - 1;
    size_t doff = 0, coff = 0, xoff = 0;
    for (auto& d : L) {
        if (d.dn_state) {
            q.memcpy(d.dn_state,
                     spec_dn_steps + size_t(step) * spec_dn_elems + doff,
                     dn_n * sizeof(float));
            doff += dn_n;
        }
        if (d.conv_ring) {
            float* dst = d.conv_ring;
            const float* base = spec_conv_ring + coff;
            const float* inputs = spec_conv_inputs + xoff;
            const int count = accepted;
            q.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(cv_n), [=](sycl::id<1> id) {
                    const int c = int(id[0]) / hist;
                    const int j = int(id[0]) % hist;
                    if (count >= hist)
                        dst[int64_t(c) * hist + j] =
                            inputs[int64_t(count - hist + j) * channels + c];
                    else if (j < hist - count)
                        dst[int64_t(c) * hist + j] =
                            base[int64_t(c) * hist + j + count];
                    else
                        dst[int64_t(c) * hist + j] =
                            inputs[int64_t(j - (hist - count)) * channels + c];
                });
            });
            coff += cv_n;
            xoff += size_t(kSpecBatch) * channels;
        }
    }
    // Restoring the accepted token's hidden state matters only to the NEXT
    // draft, not to correctness: the target reads the KV cache and the
    // token id.  The sequential verify fallback never captures these steps
    // (it has no batch to capture from), so skip rather than read a buffer
    // that was never written -- speculation stays exact there, the drafts
    // after a partial rejection are just worse.
    if (spec_hidden_steps && spec_hidden_valid)
        q.memcpy(s.h, spec_hidden_steps + size_t(step) * cfg.hidden,
                 size_t(cfg.hidden) * sizeof(float));
    pos = saved_pos + accepted;
    set_cursor(pos);
}

void Grimoire::release() {
    if(!draft_slots.empty()) {
        q.wait();
        bind_seq_slot(0);
        for(auto& slot:draft_slots) for(void* ptr:slot.owned) sycl::free(ptr,q);
        draft_slots.clear();
        if(batch_conv_steps) { sycl::free(batch_conv_steps,q); batch_conv_steps=nullptr; }
    }
    // DRAIN BEFORE FREEING.  Everything below hands device pointers to
    // sycl::free while the queue may still hold work that reads them --
    // undefined, and on this runtime it surfaces as a SIGSEGV inside the
    // JIT at some LATER point, usually in the next engine that loads.
    // That is what made it look environmental: the crash never lands
    // where the fault is, and it only shows up when the timing is right,
    // which was one run in three.
    //
    // wait(), not wait_and_throw(): release() is called from a destructor
    // path (grimoire_delete) and an exception escaping there would take
    // the process down in a worse way than the leak it is reporting.
    try { q.wait(); if (q1) q1->wait(); q_aux.wait(); } catch (...) {}
    if(dflash2.fc_plan){
        auto od=load_onednn_bf16();
        if(od)od.destroy(dflash2.fc_plan);
        dflash2.fc_plan=nullptr;
    }
    for (void** p : {(void**)&gemma_vnorm, (void**)&muse_zero,
                     (void**)&muse_zero_f16})
        if (*p) { sycl::free(*p, q); *p = nullptr; }
    if(pp_prev_fd>=0){::close(pp_prev_fd);pp_prev_fd=-1;}
    if(pp_next_fd>=0){::close(pp_next_fd);pp_next_fd=-1;}
    for(int& fd:tp_peer_fd)if(fd>=0){::close(fd);fd=-1;}
    if(pp_enabled()&&pp_rank>0&&!pp_socket.empty())
        ::unlink((pp_socket+"-"+std::to_string(pp_rank)).c_str());
    if(tp_enabled()&&tp_rank==0&&!pp_socket.empty())::unlink(pp_socket.c_str());
    if (pipe_host) { sycl::free(pipe_host, q); pipe_host = nullptr; }
    if(tp_expert){sycl::free(tp_expert,q);tp_expert=nullptr;}
    if(tp_weight){sycl::free(tp_weight,q);tp_weight=nullptr;}
    if (g_argmax_pv) { sycl::free(g_argmax_pv, q); g_argmax_pv = nullptr; }
    if (g_argmax_pi) { sycl::free(g_argmax_pi, q); g_argmax_pi = nullptr; }
    // USM frees are cheap; the process usually exits right after, but a
    // server reloading models needs this to not leak 20 GB per swap.
    for (auto& d : L) {
        d.la_qkv.release(q); d.la_z.release(q); d.la_out.release(q); d.la_all.release(q);
        d.q_proj.release(q); d.k_proj.release(q);
        d.v_proj.release(q); d.qkv_proj.release(q); d.o_proj.release(q);
        d.sh_gu.release(q); d.sh_down.release(q);
        d.la_ab.release(q); d.sh_gate_q.release(q);
        d.router.release(q);
        d.v_router.release(q);
        d.v_experts_packed.release(q);
        for (auto& ve : d.v_experts) ve.release(q);
        for (void* p : {(void*)d.in_norm, (void*)d.post_norm,
                        // The bf16 sandwich norms were missing from this
                        // list entirely -- leaked by Muse as well as by
                        // gemma-4, since only their _f16 companions were
                        // named.
                        (void*)d.pre_ff_norm, (void*)d.post_ff_norm,
                        (void*)d.in_norm_f16, (void*)d.post_norm_f16,
                        (void*)d.pre_ff_norm_f16, (void*)d.post_ff_norm_f16,
                        (void*)d.la_conv, (void*)d.la_Alog, (void*)d.la_dtb,
                        (void*)d.la_norm,  (void*)d.q_norm, (void*)d.k_norm, (void*)d.gu_pack,
                        (void*)d.gu_scale, (void*)d.dn_pack, (void*)d.dn_scale,
                        (void*)d.gu_zero, (void*)d.dn_zero,
                        // Bases, not views, for the same reason as the
                        // cache below: a view points into the middle of
                        // the allocation that owns it.
                        (void*)d.dn_base, (void*)d.conv_base,
                        // k_base, NOT k_cache: the cache is a view into
                        // whichever slot was last bound, and freeing a
                        // view frees the middle of an allocation.
                        (void*)d.k_base, (void*)d.v_base,
                        (void*)d.k_cache_f16, (void*)d.v_cache_f16,
                        // Qwen4-Exp.  ple_table is HOST memory and is
                        // the largest single allocation this engine ever
                        // makes -- ~51 GB for the published checkpoint.
                        // A server that reloads a model and leaks that
                        // does not leak a buffer, it runs the box out of
                        // RAM on the second load.
                        (void*)d.hc_attn.norm, (void*)d.hc_mlp.norm,
                        (void*)d.ix_qn, (void*)d.ix_kn,
                        (void*)d.ix_raw_base, (void*)d.ix_cmp_base,
                        (void*)d.ple_nk, (void*)d.ple_nq, (void*)d.ple_nc,
                        (void*)d.ple_cw, (void*)d.ple_hist_base,
                        (void*)d.ple_mul, (void*)d.ple_size, (void*)d.ple_off,
                        const_cast<void*>(d.ple_table)})
            if (p) sycl::free(p, q);
        d.hc_attn.down.release(q); d.hc_attn.inject.release(q);
        d.hc_attn.up.release(q);
        d.hc_mlp.down.release(q);  d.hc_mlp.inject.release(q);
        d.hc_mlp.up.release(q);
        d.ix_qk.release(q);
        d.ple_key.release(q); d.ple_value.release(q);
    }
    hc_final.down.release(q); hc_final.up.release(q);
    for (void** p : {(void**)&hc_final.norm, (void**)&q4_hyper,
                     (void**)&q4_normed, (void**)&q4_gate, (void**)&q4_lora,
                     (void**)&q4_inj, (void**)&q4_pinj, (void**)&q4_pend,
                     (void**)&q4_ixqk, (void**)&q4_pool, (void**)&q4_lg,
                     (void**)&q4_blk, (void**)&q4_idx, (void**)&q4_vis,
                     (void**)&q4_seq, (void**)&q4_qpos, (void**)&q4_emb,
                     (void**)&q4_kv, (void**)&q4_gated, (void**)&q4_conv,
                     (void**)&q4_ids, (void**)&q4_tok_base})
        if (*p) { sycl::free(*p, q); *p = nullptr; }
    {
        LayerDev& d = mtp.L;
        mtp.fc.release(q);
        d.q_proj.release(q); d.k_proj.release(q);
        d.v_proj.release(q); d.o_proj.release(q);
        d.sh_gu.release(q); d.sh_down.release(q);
        d.sh_gate_q.release(q); d.router.release(q);
        for (void* p : {(void*)mtp.pre_h, (void*)mtp.pre_e, (void*)mtp.norm,
                        (void*)mtp.cat, (void*)mtp.x, (void*)mtp.h2,
                        (void*)mtp.resid, (void*)d.in_norm,
                        (void*)d.post_norm, (void*)d.q_norm, (void*)d.k_norm,
                        (void*)d.gu_pack, (void*)d.gu_scale,
                        (void*)d.dn_pack, (void*)d.dn_scale,
                        (void*)d.gu_zero, (void*)d.dn_zero,
                        (void*)d.k_cache, (void*)d.v_cache})
            if (p) sycl::free(p, q);
        mtp = {};
    }
    OneDnnW4Api muse_od=load_onednn_w4();
    if(muse_od){
        for(auto& p:muse_od_plans){
            if(p.scratch)sycl::free(p.scratch,q);
            if(p.plan)muse_od.destroy(p.plan);
        }
    }
    muse_od_plans.clear();
    OneDnnF16Api dflash_f16=load_onednn_f16();
    if(dflash_f16){
        for(auto& p:dflash_f16_plans){
            if(p.scratch)sycl::free(p.scratch,q);
            if(p.plan)dflash_f16.destroy(p.plan);
        }
    }
    dflash_f16_plans.clear();
    if(muse_od_zp){sycl::free(muse_od_zp,q);muse_od_zp=nullptr;}
    // DFlash is optional, but release every field unconditionally so a
    // partially loaded sidecar cannot leak USM when build() reports an error.
    dflash2.fc.release(q);
    dflash2.selector_hidden.release(q);
    if(dflash2.sel_ids)sycl::free(dflash2.sel_ids,q);
    if(dflash2.sel_unary)sycl::free(dflash2.sel_unary,q);
    if(dflash2.sel_hidden)sycl::free(dflash2.sel_hidden,q);
    if(dflash2.sel_scores)sycl::free(dflash2.sel_scores,q);
    dflash2.fused_context_kv.release(q);
    dflash2.shared_embed_f16.release(q);
    dflash2.shared_lm_head_f16.release(q);
    dflash2.draft_lm_head_i4.release(q);
    dflash2.draft_lm_head.release(q);
    if(dflash2.draft_embed)sycl::free(dflash2.draft_embed,q);
    if(dflash2.draft_vocab_map)sycl::free(dflash2.draft_vocab_map,q);
    dflash2.draft_embed=nullptr;
    dflash2.draft_vocab_map=nullptr;
    for (auto& d : dflash2.layers) {
        d.q.release(q); d.k.release(q); d.v.release(q); d.qkv.release(q);
        d.o.release(q);
        d.gate_up.release(q); d.down.release(q);
        d.attn_conv_proj.release(q); d.mlp_conv_proj.release(q);
        for (void* p : {(void*)d.in_norm, (void*)d.post_norm,
                        (void*)d.q_norm, (void*)d.k_norm,
                        (void*)d.in_norm_f16, (void*)d.post_norm_f16,
                        (void*)d.q_norm_f16, (void*)d.k_norm_f16,
                        (void*)d.attn_conv_base, (void*)d.mlp_conv_base,
                        (void*)d.k_cache, (void*)d.v_cache,
                        (void*)d.k_cache_f16, (void*)d.v_cache_f16})
            if (p) sycl::free(p, q);
    }
    for (void* p : {(void*)dflash2.hidden_norm, (void*)dflash2.norm,
                    (void*)dflash2.hidden_norm_f16, (void*)dflash2.norm_f16,
                    (void*)dflash2.predecessor, (void*)dflash2.successor,
                    (void*)dflash2.target_aux, (void*)dflash2.ctx,
                    (void*)dflash2.context_kv_all,
                    (void*)dflash2.h, (void*)dflash2.resid,
                    (void*)dflash2.normed, (void*)dflash2.q,
                    (void*)dflash2.k, (void*)dflash2.v,
                    (void*)dflash2.attn, (void*)dflash2.proj,
                    (void*)dflash2.q_f16, (void*)dflash2.k_f16,
                    (void*)dflash2.v_f16, (void*)dflash2.attn_f16,
                    (void*)dflash2.linear_in_f16,
                    (void*)dflash2.linear_out_f16,
                    (void*)dflash2.context_k_all_f16,
                    (void*)dflash2.context_v_all_f16,
                    (void*)dflash2.k_norm_all, (void*)dflash2.k_norm_all_f16,
                    (void*)dflash2.gate_up, (void*)dflash2.mlp,
                    (void*)dflash2.logits, (void*)dflash2.bf,
                    (void*)dflash2.a8, (void*)dflash2.a8s,
                    (void*)dflash2.tokens, (void*)dflash2.draft_ids,
                    (void*)dflash2.draft_head_i4,
                    (void*)dflash2.draft_head_i4s,
                    (void*)dflash2.draft_head_token_ids,
                    (void*)dflash2.conv_delta, (void*)dflash2.conv_scratch,
                    (void*)dflash2.block_table, (void*)dflash2.cu_q,
                    (void*)dflash2.cu_k, (void*)dflash2.seqused_k,
                    (void*)dflash2.verify_logits, (void*)dflash2.verify_bf,
                    (void*)dflash2.verify_bf_out,
                    (void*)dflash2.verify_a8, (void*)dflash2.verify_a8s,
                    (void*)dflash2.verify_ids, (void*)dflash2.fc_scratch})
        if (p) sycl::free(p, q);
    dflash2 = {};
    // A tied lm_head points AT embed; freeing both would double-free.
    if (tied_lm_head) { lm_head.payload = nullptr; lm_head.w.payload = nullptr; }
    lm_head.release(q);
    if (embed) sycl::free(embed, q);
    if (fnorm) sycl::free(fnorm, q);
    if (fnorm_f16) sycl::free(fnorm_f16, q);
    if (spec_dn_state) sycl::free(spec_dn_state, q);
    if (spec_conv_ring) sycl::free(spec_conv_ring, q);
    if (spec_dn_steps) sycl::free(spec_dn_steps, q);
    if (spec_conv_inputs) sycl::free(spec_conv_inputs, q);
    if (spec_hidden_steps) sycl::free(spec_hidden_steps, q);
    // Every slot, not just one.  N slots is N full copies of the KV and
    // recurrent state, so a server reloading a model and freeing only the
    // first would leak the rest -- and the rest is where the memory is.
    for (auto& pc : prefix_slots) {
        for (auto& c : pc.layers)
            for (void* p : {(void*)c.dn, (void*)c.conv})
                if (p) sycl::free(p, q);
        if (pc.hidden) sycl::free(pc.hidden, q);
        if (pc.logits) sycl::free(pc.logits, q);
    }
    prefix_slots.clear();
    prefix_hit = -1;
}

} // namespace b70

// =====================================================================
//  Load-and-report entry point. Uploads the model and prints what the
//  decode step will cost, from measured kernel rates.
// =====================================================================
namespace b70 {

static double meas_rate(Fmt f) {
    switch (f) {                      // GB/s, measured on Arc Pro B70
        case Fmt::BF16:  return 542.0;
        case Fmt::INT8:  return 527.0;
        case Fmt::INT4:  return 478.0;
        case Fmt::MXFP4: return 352.0;
        case Fmt::MXFP8: return 386.0;
        default:         return 380.0;
    }
}

// Run N steps and report measured tokens/sec. Uses raw token ids, so it
// exercises the whole forward pass without needing the tokenizer.
// Prompt processing. Measures the batched projection kernel at the
// model's real dimensions, which is what sets the pp ceiling: at batch M
// each weight is read once and used M times, so prefill is compute bound
// where decode is bandwidth bound.
int grimoire_bench_prefill(Grimoire& e, int M) {
    std::printf("\n  prompt processing (batch %d)\n", M);
    std::fflush(stdout);

    const Qwen35Config& c = e.cfg;
    const int H = c.hidden;
    int nlin = 0;
    for (LayerKind k : c.layer_types) if (k == LayerKind::LINEAR_ATTN) ++nlin;

    float* X = sycl::malloc_device<float>(size_t(M) * H, e.q);
    if (!X) { std::printf("    cannot allocate activations\n"); return 1; }
    // Real values, not zeros. A zero-filled input makes both kernels
    // emit zeros and any comparison between them passes trivially --
    // which would hide exactly the VNNI-layout and tile-indexing bugs
    // the check exists to catch.
    {
        std::vector<float> hx(size_t(M) * H);
        std::mt19937 rg(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : hx) v = nd(rg);
        e.q.memcpy(X, hx.data(), hx.size() * sizeof(float)).wait();
    }

    const Grimoire::LayerDevRef d0 = e.first_linear_layer();
    if (!d0.ok) { std::printf("    no linear layer found\n"); sycl::free(X, e.q); return 1; }

    // Consolidated correctness probe for the prompt-wide MoE path. One
    // run checks projection -> routing -> both expert kernels against M
    // independent decode launches before any prefill timing is trusted.
    if (c.is_moe()) {
        const Grimoire::LayerDev* md = nullptr;
        for (const auto& d : e.L) if (d.moe.cfg.num_experts) { md = &d; break; }
        const int PM = std::min(M, 4);
        if (md && PM > 0) {
            const int E = c.n_experts, Kt = c.top_k, I = c.moe_inter;
            float* rl = sycl::malloc_device<float>(size_t(PM) * E, e.q);
            int32_t *re_b = sycl::malloc_device<int32_t>(size_t(PM) * Kt, e.q);
            int32_t *re_s = sycl::malloc_device<int32_t>(size_t(PM) * Kt, e.q);
            float *rw_b = sycl::malloc_device<float>(size_t(PM) * Kt, e.q);
            float *rw_s = sycl::malloc_device<float>(size_t(PM) * Kt, e.q);
            float *mh_b = sycl::malloc_device<float>(size_t(PM) * Kt * I, e.q);
            float *mh_s = sycl::malloc_device<float>(size_t(PM) * Kt * I, e.q);
            float *my_b = sycl::malloc_device<float>(size_t(PM) * H, e.q);
            float *my_s = sycl::malloc_device<float>(size_t(PM) * H, e.q);
            bool alloc_ok = rl && re_b && re_s && rw_b && rw_s && mh_b && mh_s && my_b && my_s;
            if (alloc_ok) {
                launch_gemm_batched(e.q, md->router.w, X, rl, PM);
                launch_router_topk_batched(e.q, rl, PM, E, Kt, re_b, rw_b, true);
                launch_moe_gate_up_batched(e.q, md->moe, re_b, X, mh_b, PM);
                launch_moe_down_batched(e.q, md->moe, re_b, rw_b, mh_b, my_b, PM);
                for (int t = 0; t < PM; ++t) {
                    launch_router_topk(e.q, rl + int64_t(t) * E, E, Kt,
                                       re_s + int64_t(t) * Kt,
                                       rw_s + int64_t(t) * Kt, true, {});
                    launch_moe_gate_up(e.q, md->moe, re_s + int64_t(t) * Kt,
                                       X + int64_t(t) * H,
                                       mh_s + int64_t(t) * Kt * I);
                    launch_moe_down(e.q, md->moe, re_s + int64_t(t) * Kt,
                                    rw_s + int64_t(t) * Kt,
                                    mh_s + int64_t(t) * Kt * I,
                                    my_s + int64_t(t) * H);
                }
                e.q.wait();
                std::vector<int32_t> he_b(size_t(PM) * Kt), he_s(he_b.size());
                std::vector<float> hw_b(size_t(PM) * Kt), hw_s(hw_b.size());
                std::vector<float> hy_b(size_t(PM) * H), hy_s(hy_b.size());
                e.q.memcpy(he_b.data(), re_b, he_b.size() * sizeof(int32_t));
                e.q.memcpy(he_s.data(), re_s, he_s.size() * sizeof(int32_t));
                e.q.memcpy(hw_b.data(), rw_b, hw_b.size() * sizeof(float));
                e.q.memcpy(hw_s.data(), rw_s, hw_s.size() * sizeof(float));
                e.q.memcpy(hy_b.data(), my_b, hy_b.size() * sizeof(float));
                e.q.memcpy(hy_s.data(), my_s, hy_s.size() * sizeof(float)).wait();
                int route_bad = 0; double wse = 0, yse = 0, ysr = 0;
                for (size_t i = 0; i < he_b.size(); ++i) {
                    route_bad += he_b[i] != he_s[i];
                    const double d = double(hw_b[i]) - hw_s[i]; wse += d * d;
                }
                for (size_t i = 0; i < hy_b.size(); ++i) {
                    const double d = double(hy_b[i]) - hy_s[i]; yse += d * d;
                    ysr += double(hy_s[i]) * hy_s[i];
                }
                const double wrms = std::sqrt(wse / std::max<size_t>(1, hw_b.size()));
                const double yrel = std::sqrt(yse / (ysr + 1e-30));
                std::printf("    batched MoE probe: routes %s, weight RMS %.2e, output rel %.2e %s\n",
                            route_bad ? "WRONG" : "exact", wrms, yrel,
                            (!route_bad && wrms < 1e-7 && yrel < 1e-5) ? "PASS" : "FAIL");
                if (route_bad || wrms >= 1e-7 || yrel >= 1e-5) {
                    for (void* p : {static_cast<void*>(rl), static_cast<void*>(re_b), static_cast<void*>(re_s),
                                    static_cast<void*>(rw_b), static_cast<void*>(rw_s), static_cast<void*>(mh_b),
                                    static_cast<void*>(mh_s), static_cast<void*>(my_b), static_cast<void*>(my_s)})
                        sycl::free(p, e.q);
                    sycl::free(X, e.q);
                    return 1;
                }
            } else std::printf("    batched MoE probe: allocation failed\n");
            for (void* p : {static_cast<void*>(rl), static_cast<void*>(re_b), static_cast<void*>(re_s),
                            static_cast<void*>(rw_b), static_cast<void*>(rw_s), static_cast<void*>(mh_b),
                            static_cast<void*>(mh_s), static_cast<void*>(my_b), static_cast<void*>(my_s)})
                if (p) sycl::free(p, e.q);
        }
    }

    // The projection micro-benchmark below reads the MXFP4 payload directly.
    // Once W4A8 has converted a weight that payload is FREED, and touching it
    // on device is a DEVICE_LOST rather than a clean null check.  It is
    // diagnostic only -- the FULL E2E PP below is what gates completion.
    if (d0.qkv->payload) {
    const int Nq = d0.qkv->w.N;
    std::printf("    qkv projection [%d x %d]\n", Nq, H);
    std::fflush(stdout);
    float* Y = sycl::malloc_device<float>(size_t(M) * Nq, e.q);
    if (!Y) { std::printf("    cannot allocate output\n"); sycl::free(X, e.q); return 1; }

    // ---- scalar sub-group path (baseline) ---------------------------
    auto time_it = [&](auto fn) {
        fn(); e.q.wait();
        double best = 1e30;
        for (int t = 0; t < 3; ++t) {
            const auto a = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 5; ++i) fn();
            e.q.wait();
            const auto b = std::chrono::high_resolution_clock::now();
            const double m = std::chrono::duration<double, std::milli>(b - a).count() / 5;
            if (m < best) best = m;
        }
        return best;
    };

    const double flop = 2.0 * double(M) * Nq * H;
    double g_xmx_ms = 0.0;

    // ---- scalar sub-group path -------------------------------------
    const double t_scalar = time_it([&]{ launch_gemm_batched(e.q, d0.qkv->w, X, Y, M); });
    std::printf("    scalar  [%d x %d] x %d: %8.3f ms  %8.1f GFLOP/s\n",
                Nq, H, M, t_scalar, flop / (t_scalar / 1e3) / 1e9);
    std::fflush(stdout);

    // keep the scalar result to check XMX against
    std::vector<float> ref(size_t(M) * Nq);
    e.q.memcpy(ref.data(), Y, ref.size() * sizeof(float)).wait();

    // ---- XMX joint_matrix path -------------------------------------
    // This kernel has never executed before. Prefill is the only phase
    // where the matrix engines matter, so it is worth the risk -- but
    // the result is checked against the scalar path before any timing
    // is believed.
    sycl_bf16* Xb = sycl::malloc_device<sycl_bf16>(size_t(M) * H, e.q);
    if (Xb) {
        launch_f32_to_bf16(e.q, X, Xb, size_t(M) * H);
        e.q.memset(Y, 0, size_t(M) * Nq * sizeof(float)).wait();

        bool xmx_ok = true;
        try {
            launch_gemm_xmx(e.q, d0.qkv->w, Xb, Y, M);
            e.q.wait();
        } catch (const sycl::exception& ex) {
            std::printf("    xmx     FAILED: %s\n", ex.what());
            xmx_ok = false;
        }

        if (xmx_ok) {
            std::vector<float> got(size_t(M) * Nq);
            e.q.memcpy(got.data(), Y, got.size() * sizeof(float)).wait();
            double se = 0, sr = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                se += double(got[i] - ref[i]) * (got[i] - ref[i]);
                sr += double(ref[i]) * ref[i];
            }
            const double rel = std::sqrt(se / (sr + 1e-30));
            // Sweep the M-blocking factor. Register pressure against
            // fragment reuse has no closed form on this hardware, so it
            // is measured rather than reasoned about.
            // 2-D sweep. M-blocking trades register pressure against
            // fragment reuse; N-blocking trades registers against how
            // many times the activation matrix is re-read from global.
            // Neither was predictable, so both are measured.
            double t_xmx = 1e30;
            int best_mp = 1, best_np = 4;
            for (int np : {4, 8}) {
                for (int mp : {1}) {
                    char b1[8], b2[8];
                    std::snprintf(b1, sizeof b1, "%d", mp);
                    std::snprintf(b2, sizeof b2, "%d", np);
                    setenv("GRIMOIRE_MPSG", b1, 1);
                    setenv("GRIMOIRE_NPSG", b2, 1);
                    const double t = time_it([&]{ launch_gemm_xmx(e.q, d0.qkv->w, Xb, Y, M); });
                    std::printf("      M=%d N=%d  tile %3dx%3d: %8.3f ms  %8.1f GFLOP/s\n",
                                mp, np, 8 * mp * 8, 16 * np, t, flop / (t / 1e3) / 1e9);
                    if (t < t_xmx) { t_xmx = t; best_mp = mp; best_np = np; }
                }
            }
            std::printf("    best M_PER_SG=%d N_PER_SG=%d\n", best_mp, best_np);
            {
                char b1[8], b2[8];
                std::snprintf(b1, sizeof b1, "%d", best_mp);
                std::snprintf(b2, sizeof b2, "%d", best_np);
                setenv("GRIMOIRE_MPSG", b1, 1);
                setenv("GRIMOIRE_NPSG", b2, 1);
            }
            if (rel < 2e-2) g_xmx_ms = t_xmx;
            std::printf("    xmx     [%d x %d] x %d: %8.3f ms  %8.1f GFLOP/s   rel-err %.2e %s\n",
                        Nq, H, M, t_xmx, flop / (t_xmx / 1e3) / 1e9, rel,
                        rel < 2e-2 ? "" : "<-- WRONG, ignore timing");
            if (rel >= 2e-2)
                std::printf("    xmx output disagrees with the scalar reference;\n"
                            "    the tiling or VNNI staging is wrong, not the speed.\n");
            if (rel < 2e-2)
                std::printf("    -> xmx is %.1fx the scalar path\n", t_scalar / t_xmx);

            // INT8 DPAS candidate: quantize activations once per prompt
            // block, then use the 2x-rate integer matrix pipeline for
            // INT8/INT4 weights.  Both conversion and GEMM are included.
            if (d0.qkv->w.fmt == Fmt::INT8 || d0.qkv->w.fmt == Fmt::INT4) {
                int8_t* Xq = sycl::malloc_device<int8_t>(size_t(M) * H, e.q);
                float* Xs = sycl::malloc_device<float>(M, e.q);
                if (Xq && Xs) {
                    auto int_run = [&] {
                        launch_quantize_rows_int8(e.q, X, Xq, Xs, M, H);
                        launch_gemm_xmx_int(e.q, d0.qkv->w, Xq, Xs, Y, M);
                    };
                    int_run(); e.q.wait();
                    e.q.memcpy(got.data(), Y, got.size() * sizeof(float)).wait();
                    double ise = 0.0;
                    for (size_t i = 0; i < got.size(); ++i)
                        ise += double(got[i] - ref[i]) * (got[i] - ref[i]);
                    const double irel = std::sqrt(ise / (sr + 1e-30));
                    const double ti = time_it(int_run);
                    std::printf("    int8-DPAS + row quant: %8.3f ms  %8.1f GFLOP/s   rel-err %.2e %s\n",
                                ti, flop / (ti / 1e3) / 1e9, irel,
                                irel < 2e-2 ? "" : "<-- WRONG, ignore timing");
                    if (irel < 2e-2 && ti < g_xmx_ms) g_xmx_ms = ti;
                }
                if (Xq) sycl::free(Xq, e.q);
                if (Xs) sycl::free(Xs, e.q);
            }
        }
        // A pre-dequantize-to-bf16 variant was tried here and removed:
        // once M_PER_SG=2 fixed the blocking it measured 6754 GFLOP/s
        // against 8778 for decoding in-kernel. bf16 weights are twice
        // the bytes of MXFP4, so it trades bandwidth for compute in the
        // wrong direction.
        sycl::free(Xb, e.q);
    }

    const double best_ms = g_xmx_ms > 0.0 ? g_xmx_ms : t_scalar;
    const double per_layer = best_ms * (1.0 + double(c.lin_v_dim * c.lin_v_heads) / Nq
                                          + double(H) / Nq);
    const double total_ms = per_layer * nlin + best_ms * 0.5 * (c.n_layers - nlin);
    std::printf("    projections across %d layers ~ %.0f ms -> ~%.0f tok/s pp  (%s)\n",
                c.n_layers, total_ms, 1000.0 * M / total_ms,
                g_xmx_ms > 0.0 ? "xmx" : "scalar");
    // bf16 DPAS peak on this part is ~180 TFLOP/s; the gap is the tiling
    // headroom that remains.
    std::printf("    %.1f%% of bf16 XMX peak -- the rest is tile/occupancy tuning\n",
                100.0 * (flop / (best_ms / 1e3) / 1e12) / 180.0);

    sycl::free(X, e.q); sycl::free(Y, e.q);
    } else {
        std::printf("    qkv micro-bench skipped (int4 weights; MXFP4 payload freed)\n");
    }

    // The projection number above is diagnostic only. Completion is gated on
    // this real model-wide pass: embedding, every attention/DeltaNet layer,
    // routed and shared experts, residuals, final norm, and logits.
    std::vector<int32_t> warm_ids(32), prompt_ids(M);
    for(size_t i=0;i<warm_ids.size();++i) warm_ids[i]=int32_t(1000+i%97);
    for(size_t i=0;i<prompt_ids.size();++i) prompt_ids[i]=int32_t(1000+i%97);
    e.reset();
    const bool skip_warmup=std::getenv("GRIMOIRE_SKIP_PREFILL_WARMUP") != nullptr;
    if(!skip_warmup){
        if(!e.prefill(warm_ids)){std::printf("    full prefill warmup FAILED\n");return 1;}
        e.sync(); e.reset();
    }else{
        std::printf("    full prefill warmup skipped (single-pass mode)\n");
    }
    const auto p0=std::chrono::high_resolution_clock::now();
    const bool full_ok=e.prefill(prompt_ids);
    e.sync();
    const auto p1=std::chrono::high_resolution_clock::now();
    const double full_ms=std::chrono::duration<double,std::milli>(p1-p0).count();
    std::printf("    FULL E2E PP: %s, %d tokens in %.1f ms -> %.1f tok/s\n",
                full_ok?"PASS":"FAIL",M,full_ms,full_ok?1000.0*M/full_ms:0.0);
    return 0;
}

int grimoire_bench_decode(Grimoire& e, int n_tokens) {
    auto run = [&](bool use_graph) {
        e.reset();
        int tok = 1;
        const auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n_tokens; ++i) {
            if (use_graph) e.step(); else e.forward(tok);
            tok = e.argmax_token();
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        return std::pair<double,int>(
            std::chrono::duration<double, std::milli>(t1 - t0).count() / n_tokens, tok);
    };

    std::printf("\n  measured decode (%d steps, greedy)\n", n_tokens);
    std::fflush(stdout);

    e.reset();
    e.forward(1);                       // warm up first-touch costs
    e.argmax_token();

    const bool had_graph = e.graph_ok;
    e.graph_ok = false;
    const auto direct = run(false);
    std::printf("    direct submission   %7.3f ms/token  -> %6.1f tok/s\n",
                direct.first, 1000.0 / direct.first);
    std::fflush(stdout);

    std::printf("    recording command graph ... ");
    std::fflush(stdout);
    const bool g = had_graph || e.build_graph();
    std::printf("%s\n", g ? "ok" : "unavailable");
    if (g) {
        const auto gr = run(true);
        std::printf("    graph replay        %7.3f ms/token  -> %6.1f tok/s\n",
                    gr.first, 1000.0 / gr.first);
        // Graph replay trades per-launch submission cost for a FIXED
        // attention split count. At short context the fixed 16 splits do
        // more empty work than the saving is worth; at long context the
        // split count is right and the graph wins. Keep whichever the
        // machine actually prefers rather than assuming.
        e.graph_ok = (gr.first < direct.first);
        std::printf("    -> using %s (%.1f%% %s)\n",
                    e.graph_ok ? "graph replay" : "direct submission",
                    100.0 * std::fabs(gr.first - direct.first) / direct.first,
                    e.graph_ok ? "faster" : "faster direct");
    }

    if (e.dag && std::getenv("GRIMOIRE_DAG_SWEEP")) {
        struct DagResult { int mask, epl, unroll; double ms; uint64_t hash; };
        std::vector<DagResult> dr;
        auto measure = [&](int mask, int epl, int unroll) {
            e.dag_mask = mask;
            set_gemv_tuning(epl, unroll, -1);
            std::array<double, 3> samples{};
            uint64_t final_hash = 0;
            for (int rep = 0; rep < 3; ++rep) {
                e.reset();
                int tok = 1;
                uint64_t hash = 1469598103934665603ull;
                e.forward(tok); tok = e.argmax_token(); // first-touch warmup
                e.reset(); tok = 1;
                const auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < n_tokens; ++i) {
                    e.forward(tok); tok = e.argmax_token();
                    hash ^= uint32_t(tok); hash *= 1099511628211ull;
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                samples[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count()
                             / n_tokens;
                if (rep == 0) final_hash = hash;
                else if (hash != final_hash) final_hash = 0; // nondeterministic => reject
            }
            std::sort(samples.begin(), samples.end());
            dr.push_back({mask, epl, unroll, samples[1], final_hash});
        };

        std::printf("\n  dependency-DAG sweep (median of 3 x %d tokens)\n", n_tokens);
        std::printf("    bits: 1=linear branches, 2=Q/K/V, 4=routed/shared MoE\n");
        for (int mask : {0, 1, 2, 4, 3, 5, 6, 7}) measure(mask, 0, 0);
        measure(7, 16, 1); // combined DAG plus winner from GEMV sweep
        const uint64_t ref = dr.front().hash;
        std::printf("    %-6s %-5s %-7s %10s %10s  %s\n",
                    "mask", "EPL", "unroll", "ms/token", "tok/s", "output");
        for (const auto& r : dr)
            std::printf("    0x%-4x %-5d %-7d %10.3f %10.1f  %s\n",
                r.mask, r.epl, r.unroll, r.ms, 1000.0 / r.ms,
                r.hash != 0 && r.hash == ref ? "MATCH" : "MISMATCH");
        e.dag_mask = 0;
        set_gemv_tuning(0, 0, -1);
    }

    if (!e.dag && std::getenv("GRIMOIRE_FUSION_SWEEP")) {
        struct FusionResult { int mask, epl, unroll; double ms; uint64_t hash; };
        std::vector<FusionResult> fr;
        auto measure = [&](int mask, int epl, int unroll) {
            e.fusion_mask = mask;
            set_gemv_tuning(epl, unroll, -1);
            std::array<double, 3> samples{};
            uint64_t final_hash = 0;
            for (int rep = 0; rep < 3; ++rep) {
                e.reset(); int tok = 1;
                uint64_t hash = 1469598103934665603ull;
                e.forward(tok); tok = e.argmax_token();
                e.reset(); tok = 1;
                const auto t0 = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < n_tokens; ++i) {
                    e.forward(tok); tok = e.argmax_token();
                    hash ^= uint32_t(tok); hash *= 1099511628211ull;
                }
                const auto t1 = std::chrono::high_resolution_clock::now();
                samples[rep] = std::chrono::duration<double, std::milli>(t1 - t0).count()
                             / n_tokens;
                if (rep == 0) final_hash = hash;
                else if (hash != final_hash) final_hash = 0;
            }
            std::sort(samples.begin(), samples.end());
            fr.push_back({mask, epl, unroll, samples[1], final_hash});
        };

        std::printf("\n  fusion sweep (all 16 masks, median of 3 x %d tokens)\n", n_tokens);
        std::printf("    bits: 1=DN norm+gate, 2=QK norm+rope, 4=MoE join+norm, 8=position, 16=DN conv+L2\n");
        for (int mask = 0; mask < 16; ++mask) measure(mask, 0, 0);
        measure(15, 16, 1);
        measure(31, 16, 1);
        const uint64_t ref = fr.front().hash;
        std::sort(fr.begin() + 1, fr.end(),
                  [](const FusionResult& a, const FusionResult& b) { return a.ms < b.ms; });
        std::printf("    %-6s %-5s %-7s %10s %10s  %s\n",
                    "mask", "EPL", "unroll", "ms/token", "tok/s", "output");
        for (const auto& r : fr)
            std::printf("    0x%-4x %-5d %-7d %10.3f %10.1f  %s\n",
                r.mask, r.epl, r.unroll, r.ms, 1000.0 / r.ms,
                r.hash != 0 && r.hash == ref ? "MATCH" : "MISMATCH");
        e.fusion_mask = 15;
        set_gemv_tuning(16, 1, -1);
    }

    if (std::getenv("GRIMOIRE_AUTOTUNE")) {
        struct Result { int epl, unroll, wide; double ms; uint64_t hash; };
        std::vector<Result> results;
        auto sweep_run = [&](int epl, int unroll, int wide) {
            set_gemv_tuning(epl, unroll, wide);
            e.graph_ok = false;
            e.reset();
            int tok = 1;
            uint64_t hash = 1469598103934665603ull;
            // Warm-up after changing kernels is outside the measurement.
            e.forward(tok); tok = e.argmax_token();
            e.reset(); tok = 1;
            const auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_tokens; ++i) {
                e.forward(tok);
                tok = e.argmax_token();
                hash ^= uint32_t(tok);
                hash *= 1099511628211ull;
            }
            const auto t1 = std::chrono::high_resolution_clock::now();
            results.push_back({epl, unroll, wide,
                std::chrono::duration<double, std::milli>(t1 - t0).count() / n_tokens,
                hash});
        };

        std::printf("\n  runtime GEMV autotune (%d-token full-model runs)\n", n_tokens);
        std::printf("    sweeping EPL {16,32,64} x unroll {1,2,4,8} x wide {off,auto,on}\n");
        sweep_run(0, 0, -1); // exact current default and correctness reference
        const uint64_t reference = results.front().hash;
        for (int wide : {0, -1, 1})
            for (int epl : {16, 32, 64})
                for (int unroll : {1, 2, 4, 8})
                    sweep_run(epl, unroll, wide);

        std::sort(results.begin() + 1, results.end(),
                  [](const Result& a, const Result& b) { return a.ms < b.ms; });
        std::printf("    %-5s %-7s %-6s %10s %10s  %s\n",
                    "EPL", "unroll", "wide", "ms/token", "tok/s", "output");
        for (const auto& r : results) {
            const char* w = r.wide < 0 ? "auto" : (r.wide ? "on" : "off");
            std::printf("    %-5d %-7d %-6s %10.3f %10.1f  %s\n",
                        r.epl, r.unroll, w, r.ms, 1000.0 / r.ms,
                        r.hash == reference ? "MATCH" : "MISMATCH");
        }
        set_gemv_tuning(16, 1, -1);
    }
    return 0;
}

int grimoire_load_report(const std::string& dir, Fmt proj_fmt, int max_seq) {
    // Unbuffered: a crash mid-upload must not swallow the progress that
    // says WHERE it crashed.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Grimoire e;
    const auto& dev = e.q.get_device();
    std::printf("grimoire\n");
    std::printf("  device   : %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("  VRAM     : %.1f GiB\n",
                double(dev.get_info<sycl::info::device::global_mem_size>()) / (1 << 30));
    std::printf("  model    : %s\n", dir.c_str());
    std::printf("  proj fmt : %s\n\n", fmt_name(proj_fmt));

    UploadOptions opt;
    opt.lm_head_fmt = proj_fmt;
    opt.quantize_lm_head = (proj_fmt != Fmt::BF16);
    opt.max_seq = max_seq;

    std::string err;
    try {
        if (!e.build(dir, opt, err)) {
            std::printf("\nLOAD FAILED: %s\n", err.c_str());
            return 1;
        }
    } catch (const sycl::exception& ex) {
        std::printf("\nSYCL EXCEPTION during upload: %s\n", ex.what());
        return 1;
    } catch (const std::exception& ex) {
        std::printf("\nEXCEPTION during upload: %s\n", ex.what());
        return 1;
    }

    const Qwen35Config& c = e.cfg;
    int nlin = 0;
    for (LayerKind k : c.layer_types) if (k == LayerKind::LINEAR_ATTN) ++nlin;
    const int nfull = c.n_layers - nlin;

    std::printf("\n  loaded in %.1f s, %.2f GiB resident\n", e.load_seconds, e.vram_gb);
    std::printf("  %d layers (%d linear, %d full), %d experts top-%d\n\n",
                c.n_layers, nlin, nfull, c.n_experts, c.top_k);

    // ---- decode budget from what was actually uploaded ----------------
    const int H = c.hidden, Hk = c.lin_k_heads, Dk = c.lin_k_dim;
    const int Hv = c.lin_v_heads, Dv = c.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;
    const double bpe = bits_per_elem(proj_fmt) / 8.0;
    const double r   = meas_rate(proj_fmt);

    const double dn_proj = (double(qkv_ch + Hv * Dv) * H + double(H) * Hv * Dv)
                         * bpe / 1e9 / r * 1e3 * nlin;
    const double moe     = 0.0479 * c.n_layers;      // measured, fused, mxfp4
    const double lm      = double(c.vocab) * H * bpe / 1e9 / r * 1e3;
    const double shared  = double(2 * c.shared_inter + H) * H * bpe / 1e9 / r * 1e3 * c.n_layers;
    const double state   = 2.0 * double(Hv) * Dk * Dv * 4 / 1e9 / 542.0 * 1e3 * nlin;
    const double fullp   = 4.0 * double(H) * H * bpe / 1e9 / r * 1e3 * nfull;
    // Launch count after fusion. Each removal is a real kernel that no
    // longer exists, not an estimate:
    //   residual adds folded into the norms      -2 per layer
    //   q and k L2-normalised in one launch      -1 per linear layer
    //   a|b concatenated at load                 -1 per linear layer
    //   gate|up concatenated at load             -1 per layer
    const int per_layer_common = 2 /*norms*/ + 1 /*router*/ + 1 /*topk*/
                               + 2 /*moe*/ + 2 /*shared gemv*/ + 1 /*swiglu*/
                               + 1 /*sh gate*/;
    const int per_linear = 1 /*qkv*/ + 1 /*conv*/ + 1 /*l2*/ + 1 /*ab*/
                         + 1 /*gates*/ + 1 /*step*/ + 1 /*z*/ + 1 /*gate_silu*/
                         + 1 /*head norm*/ + 1 /*out*/;
    const int per_full   = 3 /*qkv*/ + 2 /*rope*/ + 1 /*kv append*/
                         + 2 /*attn*/ + 1 /*o*/;
    const int launches = per_layer_common * c.n_layers
                       + per_linear * nlin + per_full * nfull + 3;
    const double lat     = launches * 5.0 / 1000.0;
    const double total   = dn_proj + moe + lm + shared + state + fullp + lat;

    std::printf("  projected decode step\n");
    std::printf("    %-22s %7.3f ms\n", "deltanet projections", dn_proj);
    std::printf("    %-22s %7.3f ms\n", "fused MoE", moe);
    std::printf("    %-22s %7.3f ms\n", "shared expert", shared);
    std::printf("    %-22s %7.3f ms\n", "full attn projections", fullp);
    std::printf("    %-22s %7.3f ms\n", "deltanet state", state);
    std::printf("    %-22s %7.3f ms\n", "lm_head", lm);
    std::printf("    %-22s %7.3f ms  (%d launches)\n", "launch latency", lat, launches);
    std::printf("    %-22s %7.3f ms  ->  %.0f tok/s\n", "TOTAL", total, 1000.0 / total);

    if (proj_fmt == Fmt::BF16)
        std::printf("\n  Re-run with --proj int4 to quantize the projections.\n");

    // Does a BATCHED GEMV at M=4 cost what M=1 costs?  That is the whole
    // premise of speculative verification: the batch is weight-bound, so
    // loading each weight once and doing M dot products should be nearly
    // free per extra token.  Measured over all 64 layers' FFN weights, i.e.
    // production streaming conditions, not a reused matrix.
    if (std::getenv("GRIMOIRE_MTP_BENCH") && e.L[0].sh_gu_i4) {
        const int H = e.cfg.hidden;
        int maxN = 0, maxK = 0;
        for (auto& d : e.L) {
            if (!d.sh_gu_i4) continue;
            maxN = std::max(maxN, std::max(d.sh_gu.w.N, d.sh_down.w.N));
            maxK = std::max(maxK, std::max(d.sh_gu.w.K, d.sh_down.w.K));
        }
        float* bx = sycl::malloc_device<float>(size_t(4) * maxK, e.q);
        float* by = sycl::malloc_device<float>(size_t(4) * maxN, e.q);
        e.q.memset(bx, 0x3c, size_t(4) * maxK * sizeof(float)).wait();
        std::printf("\n  batched GEMV over 64 layers of FFN weights\n");
        double base = 0;
        for (int mb = 1; mb <= 4; ++mb) {
            for (int rep = 0; rep < 2; ++rep) {
                if (rep) e.q.wait();
                const auto t0 = std::chrono::high_resolution_clock::now();
                for (auto& d : e.L) {
                    if (!d.sh_gu_i4) continue;
                    launch_gemv_int4sym_batch(e.q, d.sh_gu_i4, d.sh_gu_ws, bx, by,
                                              d.sh_gu.w.N, d.sh_gu.w.K, mb, {});
                    launch_gemv_int4sym_batch(e.q, d.sh_dn_i4, d.sh_dn_ws, bx, by,
                                              d.sh_down.w.N, d.sh_down.w.K, mb, {});
                }
                e.q.wait();
                if (rep) {
                    const auto t1 = std::chrono::high_resolution_clock::now();
                    const double ms =
                        std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (mb == 1) base = ms;
                    std::printf("    M=%d  %7.2f ms   %.2fx of M=1   %.2f ms/token\n",
                                mb, ms, ms / base, ms / mb);
                }
            }
        }
        sycl::free(bx, e.q); sycl::free(by, e.q);
        (void)H;
    }

    grimoire_bench_decode(e, 32);
    int bench_m=4096;
    if(const char* v=std::getenv("GRIMOIRE_BENCH_PREFILL_TOKENS"))
        bench_m=std::max(1,std::min(max_seq,std::atoi(v)));
    grimoire_bench_prefill(e, bench_m);

    e.release();
    return 0;
}

// Minimal production-path prefill benchmark. Unlike grimoire_load_report(),
// this deliberately skips the standalone projection/MoE microbenchmarks;
// those stress each PP rank independently and are not representative of the
// staged model (one such M=4096 diagnostic caused DEVICE_LOST on USB4).
int grimoire_prefill_only(const std::string& dir, Fmt proj_fmt, int max_seq,
                          int n_tokens) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Grimoire e;
    UploadOptions opt;
    opt.lm_head_fmt = proj_fmt;
    opt.quantize_lm_head = (proj_fmt != Fmt::BF16);
    opt.max_seq = max_seq;
    std::string err;
    try {
        if (!e.build(dir, opt, err)) {
            std::printf("\nLOAD FAILED: %s\n", err.c_str()); return 1;
        }
        std::vector<int32_t> ids(size_t(n_tokens), int32_t(0));
        for (int i = 0; i < n_tokens; ++i) ids[size_t(i)] = int32_t(1000 + i % 97);
        e.reset();
        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool ok = e.prefill(ids);
        e.sync();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::printf("\n  FULL E2E PP ONLY: %s, %d tokens in %.1f ms -> %.1f tok/s\n",
                    ok ? "PASS" : "FAIL", n_tokens, ms,
                    ok ? 1000.0 * n_tokens / ms : 0.0);
        e.release();
        return ok ? 0 : 1;
    } catch (const sycl::exception& ex) {
        std::printf("\nSYCL EXCEPTION: %s\n", ex.what()); return 1;
    }
}

int grimoire_prefix_cache_test(const std::string& dir, Fmt proj_fmt, int max_seq,
                               int n_tokens) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Grimoire e;
    UploadOptions opt;
    opt.lm_head_fmt = proj_fmt;
    opt.quantize_lm_head = (proj_fmt != Fmt::BF16);
    opt.max_seq = max_seq;
    std::string err;
    if (!e.build(dir, opt, err)) { std::printf("\nLOAD FAILED: %s\n", err.c_str()); return 1; }
    std::vector<int32_t> ids(size_t(n_tokens), int32_t(0));
    for (int i = 0; i < n_tokens; ++i) ids[size_t(i)] = int32_t(1000 + i % 97);
    auto run = [&]() {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool ok = e.prefill(ids); e.sync();
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        const int tok = ok ? e.argmax_token() : -1;
        return std::tuple<bool,double,int>{ok,ms,tok};
    };
    e.reset(); const auto cold = run();
    e.reset(); const auto hit = run();
    const bool pass = std::get<0>(cold) && std::get<0>(hit) &&
                      std::get<2>(cold) == std::get<2>(hit);
    std::printf("\n  PREFIX CACHE: %s, %d tokens cold %.1f ms, hit %.1f ms, token %d/%d\n",
        pass ? "PASS" : "FAIL", n_tokens, std::get<1>(cold), std::get<1>(hit),
        std::get<2>(cold), std::get<2>(hit));
    e.release(); return pass ? 0 : 1;
}

} // namespace b70

// =====================================================================
//  Forward pass
// =====================================================================
namespace b70 {

const float* Grimoire::forward_dag(int token) {
    // This path walks all cfg.n_layers with no stage bounds, so it cannot
    // run a pipeline rank that only holds a slice.  Refuse rather than
    // read weights this rank never loaded.
    if (pp_enabled()) {
        std::fprintf(stderr,
            "forward_dag does not implement pipeline stages; use forward()\n");
        return nullptr;
    }
    const int H = cfg.hidden;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    auto deps = [](std::initializer_list<sycl::event> es) {
        return std::vector<sycl::event>(es);
    };

    sycl::event e_embed = launch_embed(q, embed, token, s.h, H, dag_tail);
    sycl::event e_moe = q.submit([&](sycl::handler& h) {
        h.depends_on(e_embed);
        h.memset(s.moe_y, 0, size_t(H) * sizeof(float));
    });
    sycl::event e_h = e_embed;

    for (int i = 0; i < cfg.n_layers; ++i) {
        LayerDev& d = L[i];
        sycl::event e_in = launch_rmsnorm_residual(
            q, s.h, s.moe_y, d.in_norm, s.h2, H, cfg.rms_eps, deps({e_h, e_moe}));
        sycl::event e_attn;

        if (d.kind == LayerKind::LINEAR_ATTN) {
            const int qkv_ch = d.la_qkv.w.N;
            sycl::event e_qkv = gemv_any(d.la_qkv, s.h2, s.qkv, deps({e_in}));
            ConvParams cp{};
            cp.x = s.qkv; cp.weight = d.la_conv; cp.ring = d.conv_ring;
            cp.out = s.qkv; cp.channels = qkv_ch; cp.kernel = cfg.conv_kernel;
            sycl::event e_conv = launch_causal_conv1d(q, cp, deps({e_qkv}));
            sycl::event e_l2 = launch_l2norm_heads(q, s.qkv, 2 * Hk, Dk, deps({e_conv}));

            sycl::event e_ab = gemv_any(d.la_ab, s.h2, s.abuf,
                                           deps({(dag_mask & 1) ? e_in : e_l2}));
            sycl::event e_gates = launch_deltanet_gates(
                q, s.abuf, s.abuf + Hv, d.la_Alog, d.la_dtb,
                s.alpha, s.beta, Hv, deps({e_ab}));

            DeltaNetParams dp{};
            dp.q = s.qkv; dp.k = s.qkv + int64_t(Hk) * Dk;
            dp.v = s.qkv + int64_t(2) * Hk * Dk;
            dp.a = s.alpha; dp.beta = s.beta; dp.state = d.dn_state;
            dp.out = s.attn_out; dp.n_heads = Hv; dp.k_dim = Dk;
            dp.v_dim = Dv; dp.n_k_heads = Hk;
            sycl::event e_dn = launch_deltanet_step(q, dp, deps({e_l2, e_gates}));
            sycl::event e_z = gemv_any(d.la_z, s.h2, s.zbuf,
                                          deps({(dag_mask & 1) ? e_in : e_dn}));
            sycl::event e_norm = launch_rmsnorm_heads(
                q, s.attn_out, d.la_norm, Hv, Dv, cfg.rms_eps, false, deps({e_dn}));
            sycl::event e_gate = launch_gate_silu(
                q, s.attn_out, s.zbuf, s.attn_out, Hv * Dv, deps({e_norm, e_z}));
            e_attn = gemv_any(d.la_out, s.attn_out, s.moe_y, deps({e_gate}));
        } else {
            const int QD = d.q_proj.w.N;
            const bool gated = QD == 2 * cfg.n_heads * d.head_dim;
            sycl::event e_qp = gemv_any(d.q_proj, s.h2, s.qkv, deps({e_in}));
            sycl::event e_qready = e_qp;
            if (gated)
                e_qready = launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                    cfg.n_heads, d.head_dim, deps({e_qp}));
            float* qvec = gated ? s.qsplit : s.qkv;

            sycl::event e_kp = gemv_any(d.k_proj, s.h2, s.zbuf,
                deps({(dag_mask & 2) ? e_in : e_qready}));
            sycl::event e_vp = d.k2_sparse
                ? mova_value_m1(d, s.h2, s.bbuf, deps({(dag_mask & 2) ? e_in : e_kp}))
                : gemv_any(d.v_proj, s.h2, s.bbuf,
                    deps({(dag_mask & 2) ? e_in : e_kp}));
            sycl::event e_qn = e_qready;
            if (d.q_norm)
                e_qn = launch_rmsnorm_heads(q, qvec, d.q_norm, cfg.n_heads,
                    d.head_dim, cfg.rms_eps, true,
                    deps({(dag_mask & 2) ? e_qready : e_vp}));
            sycl::event e_kn = e_kp;
            if (d.k_norm)
                e_kn = launch_rmsnorm_heads(q, s.zbuf, d.k_norm, d.kv_heads,
                    d.head_dim, cfg.rms_eps, true,
                    deps({(dag_mask & 2) ? e_kp : e_qn}));
            // gemma-4 full-attention layers rotate with different
            // frequencies and a different pairing (see
            // launch_rope_proportional); calling the default kernel here
            // would be silently wrong rather than an error.  Written out
            // rather than selected through a function pointer: the two
            // kernels no longer share a signature, and a pointer that
            // happens to match today is a trap for whoever adds the next
            // parameter.
            const auto qdeps = deps({(dag_mask & 2) ? e_qn : e_kn});
            sycl::event e_qr = d.rope_proportional
                ? launch_rope_proportional(q, qvec, cfg.n_heads, d.head_dim,
                      s.d_pos, d.rope_theta, d.partial_rope, qdeps, d.rope_factor)
                : launch_rope_dev(q, qvec, cfg.n_heads, d.head_dim,
                      s.d_pos, d.rope_theta, d.partial_rope, qdeps);
            const auto kdeps = deps({(dag_mask & 2) ? e_kn : e_qr});
            sycl::event e_kr = d.rope_proportional
                ? launch_rope_proportional(q, s.zbuf, d.kv_heads, d.head_dim,
                      s.d_pos, d.rope_theta, d.partial_rope, kdeps, d.rope_factor)
                : launch_rope_dev(q, s.zbuf, d.kv_heads, d.head_dim,
                      s.d_pos, d.rope_theta, d.partial_rope, kdeps);
            sycl::event e_kv = launch_kv_append_dev(q, s.zbuf, s.bbuf,
                d.k_cache, d.v_cache, s.d_pos, d.kv_heads, d.head_dim,
                max_seq, deps({e_kr, e_vp}));

            AttnParams ap{};
            ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
            ap.out = s.attn_out; ap.seq_len = pos + 1; ap.seq_cap = max_seq;
            ap.head_dim = d.head_dim; ap.num_heads = cfg.n_heads;
            ap.num_kv_heads = d.kv_heads;
            ap.softmax_scale = cfg.attn_softmax_scale(d.head_dim);
            ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
            ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
            sycl::event e_fd = launch_flash_decode(q, ap, deps({e_qr, e_kv}));
            sycl::event e_fm = launch_flash_merge(q, ap, deps({e_fd}));
            sycl::event e_gate = e_fm;
            if (gated)
                e_gate = launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit,
                    cfg.n_heads * d.head_dim, deps({e_fm, e_qready}));
            e_attn = gemv_any(d.o_proj, s.attn_out, s.moe_y, deps({e_gate}));
        }

        e_h = launch_rmsnorm_residual(q, s.h, s.moe_y, d.post_norm,
                                      s.h2, H, cfg.rms_eps, deps({e_attn}));
        // per LAYER, not per model -- see LayerDev::moe_layer
        if (d.moe_layer) {
            sycl::event e_router = gemv_any(d.router, s.h2, s.rlogits, deps({e_h}));
            sycl::event e_top = launch_router_topk(q, s.rlogits, cfg.n_experts,
                cfg.top_k, s.d_expert, s.d_weight, true, deps({e_router}));
            sycl::event e_gu = launch_moe_gate_up(
                q, d.moe, s.d_expert, s.h2, s.moe_h, deps({e_top}));
            sycl::event e_routed = launch_moe_down(
                q, d.moe, s.d_expert, s.d_weight, s.moe_h, s.moe_y, deps({e_gu}));

            sycl::event e_shgu = gemv_any(d.sh_gu, s.h2, s.sh_g,
                deps({(dag_mask & 4) ? e_h : e_routed}));
            const int SI = d.sh_gu.output_rows() / 2;
            sycl::event e_sw = launch_swiglu(
                q, s.sh_g, s.sh_g + SI, s.sh_g, SI, deps({e_shgu}));
            sycl::event e_shdown = launch_gemv(
                q, d.sh_down.w, s.sh_g, s.sh_out, deps({e_sw}));
            sycl::event e_shared = e_shdown;
            if (d.has_sh_gate) {
                sycl::event e_gq = gemv_any(d.sh_gate_q, s.h2,
                    s.sh_gate_val, deps({(dag_mask & 4) ? e_h : e_shdown}));
                e_shared = launch_scale_by_sigmoid(
                    q, s.sh_out, s.sh_gate_val, H, deps({e_shdown, e_gq}));
            }
            e_moe = launch_add(q, s.moe_y, s.sh_out, H, deps({e_routed, e_shared}));
        } else {
            const int FI = d.sh_gu.output_rows() / 2;
            sycl::event e_gu = ffn_gemv(d, true, s.h2, s.sh_g, deps({e_h}));
            sycl::event e_sw = launch_swiglu(q, s.sh_g, s.sh_g + FI, s.sh_g, FI, deps({e_gu}));
            e_moe = ffn_gemv(d, false, s.sh_g, s.moe_y, deps({e_sw}));
        }
    }

    sycl::event e_fn = launch_rmsnorm_residual(
        q, s.h, s.moe_y, fnorm, s.h2, H, cfg.rms_eps, deps({e_h, e_moe}));
    dag_logits = gemv_any(lm_head, s.h2, s.logits, deps({e_fn}));
    sycl::event e_pos = launch_incr_pos(q, s.d_pos, deps({e_fn}));
    sycl::event e_len = launch_incr_pos(q, s.d_seq_len, deps({e_fn}));
    dag_tail = deps({e_pos, e_len});
    ++pos;
    return s.logits;
}

// One decode step. Returns the device logits pointer.
const float* Grimoire::forward_muse(int token) {
    check_token(token);
    const int H = cfg.hidden, HD = cfg.head_dim;
    const int QH = cfg.n_heads, KVH = cfg.n_kv_heads;
    const int QW = QH * HD;                 // attention output width
    const std::vector<sycl::event> none{};
    const float eps = cfg.rms_eps;
    const float sm_scale = cfg.query_prescale / std::sqrt(float(HD));

    // ---- pipeline stage ownership ---------------------------------
    // build() loads ONLY [pp_begin, pp_end) on this rank, omits the
    // embedding table on every rank but the first, and omits the final
    // norm and lm_head on every rank but the last.  This function used to
    // embed, walk all cfg.n_layers and run the head unconditionally,
    // which on a later rank dereferences weights and caches that were
    // never allocated.  Mirror forward()'s stage discipline exactly.
    const bool pp_first = !pp_enabled() || pp_rank == 0;
    const bool pp_last  = !pp_enabled() || pp_rank == pp_world - 1;

    if (pp_first) {
        // embed, then SCALELESS RMSNorm on the token embedding (Muse: no sqrt(H)).
        if(!embed_one(token,s.h2))return nullptr;
        launch_rmsnorm_residual(q, s.h2, nullptr, muse_zero, s.h, H, eps, none);
    } else {
        // A later stage joins mid-network: it must NOT re-embed (no table
        // here) and must NOT re-apply the scaleless input norm, which
        // belongs to the first stage only.  s.h is the residual stream.
        if (!pp_recv_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank %d: Muse hidden receive failed\n", pp_rank);
            return nullptr;
        }
        // Same order on both sides of every boundary: hidden, then taps.
        // The socket is a byte stream, so a mismatch here would not fail
        // where it happened -- it would misalign every later message.
        if (!pp_recv_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: Muse tap receive failed\n", pp_rank);
            return nullptr;
        }
    }

    const int layer_begin = pp_enabled() ? pp_begin : 0;
    const int layer_end   = pp_enabled() ? pp_end   : cfg.n_layers;
    for (int i = layer_begin; i < layer_end; ++i) {
        LayerDev& d = L[i];
        // target_aux, not dflash2.ok.  Under PP an earlier stage owns the
        // tap buffer and the tap set but hosts no drafter, and it is
        // exactly that stage's taps the last stage cannot compute for
        // itself -- gating on .ok would leave them zero and the drafter
        // would run on a tap row that is mostly zeros, silently.
        if (dflash2.target_aux) {
            for (size_t tap = 0; tap < dflash2.target_layers.size(); ++tap) {
                // vLLM requests aux layer target_layer_id + 1 and Muse emits
                // it after completing target_layer_id.  This loop observes
                // that same residual stream at entry to the following layer.
                if (dflash2.target_layers[tap] + 1 == i) {
                    launch_dflash_store_tap_dev(q, s.h, dflash2.target_aux, H,
                        int(dflash2.target_layers.size()),s.d_pos,int(tap),none,true);
                    break;
                }
            }
        }
        // --- attention block (residual added AFTER post_attention_layernorm)
        launch_rmsnorm_residual(q, s.h, nullptr, d.in_norm, s.h2, H, eps, none);
        gemv_any(d.q_proj, s.h2, s.qkv, none);
        gemv_any(d.k_proj, s.h2, s.zbuf, none);
        if (d.k2_sparse) mova_value_m1(d, s.h2, s.bbuf, none);
        else gemv_any(d.v_proj, s.h2, s.bbuf, none);
        // scaleless QK-norm over head_dim (zero weight -> (1+0)), BEFORE RoPE.
        launch_rmsnorm_heads(q, s.qkv,  muse_zero, QH,  HD, eps, true, none);
        launch_rmsnorm_heads(q, s.zbuf, muse_zero, KVH, HD, eps, true, none);
        // Muse iRoPE: sliding layers use NeoX RoPE; every fourth full-
        // attention layer is NoPE.  vLLM keys this from no_rope_layers.
        if (d.muse_sliding) {
            launch_rope_dev(q, s.qkv,  QH,  HD, s.d_pos, cfg.rope_theta,
                            cfg.partial_rope, none);
            launch_rope_dev(q, s.zbuf, KVH, HD, s.d_pos, cfg.rope_theta,
                            cfg.partial_rope, none);
        }
        launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                             s.d_pos, KVH, HD, max_seq, none);
        AttnParams ap{};
        ap.q = s.qkv; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
        ap.out = s.attn_out; ap.seq_len = pos + 1; ap.seq_cap = max_seq;
        ap.head_dim = HD; ap.num_heads = QH; ap.num_kv_heads = KVH;
        ap.softmax_scale = sm_scale;
        ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
        ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
        // Muse iRoPE: the SLIDING layers are the rope-using ones and they
        // are the ones with a window (ref/muse_glimmer.py:1191 --
        // `sliding_window = None if not self.use_rope else
        // config.sliding_window`).  The full-attention layers are NoPE and
        // see everything.
        //
        // This line did not exist: AttnParams is zero-initialised, so every
        // Muse layer attended to the WHOLE history while prefill_muse
        // windowed its sliding layers at a hardcoded 2047.  Prefill and
        // decode therefore disagreed, and only past the window -- a short
        // prompt cannot show it, which is why nothing caught it.
        ap.window_left = d.muse_sliding ? cfg.sliding_window : 0;
        launch_flash_decode(q, ap, none);
        launch_flash_merge(q, ap, none);
        // per-head attention output gate (separate projection).  K2 uses
        // softplus(beta=log 2) here where Qwen/Muse use sigmoid; applying
        // the wrong one is silent, it just reshapes every attention output.
        gemv_any(d.o_gate, s.h2, s.gsplit, none);
        if (cfg.attn_gate == 2)
            launch_softplus_gate(q, s.attn_out, s.gsplit, s.attn_out,
                                 QW, kK2GateBeta, none);
        else
            launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit, QW, none);
        gemv_any(d.o_proj, s.attn_out, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
        // --- feed-forward block (sandwich: pre_ff -> mlp -> post_ff -> +res)
        launch_rmsnorm_residual(q, s.h, nullptr, d.pre_ff_norm, s.h2, H, eps, none);
        const int I=d.sh_gu.output_rows()/2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        launch_swiglu(q, s.sh_g, s.sh_g + I, s.sh_g, I, none);
        gemv_any(d.sh_down, s.sh_g, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_ff_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
    }
    if (!pp_last) {
        // hand the residual stream to the next stage; fnorm and lm_head
        // live on the last rank only.
        if (!pp_send_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank %d: Muse hidden send failed\n", pp_rank);
            return nullptr;
        }
        if (!pp_send_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: Muse tap send failed\n", pp_rank);
            return nullptr;
        }
        launch_incr_pos(q, s.d_pos, none);
        launch_incr_pos(q, s.d_seq_len, none);
        ++pos;
        return s.logits;
    }
    // final norm + lm_head
    launch_rmsnorm_residual_batched(q, s.h, nullptr, nullptr, fnorm, s.h2,
                                    1, H, eps, nullptr, none, 0.0f);
    gemv_any(lm_head, s.h2, s.logits, none);
    launch_incr_pos(q, s.d_pos, none);
    launch_incr_pos(q, s.d_seq_len, none);
    ++pos;
    return s.logits;
}


// ---------------------------------------------------------------------
//  gemma-4.  A dense Gemma-sandwich transformer, which is NOT forward()'s
//  residual graph:
//
//      h   = x + post_attention_layernorm(attn(input_layernorm(x)))
//      out = h + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(h)))
//      out *= layer_scalar
//
//  forward() adds the attention output raw and normalises on the way INTO
//  the FFN; running gemma-4 there loads cleanly and emits fluent text that
//  is not this model's output.  It IS forward_muse()'s graph, but
//  forward_muse carries query_prescale, a scaleless embedding norm, f16 KV
//  caches and NoPE layers -- making those conditional would put a working,
//  tested Muse path at risk for no gain.
//
//  Everything gemma-4-specific beyond the graph comes from LayerDev, which
//  resolved it once at load: head_dim and KV heads differ per layer type,
//  and so do the RoPE base, the rotated fraction and the rotation itself.
// ---------------------------------------------------------------------
const float* Grimoire::forward_gemma4(int token) {
    check_token(token);
    const int H = cfg.hidden;
    const std::vector<sycl::event> none{};
    const float eps = cfg.rms_eps;

    const bool pp_first = !pp_enabled() || pp_rank == 0;
    const bool pp_last  = !pp_enabled() || pp_rank == pp_world - 1;

    if (pp_first) {
        // Gemma4TextScaledWordEmbedding: the lookup is multiplied by
        // sqrt(hidden) BEFORE the first norm.  Omitting it changes the
        // scale every later RMSNorm sees.
        if (tp_enabled()) {
            if (!embed_one(token, s.h)) return nullptr;
        } else if (recording) {
            // Same reason forward() does this: a captured graph replays
            // the kernels it recorded, so an embedding taken from the HOST
            // token bakes that one token into every replay.  s.d_tok is
            // the buffer argmax_token() writes, which makes one recorded
            // graph valid for every token.
            launch_embed_batched(q, embed, s.d_tok, s.h, 1, H, none);
        } else {
            launch_embed(q, embed, token, s.h, H, none);
        }
        if (cfg.embed_scale != 1.0f)
            launch_scale(q, s.h, cfg.embed_scale, H, none);
    } else {
        if (!pp_recv_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank %d: gemma-4 hidden receive failed\n", pp_rank);
            return nullptr;
        }
        if (!pp_recv_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: gemma-4 tap receive failed\n", pp_rank);
            return nullptr;
        }
    }

    const int layer_begin = pp_enabled() ? pp_begin : 0;
    const int layer_end   = pp_enabled() ? pp_end   : cfg.n_layers;
    for (int i = layer_begin; i < layer_end; ++i) {
        LayerDev& d = L[i];
        // target_aux, not dflash2.ok -- see forward_muse: under PP an
        // earlier stage owns taps it cannot draft with, and gating on .ok
        // would leave them zero and the drafter would run on a mostly
        // zero row, silently.
        if (dflash2.target_aux) {
            for (size_t tap = 0; tap < dflash2.target_layers.size(); ++tap)
                if (dflash2.target_layers[tap] + 1 == i) {
                    launch_dflash_store_tap_dev(q, s.h, dflash2.target_aux, H,
                        int(dflash2.target_layers.size()), s.d_pos, int(tap),
                        none, true);
                    break;
                }
        }

        const int HD = d.head_dim, KVH = d.kv_heads;
        const int QH = cfg.n_heads, QW = QH * HD;

        // ---- attention (residual added AFTER post_attention_layernorm) --
        if (i == probe_layer) probe("G4 embed", s.h, H);
        launch_rmsnorm_residual(q, s.h, nullptr, d.in_norm, s.h2, H, eps, none);
        gemv_any(d.q_proj, s.h2, s.qkv, none);
        gemv_any(d.k_proj, s.h2, s.zbuf, none);

        // attention_k_eq_v: a full-attention layer ships NO v_proj, and V
        // is the k_proj output taken BEFORE k_norm and BEFORE RoPE, then
        // put through a scaleless RMSNorm of its own.  Copying it after
        // either step is the mistake this comment exists to prevent --
        // both produce fluent text.
        // A DevQuant that was never uploaded keeps w.N == 0.  Rule 1
        // applies to the PAYLOAD, not the format, so test the shape here
        // and let gemv_any decide how to read it.
        const bool k_is_v = d.v_proj.w.N <= 0;
        // No .wait() here.  The queue is in-order (see the property list in
        // make_queue), so the copy is already ordered after the k_proj that
        // produced s.zbuf -- and a HOST-SIDE WAIT while a SYCL graph is
        // recording throws, which would make gemma-4 the one architecture
        // that cannot be captured.
        if (k_is_v) q.memcpy(s.bbuf, s.zbuf, size_t(KVH) * HD * sizeof(float));
        else        gemv_any(d.v_proj, s.h2, s.bbuf, none);

        if (d.q_norm)
            launch_rmsnorm_heads(q, s.qkv, d.q_norm, QH, HD, eps, false, none);
        if (d.k_norm)
            launch_rmsnorm_heads(q, s.zbuf, d.k_norm, KVH, HD, eps, false, none);
        // v_norm runs on EVERY non-kv-shared layer, not only the ones
        // where V came from k_proj.  ref/gemma4.py:1249 applies it
        // unconditionally after the branch that chose the value source --
        // reading it as "the replacement for the missing v_proj" is the
        // natural misreading, and it leaves 50 of 60 layers unnormalised.
        launch_rmsnorm_heads(q, s.bbuf, gemma_vnorm, KVH, HD, eps, false, none);

        if (d.rope_proportional) {
            launch_rope_proportional(q, s.qkv,  QH,  HD, s.d_pos,
                                     d.rope_theta, d.partial_rope, none,
                                     d.rope_factor);
            launch_rope_proportional(q, s.zbuf, KVH, HD, s.d_pos,
                                     d.rope_theta, d.partial_rope, none,
                                     d.rope_factor);
        } else {
            launch_rope_dev(q, s.qkv,  QH,  HD, s.d_pos,
                            d.rope_theta, d.partial_rope, none);
            launch_rope_dev(q, s.zbuf, KVH, HD, s.d_pos,
                            d.rope_theta, d.partial_rope, none);
        }
        launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                             s.d_pos, KVH, HD, max_seq, none);

        AttnParams ap{};
        ap.q = s.qkv; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
        ap.out = s.attn_out; ap.seq_len = pos + 1; ap.seq_cap = max_seq;
        ap.head_dim = HD; ap.num_heads = QH; ap.num_kv_heads = KVH;
        // scaling = 1.0 in the reference, NOT 1/sqrt(head_dim).
        ap.softmax_scale = cfg.attn_softmax_scale(HD);
        ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
        ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
        // Only the SLIDING layers have a window; the full-attention ones
        // see the whole history.  Identical until the context passes the
        // window, then quietly wrong -- which is why a short-prompt smoke
        // test cannot find it.
        ap.window_left = cfg.layer_global(i) ? 0 : cfg.sliding_window;
        launch_flash_decode(q, ap, none);
        launch_flash_merge(q, ap, none);

        if (i == probe_layer) probe("G4 attn_core", s.attn_out, QH * HD);
        gemv_any(d.o_proj, s.attn_out, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
        if (i == probe_layer) probe("G4 after attn", s.h, H);

        // ---- feed-forward (sandwich: pre_ff -> GeGLU mlp -> post_ff) ----
        launch_rmsnorm_residual(q, s.h, nullptr, d.pre_ff_norm, s.h2, H, eps, none);
        const int I = d.sh_gu.output_rows() / 2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        // gelu_pytorch_tanh, not silu.  The two differ by up to 0.77 on
        // the same input (bin/test_k2_kernels measures it), and nothing
        // downstream would notice the substitution.
        launch_geglu(q, s.sh_g, s.sh_g + I, s.sh_g, I, none);
        gemv_any(d.sh_down, s.sh_g, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_ff_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
        if (i == probe_layer) probe("G4 after ffn", s.h, H);

        // hidden_states *= layer_scalar, the LAST act of the layer.  A
        // nn.Buffer of ones in the reference source, and NOT ones in the
        // checkpoint -- which is why it is read rather than assumed.
        if (d.layer_scalar != 1.0f)
            launch_scale(q, s.h, d.layer_scalar, H, none);
    }

    if (!pp_last) {
        if (!pp_send_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank %d: gemma-4 hidden send failed\n", pp_rank);
            return nullptr;
        }
        if (!pp_send_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: gemma-4 tap send failed\n", pp_rank);
            return nullptr;
        }
        launch_incr_pos(q, s.d_pos, none);
        launch_incr_pos(q, s.d_seq_len, none);
        ++pos;
        return s.logits;
    }
    launch_rmsnorm_residual_batched(q, s.h, nullptr, nullptr, fnorm, s.h2,
                                    1, H, eps, nullptr, none, 0.0f);
    probe("G4 final h2", s.h2, H);
    gemv_any(lm_head, s.h2, s.logits, none);
    probe("G4 logits", s.logits, cfg.vocab);
    // final_logit_softcapping.  Monotonic, so greedy picks the same token
    // either way -- but not a no-op for sampling.
    if (cfg.logit_softcap > 0.0f)
        launch_logit_softcap(q, s.logits, cfg.logit_softcap, cfg.vocab, none);
    launch_incr_pos(q, s.d_pos, none);
    launch_incr_pos(q, s.d_seq_len, none);
    ++pos;
    return s.logits;
}

// ---------------------------------------------------------------------
//  Qwen4-Exp (Qwen3.8-Flash-Next).
//
//  The BODY is Qwen3-Next -- Gated DeltaNet interleaved with attention,
//  MoE FFNs -- and every kernel below is the one the generic forward()
//  uses.  What is different is the RESIDUAL STRUCTURE around them, and
//  it is different enough that running this checkpoint through forward()
//  would produce fluent text that is not the model's output:
//
//   * There is no input_layernorm and no post_attention_layernorm.  Each
//     block is wrapped by a HYPER-CONNECTION whose grouped norm stands
//     where those would be, and the residual stream is hc_count STREAMS
//     wide.  The embedding is REPEATED into every stream (ref:510), not
//     projected.
//   * A combine is DEFERRED.  A layer returns its FFN output and the
//     injection that belongs with it, and the NEXT hyper-connection
//     consumes the pair (ref:276-332).  Only PLE, a stage boundary and
//     the tail force it early.
//   * A PLE layer ADDS TO the multi-stream state directly, so any
//     pending combine must be materialised first (ref:290-296).
//   * There is no model.norm.  The tail mixer's norm plus gated mean is
//     the final norm and what the head reads (ref:576).
//
//  Reference: ref/qwen4_exp_nvidia_model.py and the three modules it
//  pulls in; the derivation of every operator is in b70/qwen4_exp.hpp.
// ---------------------------------------------------------------------
const float* Grimoire::forward_qwen4_exp(int token) {
    const int H  = cfg.hidden;
    const int HC = cfg.hc_count, WIDE = HC * H, LR = cfg.hc_lowrank;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const std::vector<sycl::event> none{};

    launch_embed(q, embed, token, s.h, H, none);
    // The n-gram hash walks backwards through the request's own tokens,
    // so the engine has to keep them.  One int per position.
    {
        int32_t* tok = q4_tok; const int at = pos; const int32_t tv = token;
        q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { tok[at] = tv; });
    }
    // hidden_states = embed(id).repeat(1, hc_count) -- the SAME row in
    // every stream.  A projection here would be a different model.
    for (int c = 0; c < HC; ++c)
        q.memcpy(q4_hyper + size_t(c) * H, s.h, size_t(H) * sizeof(float));

    bool pending = false;          // is a deferred combine outstanding?

    // mix(), fused with a pending combine when there is one.  Returns the
    // block input in `block_out` and, when this hyper-connection has an
    // injection projection, the injection logits in `inj_out`.
    auto hc_mix = [&](const LayerDev::HCDev& hc, const float* pend,
                      const float* pinj, float* inj_out, float* block_out) {
        if (pend) launch_hc_combine(q, q4_hyper, pinj, pend, q4_hyper,
                                    1, HC, H, none);
        launch_hc_norm(q, q4_hyper, hc.norm, q4_normed, 1, HC, H,
                       cfg.rms_eps, none);
        gemv_any(hc.down, q4_normed, q4_lora, none);
        launch_hc_silu(q, q4_lora, LR, HC, none);
        gemv_any(hc.up, q4_lora, q4_gate, none);
        launch_hc_gated_mean(q, q4_gate, q4_normed, block_out, 1, HC, H, none);
        if (inj_out && hc.inject.w.N)
            gemv_any(hc.inject, q4_normed, inj_out, none);
    };

    for (int i = 0; i < cfg.n_layers; ++i) {
        LayerDev& d = L[i];

        // ---- PLE ------------------------------------------------------
        if (d.ple) {
            // PLE adds to the multi-stream state, so a deferred combine
            // has to land first or it would be added to a state that the
            // combine then overwrites.
            if (pending) {
                launch_hc_combine(q, q4_hyper, q4_pinj, q4_pend, q4_hyper,
                                  1, HC, H, none);
                pending = false;
            }
            const int NH = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
            const int HD = cfg.ple_embed_dim / NH;
            const int state_len = (cfg.ple_conv_kernel - 1) * cfg.ngram_size;
            launch_ple_ngram_ids(q, q4_tok, q4_ids, pos, 1, d.ple_mul,
                                 d.ple_size, d.ple_off, cfg.ngram_size - 1,
                                 cfg.heads_per_ngram, NH, cfg.eos_token_id, none);
            launch_ple_embed_gather(q, d.ple_table, d.ple_fp8, d.ple_scale,
                                    q4_ids, q4_emb, 1, NH, HD, d.ple_rows, none);
            gemv_any(d.ple_key,   q4_emb, q4_kv, none);
            gemv_any(d.ple_value, q4_emb, q4_kv + WIDE, none);
            // key is the first hc*H rows of the merged projection, value
            // the H after it.
            launch_ple_gate(q, q4_kv, q4_kv + WIDE, q4_hyper, d.ple_nk,
                            d.ple_nq, d.ple_nc, q4_gated,
                            q4_conv + size_t(state_len) * WIDE, 1, HC, H,
                            cfg.rms_eps, none);
            // The history rows sit in front of this token's row, so the
            // conv reads one contiguous array and a tap that falls before
            // the start of the sequence lands on a zeroed row.
            q.memcpy(q4_conv, d.ple_hist, size_t(state_len) * WIDE * sizeof(float));
            launch_ple_conv(q, q4_conv, q4_gated, q4_hyper, d.ple_cw,
                            q4_hyper, state_len, 1, WIDE,
                            cfg.ple_conv_kernel, cfg.ngram_size, none);
            // roll the window: drop the oldest row, append this one
            if (state_len > 0) {
                q.memcpy(d.ple_hist, q4_conv + size_t(WIDE),
                         size_t(state_len) * WIDE * sizeof(float));
                q.wait();
            }
        }

        // ---- attention hyper-connection -------------------------------
        hc_mix(d.hc_attn, pending ? q4_pend : nullptr,
               pending ? q4_pinj : nullptr, q4_inj, s.h2);
        pending = false;

        if (d.kind == LayerKind::LINEAR_ATTN) {
            const int qkv_ch = d.la_qkv.output_rows();
            gemv_any(d.la_qkv, s.h2, s.qkv, none);
            ConvParams cp{};
            cp.x = s.qkv; cp.weight = d.la_conv; cp.ring = d.conv_ring;
            cp.out = s.qkv; cp.channels = qkv_ch; cp.kernel = cfg.conv_kernel;
            launch_causal_conv1d_l2norm(q, cp, 2 * Hk, Dk, none);
            float* qv = s.qkv;
            float* kv = s.qkv + int64_t(Hk) * Dk;
            float* vv = s.qkv + int64_t(2) * Hk * Dk;
            gemv_any(d.la_ab, s.h2, s.abuf, none);
            launch_deltanet_gates(q, s.abuf, s.abuf + Hv, d.la_Alog, d.la_dtb,
                                  s.alpha, s.beta, Hv, none);
            DeltaNetParams dp{};
            dp.q = qv; dp.k = kv; dp.v = vv;
            dp.a = s.alpha; dp.beta = s.beta;
            dp.state = d.dn_state; dp.out = s.attn_out;
            dp.n_heads = Hv; dp.k_dim = Dk; dp.v_dim = Dv;
            dp.n_k_heads = Hk;
            launch_deltanet_step(q, dp, none);
            gemv_any(d.la_z, s.h2, s.zbuf, none);
            launch_rmsnorm_gate_silu(q, s.attn_out, s.zbuf, d.la_norm,
                                     Hv, Dv, cfg.rms_eps, none);
            gemv_any(d.la_out, s.attn_out, s.moe_y, none);
        } else {
            const int QD = d.q_proj.output_rows();
            const bool gated = (QD == 2 * cfg.n_heads * d.head_dim);
            gemv_any(d.q_proj, s.h2, s.qkv, none);
            if (gated)
                launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                                   cfg.n_heads, d.head_dim, none);
            float* qvec = gated ? s.qsplit : s.qkv;
            gemv_any(d.k_proj, s.h2, s.zbuf, none);
            gemv_any(d.v_proj, s.h2, s.bbuf, none);
            const int qheads = cfg.n_heads;
            launch_qk_norm_rope(q, qvec, s.zbuf, d.q_norm, d.k_norm,
                qheads, d.kv_heads, d.head_dim, s.d_pos,
                d.rope_theta, d.partial_rope, cfg.rms_eps, none);
            launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                                 s.d_pos, d.kv_heads, d.head_dim, max_seq, none);

            if (d.qsa) {
                // ---- the indexer --------------------------------------
                // Its q and k come from the BLOCK INPUT, not from the
                // attention's own q/k: index_qk_proj is a separate,
                // narrower projection (ref/qwen4_exp_nvidia_qsa.py:500).
                const int IH  = cfg.indexer_n_heads;
                const int IHD = cfg.indexer_head_dim;
                const int RAT = cfg.indexer_compress_ratio;
                const int BTK = cfg.indexer_budget / RAT;
                gemv_any(d.ix_qk, s.h2, q4_ixqk, none);
                float* iq = q4_ixqk;
                float* ik = q4_ixqk + int64_t(IH) * IHD;
                launch_rmsnorm_heads(q, iq, d.ix_qn, IH, IHD, cfg.rms_eps,
                                     true, none);
                // launch_rope_rows, not launch_rope_dev: the batched
                // prefill ropes the indexer query with this kernel and
                // the two must be the SAME arithmetic, or the two paths
                // rank blocks differently and disagree on which tokens
                // the layer attends to.
                launch_rope_rows(q, iq, 1, IH, IHD, pos, d.rope_theta,
                                 d.partial_rope, none);
                // The RAW key is cached exactly as projected: the pooling
                // averages raw keys and norms and ropes the POOLED result.
                q.memcpy(d.ix_kraw + int64_t(pos) * IHD, ik,
                         size_t(IHD) * sizeof(float));
                if ((pos + 1) % RAT == 0) {
                    const int b = pos / RAT;
                    launch_qsa_pool_blocks(q, d.ix_kraw, q4_pool, b, 1, RAT,
                                           IHD, none);
                    launch_rmsnorm_heads(q, q4_pool, d.ix_kn, 1, IHD,
                                         cfg.rms_eps, true, none);
                    launch_qsa_rope_blocks(q, q4_pool, b, 1, IHD, RAT,
                                           d.rope_theta, d.partial_rope, none);
                    q.memcpy(d.ix_kcmp + int64_t(b) * IHD, q4_pool,
                             size_t(IHD) * sizeof(float));
                }
                // Only COMPLETE blocks are rankable; the block still being
                // filled reaches attention through the expand's tail.
                const int seq_len = pos + 1;
                const int visible = seq_len / RAT;
                {
                    int32_t* v = q4_vis; int32_t* sq = q4_seq; int32_t* qp = q4_qpos;
                    const int32_t vv2 = visible, sv = seq_len, pv = pos;
                    q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
                        v[0] = vv2; sq[0] = sv; qp[0] = pv;
                    });
                }
                launch_qsa_index_logits(q, iq, d.ix_kcmp, q4_lg, 1, IH, IHD,
                                        visible > 0 ? visible : 1, q4_vis, none);
                launch_qsa_topk_blocks(q, q4_lg, q4_blk, 1,
                                       visible > 0 ? visible : 1, BTK,
                                       q4_vis, none);
                launch_qsa_expand_blocks(q, q4_blk, q4_idx, 1, BTK, RAT,
                                         cfg.indexer_budget, q4_seq, q4_qpos,
                                         none);
                launch_qsa_attention(q, qvec, d.k_cache, d.v_cache, q4_idx,
                                     s.attn_out, 1, qheads, d.kv_heads,
                                     d.head_dim, max_seq, q4_expand_w,
                                     cfg.attn_softmax_scale(d.head_dim), none);
            } else {
                AttnParams ap{};
                ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
                ap.out = s.attn_out;
                ap.seq_len = pos + 1; ap.seq_cap = max_seq;
                ap.head_dim = d.head_dim; ap.num_heads = qheads;
                ap.num_kv_heads = d.kv_heads;
                ap.softmax_scale = cfg.attn_softmax_scale(d.head_dim);
                ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
                ap.splits = GRAPH_SPLITS;
                ap.d_seq_len = s.d_seq_len;
                launch_flash_decode(q, ap, none);
                launch_flash_merge(q, ap, none);
            }
            if (gated) {
                const int gn = cfg.n_heads * d.head_dim;
                launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit, gn, none);
            }
            gemv_any(d.o_proj, s.attn_out, s.moe_y, none);
        }

        // ---- FFN hyper-connection -------------------------------------
        // combine_and_mix: this layer's attention output and the
        // injection its own mix produced, then the next block input.
        hc_mix(d.hc_mlp, s.moe_y, q4_inj, q4_pinj, s.h2);

        if (d.moe_layer) {
            const int I = cfg.moe_inter;
            gemv_any(d.router, s.h2, s.rlogits, none);
            launch_router_topk(q, s.rlogits, cfg.n_experts, cfg.top_k,
                               s.d_expert, s.d_weight, true, none);
            launch_moe_gate_up(q, d.moe, s.d_expert, s.h2, s.moe_h, none);
            launch_moe_down(q, d.moe, s.d_expert, s.d_weight, s.moe_h,
                            q4_pend, none);
            (void)I;
            if (d.sh_gu.w.N) {
                const int SI = d.sh_gu.output_rows() / 2;
                ffn_gemv(d, true, s.h2, s.sh_g, none);
                launch_swiglu(q, s.sh_g, s.sh_g + SI, s.sh_g, SI, none);
                ffn_gemv(d, false, s.sh_g, s.sh_out, none);
                if (d.has_sh_gate) {
                    gemv_any(d.sh_gate_q, s.h2, s.sh_gate_val, none);
                    launch_scale_by_sigmoid(q, s.sh_out, s.sh_gate_val, H, none);
                }
                launch_add(q, q4_pend, s.sh_out, H, none);
            }
        } else {
            const int FI = d.sh_gu.output_rows() / 2;
            ffn_gemv(d, true, s.h2, s.sh_g, none);
            launch_swiglu(q, s.sh_g, s.sh_g + FI, s.sh_g, FI, none);
            ffn_gemv(d, false, s.sh_g, q4_pend, none);
        }
        pending = true;    // (q4_pend, q4_pinj) travel to the next layer
    }

    // ---- the tail mixer ----------------------------------------------
    // It consumes whatever combine is still pending and produces the
    // single stream the head reads.  use_combine is false here, so it
    // emits no injection -- and there is no model.norm after it.
    hc_mix(hc_final, pending ? q4_pend : nullptr,
           pending ? q4_pinj : nullptr, nullptr, s.h2);
    gemv_any(lm_head, s.h2, s.logits, none);
    if (fusion_mask & 8) launch_incr_pos2(q, s.d_pos, s.d_seq_len, none);
    else {
        launch_incr_pos(q, s.d_pos, none);
        launch_incr_pos(q, s.d_seq_len, none);
    }
    ++pos;
    return s.logits;
}

const float* Grimoire::forward(int token) {
    check_token(token);
    if(mtp.ok && pos>0 && !recording) {
        mtp_warm(s.h,token,pos-1);
        set_cursor(pos);
    }
    if (cfg.is_muse) return forward_muse(token);
    if (cfg.is_gemma4) return forward_gemma4(token);
    if (cfg.is_qwen4_exp) return forward_qwen4_exp(token);
    if (dag && !tp_enabled()) return forward_dag(token);
    const int H  = cfg.hidden;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const std::vector<sycl::event> none{};

    // Profile ONE linear and ONE full-attention layer in detail. Marking
    // all 40 would add 600 kernels and measure the markers instead.
    int tl_lin = -1, tl_full = -1;
    if (timeline)
        for (int i = 0; i < cfg.n_layers; ++i) {
            if (tl_lin  < 0 && L[i].kind == LayerKind::LINEAR_ATTN) tl_lin  = i;
            if (tl_full < 0 && L[i].kind == LayerKind::FULL_ATTN)   tl_full = i;
        }

    mark("start");
    if (pp_enabled() && pp_rank > 0) {
        if (!pp_recv_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank 1: hidden receive failed\n");
            return nullptr;
        }
        if (!pp_recv_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: tap receive failed\n", pp_rank);
            return nullptr;
        }
    } else if (tp_enabled()) {
        if(!embed_one(token,s.h)){
            std::fprintf(stderr,"TP embedding all-reduce failed\n");return nullptr;
        }
    } else if (recording) {
        // Graph capture bakes every argument into the recorded node, so a
        // host-side token would pin the replay to the captured token.
        // s.d_tok is the same buffer argmax_token() writes, so reading the
        // embedding through it makes one recorded graph valid for every
        // subsequent token.
        launch_embed_batched(q, embed, s.d_tok, s.h, 1, H, none);
    } else {
        launch_embed(q, embed, token, s.h, H, none);
    }
    // layer 0 has no previous block output to add
    q.memset(s.moe_y, 0, size_t(H) * sizeof(float));
    if (fusion_mask & 4) q.memset(s.sh_out, 0, size_t(H) * sizeof(float));
    mark("embed");
    const int layer_begin = pp_enabled() ? pp_begin : 0;
    const int layer_end   = pp_enabled() ? pp_end   : cfg.n_layers;
    for (int i = layer_begin; i < layer_end; ++i) {
        LayerDev& d = L[i];
        const bool mk = timeline && (i == tl_lin || i == tl_full);
        auto MK = [&](const char* t) { if (mk) mark(t); };

        // ---- attention block ------------------------------------------
        // h is the residual stream throughout. rmsnorm_residual folds
        // the PREVIOUS block's output (moe_y) back into h and normalises
        // in one pass.
        //
        // Passing nullptr here silently discarded every layer's FFN
        // output -- 40 MoE blocks contributing nothing. The model still
        // ran at full speed and emitted fluent-looking garbage, which is
        // exactly how this class of bug hides.
        if ((fusion_mask & 4) && i > 0)
            launch_rmsnorm_residual2(q, s.h, s.moe_y, s.sh_out,
                                     d.in_norm, s.h2, H, cfg.rms_eps, none);
        else
            launch_rmsnorm_residual(q, s.h, s.moe_y, d.in_norm, s.h2,
                                    H, cfg.rms_eps, none);
        // vLLM turns target_layer_id n into aux layer n+1.  At entry to layer
        // n+1 Grimoire has folded layer n's output into s.h, matching Muse's
        // post-layer aux hidden state exactly.
        if (dflash2.target_aux) {
            for (size_t tap = 0; tap < dflash2.target_layers.size(); ++tap) {
                if (dflash2.target_layers[tap] + 1 == i) {
                    launch_dflash_store_tap_dev(q,s.h,dflash2.target_aux,H,
                        int(dflash2.target_layers.size()),s.d_pos,int(tap),none,
                        cfg.is_muse);
                    break;
                }
            }
        }
        if (i == probe_layer) probe("L0 in_norm", s.h2, H);
        MK("  in_norm");

        if (d.kind == LayerKind::LINEAR_ATTN) {
            const int qkv_ch = d.la_qkv.output_rows(); // logical full shape
            gemv_any(d.la_qkv, s.h2, s.qkv, none);
            if (i == probe_layer) probe("L0 qkv proj", s.qkv, qkv_ch);
            MK("  la_qkv gemv");

            // causal depthwise conv over the packed qkv, then SiLU
            ConvParams cp{};
            cp.x = s.qkv; cp.weight = d.la_conv; cp.ring = d.conv_ring;
            cp.out = s.qkv; cp.channels = qkv_ch; cp.kernel = cfg.conv_kernel;
            if (fusion_mask & 16)
                launch_causal_conv1d_l2norm(q, cp, 2 * Hk, Dk, none);
            else
                launch_causal_conv1d(q, cp, none);
            if (i == probe_layer) probe("L0 conv+silu", s.qkv, qkv_ch);
            MK("  conv1d");

            float* qv = s.qkv;
            float* kv = s.qkv + int64_t(Hk) * Dk;
            float* vv = s.qkv + int64_t(2) * Hk * Dk;

            // q and k must be unit length or the delta rule is not
            // contractive and the recurrent state diverges over context.
            // q and k are 2*Hk contiguous heads of Dk; v follows and is
            // NOT normalised.
            if (!(fusion_mask & 16))
                launch_l2norm_heads(q, qv, 2 * Hk, Dk, none);
            MK("  l2norm_heads");

            // one launch produces both a and b, laid out back to back
            gemv_any(d.la_ab, s.h2, s.abuf, none);
            launch_deltanet_gates(q, s.abuf, s.abuf + Hv, d.la_Alog, d.la_dtb,
                                  s.alpha, s.beta, Hv, none);
            MK("  ab gemv + gates");

            DeltaNetParams dp{};
            dp.q = qv; dp.k = kv; dp.v = vv;
            dp.a = s.alpha; dp.beta = s.beta;
            dp.state = d.dn_state; dp.out = s.attn_out;
            dp.n_heads = Hv; dp.k_dim = Dk; dp.v_dim = Dv;
            dp.n_k_heads = Hk;
            launch_deltanet_step(q, dp, none);
            if (i == probe_layer) probe("L0 deltanet", s.attn_out, Hv * Dv);
            MK("  deltanet_step");

            // Gated RMSNorm, in the reference order:
            //     out = rms_norm(core_out) * silu(z)
            // Normalising the PRODUCT instead -- rms_norm(out * silu(z))
            // -- is a different operation and rescales by the gate's own
            // magnitude, which quietly destroys the output distribution.
            gemv_any(d.la_z, s.h2, s.zbuf, none);
            if (i == probe_layer) probe("L0 z", s.zbuf, Hv * Dv);
            MK("  z gemv");
            if (fusion_mask & 1)
                launch_rmsnorm_gate_silu(q, s.attn_out, s.zbuf, d.la_norm,
                                         Hv, Dv, cfg.rms_eps, none);
            else {
                launch_rmsnorm_heads(q, s.attn_out, d.la_norm, Hv, Dv,
                                     cfg.rms_eps, false, none);
                launch_gate_silu(q, s.attn_out, s.zbuf, s.attn_out, Hv * Dv, none);
            }
            if (i == probe_layer) probe("L0 after gate", s.attn_out, Hv * Dv);
            MK("  rmsnorm + gate");
            gemv_any(d.la_out, s.attn_out, s.moe_y, none);
            if (i == probe_layer) probe("L0 attn out", s.moe_y, H);
            MK("  out gemv");
        } else {
            const int QD = d.q_proj.output_rows();
            const int KD = d.k_proj.output_rows();
            // With attn_output_gate the projection emits [q | gate] per
            // head, so only half its rows are queries. Feeding all of
            // them to attention treats gate values as queries.
            const bool gated = (QD == 2 * cfg.n_heads * d.head_dim);
            gemv_any(d.q_proj, s.h2, s.qkv, none);
            if (gated)
                launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                                   cfg.n_heads, d.head_dim, none);
            float* qvec = gated ? s.qsplit : s.qkv;
            MK("  q gemv + split");
            gemv_any(d.k_proj, s.h2, s.zbuf, none);
            if (d.k2_sparse) mova_value_m1(d, s.h2, s.bbuf, none);
        else gemv_any(d.v_proj, s.h2, s.bbuf, none);
            MK("  k+v gemv");

            const int qheads = cfg.n_heads;
            // q_norm / k_norm come BEFORE RoPE in the reference:
            //   q = q_norm(q.view(heads, head_dim))
            //   k = k_norm(k_proj(x).view(heads, head_dim))
            //   q, k = apply_rotary_pos_emb(q, k, cos, sin)
            // Both are Qwen3_5MoeRMSNorm, so zero-centered.
            // The fused kernel bakes in launch_rope_dev's partial_rope.
            // A layer that wants proportional RoPE must take the split
            // path, or it silently gets the wrong frequencies.
            if ((fusion_mask & 2) && d.q_norm && d.k_norm &&
                !d.rope_proportional) {
                launch_qk_norm_rope(q, qvec, s.zbuf, d.q_norm, d.k_norm,
                    qheads, d.kv_heads, d.head_dim, s.d_pos,
                    d.rope_theta, d.partial_rope, cfg.rms_eps, none);
            } else {
                if (d.q_norm)
                    launch_rmsnorm_heads(q, qvec, d.q_norm, qheads, d.head_dim,
                                         cfg.rms_eps, true, none);
                if (d.k_norm)
                    launch_rmsnorm_heads(q, s.zbuf, d.k_norm, d.kv_heads,
                                         d.head_dim, cfg.rms_eps, true, none);
                if (d.rope_proportional) {
                    launch_rope_proportional(q, qvec, qheads, d.head_dim,
                        s.d_pos, d.rope_theta, d.partial_rope, none,
                        d.rope_factor);
                    launch_rope_proportional(q, s.zbuf, d.kv_heads, d.head_dim,
                        s.d_pos, d.rope_theta, d.partial_rope, none,
                        d.rope_factor);
                } else {
                    launch_rope_dev(q, qvec, qheads, d.head_dim, s.d_pos,
                                    d.rope_theta, d.partial_rope, none);
                    launch_rope_dev(q, s.zbuf, d.kv_heads, d.head_dim, s.d_pos,
                                    d.rope_theta, d.partial_rope, none);
                }
            }
            MK("  q/k norm + rope");
            launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                                 s.d_pos, d.kv_heads, d.head_dim,
                                 max_seq, none);
            MK("  kv_append");

            AttnParams ap{};
            ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
            ap.out = s.attn_out;
            ap.seq_len = pos + 1; ap.seq_cap = max_seq;
            ap.head_dim = d.head_dim; ap.num_heads = qheads;
            ap.num_kv_heads = d.kv_heads;
            ap.softmax_scale = cfg.attn_softmax_scale(d.head_dim);
            ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
            // FIXED split count so the launch geometry never changes and
            // the graph stays valid; the kernel reads the live length.
            ap.splits    = GRAPH_SPLITS;
            ap.d_seq_len = s.d_seq_len;
            if (i == probe_layer) probe("FA v", s.bbuf, d.kv_heads * d.head_dim);
            // GQA redundancy: this model is 24 query heads over 4 KV heads, so
            // launch_flash_decode's one-subgroup-per-query-head mapping fetches
            // every KV byte 6 times. The batched kernel gives a workgroup one
            // KV head and stages K/V in SLM for all q_per_kv query heads, so
            // the cache streams once. Opt-in until measured.
            static const bool batched_decode_attn =
                std::getenv("GRIMOIRE_DECODE_BATCHED_ATTN") != nullptr;
            if (batched_decode_attn) {
                static const int bdelta = [] {
                    const char* e = std::getenv("GRIMOIRE_DECODE_BATCHED_DELTA");
                    return e ? std::atoi(e) : 0;
                }();
                static const char* vs_env = std::getenv("GRIMOIRE_DECODE_BATCHED_SPLITS");
                const int vs = vs_env ? std::atoi(vs_env)
                    : std::min(MAX_SPLITS,
                        std::max(GRAPH_SPLITS, (pos + 1 + 127) / 128));
                launch_flash_decode_batched(q, qvec, d.k_cache, d.v_cache,
                    s.attn_out, 1, pos + 1 + bdelta, qheads, d.kv_heads,
                    d.head_dim, max_seq, ap.softmax_scale,
                    s.part, s.pm, s.pl, vs, {});
                MK("  flash_decode_batched");
            } else {
            launch_flash_decode(q, ap, none);
            MK("  flash_decode");
            launch_flash_merge(q, ap, none);
            MK("  flash_merge");
            }
            if (i == probe_layer) probe("FA attn core", s.attn_out, qheads * d.head_dim);

            // apply the output gate before projecting back
            if (gated) {
                const int gn = cfg.n_heads * d.head_dim;
                if (cfg.attn_gate == 2)
                    launch_softplus_gate(q, s.attn_out, s.gsplit, s.attn_out,
                                         gn, kK2GateBeta, none);
                else
                    launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit, gn, none);
            }
            if (i == probe_layer) probe("FA after gate", s.attn_out, qheads * d.head_dim);
            MK("  attn out gate");
            gemv_any(d.o_proj, s.attn_out, s.moe_y, none);
            if (i == probe_layer) probe("FA out", s.moe_y, H);
            MK("  o gemv");
            (void)KD;
        }

        // ---- FFN block -------------------------------------------------
        // fused: h += attn_out, then normalise
        launch_rmsnorm_residual(q, s.h, s.moe_y, d.post_norm, s.h2, H, cfg.rms_eps, none);
        MK("  post_norm");

        if (d.moe_layer) {
            const int I = cfg.moe_inter;
            gemv_any(d.router, s.h2, s.rlogits, none);
            MK("  router gemv");
            launch_router_topk(q, s.rlogits, cfg.n_experts, cfg.top_k,
                               s.d_expert, s.d_weight, true, none);
            MK("  router topk");
            if (i == probe_layer) probe("L0 router logits", s.rlogits, cfg.n_experts);
            if (i == probe_layer && debug) {
                q.wait();
                std::vector<int32_t> ex(cfg.top_k);
                std::vector<float>   wt(cfg.top_k);
                q.memcpy(ex.data(), s.d_expert, cfg.top_k * sizeof(int32_t)).wait();
                q.memcpy(wt.data(), s.d_weight, cfg.top_k * sizeof(float)).wait();
                std::printf("    L0 routed experts:");
                for (int t = 0; t < cfg.top_k; ++t)
                    std::printf(" %d(%.3f)", ex[t], wt[t]);
                std::printf("\n");
                std::fflush(stdout);
            }
            const int32_t* route_expert=s.d_expert;
            const float* route_weight=s.d_weight;
            if(tp_enabled()){
                const int b=d.expert_begin,e=b+d.expert_count,k=cfg.top_k;
                int32_t* dst_e=tp_expert;float* dst_w=tp_weight;
                const int32_t* src_e=s.d_expert;const float* src_w=s.d_weight;
                q.parallel_for(sycl::range<1>(size_t(k)),[=](sycl::id<1> ix){
                    const int j=int(ix[0]),g=src_e[j];const bool own=g>=b&&g<e;
                    dst_e[j]=own?g-b:-1;dst_w[j]=own?src_w[j]:0.0f;
                });
                route_expert=tp_expert;route_weight=tp_weight;
            }
            launch_moe_gate_up(q,d.moe,route_expert,s.h2,s.moe_h,none);
            MK("  moe_gate_up");
            if (i == probe_layer) probe("L0 moe_h (gate_up)", s.moe_h, cfg.top_k * I);
            launch_moe_down(q,d.moe,route_expert,route_weight,s.moe_h,s.moe_y,none);
            if(tp_enabled()&&!tp_allreduce_sum(s.moe_y,H)){
                std::fprintf(stderr,"TP MoE all-reduce failed\n");return nullptr;
            }
            MK("  moe_down");
            if (i == probe_layer) probe("L0 moe routed", s.moe_y, H);

            // always-on shared expert, added to the routed result
            const int SI = d.sh_gu.output_rows() / 2;
            ffn_gemv(d, true, s.h2, s.sh_g, none);
            launch_swiglu(q, s.sh_g, s.sh_g + SI, s.sh_g, SI, none);
            ffn_gemv(d, false, s.sh_g, s.sh_out, none);
            if (d.has_sh_gate) {
                gemv_any(d.sh_gate_q, s.h2, s.sh_gate_val, none);
                launch_scale_by_sigmoid(q, s.sh_out, s.sh_gate_val, H, none);
            }
            MK("  shared expert");
            if (i == probe_layer) probe("L0 shared out", s.sh_out, H);
            if (!(fusion_mask & 4)) launch_add(q, s.moe_y, s.sh_out, H, none);
            MK("  add shared");
            if (i == probe_layer) probe("L0 moe total", s.moe_y, H);
        } else {
            const int FI = d.sh_gu.output_rows() / 2;
            ffn_gemv(d, true, s.h2, s.sh_g, none);
            MK("  ffn gate_up");
            launch_swiglu(q, s.sh_g, s.sh_g + FI, s.sh_g, FI, none);
            MK("  ffn swiglu");
            ffn_gemv(d, false, s.sh_g, s.moe_y, none);
            MK("  ffn down");
        }

        if (timeline) {
            char t[24];
            std::snprintf(t, sizeof t, "L%02d %s", i,
                          d.kind == LayerKind::LINEAR_ATTN ? "linear" : "full  ");
            mark(t);
        }
        if (debug && (i == 0 || i == cfg.n_layers / 2 || i == cfg.n_layers - 1)) {
            char tag[32];
            std::snprintf(tag, sizeof tag, "hidden after L%d", i);
            probe(tag, s.h, H);
        }
    }

    // Rank 0 owns no output head. Materialize the last early-layer FFN
    // residual before crossing the process boundary, then advance its local
    // position counters so both ranks retain identical attention positions.
    if (pp_enabled() && pp_rank < pp_world-1) {
        if (fusion_mask & 4) {
            launch_add(q, s.h, s.moe_y, H, none);
            launch_add(q, s.h, s.sh_out, H, none);
        } else {
            launch_add(q, s.h, s.moe_y, H, none);
        }
        if (!pp_send_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank %d: hidden send failed\n",pp_rank);
            return nullptr;
        }
        if (!pp_send_taps(pos, 1)) {
            std::fprintf(stderr, "PP rank %d: tap send failed\n",pp_rank);
            return nullptr;
        }
        if (fusion_mask & 8) launch_incr_pos2(q, s.d_pos, s.d_seq_len, none);
        else {
            launch_incr_pos(q, s.d_pos, none);
            launch_incr_pos(q, s.d_seq_len, none);
        }
        ++pos;
        return s.logits;
    }

    if (fusion_mask & 4)
        launch_rmsnorm_residual2(q, s.h, s.moe_y, s.sh_out,
                                 fnorm, s.h2, H, cfg.rms_eps, none);
    else
        launch_rmsnorm_residual(q, s.h, s.moe_y, fnorm, s.h2,
                                H, cfg.rms_eps, none);
    probe("final norm", s.h2, H);
    mark("final_norm");
    gemv_any(lm_head, s.h2, s.logits, none);
    probe("logits", s.logits, cfg.vocab);
    mark("lm_head gemv");
    if (fusion_mask & 8) launch_incr_pos2(q, s.d_pos, s.d_seq_len, none);
    else {
        launch_incr_pos(q, s.d_pos, none);
        launch_incr_pos(q, s.d_seq_len, none);
    }
    mark("incr_pos");
    // No q.wait() here. The caller's argmax_token() ends in a blocking
    // memcpy, and the queue is in-order, so waiting here only drains the
    // pipeline a second time per token -- pure latency between tokens.
    // The timeline still needs the drain to read its profiling info.
    if (!recording && timeline) q.wait();
    if (timeline && !tl_done && tl_tok == kTlToken) {
        dump_timeline(); tl_done = true; tl.clear();
    }
    if (timeline) ++tl_tok;
    ++pos;
    return s.logits;
}

// ---------------------------------------------------------------------
//  MTP draft.  One decoder layer over fc([norm(h_t) ; norm(embed(t+1))]),
//  then the model's own lm_head.  Every kernel here is the one the main
//  full-attention path uses; only the weights and the KV cache differ.
//
//  s.h must still hold h_t, so this runs AFTER forward() and before the
//  next one.  Everything else it borrows (s.h2, s.qkv, s.attn_out, s.moe_y,
//  ...) is transient within a step and re-initialised at the top of forward.
// ---------------------------------------------------------------------
// ---------------------------------------------------------------------
//  MTP KV warm-up.
//
//  The MTP head attends over its OWN KV cache, and nothing ever wrote that
//  cache for prompt positions: prefill did not touch it and reset() cleared
//  it, so the drafter began every request with no context and its first
//  drafts were near-worthless. Commit 3b5b510 named this and never built it:
//  "the real fix is to populate mtp.L's KV cache during prefill ... That is
//  what vLLM does and why it holds ~52 TG at 4.4k context."
//
//  MEASURED 2026-09-05, 4080-token prompt, k=3, EXACT_VERIFY, same config,
//  only the generation length changed:
//      n=32   31.4 tok/s     n=64  37.8 tok/s     n=128  42.0 tok/s
//  A per-request fixed cost that dilutes as generation grows -- the warm-up
//  penalty. vLLM shows no such slope and holds ~44 at n=32.
//
//  Only K and V are needed to fill the cache, so this runs the front of
//  mtp_draft and stops: fc([norm(embed) ; norm(hidden)]) -> in_norm ->
//  k_proj/v_proj -> k_norm -> RoPE -> kv_append. No attention, no FFN and no
//  lm_head, which is what makes warming a few hundred positions affordable.
// ---------------------------------------------------------------------
void Grimoire::mtp_warm(const float* hidden, int next_token, int position) {
    if (!mtp.ok) return;
    if(position<0||position>=max_seq||next_token<0||next_token>=cfg.vocab)throw std::out_of_range("invalid MTP warm position/token");
    const int H = cfg.hidden;
    const std::vector<sycl::event> none{};
    LayerDev& d = mtp.L;

    set_cursor(position);

    // Same concat order as mtp_draft: EMBEDDING FIRST.
    launch_rmsnorm_residual(q, const_cast<float*>(hidden), nullptr, mtp.pre_h,
                            mtp.cat + H, H, cfg.rms_eps, none);
    // embed_one, not launch_embed: under TP the embedding table is row
    // sharded, so a global token id indexes the WRONG row on every rank
    // that does not own it (and off the end of the last one).
    if(!embed_one(next_token, mtp.resid))
        throw std::runtime_error("MTP warm embedding all-reduce failed");
    launch_rmsnorm_residual(q, mtp.resid, nullptr, mtp.pre_e, mtp.cat,
                            H, cfg.rms_eps, none);
    launch_gemv(q, mtp.fc.w, mtp.cat, mtp.x, none);
    launch_rmsnorm_residual(q, mtp.x, nullptr, d.in_norm, s.h2, H, cfg.rms_eps, none);

    gemv_any(d.k_proj, s.h2, s.zbuf, none);
    if (d.k2_sparse) mova_value_m1(d, s.h2, s.bbuf, none);
    else gemv_any(d.v_proj, s.h2, s.bbuf, none);
    if (d.k_norm)
        launch_rmsnorm_heads(q, s.zbuf, d.k_norm, d.kv_heads,
                             d.head_dim, cfg.rms_eps, true, none);
    launch_rope_dev(q, s.zbuf, d.kv_heads, d.head_dim, s.d_pos,
                    d.rope_theta, d.partial_rope, none);
    launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                         s.d_pos, d.kv_heads, d.head_dim, max_seq, none);
}

int Grimoire::mtp_draft(int next_token, int position, bool from_mtp_hidden) {
    // Only the last pipeline stage owns the MTP head and the final hidden
    // state, so it drafts and the earlier stages receive -- the same
    // backward hop argmax_token already uses to keep every rank's decode
    // loop in lockstep.  Every rank ends up with identical candidates,
    // which is what makes the acceptance count agree without a second
    // round trip.
    if (pp_enabled() && pp_rank < pp_world - 1) return pp_sync_token(-1);
    if (!mtp.ok) return -1;
    if(position<0||position>=max_seq||next_token<0||next_token>=cfg.vocab)throw std::out_of_range("invalid MTP draft position/token");
    const int H = cfg.hidden;
    const std::vector<sycl::event> none{};
    LayerDev& d = mtp.L;

    // position for this draft: token t+1 sits at `position`
    set_cursor(position);

    // cat order.  DeepSeek/Qwen MTP is fc([norm(embedding) ; norm(hidden)]),
    // i.e. EMBEDDING FIRST.  Hidden-first measured 0/159 acceptance, which is
    // what a wrong concat looks like -- not a weak head.
    // residual == nullptr, so s.h is read and NOT modified.
    static const bool hid_first = std::getenv("GRIMOIRE_MTP_HID_FIRST") != nullptr;
    float* p_emb = hid_first ? mtp.cat + H : mtp.cat;
    float* p_hid = hid_first ? mtp.cat     : mtp.cat + H;
    // CHAINED-DRAFT HIDDEN SOURCE.
    //
    // vLLM feeds the MTP layer's OUTPUT into the next draft step, not its
    // input. Confirmed in the live container 2026-09-05:
    //   qwen3_5_mtp.py:179  hidden_states, _ = self.norm(hidden_states, residual)
    //                :183  return hidden_states          <- post-layer, post-norm
    //   llm_base_proposer.py:600-601  hidden_states = ret_hidden_states
    //                                 ... carried into the next draft step
    // GRIMOIRE used mtp.x, which is the fc OUTPUT = the decoder layer's INPUT,
    // so every chained draft (position 2 onward) started from the wrong
    // tensor. The correct value was already computed for the lm_head but
    // written to the shared scratch s.h2 and clobbered; mtp.h2 was allocated
    // (grimoire.cpp ~3161) and never used. It now holds it across calls.
    // GRIMOIRE_MTP_CHAIN_OLD=1 restores the previous behaviour for A/B.
    static const bool chain_old = std::getenv("GRIMOIRE_MTP_CHAIN_OLD") != nullptr;
    const float* hsrc = from_mtp_hidden ? (chain_old ? mtp.x : mtp.h2) : s.h;
    launch_rmsnorm_residual(q, const_cast<float*>(hsrc), nullptr, mtp.pre_h,
                            p_hid, H, cfg.rms_eps, none);
    // TP-aware: see the note in mtp_warm.
    if(!embed_one(next_token, mtp.resid))
        throw std::runtime_error("MTP draft embedding all-reduce failed");
    launch_rmsnorm_residual(q, mtp.resid, nullptr, mtp.pre_e, p_emb,
                            H, cfg.rms_eps, none);
    launch_gemv(q, mtp.fc.w, mtp.cat, mtp.x, none);      // [H] = [H][2H] x [2H]

    // ---- one full-attention decoder layer over mtp.x --------------------
    q.memset(s.moe_y, 0, size_t(H) * sizeof(float));
    launch_rmsnorm_residual(q, mtp.x, nullptr, d.in_norm, s.h2, H, cfg.rms_eps, none);

    const int QD = d.q_proj.w.N;
    const bool gated = (QD == 2 * cfg.n_heads * d.head_dim);
    gemv_any(d.q_proj, s.h2, s.qkv, none);
    if (gated)
        launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                           cfg.n_heads, d.head_dim, none);
    float* qvec = gated ? s.qsplit : s.qkv;
    gemv_any(d.k_proj, s.h2, s.zbuf, none);
    if (d.k2_sparse) mova_value_m1(d, s.h2, s.bbuf, none);
    else gemv_any(d.v_proj, s.h2, s.bbuf, none);

    if ((fusion_mask & 2) && d.q_norm && d.k_norm) {
        launch_qk_norm_rope(q, qvec, s.zbuf, d.q_norm, d.k_norm,
            cfg.n_heads, d.kv_heads, d.head_dim, s.d_pos,
            d.rope_theta, d.partial_rope, cfg.rms_eps, none);
    } else {
        if (d.q_norm)
            launch_rmsnorm_heads(q, qvec, d.q_norm, cfg.n_heads, d.head_dim,
                                 cfg.rms_eps, true, none);
        if (d.k_norm)
            launch_rmsnorm_heads(q, s.zbuf, d.k_norm, d.kv_heads,
                                 d.head_dim, cfg.rms_eps, true, none);
        launch_rope_dev(q, qvec, cfg.n_heads, d.head_dim, s.d_pos,
                        d.rope_theta, d.partial_rope, none);
        launch_rope_dev(q, s.zbuf, d.kv_heads, d.head_dim, s.d_pos,
                        d.rope_theta, d.partial_rope, none);
    }
    launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                         s.d_pos, d.kv_heads, d.head_dim, max_seq, none);

    AttnParams ap{};
    ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
    ap.out = s.attn_out;
    ap.seq_len = position + 1; ap.seq_cap = max_seq;
    ap.head_dim = d.head_dim; ap.num_heads = cfg.n_heads;
    ap.num_kv_heads = d.kv_heads;
    ap.softmax_scale = cfg.attn_softmax_scale(d.head_dim);
    ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
    ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
    launch_flash_decode(q, ap, none);
    launch_flash_merge(q, ap, none);
    if (gated)
        launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit,
                                cfg.n_heads * d.head_dim, none);
    gemv_any(d.o_proj, s.attn_out, s.moe_y, none);

    // ---- FFN ------------------------------------------------------------
    launch_rmsnorm_residual(q, mtp.x, s.moe_y, d.post_norm, s.h2, H, cfg.rms_eps, none);
    if (cfg.is_moe() && d.moe.cfg.num_experts > 0) {
        const int I = cfg.moe_inter;
        gemv_any(d.router, s.h2, s.rlogits, none);
        launch_router_topk(q, s.rlogits, cfg.n_experts, cfg.top_k,
                           s.d_expert, s.d_weight, true, none);
        launch_moe_gate_up(q, d.moe, s.d_expert, s.h2, s.moe_h, none);
        launch_moe_down(q, d.moe, s.d_expert, s.d_weight,
                        s.moe_h, s.moe_y, none);

        const int SI = d.sh_gu.output_rows() / 2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        launch_swiglu(q, s.sh_g, s.sh_g + SI, s.sh_g, SI, none);
        gemv_any(d.sh_down, s.sh_g, s.sh_out, none);
        if (d.has_sh_gate) {
            gemv_any(d.sh_gate_q, s.h2, s.sh_gate_val, none);
            launch_scale_by_sigmoid(q, s.sh_out, s.sh_gate_val, H, none);
        }
        launch_add(q, s.moe_y, s.sh_out, H, none);
        (void)I;
    } else {
        const int FI = d.sh_gu.output_rows() / 2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        launch_swiglu(q, s.sh_g, s.sh_g + FI, s.sh_g, FI, none);
        gemv_any(d.sh_down, s.sh_g, s.moe_y, none);
    }

    // ---- final norm + the model's own lm_head ---------------------------
    // Into mtp.h2, not s.h2: this value is both the lm_head input AND the
    // hidden state the next chained draft consumes, and s.h2 is shared
    // scratch that the next call overwrites before it is read.
    launch_rmsnorm_residual(q, mtp.x, s.moe_y, mtp.norm, mtp.h2, H, cfg.rms_eps, none);
    static const int draft_vocab = [] {
        const char* v = std::getenv("GRIMOIRE_MTP_DRAFT_VOCAB");
        return v && *v ? std::atoi(v) : 0;
    }();
    const int dv = draft_vocab > 0 ? std::min(draft_vocab, cfg.vocab) : cfg.vocab;
    // The reduced-vocabulary shortcut writes s.logits directly and so
    // skips the all-gather gemv_any would have done: under TP that
    // leaves every rank holding only its own slice of the draft logits
    // and the argmax below picks from a fraction of the vocabulary.
    if (dv < cfg.vocab && lm_head.has_i4() && !lm_head.tp_sharded())
        launch_gemv_int4sym(q, lm_head.i4, lm_head.i4s, mtp.h2, s.logits,
                            dv, H, none);
    else
        gemv_any(lm_head, mtp.h2, s.logits, none);
    launch_argmax(q, s.logits, dv, s.d_tok, s.d_val, none);
    int32_t tok = 0;
    q.memcpy(&tok, s.d_tok, sizeof(int32_t)).wait();
    return pp_enabled() ? pp_sync_token(int(tok)) : int(tok);
}

int Grimoire::argmax_token() {
    const std::vector<sycl::event> none{};
    // Only the late-stage rank owns valid logits. Send its selected token
    // back so rank 0's independent generation loop stays in lockstep.
    if(pp_enabled()&&pp_rank<pp_world-1)return pp_sync_token(-1);
    if (dag) {
        std::vector<sycl::event> ready = dag_tail;
        ready.push_back(dag_logits);
        sycl::event e_arg = launch_argmax(
            q, s.logits, cfg.vocab, s.d_tok, s.d_val, ready);
        int32_t tok = 0;
        sycl::event e_copy = q.submit([&](sycl::handler& h) {
            h.depends_on(e_arg);
            h.memcpy(&tok, s.d_tok, sizeof(int32_t));
        });
        e_copy.wait();
        dag_tail.clear();
        if (debug) {
            float val = 0;
            q.memcpy(&val, s.d_val, sizeof(float)).wait();
            std::printf("    argmax -> id %d  logit %.4f\n", tok, val);
        }
        return pp_enabled() ? pp_sync_token(int(tok)) : int(tok);
    }
    launch_argmax(q, s.logits, cfg.vocab, s.d_tok, s.d_val, none);
    // ONE blocking round trip per token, not two. The logit value is only
    // ever printed under debug, and each `.wait()` here is a full device
    // round trip on the critical path between one token and the next.
    int32_t tok = 0;
    q.memcpy(&tok, s.d_tok, sizeof(int32_t)).wait();
    if (debug) {
        float val = 0;
        q.memcpy(&val, s.d_val, sizeof(float)).wait();
        std::printf("    argmax -> id %d  logit %.4f\n", tok, val);
        std::fflush(stdout);
    }
    return pp_enabled() ? pp_sync_token(int(tok)) : int(tok);
}

// Original DFlash: ingest any target taps not yet present in the six draft KV
// caches, then evaluate [bonus, mask x 15] in one non-causal block. DFlash2
// has additional grouped-conv/selector stages and deliberately does not enter
// this path.
// Ornith's verify slope is ~2.95 ms per extra verified token, so the
// profitable block is far shorter than the drafter's native 16. Let the
// width be tuned; M is the block INCLUDING the bonus row, so M=4 verifies
// 3 drafts.
int Grimoire::dflash_block_rows() const {
    constexpr int MMAX=16;
    static const int m_env=[]{const char* v=std::getenv("GRIMOIRE_DFLASH_M");
        return v&&*v?std::atoi(v):0;}();
    const int wide=dflash2.draft_head_rows?8:MMAX;
    return (m_env>=2&&m_env<=MMAX)?m_env:wide;
}

bool Grimoire::dflash_draft(int bonus_token, int position,
                            std::vector<int32_t>& draft_tokens,
                            bool context_only) {
    constexpr int MMAX=16;   // the host_tokens array below is this wide
    // Under PP only the LAST stage owns the final hidden state, so only it
    // can run the drafter.  Earlier stages take the drafted block from the
    // same backward hop mtp_draft uses, so every rank ends up with the
    // identical candidate list and therefore the identical acceptance
    // count -- which is what lets them all roll back to the same position
    // without a second round trip.
    if(pp_enabled()&&pp_rank<pp_world-1){
        draft_tokens.clear();
        if(context_only)return true;     // nothing to ingest without a drafter
        if(pp_dflash<2)return false;
        draft_tokens.assign(size_t(pp_dflash-1),0);
        if(!pp_sync_tokens(draft_tokens)){
            std::fprintf(stderr,"PP rank %d: draft block receive failed\n",pp_rank);
            return false;
        }
        // The last stage marks a refusal with -1, which is not a legal
        // token id.  Fail the same round it did, rather than verifying a
        // block of -1 and throwing "invalid draft token" on every rank.
        if(!draft_tokens.empty()&&draft_tokens[0]<0){
            draft_tokens.clear();
            return false;
        }
        return true;
    }
    const int M=dflash_block_rows();
    // Every refusal from here on must ALSO unblock the earlier stages, which
    // are already sitting in pp_sync_tokens() waiting for a block: they
    // committed to receiving one before this stage decided it could not
    // produce it.  Without this they wait until the socket tears down --
    // a CLI process exiting frees them by EOF, a surviving server does not.
    // A block of -1 is the failure marker; it is not a legal token id, and
    // the receiver turns it back into `return false` so every stage fails
    // the same round the same way.
    // THE LAST STAGE OWES A REPLY, AND ITS SHAPE DEPENDS ON THE PHASE.
    //
    // Earlier stages are never idle after calling in here.  On a DRAFTING
    // call they are inside pp_sync_tokens() waiting for M-1 integers.  On
    // a CONTEXT-ONLY call they return immediately and go straight to
    // argmax_token(), which waits for exactly ONE.  So:
    //
    //   drafting, failed   -> M-1 negative tokens
    //   context-only, failed -> ONE negative token
    //   context-only, ok   -> nothing; argmax_token sends the scalar next
    //
    // Both wrong answers have been made here.  Sending a block during a
    // context-only call put the first -1 where the scalar was expected and
    // left the rest in the stream for the next message to misread; sending
    // NOTHING left the peers blocked on a scalar that never came.  Neither
    // fails where it happens.
    //
    // A destructor, not a lambda at each exit: it covers every `return
    // false` AND every exception out of a bridge, an allocation, a kernel
    // submission or a wait -- paths that have no explicit exit to annotate.
    // A fixed buffer, so the failure path allocates nothing even after a
    // bad_alloc; clamped, so the count can never outrun it.
    struct Owed {
        Grimoire* g; bool active; bool context_only; int count; bool owed = true;
        ~Owed() {
            if (!active || !owed) return;
            owed = false;
            constexpr int kMax = 16;          // MMAX; pp_dflash cannot exceed it
            int32_t dead[kMax];
            const int n = context_only ? 1 : std::min(std::max(count, 0), kMax);
            if (n <= 0) return;
            for (int i = 0; i < n; ++i) dead[i] = -1;
            // Best effort, and never throw out of a destructor: if the
            // transport is already broken there is nothing left to say.
            fd_write_all(g->pp_prev_fd, dead, size_t(n) * sizeof(int32_t));
        }
    } reply{this, pp_enabled() && pp_rank == pp_world - 1 && pp_world > 1,
            context_only, pp_dflash - 1};
    auto refuse=[&]()->bool{ return false; };   // ~Owed does the notifying
    if(!dflash2.ok||position<0||position+M>max_seq){
        static bool once2=false;
        if(!once2){once2=true;std::fprintf(stderr,
            "  DFlash2: entry guard ok=%d pos=%d M=%d max_seq=%d\n",
            int(dflash2.ok),position,M,max_seq);}
        return refuse();}
    // DFlash2 needs its dynamic convolutions; without them the draft is the
    // wrong function and acceptance collapses silently.
    if(dflash2.v2&&(!dflash2.conv_delta||!dflash2.conv_scratch)){
        static bool once=false;
        if(!once){once=true;std::fprintf(stderr,
            "  DFlash2: conv buffers missing (taps %d groups %d delta %p scratch %p)\n",
            dflash2.conv_taps,dflash2.conv_groups,
            (void*)dflash2.conv_delta,(void*)dflash2.conv_scratch);}
        return refuse();}
    if(dflash2.v2&&cfg.is_muse)return refuse();
    const int H=dflash2.hidden,QH=dflash2.q_heads,KVH=dflash2.kv_heads;
    const int HD=dflash2.head_dim,QW=QH*HD,KVW=KVH*HD,I=dflash2.inter;
    const int MASK=dflash2.mask_token;
    const int NT=int(dflash2.target_layers.size());
    const float eps=dflash2.rms_eps;
    const float theta=dflash2.rope_theta;
    const auto fa2_paged=cfg.is_muse?load_xe2_dflash_paged_f16():nullptr;
    if(cfg.is_muse&&!fa2_paged)return refuse();
    const bool time_draft=std::getenv("GRIMOIRE_DFLASH_TIME")!=nullptr;
    static int time_draft_call=0;
    const int timed_call=time_draft_call++;
    std::map<std::string,double> draft_times;
    std::chrono::high_resolution_clock::time_point draft_prev;
    if(time_draft){
        q.wait_and_throw();
        draft_prev=std::chrono::high_resolution_clock::now();
    }
    auto draft_mark=[&](const char* stage){
        if(!time_draft)return;
        q.wait_and_throw();
        const auto now=std::chrono::high_resolution_clock::now();
        draft_times[stage]+=std::chrono::duration<double,std::milli>(
            now-draft_prev).count();
        draft_prev=now;
    };
    const bool trace=std::getenv("GRIMOIRE_DFLASH_TRACE")!=nullptr;
    auto checkpoint=[&](const char* stage){
        if(!trace)return;
        std::fprintf(stderr,"  dflash trace: %s ... ",stage);
        std::fflush(stderr);
        q.wait_and_throw();
        std::fprintf(stderr,"ok\n");
    };
    const char* dump_dir=std::getenv("GRIMOIRE_DFLASH_DUMP");
    // Two latches, not one.  generate_tokens() opens with a context-only
    // call, so a single "first call" latch was spent entirely on the
    // context ingest and the draft stages (the block embedding, the final
    // norm, the logits, the draft ids) were NEVER written -- the tensor
    // comparison the 2026-08-28 handoff asks for could only ever see its
    // first half.  Latch the two halves independently, and set each only
    // once the work it describes has actually happened.
    static bool dumped_ctx=false, dumped_draft=false;
    const bool want_dump=dump_dir&&*dump_dir;
    bool dumping=want_dump&&!dumped_ctx;
    auto dump_write=[&](const std::string& name,const std::vector<float>& h){
        std::string fn=std::string(dump_dir)+"/g_"+name+".f32";
        std::FILE* f=std::fopen(fn.c_str(),"wb");
        if(!f){std::fprintf(stderr,"  dflash dump: cannot open %s\n",fn.c_str());return;}
        std::fwrite(h.data(),sizeof(float),h.size(),f);
        std::fclose(f);
        std::fprintf(stderr,"  dflash dump: %s [%zu]\n",name.c_str(),h.size());
    };
    auto dump_f32=[&](const std::string& name,const float* p,size_t n){
        if(!dumping||!p||!n)return;
        std::vector<float> h(n);
        q.memcpy(h.data(),p,n*sizeof(float)).wait();
        dump_write(name,h);
    };
    auto dump_f16=[&](const std::string& name,const sycl::half* p,size_t n){
        if(!dumping||!p||!n)return;
        std::vector<sycl::half> hh(n);
        q.memcpy(hh.data(),p,n*sizeof(sycl::half)).wait();
        std::vector<float> h(n);
        for(size_t i=0;i<n;++i)h[i]=float(hh[i]);
        dump_write(name,h);
    };
    auto dense=load_xe2_dense_mxfp4_f32();
    auto mm_f16_raw=[&](const DevQuant& w,const float* x,int rows){
        auto od=load_onednn_f16();
        if(!od||!w.fp16||!dflash2.linear_in_f16||!dflash2.linear_out_f16)
            throw std::runtime_error("Muse DFlash FP16 linear unavailable");
        launch_f32_to_f16(q,x,dflash2.linear_in_f16,size_t(rows)*w.w.K,{});
        auto it=std::find_if(dflash_f16_plans.begin(),
            dflash_f16_plans.end(),[&](const OneDnnPlan& p){
                return p.m==rows&&p.n==w.w.N&&p.k==w.w.K;
            });
        if(it==dflash_f16_plans.end()){
            void* plan=od.create(&q,rows,w.w.N,w.w.K);
            if(!plan)throw std::runtime_error(
                "Muse DFlash FP16 linear plan creation failed");
            const size_t bytes=od.scratch_size(plan);
            void* scratch=bytes?sycl::malloc_device<uint8_t>(bytes,q):nullptr;
            if(bytes&&!scratch){
                od.destroy(plan);
                throw std::runtime_error(
                    "Muse DFlash FP16 linear scratch allocation failed");
            }
            dflash_f16_plans.push_back({rows,w.w.N,w.w.K,plan,scratch});
            it=dflash_f16_plans.end()-1;
        }
        od.execute(it->plan,dflash2.linear_in_f16,w.fp16,
            dflash2.linear_out_f16,it->scratch);
        return dflash2.linear_out_f16;
    };
    // Same predicate the batched prefill uses: a device with no matrix
    // hardware cannot run a joint_matrix kernel at all -- the runtime
    // throws on submission -- so the whole draft forward was unreachable
    // anywhere but a B70, which is why none of it had ever executed off
    // the card.  A B70 is a GPU, so on the card this is false and the
    // behaviour is exactly as before.  The row loop is the same math the
    // single-token decode path already runs.
    static const bool no_matrix = !device_can_matrix(q);
    auto mm=[&](const DevQuant& w,const float* x,float* y,int rows){
        if(cfg.is_muse&&w.fp16){
            const sycl::half* out=mm_f16_raw(w,x,rows);
            launch_f16_to_f32(q,out,y,size_t(rows)*w.w.N,{});
            return;
        }
        if(no_matrix){
            for(int r=0;r<rows;++r)
                gemv_any(w,x+int64_t(r)*w.w.K,y+int64_t(r)*w.w.N,{});
            return;
        }
        launch_f32_to_bf16(q,x,dflash2.bf,size_t(rows)*w.w.K);
        if(dense&&w.w.fmt==Fmt::MXFP4&&w.w.payload){
            dense(&q,dflash2.bf,w.w.payload,
                  static_cast<const unsigned char*>(w.w.scales),y,
                  rows,w.w.N,w.w.K);
        }else launch_gemm_xmx(q,w.w,dflash2.bf,y,rows);
    };
    auto fc_mm=[&](const float* x,float* y,int rows){
        if(cfg.is_muse){
            mm(dflash2.fc,x,y,rows);
        }else if(dflash2.fc_plan&&dflash2.fc_scratch){
            launch_f32_to_bf16(q,x,dflash2.bf,size_t(rows)*dflash2.fc.w.K);
            auto od=load_onednn_bf16();
            od.execute(dflash2.fc_plan,dflash2.bf,dflash2.fc.w.payload,y,
                       dflash2.fc_scratch);
        }else mm(dflash2.fc,x,y,rows);
    };
    auto norm=[&](float* h,const float* residual,const bf16_t* weight,
                  const sycl::half* weight_f16,float* out,int rows){
        if(cfg.is_muse)
            launch_rmsnorm_residual_f16w_batched(
                q,h,residual,weight_f16,out,rows,H,eps,{});
        else
            launch_rmsnorm_residual_batched(
                q,h,residual,nullptr,weight,out,rows,H,eps,nullptr,{},0.0f);
    };

    // Target context is projected once per newly accepted token. The query
    // block later overwrites speculative cache slots, so rejected suffixes do
    // not require a separate draft-cache rollback image.
    while(dflash2.context_pos<position){
        const int start=dflash2.context_pos;
        const int rows=std::min(std::max(M,dflash2.ctx_chunk),position-start);
        dump_f32("01_aux_"+std::to_string(start),
                 dflash2.target_aux+int64_t(start)*NT*H,size_t(rows)*NT*H);
        fc_mm(dflash2.target_aux+int64_t(start)*NT*H,dflash2.ctx,rows);
        norm(dflash2.ctx,nullptr,dflash2.hidden_norm,dflash2.hidden_norm_f16,
             dflash2.normed,rows);
        dump_f32("02_fc_"+std::to_string(start),dflash2.ctx,size_t(rows)*H);
        dump_f32("03_ctxnorm_"+std::to_string(start),dflash2.normed,size_t(rows)*H);
        if(cfg.is_muse){
            // Exact vLLM context path: one [H -> L*2*KV] projection, one
            // contiguous [row,L,2,KV] -> [2,L,row,KV] transform, grouped K
            // RMSNorm, one logical RoPE batch, then per-layer cache inserts.
            mm(dflash2.fused_context_kv,dflash2.normed,
               dflash2.context_kv_all,rows);
            dump_f32("04_ctxkv_"+std::to_string(start),dflash2.context_kv_all,
                     size_t(rows)*dflash2.fused_context_kv.w.N);
            launch_dflash_context_kv_f16w(q,dflash2.context_kv_all,
                dflash2.context_k_all_f16,dflash2.context_v_all_f16,
                dflash2.k_norm_all_f16,int(dflash2.layers.size()),rows,KVH,HD,
                start,theta,eps,{});
            dump_f16("05_ctxk_"+std::to_string(start),dflash2.context_k_all_f16,
                     dflash2.layers.size()*size_t(rows)*KVW);
            dump_f16("06_ctxv_"+std::to_string(start),dflash2.context_v_all_f16,
                     dflash2.layers.size()*size_t(rows)*KVW);
            for(size_t li=0;li<dflash2.layers.size();++li){
                auto& d=dflash2.layers[li];
                const size_t off=li*size_t(rows)*KVW;
                launch_kv_append_f16_paged(q,dflash2.context_k_all_f16+off,
                    dflash2.context_v_all_f16+off,d.k_cache_f16,d.v_cache_f16,
                    rows,start,KVH,HD,dflash2.block_size,{});
            }
        }else{
            // Stages 04-06 used to be dumped only on the Muse path, which
            // left the comparison harness blind on exactly the model it
            // was written for.  The per-layer projections here produce the
            // same tensors the reference's fused one does, so stack them
            // into the reference's layouts: 04 is all_kv_flat,
            // [rows][L,2,nkv,hd]; 05 and 06 are [L][rows][nkv*hd].  Debug
            // path, first draft call only.
            const size_t LN=dflash2.layers.size();
            std::vector<float> ctxkv, ctxk, ctxv, hk, hv;
            if(dumping){
                ctxkv.assign(size_t(rows)*LN*2*size_t(KVW),0.0f);
                ctxk.assign(LN*size_t(rows)*size_t(KVW),0.0f);
                ctxv.assign(LN*size_t(rows)*size_t(KVW),0.0f);
                hk.resize(size_t(rows)*size_t(KVW));
                hv.resize(size_t(rows)*size_t(KVW));
            }
            for(size_t li=0;li<LN;++li){
                auto& d=dflash2.layers[li];
                mm(d.k,dflash2.normed,dflash2.k,rows);
                mm(d.v,dflash2.normed,dflash2.v,rows);
                if(dumping){
                    // Pre-norm, pre-RoPE: this is what the reference's
                    // fused KV GEMM emits, before _normalize_context_k.
                    q.memcpy(hk.data(),dflash2.k,hk.size()*sizeof(float)).wait();
                    q.memcpy(hv.data(),dflash2.v,hv.size()*sizeof(float)).wait();
                    for(int r=0;r<rows;++r)
                        for(int c=0;c<KVW;++c){
                            const size_t base=((size_t(r)*LN+li)*2)*size_t(KVW);
                            ctxkv[base+size_t(c)]=hk[size_t(r)*KVW+c];
                            ctxkv[base+size_t(KVW)+size_t(c)]=hv[size_t(r)*KVW+c];
                        }
                }
                launch_qk_norm_rope_batched(q,dflash2.q,dflash2.k,
                    d.q_norm,d.k_norm,rows,QH,KVH,HD,start,theta,1.0f,eps,{},0.0f);
                if(dumping){
                    q.memcpy(ctxk.data()+li*size_t(rows)*KVW,dflash2.k,
                             size_t(rows)*KVW*sizeof(float)).wait();
                    q.memcpy(ctxv.data()+li*size_t(rows)*KVW,dflash2.v,
                             size_t(rows)*KVW*sizeof(float)).wait();
                }
                launch_kv_append_batched(q,dflash2.k,dflash2.v,
                    d.k_cache,d.v_cache,rows,start,KVH,HD,max_seq);
            }
            if(dumping){
                dump_write("04_ctxkv_"+std::to_string(start),ctxkv);
                dump_write("05_ctxk_"+std::to_string(start),ctxk);
                dump_write("06_ctxv_"+std::to_string(start),ctxv);
            }
        }
        checkpoint("context KV");
        dflash2.context_pos+=rows;
        if(dumping)dumped_ctx=true;
    }
    draft_mark("context ingest");
    // From here on the draft stages get their own first-call latch.
    dumping=want_dump&&!context_only&&!dumped_draft;
    if(context_only){
        reply.owed=false;      // argmax_token sends the scalar next
        if(time_draft){
            std::fprintf(stderr,
                "  DFlash context preparation call %d: %.3f ms\n",
                timed_call,draft_times["context ingest"]);
        }
        return true;
    }

    std::array<int32_t,MMAX> host_tokens{};
    host_tokens[0]=bonus_token;
    for(int i=1;i<M;++i)host_tokens[size_t(i)]=MASK;
    q.memcpy(dflash2.tokens,host_tokens.data(),size_t(M)*sizeof(int32_t));
    // shared_embed_f16 is populated for Muse ONLY. On a non-Muse target it is
    // a null device pointer, and feeding it to the embed kernel faults the
    // GPU (UR_RESULT_ERROR_DEVICE_LOST) rather than failing cleanly. The
    // non-Muse drafter shares the target's own bf16 embedding table.
    if(cfg.is_muse)
        launch_embed_f16_batched(q,dflash2.shared_embed_f16.fp16,dflash2.tokens,
                                 dflash2.resid,M,H);
    else if(!embed_rows(dflash2.draft_embed?dflash2.draft_embed:embed,
                        dflash2.tokens,dflash2.resid,M)) {
        std::fprintf(stderr,"  DFlash: block embed all-reduce failed\n");
        return refuse();
    }
    checkpoint("block embedding");
    draft_mark("embed + setup");
    dump_f32("07_blockembed",dflash2.resid,size_t(M)*H);

    if(cfg.is_muse){
        // The paged FA2 kernel reads the query batch layout from cu_q.
        // prefill_muse leaves it describing its own 64-token prefill, and the
        // draft never restored it, so the kernel bounded the key range by a
        // 64-row query length and never read the query block's own K/V: the
        // 16 draft rows attended to the 64 context keys only, never to the
        // bonus token or each other. Both copies source host stack values, so
        // they must complete before the value dies.
        const int32_t cuq[2]={0,M};
        const int32_t used_all=position+M;
        q.memcpy(dflash2.cu_q,cuq,sizeof(cuq)).wait();
        q.memcpy(dflash2.seqused_k,&used_all,sizeof(used_all)).wait();
    }
    // DFlash2 grouped dynamic convolution.  coefficient[tap] is the layer's
    // base kernel (per channel) plus a per-token, per-group delta produced by
    // kernel_projection; the sum is a causal FIR over positions WITHIN the
    // draft block, so tap t only contributes where (row % block) >= t.
    // `prepare` runs side 0 and hands its side-1 coefficients to `finish`,
    // which is why both sides read the same delta buffer.
    auto dyn_conv=[&](const float* x,const bf16_t* base,float* out,int side){
        const int HH=dflash2.hidden,T=dflash2.conv_taps,G=dflash2.conv_groups;
        const int GS=HH/G,BS=dflash2.conv_block;
        const float* delta=dflash2.conv_delta;
        q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<1>(size_t(M)*size_t(HH)),
                [=](sycl::id<1> id){
                    const int m=int(id[0]/HH),c=int(id[0]%HH),g=c/GS;
                    const int posb=m%BS;
                    float acc=0.0f;
                    for(int t=0;t<T;++t){
                        if(t>posb)break;
                        const float b=bf16_to_f32(
                            base[(size_t(side)*T+size_t(t))*HH+c]);
                        const float dv=delta[((size_t(m)*2+size_t(side))*T+
                                              size_t(t))*G+g];
                        acc+=(b+dv)*x[size_t(m-t)*HH+c];
                    }
                    out[size_t(m)*HH+c]=acc;
                });
        });
    };
    auto conv_back=[&](float* dst){
        q.memcpy(dst,dflash2.conv_scratch,size_t(M)*dflash2.hidden*sizeof(float));
    };
    for(size_t li=0;li<dflash2.layers.size();++li){
        auto& d=dflash2.layers[li];
        if(li==0)
            norm(dflash2.resid,nullptr,d.in_norm,d.in_norm_f16,dflash2.normed,M);
        else
            norm(dflash2.resid,dflash2.mlp,d.in_norm,d.in_norm_f16,
                 dflash2.normed,M);
        draft_mark("input norm");
        dump_f32("08_L"+std::to_string(li)+"_innorm",dflash2.normed,size_t(M)*H);
        if(dflash2.v2){
            mm(d.attn_conv_proj,dflash2.normed,dflash2.conv_delta,M);
            dyn_conv(dflash2.normed,d.attn_conv_base,dflash2.conv_scratch,0);
            conv_back(dflash2.normed);
        }
        if(cfg.is_muse){
            const sycl::half* qkv=mm_f16_raw(d.qkv,dflash2.normed,M);
            draft_mark("qkv f16 gemm");
            launch_qkv_norm_rope_f16w_fused(q,qkv,dflash2.q_f16,
                dflash2.k_f16,dflash2.v_f16,d.q_norm_f16,d.k_norm_f16,M,QH,KVH,HD,
                position,theta,eps,{});
            dump_f16("09_L"+std::to_string(li)+"_qkv",qkv,
                     size_t(M)*d.qkv.w.N);
            dump_f16("10_L"+std::to_string(li)+"_q",dflash2.q_f16,size_t(M)*QW);
            dump_f16("11_L"+std::to_string(li)+"_k",dflash2.k_f16,size_t(M)*KVW);
            dump_f16("12_L"+std::to_string(li)+"_v",dflash2.v_f16,size_t(M)*KVW);
        }else{
            mm(d.q,dflash2.normed,dflash2.q,M);
            mm(d.k,dflash2.normed,dflash2.k,M);
            mm(d.v,dflash2.normed,dflash2.v,M);
        }
        checkpoint("QKV projections");
        draft_mark("qkv norm + rope");
        if(cfg.is_muse){
            launch_kv_append_f16_paged(q,dflash2.k_f16,dflash2.v_f16,
                d.k_cache_f16,d.v_cache_f16,M,position,KVH,HD,
                dflash2.block_size,{});
            const int32_t used=position+M;
            const int window=d.window?d.window-1:-1;
            if(dumping&&li==0){
                // Read the KV cache back through the same linear indexing the
                // append uses, for the 64 context slots and the 16 query slots.
                dump_f16("18_cache_ctx_k",d.k_cache_f16,size_t(position)*KVW);
                dump_f16("19_cache_qry_k",
                         d.k_cache_f16+size_t(position)*KVW,size_t(M)*KVW);
                dump_f16("20_cache_qry_v",
                         d.v_cache_f16+size_t(position)*KVW,size_t(M)*KVW);
            }
            if(dumping&&li==0){
                // Read back exactly what the paged kernel will see. The
                // memcpy above sources a stack local asynchronously, and
                // cu_q is shared with prefill_muse, so neither value can be
                // assumed from the code alone.
                int32_t sk=-1,cq[2]={-1,-1},ck2[2]={-1,-1};
                q.memcpy(&sk,dflash2.seqused_k,sizeof(sk)).wait();
                q.memcpy(cq,dflash2.cu_q,sizeof(cq)).wait();
                q.memcpy(ck2,dflash2.cu_k,sizeof(ck2)).wait();
                std::fprintf(stderr,
                    "  dflash probe: position=%d M=%d used=%d | device "
                    "seqused_k=%d cu_q=[%d,%d] cu_k=[%d,%d] | block_size=%d "
                    "num_blocks=%d window=%d causal=%d\n",
                    position,M,used,sk,cq[0],cq[1],ck2[0],ck2[1],
                    dflash2.block_size,dflash2.num_blocks,window,int(d.causal));
                std::fflush(stderr);
            }
            const int rc=fa2_paged(&q,dflash2.q_f16,d.k_cache_f16,
                d.v_cache_f16,dflash2.attn_f16,M,used,QH,KVH,HD,
                dflash2.block_size,dflash2.num_blocks,dflash2.block_table,
                dflash2.cu_q,dflash2.cu_k,dflash2.seqused_k,
                1.0f/std::sqrt(float(HD)),window,window,false);
            if(rc)return refuse();
            draft_mark("kv append + attention");
            launch_f16_to_f32(q,dflash2.attn_f16,dflash2.attn,
                size_t(M)*QW,{});
        }else{
            launch_qk_norm_rope_batched(q,dflash2.q,dflash2.k,d.q_norm,d.k_norm,
                                        M,QH,KVH,HD,position,theta,1.0f,eps,{},0.0f);
            launch_kv_append_batched(q,dflash2.k,dflash2.v,d.k_cache,d.v_cache,
                                     M,position,KVH,HD,max_seq);
            launch_dflash2_block_attention(q,dflash2.q,d.k_cache,d.v_cache,
                dflash2.attn,M,position,QH,KVH,HD,max_seq,
                d.window,d.causal,
                1.0f/std::sqrt(float(HD)));
        }
        checkpoint("block attention");
        draft_mark("attention convert");
        dump_f32("13_L"+std::to_string(li)+"_attn",dflash2.attn,size_t(M)*QW);
        mm(d.o,dflash2.attn,dflash2.proj,M);
        draft_mark("o proj");
        if(dflash2.v2){
            dyn_conv(dflash2.proj,d.attn_conv_base,dflash2.conv_scratch,1);
            conv_back(dflash2.proj);
        }
        dump_f32("14_L"+std::to_string(li)+"_o",dflash2.proj,size_t(M)*H);
        norm(dflash2.resid,dflash2.proj,d.post_norm,d.post_norm_f16,
             dflash2.normed,M);
        draft_mark("post-attn norm");
        if(dflash2.v2){
            mm(d.mlp_conv_proj,dflash2.normed,dflash2.conv_delta,M);
            dyn_conv(dflash2.normed,d.mlp_conv_base,dflash2.conv_scratch,0);
            conv_back(dflash2.normed);
        }
        mm(d.gate_up,dflash2.normed,dflash2.gate_up,M);
        draft_mark("gate_up proj");
        if(cfg.is_muse)
            launch_swiglu_f16_batched(q,dflash2.gate_up,dflash2.h,M,I,{});
        else launch_swiglu_batched(q,dflash2.gate_up,dflash2.h,M,I);
        draft_mark("swiglu");
        mm(d.down,dflash2.h,dflash2.mlp,M);
        draft_mark("down proj");
        if(dflash2.v2){
            dyn_conv(dflash2.mlp,d.mlp_conv_base,dflash2.conv_scratch,1);
            conv_back(dflash2.mlp);
        }
        checkpoint("MLP");
        dump_f32("15_L"+std::to_string(li)+"_mlp",dflash2.mlp,size_t(M)*H);
    }
    norm(dflash2.resid,dflash2.mlp,dflash2.norm,dflash2.norm_f16,
         dflash2.normed,M);
    draft_mark("final norm");

    auto w4=load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m16");
    dump_f32("16_finalnorm",dflash2.normed,size_t(M)*H);
    if(dflash2.draft_head_rows){
        auto head_w4=load_xe2_dense_w4a8(
            "grimoire_xe2_dense_w4a8_f32_m8g64");
        if(!head_w4)return refuse();
        launch_quantize_rows_int8(q,dflash2.normed+H,dflash2.a8,dflash2.a8s,
                                  M-1,H,{});
        head_w4(&q,dflash2.a8,dflash2.draft_head_i4,
                dflash2.draft_head_i4s,dflash2.a8s,dflash2.logits,
                M-1,dflash2.draft_head_rows,H);
    }else if(dflash2.draft_lm_head.w.N){
        // The drafter's OWN head.  Row 0 is the anchor, which is a verified
        // token and never sampled, so only rows 1..M-1 are projected.
        mm(dflash2.draft_lm_head,dflash2.normed+H,dflash2.logits,M-1);
    }else if(cfg.is_muse&&dflash2.draft_lm_head_i4.has_i4()&&w4){
        const auto& head=dflash2.draft_lm_head_i4;
        launch_quantize_rows_int8(q,dflash2.normed+H,dflash2.a8,dflash2.a8s,
                                  M-1,H,{});
        w4(&q,dflash2.a8,head.i4,head.i4s,dflash2.a8s,dflash2.logits,
           M-1,head.w.N,H);
    }else if(cfg.is_muse){
        mm(dflash2.shared_lm_head_f16,dflash2.normed+H,
           dflash2.logits,M-1);
    }else if(tp_enabled()&&lm_head.tp_sharded()){
        // Under TP the target's lm_head is row-sharded over the vocabulary,
        // so lm_head.w.N is THIS rank's slice.  Both wide paths below write
        // a full-width logits row from it and would silently emit one rank's
        // fraction of the vocabulary -- every rank then drafts a different
        // token, the ranks' KV caches stop describing the same sequence, and
        // nothing reports an error.  gemv_any is the one path that knows to
        // compute the local rows and all-gather them, so take it.  It costs
        // M-1 <= 15 row projections per step; correctness first, and the
        // batched sharded head is a measurement to make on the card.
        for(int r=1;r<M;++r)
            gemv_any(lm_head,dflash2.normed+int64_t(r)*H,
                     dflash2.logits+int64_t(r-1)*cfg.vocab,{});
    }else if(lm_head.has_i4()&&w4){
        launch_quantize_rows_int8(q,dflash2.normed+H,dflash2.a8,dflash2.a8s,
                                  M-1,H,{});
        w4(&q,dflash2.a8,lm_head.i4,lm_head.i4s,dflash2.a8s,dflash2.logits,
           M-1,lm_head.w.N,lm_head.w.K);
    }else if(lm_head.w.fmt==Fmt::MXFP4&&lm_head.w.payload){
        // Project all draft rows together.  The old per-row GEMV streamed the
        // full vocabulary matrix fifteen times per speculative step.
        mm(lm_head,dflash2.normed+H,dflash2.logits,M-1);
    }else for(int r=1;r<M;++r)
        gemv_any(lm_head,dflash2.normed+int64_t(r)*H,
                 dflash2.logits+int64_t(r-1)*cfg.vocab,{});
    draft_mark("lm head");
    const int draft_vocab=dflash2.draft_head_rows?dflash2.draft_head_rows:
        (dflash2.draft_vocab_rows?dflash2.draft_vocab_rows:cfg.vocab);
    const int draft_stride=dflash2.draft_logits_stride?
        dflash2.draft_logits_stride:draft_vocab;
    // Either head can emit ids in its own reduced vocabulary: the NInfer
    // extract carries absolute target ids, the checkpoint's d2t a delta
    // already folded into absolute ids at load.  Everything downstream --
    // the selector codebooks, the verifier, the caller -- speaks target
    // ids, so map exactly once, here.
    const int32_t* vocab_map=dflash2.draft_head_rows?
        dflash2.draft_head_token_ids:dflash2.draft_vocab_map;
    // ---- candidate selection -------------------------------------
    // DFlash2 does NOT take a bare argmax per position.  It keeps the
    // top-k candidates at each draft position and scores the EDGE from
    // the previous choice to each candidate:
    //
    //   score[l,p,c] = unary[l,c]
    //                + sum_r pred[pred_id[l,p],r] * hidden[l,r] * succ[cand[l,c],r]
    //
    // with pred_id[0,*] = the verified anchor, then walks greedily from
    // the anchor, each step conditioned on the previous pick.  That is
    // vLLM's _score_edges plus its _selector_walk_kernel; independent
    // argmax is a different algorithm that ignores the codebooks
    // entirely, which is why the weights loaded but nothing used them.
    if(dflash2.selector_ok){
        const int steps=M-1, K=16, TK=dflash2.selector_top_k;
        const int rank=dflash2.selector_rank;
        // topk16_rows assumes a packed row; the draft logits can be padded
        // (draft_logits_stride != draft_vocab), so feed it one row at a time.
        for(int r=0;r<steps;++r)
            launch_topk16_rows(q,dflash2.logits+int64_t(r)*draft_stride,1,
                               draft_vocab,dflash2.sel_ids+int64_t(r)*K,
                               dflash2.sel_unary+int64_t(r)*K,{});
        if(vocab_map){
            // candidates are rows of the REDUCED head; the codebooks are
            // indexed by real vocabulary id, so map before scoring.
            const int32_t* map=vocab_map;
            int32_t* ids=dflash2.sel_ids;
            const int64_t n=int64_t(steps)*K;
            q.parallel_for(sycl::range<1>(size_t(n)),[=](sycl::id<1> i){
                ids[i[0]]=map[ids[i[0]]];
            });
        }
        // hidden_projection(hidden_states) -> [steps, rank]
        mm(dflash2.selector_hidden,dflash2.normed+H,dflash2.sel_hidden,steps);
        launch_dflash2_selector_edges(q,dflash2.predecessor,dflash2.successor,
            dflash2.sel_ids,dflash2.sel_unary,dflash2.sel_hidden,
            int32_t(bonus_token),dflash2.sel_scores,steps,TK,rank,{});
        launch_dflash2_path_walk(q,dflash2.sel_scores,dflash2.sel_ids,
            dflash2.draft_ids,steps,TK,{});
    }else for(int r=0;r<M-1;++r){
        launch_argmax(q,dflash2.logits+int64_t(r)*draft_stride,draft_vocab,
                      s.d_tok,s.d_val,{});
        if(vocab_map){
            const int32_t* map=vocab_map;
            int32_t* src=s.d_tok;
            int32_t* dst=dflash2.draft_ids+r;
            q.single_task([=](){*dst=map[*src];});
        }else q.memcpy(dflash2.draft_ids+r,s.d_tok,sizeof(int32_t));
    }
    dump_f32("17_logits",dflash2.logits,size_t(M-1)*draft_stride);
    draft_tokens.resize(M-1);
    q.memcpy(draft_tokens.data(),dflash2.draft_ids,
             size_t(M-1)*sizeof(int32_t)).wait();
    draft_mark("argmax + readback");
    if(dumping){
        std::fprintf(stderr,"  dflash dump: draft_ids");
        for(int r=0;r<M-1;++r)
            std::fprintf(stderr," %d",draft_tokens[size_t(r)]);
        std::fprintf(stderr,"\n");
    }
    checkpoint("draft logits");
    // Push the block backward so every earlier stage speculates on the
    // SAME candidates.  Sizes already agree: pp_dflash carries this M.
    if(pp_enabled()){
        reply.owed=false;         // the real block; never a sentinel after it
        if(!pp_sync_tokens(draft_tokens)){
            std::fprintf(stderr,"PP rank %d: draft block send failed\n",pp_rank);
            return false;         // transport is already broken -- do not resend
        }
    }
    if(dumping)dumped_draft=true;
    if(time_draft){
        double total=0.0;
        for(const auto& kv:draft_times)total+=kv.second;
        std::fprintf(stderr,
            "  DFlash native timing call %d, %d layers, M=%d:\n",
            timed_call,int(dflash2.layers.size()),M);
        for(const auto& kv:draft_times)
            std::fprintf(stderr,"    %-24s %7.3f ms  (%5.1f%%)\n",
                kv.first.c_str(),kv.second,total?100.0*kv.second/total:0.0);
        std::fprintf(stderr,"    %-24s %7.3f ms\n","TOTAL",total);
    }
    return true;
}

// Muse Glimmer is a dense Gemma-sandwich transformer.  Its verifier must keep
// the exact Muse norm placement/epsilons while evaluating all speculative rows
// together; routing it through the Qwen/DeltaNet prefill allocates irrelevant
// recurrent buffers and computes the wrong residual graph.
bool Grimoire::prefill_muse(const std::vector<int32_t>& tokens,
                            std::vector<int32_t>* next_tokens,
                            bool allow_exact_restore) {
    const int M=int(tokens.size());
    if(M<=0||pos+M>max_seq)return false;
    const int start_pos=pos;
    // allow_exact_restore threaded through from prefill() (external audit
    // follow-up, 2026-09-21): this function has its own restore_prefix()
    // shortcut, bypassed by prefill()'s dispatch BEFORE its own guarded
    // one is ever reached, so the F3 fix there did not cover it. Latent
    // today only because batch_unsupported_reason() always refuses Muse,
    // which keeps a caller with its own slot-ownership model from ever
    // reaching this function concurrently -- must not go stale the day
    // Muse gets a batched path.
    if(!next_tokens&&start_pos==0&&allow_exact_restore&&restore_prefix(tokens))return true;
    const int H=cfg.hidden,HD=cfg.head_dim,QH=cfg.n_heads,KVH=cfg.n_kv_heads;
    const int QW=QH*HD,KVW=KVH*HD,I=cfg.dense_inter;
    int W=std::max({H,QW,KVW,2*I});
    for(const auto&d:L){
        const DevQuant* ws[]={&d.q_proj,&d.k_proj,&d.v_proj,&d.o_proj,
                              &d.o_gate,&d.sh_gu,&d.sh_down};
        for(const auto* w:ws){W=std::max(W,w->w.N);W=std::max(W,w->w.K);}
    }
    // The verifier also projects the shared lm_head through xb/yb. lm_head is
    // not in the layer list above, so W must account for the vocabulary or the
    // staging buffers overflow by ~5x on every verify batch.
    if(next_tokens)W=std::max(W,cfg.vocab);
    const bool reuse=next_tokens&&M<=kSpecBatch&&dflash2.ok&&
        dflash2.verify_logits&&dflash2.verify_bf&&dflash2.verify_bf_out&&
        dflash2.verify_a8;
    // Activations are carried in FP16, matching Fusion's float16 dataflow. The
    // f32 kernels this replaces already rounded every intermediate through
    // sycl::half before storing, so this is numerically identical and removes
    // the fp32<->fp16 conversion around every oneDNN GEMM. The dflash2 reuse
    // buffers are f32 allocations, i.e. twice the bytes an fp16 view needs, so
    // reinterpreting them here is always in-bounds.
    auto df=[&](size_t n){return sycl::malloc_device<float>(n,q);};
    auto dh=[&](size_t n){return sycl::malloc_device<sycl::half>(n,q);};
    auto asH=[](float* p){return reinterpret_cast<sycl::half*>(p);};
    sycl::half* hidden=reuse?asH(dflash2.resid):dh(size_t(M)*H);
    sycl::half* normed=reuse?asH(dflash2.normed):dh(size_t(M)*H);
    sycl::half* tmp=reuse?asH(dflash2.ctx):dh(size_t(M)*H);
    float* qv=reuse?dflash2.q:df(size_t(M)*QW);
    float* kv=reuse?dflash2.k:df(size_t(M)*KVW);
    float* vv=reuse?dflash2.v:df(size_t(M)*KVW);
    sycl::half* attn=reuse?asH(dflash2.attn):dh(size_t(M)*QW);
    sycl::half* gate=reuse?asH(dflash2.proj):dh(size_t(M)*QW);
    sycl::half* proj=reuse?asH(dflash2.mlp):dh(size_t(M)*H);
    sycl::half* ff=reuse?asH(dflash2.gate_up):dh(size_t(M)*2*I);
    // Scratch for the DFlash tap, which still writes f32 target_aux.
    float* tapbuf=(dflash2.ok&&!reuse)?df(size_t(M)*H):nullptr;
    float* lastrow=df(size_t(H));
    float* batch_logits=next_tokens?(reuse?dflash2.verify_logits:
        df(size_t(M)*cfg.vocab)):nullptr;
    sycl_bf16* xb=reuse?dflash2.verify_bf:
        sycl::malloc_device<sycl_bf16>(size_t(M)*W,q);
    sycl_bf16* yb=reuse?dflash2.verify_bf_out:
        sycl::malloc_device<sycl_bf16>(size_t(M)*W,q);
    int8_t* a8=reuse?dflash2.verify_a8:
        sycl::malloc_device<int8_t>(size_t(M)*W,q);
    float* a8s=reuse?dflash2.verify_a8s:sycl::malloc_device<float>(M,q);
    int32_t* dtok=reuse?dflash2.tokens:sycl::malloc_device<int32_t>(M,q);
    int32_t* outtok=next_tokens?(reuse?dflash2.verify_ids:
        sycl::malloc_device<int32_t>(M,q)):nullptr;
    sycl::half* qh=reuse?dflash2.q_f16:
        sycl::malloc_device<sycl::half>(size_t(M)*QW,q);
    sycl::half* kh=reuse?dflash2.k_f16:
        sycl::malloc_device<sycl::half>(size_t(M)*KVW,q);
    sycl::half* vh=reuse?dflash2.v_f16:
        sycl::malloc_device<sycl::half>(size_t(M)*KVW,q);
    sycl::half* oh=reuse?dflash2.attn_f16:
        sycl::malloc_device<sycl::half>(size_t(M)*QW,q);
    std::vector<void*> mem={(void*)hidden,(void*)normed,(void*)tmp,(void*)qv,
        (void*)kv,(void*)vv,(void*)attn,(void*)gate,(void*)proj,(void*)ff,
        (void*)tapbuf,(void*)lastrow,
        (void*)batch_logits,(void*)xb,(void*)yb,(void*)a8,(void*)a8s,
        (void*)dtok,(void*)outtok,(void*)qh,(void*)kh,(void*)vh,(void*)oh};
    auto cleanup=[&](){if(!reuse)for(void* p:mem)if(p)sycl::free(p,q);};
    if(!hidden||!normed||!tmp||!qv||!kv||!vv||!attn||!gate||!proj||!ff||
       !xb||!yb||!a8||!a8s||!dtok||!qh||!kh||!vh||!oh||
       (next_tokens&&(!batch_logits||!outtok))){
        cleanup();return false;
    }

    // Host-side region timing for the Muse prefill path, same drain-and-clock
    // approach as GRIMOIRE_TIME_LAYER in the generic prefill(): retains no
    // sycl::events and needs no profiling-enabled queue, so it does not hit
    // the M>=64 host-spin documented in the generic path above.
    //   GRIMOIRE_MUSE_TIME_LAYER=<n>   time only layer n
    //   GRIMOIRE_MUSE_TIME_LAYER=all   time every layer, print per-region sums
    int mt_layer=-1; bool mt_all=false;
    if(const char* v=std::getenv("GRIMOIRE_MUSE_TIME_LAYER")){
        if(std::strcmp(v,"all")==0){mt_all=true;} else {mt_layer=std::atoi(v);}
    }
    const bool mt_on=mt_all||mt_layer>=0;
    std::chrono::high_resolution_clock::time_point mt_prev;
    std::map<std::string,double> mt_sums;
    std::vector<std::string> mt_order;
    int mt_cur=-1;
    auto mt_mark=[&](const char* region){
        if(!mt_on||!(mt_all||mt_cur==mt_layer))return;
        q.wait();
        const auto now=std::chrono::high_resolution_clock::now();
        const double ms=std::chrono::duration<double,std::milli>(now-mt_prev).count();
        if(mt_sums.find(region)==mt_sums.end())mt_order.push_back(region);
        mt_sums[region]+=ms;
        if(!mt_all)std::fprintf(stderr,"      [layer %d] %-24s %9.3f ms\n",
                                 mt_cur,region,ms);
        mt_prev=now;
    };

    auto dense=load_xe2_dense_mxfp4_f32();
    OneDnnW4Api od=load_onednn_w4();
    if(od&&!muse_od_zp){
        muse_od_zp=sycl::malloc_device<int8_t>(1,q);
        if(muse_od_zp){const int8_t z=8;q.memcpy(muse_od_zp,&z,1);}
    }
    const bool w4n128=std::getenv("GRIMOIRE_W4A8_N128")!=nullptr;
    auto w4=load_xe2_dense_w4a8(M<=kSpecBatch
        ?(w4n128?"grimoire_xe2_dense_w4a8_f32_m16n128"
                 :"grimoire_xe2_dense_w4a8_f32_m16")
        :"grimoire_xe2_dense_w4a8_f32");
    const bool exact_gemv=std::getenv("GRIMOIRE_MUSE_PREFILL_GEMV")!=nullptr;
    const bool exact_bf16=std::getenv("GRIMOIRE_MUSE_PREFILL_EXACT_BF16")!=nullptr;
    const auto fa2_paged=load_xe2_dflash_paged_f16();
    const bool exact_attn=fa2_paged&&dflash2.block_table&&dflash2.cu_q&&
        dflash2.cu_k&&dflash2.seqused_k;
    if(!exact_attn){cleanup();return false;}
    const int32_t cuq[2]={0,M};
    q.memcpy(dflash2.cu_q,cuq,sizeof(cuq)).wait();
    auto mm_w4_f16=[&](const DevQuant& w,const sycl::half* x,
                       sycl::half* dst=nullptr)->sycl::half*{
        if(!od||!muse_od_zp||!w.od_w4||!w.payload||!w.od_scales_fp16)
            throw std::runtime_error("Muse target W4A16 parity path unavailable");
        auto* yh=dst?dst:reinterpret_cast<sycl::half*>(yb);
        const sycl::half* xh=x;   // activations are already FP16
        auto it=std::find_if(muse_od_plans.begin(),muse_od_plans.end(),
            [&](const OneDnnPlan& p){
                return p.m==M&&p.n==w.w.N&&p.k==w.w.K;
            });
        if(it==muse_od_plans.end()){
            void* plan=od.create(&q,M,w.w.N,w.w.K,kInt4Group,0);
            if(!plan)throw std::runtime_error(
                "Muse target W4A16 plan creation failed");
            const size_t bytes=od.scratch_size(plan);
            void* scratch=bytes?sycl::malloc_device<uint8_t>(bytes,q):nullptr;
            if(bytes&&!scratch){od.destroy(plan);throw std::runtime_error(
                "Muse target W4A16 scratch allocation failed");}
            muse_od_plans.push_back({M,w.w.N,w.w.K,plan,scratch});
            it=muse_od_plans.end()-1;
        }
        mt_mark("  w4 convert-in");
        od.execute(it->plan,xh,w.payload,w.od_scales_fp16,muse_od_zp,yh,
                   it->scratch);
        mt_mark("  w4 gemm");
        return yh;
    };
    auto mm_f16_raw=[&](const DevQuant& w,const sycl::half* x)->sycl::half*{
        OneDnnF16Api f16=load_onednn_f16();
        if(!f16||!w.fp16)throw std::runtime_error(
            "Muse FP16 linear parity path unavailable");
        auto* yh=reinterpret_cast<sycl::half*>(yb);
        const sycl::half* xh=x;   // activations are already FP16
        auto it=std::find_if(dflash_f16_plans.begin(),dflash_f16_plans.end(),
            [&](const OneDnnPlan& p){return p.m==M&&p.n==w.w.N&&p.k==w.w.K;});
        if(it==dflash_f16_plans.end()){
            void* plan=f16.create(&q,M,w.w.N,w.w.K);
            if(!plan)throw std::runtime_error("Muse FP16 plan creation failed");
            const size_t bytes=f16.scratch_size(plan);
            void* scratch=bytes?sycl::malloc_device<uint8_t>(bytes,q):nullptr;
            if(bytes&&!scratch){f16.destroy(plan);throw std::runtime_error(
                "Muse FP16 scratch allocation failed");}
            dflash_f16_plans.push_back({M,w.w.N,w.w.K,plan,scratch});
            it=dflash_f16_plans.end()-1;
        }
        f16.execute(it->plan,xh,w.fp16,yh,it->scratch);
        return yh;
    };
    // FP16 in, FP16 out. The previous version converted the activation to
    // half on the way in and the result back to f32 on the way out; those two
    // conversions cost 546 ms of a 2198 ms Muse prefill (25%) and are pure
    // overhead now that activations are carried as half throughout.
    auto mm=[&](const DevQuant& w,const sycl::half* x,sycl::half* y){
        if(w.fp16){
            sycl::half* yh=mm_f16_raw(w,x);
            if(yh!=y)q.memcpy(y,yh,size_t(M)*w.w.N*sizeof(sycl::half));
            return;
        }
        // Match vLLM XPU's Muse compressed-INT4 dispatch exactly: FP16
        // activations/scales through oneDNN W4A16, with plans cached by shape.
        if(od&&muse_od_zp&&w.od_w4&&w.payload&&w.od_scales_fp16){
            // Write the GEMM result straight into the destination; the shared
            // yb scratch would cost a full extra pass over the output (293 MB
            // per layer for gate_up alone).
            mm_w4_f16(w,x,y);
            return;
        }
        throw std::runtime_error(
            "Muse prefill FP16 path requires the oneDNN W4A16 or FP16 weights");
    };
    auto mm_w4_prequant=[&](const DevQuant& w,float* y){
        w4(&q,a8,w.i4,w.i4s,a8s,y,M,w.w.N,w.w.K);
    };

    q.memcpy(dtok,tokens.data(),size_t(M)*sizeof(int32_t));
    launch_embed_f16_h(q,dflash2.shared_embed_f16.fp16,dtok,hidden,M,H,{});
    launch_rmsnorm_residual_f16w_h(q,hidden,nullptr,muse_zero_f16,normed,
                                   M,H,cfg.rms_eps,{},1.0f);
    std::swap(hidden,normed);
    // launch_qkv_norm_rope_f16_fused already multiplies Q by
    // query_prescale.  Fusion's Attention then applies only 1/sqrt(HD).
    // Including query_prescale here applied 3.87 twice.
    const float sm_scale=1.0f/std::sqrt(float(HD));
    // seqused_k is start_pos+M for every layer of this call -- it does not
    // vary with li. Writing it inside the loop forced a blocking host
    // round-trip and full queue drain on every one of the 52 layers, which
    // serializes what should be back-to-back GPU submissions. Hoisted to a
    // single write before the loop.
    const int32_t used=start_pos+M;
    q.memcpy(dflash2.seqused_k,&used,sizeof(used)).wait();

    for(int li=0;li<cfg.n_layers;++li){
        mt_cur=li;
        if(mt_on&&(mt_all||li==mt_layer))mt_prev=std::chrono::high_resolution_clock::now();
        LayerDev& d=L[size_t(li)];
        if(dflash2.target_aux){
            for(size_t tap=0;tap<dflash2.target_layers.size();++tap){
                // vLLM converts DFlash target IDs with i+1, producing Muse
                // auxiliary layers (2,14,26,38,50).
                if(dflash2.target_layers[tap]+1==li){
                    launch_dflash_store_tap_h(q,hidden,dflash2.target_aux,M,H,
                        int(dflash2.target_layers.size()),start_pos,int(tap),{});
                    break;
                }
            }
        }
        launch_rmsnorm_residual_f16w_h(q,hidden,nullptr,d.in_norm_f16,normed,
                                       M,H,cfg.rms_eps,{},1.0f);
        sycl::half* qkvh=mm_w4_f16(d.qkv_proj,normed);
        mt_mark("qkv proj");
        launch_qkv_norm_rope_f16_fused(q,qkvh,qh,kh,vh,muse_zero,muse_zero,
            M,QH,KVH,HD,start_pos,cfg.rope_theta,cfg.rms_eps,{},
            d.muse_sliding,cfg.query_prescale,1.0f);
        mt_mark("qkv norm+rope");
        mm(d.o_gate,normed,gate);
        mt_mark("o_gate proj");
        constexpr int target_block_size=64;
        const int target_num_blocks=(max_seq+target_block_size-1)/target_block_size;
        launch_kv_append_f16_paged(q,kh,vh,d.k_cache_f16,d.v_cache_f16,M,
            start_pos,KVH,HD,target_block_size,{});
        mt_mark("kv append");
        // FA2 counts window_left as keys STRICTLY LEFT of the query, so a
        // window of W is W-1 here plus the query itself.  This was
        // hardcoded 2047 (i.e. W = 2048) and so was wrong for any
        // checkpoint whose config says otherwise -- silently, and only
        // past the window.  cfg.sliding_window is now parsed for Muse.
        const int msw=cfg.sliding_window>0?cfg.sliding_window:2048;
        const int window_left=d.muse_sliding?msw-1:-1;
        const int window_right=d.muse_sliding?0:-1;
        const int rc=fa2_paged(&q,qh,d.k_cache_f16,d.v_cache_f16,oh,M,used,
            QH,KVH,HD,target_block_size,target_num_blocks,
            dflash2.block_table,dflash2.cu_q,dflash2.cu_k,
            dflash2.seqused_k,sm_scale,window_left,window_right,true);
        if(rc){cleanup();return false;}
        mt_mark("attention");
        q.memcpy(attn,oh,size_t(M)*QW*sizeof(sycl::half));
        launch_gate_sigmoid_mul_h(q,attn,gate,M,QW,{});
        mm(d.o_proj,attn,proj);
        mt_mark("o_proj");
        if(std::getenv("GRIMOIRE_MUSE_UNFUSED_NORMS")){
            launch_rmsnorm_residual_f16w_h(q,proj,nullptr,d.post_norm_f16,tmp,
                                           M,H,cfg.post_norm_eps,{},1.0f);
            launch_add_f16_round_h(q,hidden,tmp,M*H,{});
            launch_rmsnorm_residual_f16w_h(q,hidden,nullptr,d.pre_ff_norm_f16,
                                           normed,M,H,cfg.rms_eps,{},1.0f);
        }else{
            launch_muse_post_attn_pre_ff_h(q,hidden,proj,d.post_norm_f16,
                d.pre_ff_norm_f16,normed,M,H,cfg.post_norm_eps,cfg.rms_eps,{});
        }
        mt_mark("post-attn+pre-ff norms");

        mm(d.sh_gu,normed,ff);
        mt_mark("FFN gate_up");
        launch_swiglu_h(q,ff,ff,M,I,{});
        mt_mark("FFN swiglu");
        mm(d.sh_down,ff,proj);
        mt_mark("FFN down");
        launch_rmsnorm_residual_f16w_h(q,proj,nullptr,d.post_ff_norm_f16,tmp,
                                       M,H,cfg.post_norm_eps,{},1.0f);
        launch_add_f16_round_h(q,hidden,tmp,M*H,{});
        mt_mark("post-ff norm+add");
    }
    if(mt_all){
        double tot=0;for(auto&kv:mt_sums)tot+=kv.second;
        std::fprintf(stderr,"  GRIMOIRE_MUSE_TIME_LAYER=all over %d layers, %d tokens:\n",cfg.n_layers,M);
        for(const auto&name:mt_order)
            std::fprintf(stderr,"    %-24s %10.2f ms  (%5.1f%%)\n",
                         name.c_str(),mt_sums[name],100.0*mt_sums[name]/tot);
        std::fprintf(stderr,"    %-24s %10.2f ms\n","TOTAL",tot);
    }

    launch_rmsnorm_residual_f16w_h(q,hidden,nullptr,fnorm_f16,normed,M,H,
                                   cfg.rms_eps,{},0.0f);
    // The decode path and the argmax kernel still consume f32, so convert only
    // at these boundaries rather than throughout the layer stack.
    if(next_tokens){
        sycl::half* lg=mm_f16_raw(dflash2.shared_lm_head_f16,normed);
        launch_f16_to_f32(q,lg,batch_logits,size_t(M)*cfg.vocab,{});
        for(int r=0;r<M;++r){
            launch_argmax(q,batch_logits+int64_t(r)*cfg.vocab,cfg.vocab,
                          s.d_tok,s.d_val,{});
            q.memcpy(outtok+r,s.d_tok,sizeof(int32_t));
        }
    }else{
        launch_f16_to_f32(q,normed+int64_t(M-1)*H,lastrow,size_t(H),{});
        gemv_any(lm_head,lastrow,s.logits,{});
    }
    if(next_tokens&&M<=kSpecBatch&&spec_hidden_steps){
        launch_f16_to_f32(q,hidden,spec_hidden_steps,size_t(M)*H,{});
        spec_hidden_valid=true;
    }
    launch_f16_to_f32(q,hidden+int64_t(M-1)*H,s.h,size_t(H),{});
    pos+=M;
    set_cursor(pos);
    if(next_tokens){
        next_tokens->resize(M);
        q.memcpy(next_tokens->data(),outtok,size_t(M)*sizeof(int32_t));
    }
    q.wait_and_throw();
    cleanup();
    if(!next_tokens&&start_pos==0)save_prefix(tokens);
    return true;
}

// INVARIANT, relied on by the speculative verify fallback in
// generation.hpp: every `return false` below happens BEFORE any kernel is
// submitted, so a false return leaves pos, the KV cache and the recurrent
// state exactly as they were and the caller may redo the work
// sequentially.  Keep it that way -- a late bail-out would silently
// double-process tokens.
// Counts SUCCESSFUL gemma-4 batched prefills.  A test that only compares
// batched output against sequential output passes trivially when the
// batched path quietly declined and fell back -- which is the same shape
// of silence rule 12 was written about.  The gate reads this to prove the
// path it is comparing actually ran.
// Counts NVFP4 tensors actually DECODED.  A gate that compares an
// NVFP4 checkpoint against a bf16 twin proves nothing if the NVFP4
// arm quietly took some other path, and every failure mode of this
// format is a number rather than an error.
long g_nvfp4_tensors = 0;
// Prompt tokens NOT re-processed because a prefix was already resident.
// A gate that compares reuse against a full prefill proves nothing if no
// reuse happened, and the difference is invisible in the output by
// design -- the whole claim is that it changes nothing but the work.
long g_prefix_tokens_reused = 0;
long g_prefix_tokens_reused_calls = 0;
// Bytes a save or a restore actually copies.  It exists to be ASSERTED
// ON, not read: the claim "the KV cache is no longer copied" is
// invisible in the output -- a copying cache and a viewing one answer
// identically -- and the only thing that distinguishes them is that one
// of them moves bytes proportional to max_seq.
long g_prefix_bytes_copied = 0;
// Batched decode steps, and the rows they carried.  The ratio is the
// only thing that distinguishes a batch from a loop, and an identity
// test cannot see the difference -- a path that quietly stepped each
// sequence alone would pass every token comparison there is.
long g_batch_decode_steps = 0;
long g_batch_decode_rows = 0;
long g_gemma4_batched_prefills = 0;
// Same, for Qwen4-Exp: a prefill-vs-decode gate that cannot see the
// fallback proves nothing at all.
long g_qwen4_exp_batched_prefills = 0;

// ---------------------------------------------------------------------
// gemma-4 batched prefill.
//
// forward_gemma4()'s graph, M tokens at a time.  It is NOT the generic
// prefill() loop: that one is the Qwen residual graph (attention output
// added raw, normalised on the way INTO the FFN) and gemma-4 is a
// sandwich, so running a prompt through it would contradict
// forward_gemma4() token for token while still emitting fluent text.
//
// Every stage below is the batched twin of the decode kernel on the same
// line of forward_gemma4(), in the same order.  The three that are easy
// to get wrong, and that produce fluent output when wrong:
//
//   * k_eq_v: V is the k_proj output taken BEFORE k_norm and BEFORE RoPE.
//     The copy therefore happens before the fused norm+rope kernel, not
//     after it.
//   * v_norm runs on EVERY layer, not only the ones that took V from
//     k_proj (ref/gemma4.py:1249 applies it after the branch).
//   * the sliding window is per LAYER, and only the sliding layers have
//     one.  launch_dflash2_block_attention computes each query's own
//     [query_end - window, query_end) bound, which is what a batch needs
//     and what launch_flash_prefill (no window at all) cannot give.
//
// SCOPE: prompt prefill only.  A verify batch (next_tokens) returns false
// -- gemma-4 has no MTP head and its DFlash drafter has never been run,
// so there is nothing to verify for yet, and a wrong verify path is worse
// than none.  PP and TP return false for the same reason: unwired, not
// broken.  Each falls back to sequential decode, which is correct.
bool Grimoire::prefill_sandwich(const std::vector<int32_t>& tokens,
                              std::vector<int32_t>* next_tokens,
                              bool allow_exact_restore, const SeqBatch* seqb) {
    const int M = int(tokens.size());
    if (M <= 0 || (!seqb && pos + M > max_seq)) return false;
    if (next_tokens && !seqb) return false;
    if ((tp_enabled() || pp_enabled()) && !seqb) return false;
    const int start_pos = pos;
    // allow_exact_restore threaded through from prefill() -- see the
    // matching note in prefill_muse(); same gap, same reason it is
    // latent rather than live today.
    if (!seqb && start_pos == 0 && allow_exact_restore && restore_prefix(tokens)) return true;

    const int H = cfg.hidden, QH = cfg.n_heads;
    const int HDX = cfg.max_head_dim(), KVHX = cfg.max_kv_heads();
    const int QWX = QH * HDX, KVWX = KVHX * HDX;
    int IX = 0, WX = std::max({H, QWX, KVWX});
    for (const auto& d : L) {
        IX = std::max(IX, d.sh_gu.output_rows() / 2);
        const DevQuant* ws[] = {&d.q_proj, &d.k_proj, &d.v_proj, &d.o_proj,
                                &d.sh_gu, &d.sh_down};
        for (const auto* w : ws) {
            WX = std::max(WX, w->w.N);
            WX = std::max(WX, w->w.K);
        }
    }
    WX = std::max(WX, 2 * IX);

    auto df = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float* h    = df(size_t(M) * H);
    float* h2   = df(size_t(M) * H);
    float* qv   = df(size_t(M) * QWX);
    float* kv   = df(size_t(M) * KVWX);
    float* vv   = df(size_t(M) * KVWX);
    float* attn = df(size_t(M) * QWX);
    float* gate = df(size_t(M) * QWX);
    float* proj = df(size_t(M) * H);
    float* sh   = df(size_t(M) * H);
    float* ff   = df(size_t(M) * 2 * size_t(IX ? IX : 1));
    // GeGLU needs its OWN output.  launch_geglu_batched writes
    // out[r*I + c] while other work-items still read gu[r'*2I + ...];
    // aliasing them is a race, not an in-place op.  Decode gets away with
    // one buffer only because M is 1 and the write is at the same index
    // as the read.
    float* ffo  = df(size_t(M) * size_t(IX ? IX : 1));
    int32_t* dtok = sycl::malloc_device<int32_t>(size_t(M), q);
    sycl_bf16* xb = sycl::malloc_device<sycl_bf16>(size_t(M) * WX, q);
    // Per-row int8 activations for the W4A8 tiles, which are the fast path
    // for a converted weight and the difference between "faster than one
    // token at a time" and using the card.
    int8_t* a8  = sycl::malloc_device<int8_t>(size_t(M) * WX, q);
    float*  a8s = df(size_t(M));
    std::vector<void*> mem = {(void*)h, (void*)h2, (void*)qv, (void*)kv,
        (void*)vv, (void*)attn, (void*)gate, (void*)proj, (void*)sh, (void*)ff,
        (void*)ffo, (void*)dtok, (void*)xb, (void*)a8, (void*)a8s};
    auto cleanup = [&]() { for (void* p : mem) if (p) sycl::free(p, q); };
    for (void* p : mem) if (!p) { cleanup(); return false; }

    const std::vector<sycl::event> none{};
    const float eps = cfg.rms_eps;

    // Same opt-in escape hatch prefill() uses, for the same reason: without
    // it none of this can be executed anywhere but the card.
    const bool noxmx =
        std::getenv("GRIMOIRE_BATCHED_PREFILL_NOXMX") != nullptr &&
        !device_can_matrix(q);

    // RULE 1: GRIMOIRE_W4A8 FREES the MXFP4 payload of a converted weight,
    // and launch_gemm_xmx dereferences it.  A converted weight still
    // reports fmt == MXFP4, so the FORMAT is not a safe test -- the
    // POINTER is.  gemv_any() is the decode path and reads whatever the
    // weight actually became, so route there rather than take the card
    // off the bus.
    // The same tuned dispatch the generic prefill uses.  A plain bf16 GEMM
    // would be correct and would waste the card: a converted weight wants
    // the W4A8 tile, an MXFP4 weight wants the dense bridge.
    //
    // RULE 5: the production 128-row tile fetches 128 rows of A per tile
    // however few tokens it is given, so a short prompt takes m16 instead.
    Xe2DenseMXFP4 g4_dense_f32 = load_xe2_dense_mxfp4_f32();
    Xe2DenseW4A8  g4_w4a8 = w4a8_enabled()
        ? load_xe2_dense_w4a8(M <= 16 ? "grimoire_xe2_dense_w4a8_f32_m16"
                                      : "grimoire_xe2_dense_w4a8_f32")
        : nullptr;

    bool ok = true;
    auto mmg = [&](const DevQuant& w, const float* x, float* y) {
        if(tp_enabled() && w.tp_sharded()) { gemm_tp(w,x,y,M); return; }
        if (!ok) return;
        if (w.w.N <= 0) { ok = false; return; }
        // Correctness-only path for a device with no matrix hardware.
        if (noxmx) { launch_gemm_batched(q, w.w, x, y, M, none); return; }

        // RULE 4: every W4A8 tile is 256 wide in N and its B block loads do
        // NOT clamp to the tensor, so a weight with N % 256 != 0 reads
        // hundreds of KB past the end -- a DEVICE_LOST, not a wrong number.
        // gemma-4's o_proj is hidden-wide and need not be aligned.
        if (w.has_i4() && g4_w4a8 && a8 && a8s && (w.w.N % 256) == 0) {
            launch_quantize_rows_int8(q, x, a8, a8s, M, w.w.K, none);
            g4_w4a8(&q, a8, w.i4, w.i4s, a8s, y, M, w.w.N, w.w.K);
            return;
        }
        // RULE 1: GRIMOIRE_W4A8 FREES the MXFP4 payload of a converted
        // weight and everything below dereferences it.  A converted weight
        // still reports fmt == MXFP4, so the FORMAT is not a safe test --
        // the POINTER is.  gemv_any() reads whatever the weight became,
        // which is what decode does.
        if (!w.w.payload) {
            for (int r = 0; r < M; ++r)
                gemv_any(w, x + size_t(r) * w.w.K,
                         y + size_t(r) * w.w.N, none);
            return;
        }
        launch_f32_to_bf16(q, x, xb, size_t(M) * w.w.K, none);
        if (g4_dense_f32 && w.w.fmt == Fmt::MXFP4) {
            g4_dense_f32(&q, xb, w.w.payload,
                         static_cast<const unsigned char*>(w.w.scales),
                         y, M, w.w.N, w.w.K);
            return;
        }
        launch_gemm_xmx(q, w.w, xb, y, M, none);
    };

    q.memcpy(dtok, tokens.data(), size_t(M) * sizeof(int32_t));
    const bool first = !pp_enabled() || pp_rank==0;
    const bool last = !pp_enabled() || pp_rank==pp_world-1;
    if (first) {
        if(!embed_rows(embed,dtok,h,M)) { cleanup(); throw std::runtime_error("batch embedding failed"); }
        if (cfg.is_muse) {
            launch_rmsnorm_residual_batched(q,h,nullptr,nullptr,muse_zero,
                                            h2,M,H,eps,nullptr,none);
            q.memcpy(h,h2,size_t(M)*H*sizeof(float));
        }
    } else if (!pp_recv_hidden(h,size_t(M)*H)) {
        cleanup(); throw std::runtime_error("batch hidden receive failed");
    }
    // Gemma4TextScaledWordEmbedding: the lookup is multiplied by
    // sqrt(hidden) BEFORE the first norm.  Omitting it changes the scale
    // every later RMSNorm sees.
    if (first && cfg.embed_scale != 1.0f)
        launch_scale(q, h, cfg.embed_scale, int(size_t(M) * H), none);

    for (int i = pp_enabled()?pp_begin:0; i < (pp_enabled()?pp_end:cfg.n_layers) && ok; ++i) {
        LayerDev& d = L[i];
        const int HD = d.head_dim, KVH = d.kv_heads, QW = QH * HD;
        const int KVW = KVH * HD;

        if(seqb && dflash2.target_aux) {
            for(size_t tap=0;tap<dflash2.target_layers.size();++tap)
                if(dflash2.target_layers[tap]+1==i)
                    for(int r=0;r<M;++r)
                        launch_dflash_store_tap(q,h+int64_t(r)*H,
                            draft_slots[size_t(seqb->slot[r])].aux,1,H,
                            int(dflash2.target_layers.size()),seqb->pos[r],int(tap),{},
                            cfg.is_muse);
        }
        // ---- attention (sandwich: in_norm -> attn -> post_norm -> add) --
        launch_rmsnorm_residual_batched(q, h, nullptr, nullptr, d.in_norm,
                                        h2, M, H, eps, nullptr, none);
        mmg(d.q_proj, h2, qv);
        mmg(d.k_proj, h2, kv);
        if (!ok) break;

        // attention_k_eq_v: a full-attention layer ships NO v_proj and V is
        // the k_proj output BEFORE k_norm and BEFORE RoPE.  This copy must
        // stay above the fused norm+rope call below; moving it under that
        // call is the mistake this comment exists to prevent, and both
        // orders produce fluent text.  A DevQuant that was never uploaded
        // keeps w.N == 0, which is the shape test forward_gemma4() uses.
        if (d.v_proj.w.N <= 0)
            q.memcpy(vv, kv, size_t(M) * KVW * sizeof(float));
        else
            mmg(d.v_proj, h2, vv);
        if (!ok) break;

        // q_norm and k_norm, then RoPE -- fused, and in that order, which
        // is forward_gemma4()'s order.
        //
        // RULE 11, AND IT BIT HERE.  Unlike launch_rmsnorm_residual_batched,
        // which reads the global set_norm_convention() installed, these two
        // take the convention as a PARAMETER that DEFAULTS TO 1.0 -- i.e.
        // (1 + w), Qwen's zero-centered norm.  gemma-4 applies the weight
        // directly, and decode says so by passing zero_centered=false to
        // launch_rmsnorm_heads.  Passing 1.0 here (or omitting it) shifts
        // every normalised q and k in the model and leaves the output
        // fluent -- it cost this path its first two gate runs.
        //
        // Read what build() actually installed rather than hardcoding 0.0,
        // so the convention stays owned by one place.
        int nconv_groups = 1; float nconv_offset = 1.0f;
        get_norm_convention(&nconv_groups, &nconv_offset);
        if (cfg.is_muse) {
            launch_rmsnorm_heads(q,qv,muse_zero,M*QH,HD,eps,true,none);
            launch_rmsnorm_heads(q,kv,muse_zero,M*KVH,HD,eps,true,none);
        }
        // Rows can belong to different conversations and positions.
        // Weight projections stay M-wide; position-dependent work is per row.
        for (int r=0; r<(seqb?M:1); ++r) {
            const int count=seqb?1:M, position=seqb?seqb->pos[r]:start_pos;
            float* qr=qv+int64_t(r)*QW;
            float* kr=kv+int64_t(r)*KVW;
            if (cfg.is_muse) {
                if (d.muse_sliding) {
                    launch_rope_rows(q,qr,count,QH,HD,position,
                                     cfg.rope_theta,cfg.partial_rope,none);
                    launch_rope_rows(q,kr,count,KVH,HD,position,
                                     cfg.rope_theta,cfg.partial_rope,none);
                }
            } else if (d.rope_proportional)
                launch_qk_norm_rope_proportional_batched(q,qr,kr,d.q_norm,d.k_norm,
                    count,QH,KVH,HD,position,d.rope_theta,d.partial_rope,eps,
                    none,nconv_offset,d.rope_factor);
            else
                launch_qk_norm_rope_batched(q,qr,kr,d.q_norm,d.k_norm,count,
                    QH,KVH,HD,position,d.rope_theta,d.partial_rope,eps,
                    none,nconv_offset);
        }
        if (!cfg.is_muse && gemma_vnorm)
            launch_rmsnorm_heads(q,vv,gemma_vnorm,M*KVH,HD,eps,false,none);
        const int window=cfg.is_muse ? (d.muse_sliding?cfg.sliding_window:0)
                                    : (cfg.layer_global(i)?0:cfg.sliding_window);
        const float scale=cfg.is_muse ? cfg.query_prescale/std::sqrt(float(HD))
                                     : cfg.attn_softmax_scale(HD);
        for (int r=0; r<(seqb?M:1); ++r) {
            const int count=seqb?1:M, position=seqb?seqb->pos[r]:start_pos;
            auto* kc=seqb?d.k_base+size_t(seqb->slot[r])*d.kv_slot:d.k_cache;
            auto* vc=seqb?d.v_base+size_t(seqb->slot[r])*d.kv_slot:d.v_cache;
            launch_kv_append_batched(q,kv+int64_t(r)*KVW,vv+int64_t(r)*KVW,
                                    kc,vc,count,position,KVH,HD,max_seq,none);
            launch_dflash2_block_attention(q,qv+int64_t(r)*QW,kc,vc,
                attn+int64_t(r)*QW,count,position,QH,KVH,HD,max_seq,window,true,scale,none);
        }
        if (cfg.is_muse) {
            mmg(d.o_gate,h2,gate);
            if (cfg.attn_gate==2)
                launch_softplus_gate(q,attn,gate,attn,M*QW,kK2GateBeta,none);
            else launch_gate_sigmoid_mul(q,attn,gate,M*QW,none);
        }

        mmg(d.o_proj, attn, proj);
        if (!ok) break;
        launch_rmsnorm_residual_batched(q, proj, nullptr, nullptr,
            d.post_norm, sh, M, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, h, sh, int(size_t(M) * H), none);

        // ---- feed-forward (sandwich: pre_ff -> GeGLU mlp -> post_ff) ----
        launch_rmsnorm_residual_batched(q, h, nullptr, nullptr, d.pre_ff_norm,
                                        h2, M, H, eps, nullptr, none);
        const int I = d.sh_gu.output_rows() / 2;
        mmg(d.sh_gu, h2, ff);
        if (!ok) break;
        // gelu_pytorch_tanh, not silu.  The two differ by up to 0.77 on the
        // same input and nothing downstream would notice the substitution.
        if (cfg.is_muse) launch_swiglu_batched(q,ff,ffo,M,I,none);
        else launch_geglu_batched(q, ff, ffo, M, I, none);
        mmg(d.sh_down, ffo, proj);
        if (!ok) break;
        launch_rmsnorm_residual_batched(q, proj, nullptr, nullptr,
            d.post_ff_norm, sh, M, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, h, sh, int(size_t(M) * H), none);

        // hidden_states *= layer_scalar, the LAST act of the layer.
        if (d.layer_scalar != 1.0f)
            launch_scale(q, h, d.layer_scalar, int(size_t(M) * H), none);
    }

    if (!ok) { q.wait(); cleanup(); return false; }

    if (seqb) {
        next_tokens->resize(size_t(M));
        if (!last) {
            if (!pp_send_hidden(h,size_t(M)*H)) {
                cleanup(); throw std::runtime_error("batch hidden send failed");
            }
        } else {
            launch_rmsnorm_residual_batched(q,h,nullptr,nullptr,fnorm,h2,
                                            M,H,eps,nullptr,none);
            for (int r=0; r<M; ++r) {
                gemv_any(lm_head,h2+int64_t(r)*H,s.logits,none);
                if (cfg.logit_softcap>0)
                    launch_logit_softcap(q,s.logits,cfg.logit_softcap,cfg.vocab,none);
                launch_argmax(q,s.logits,cfg.vocab,s.d_tok,s.d_val,none);
                q.memcpy(next_tokens->data()+r,s.d_tok,sizeof(int32_t));
            }
        }
        if(seqb->verify && spec_hidden_steps) {
            q.memcpy(spec_hidden_steps,h,size_t(M)*H*sizeof(float));
            spec_hidden_valid=true;
        }
        q.wait_and_throw();
        if (pp_enabled() && !pp_sync_tokens(*next_tokens)) {
            cleanup(); throw std::runtime_error("batch token sync failed");
        }
        ++g_batch_decode_steps; g_batch_decode_rows+=M;
        cleanup();
        return true;
    }
    // prefill()'s contract includes s.logits for the LAST row: the caller
    // takes the first generated token from it and does NOT re-run the
    // prompt's final token through forward().  Leaving it stale is silent
    // -- the logits are in vocabulary and the length is right, and only a
    // token-for-token diff against sequential decode shows it (rule 12).
    // This is forward_gemma4()'s tail: final norm, head, then softcap.
    launch_rmsnorm_residual_batched(q, h + int64_t(M - 1) * H, nullptr,
        nullptr, fnorm, h2, 1, H, eps, nullptr, none);
    gemv_any(lm_head, h2, s.logits, none);
    // final_logit_softcapping.  Monotonic, so greedy picks the same token
    // either way -- but not a no-op for sampling, and sequential applies it.
    if (cfg.logit_softcap > 0.0f)
        launch_logit_softcap(q, s.logits, cfg.logit_softcap, cfg.vocab, none);

    // The UNNORMALISED hidden state of the last processed token is left in
    // s.h, the cursor moves by M, and a full prompt from position 0
    // populates the prefix cache.
    q.memcpy(s.h, h + int64_t(M - 1) * H, size_t(H) * sizeof(float));
    q.wait_and_throw();
    pos += M;
    set_cursor(pos);
    cleanup();
    if (start_pos == 0) save_prefix(tokens);
    ++g_gemma4_batched_prefills;
    return true;
}

// ---------------------------------------------------------------------
//  Qwen4-Exp batched prefill.
//
//  forward_qwen4_exp()'s graph, M tokens at a time.  It is NOT the
//  generic prefill with different weights: that one folds each block's
//  output straight into a single residual and normalises on the way into
//  the next, and this model's residual is hc_count streams wide with a
//  gated injection.  Running a prompt through the generic path would
//  contradict decode token for token and still emit fluent text.
//
//  Every line below is a line of forward_qwen4_exp(), in the same order.
//  What is genuinely different at M > 1:
//
//   * the indexer's key rows are STRIDED (the projection emits q|k per
//     token), so they are gathered rather than memcpy'd;
//   * a whole RANGE of key blocks completes inside one batch;
//   * every QSA row has its own visible-block count, so the metadata is
//     computed on the device instead of passed as a scalar;
//   * the PLE conv reads this batch's rows plus the carried history in
//     one contiguous array.
//
//  The prefix cache is deliberately NOT used here.  It snapshots the KV
//  caches and the recurrent state; it knows nothing about the indexer's
//  two key caches or a PLE layer's conv history, so a hit would restore
//  a model that is half this request and half the last one -- silently.
// ---------------------------------------------------------------------
bool Grimoire::prefill_qwen4_exp(const std::vector<int32_t>& tokens,
                                 std::vector<int32_t>* next_tokens, const SeqBatch* seqb) {
    const int M = int(tokens.size());
    if (M <= 0 || (!seqb && pos + M > max_seq)) return false;
    if (next_tokens && !seqb) return false;              // no speculative verify path
    if (pp_enabled() || tp_enabled()) return false;
    const int start_pos = pos;

    const int H = cfg.hidden, QH = cfg.n_heads;
    const int HC = cfg.hc_count, WIDE = HC * H, LR = cfg.hc_lowrank;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;
    const int HDX = cfg.max_head_dim(), KVHX = cfg.max_kv_heads();
    int IX = 0, WX = std::max({H, WIDE, QH * HDX, KVHX * HDX, qkv_ch});
    for (const auto& d : L) {
        IX = std::max(IX, d.sh_gu.output_rows() / 2);
        const DevQuant* ws[] = {&d.q_proj, &d.k_proj, &d.v_proj, &d.o_proj,
            &d.la_qkv, &d.la_z, &d.la_ab, &d.la_out, &d.sh_gu, &d.sh_down,
            &d.hc_attn.down, &d.hc_attn.up, &d.hc_attn.inject,
            &d.hc_mlp.down, &d.hc_mlp.up, &d.hc_mlp.inject,
            &d.ix_qk, &d.ple_key, &d.ple_value};
        for (const auto* w : ws) {
            WX = std::max(WX, w->w.N);
            WX = std::max(WX, w->w.K);
        }
    }
    WX = std::max(WX, 2 * IX);
    WX = std::max(WX, cfg.moe_inter > 0 ? 2 * cfg.moe_inter : 1);

    const int RAT   = cfg.indexer_compress_ratio > 0 ? cfg.indexer_compress_ratio : 1;
    const int BTK   = cfg.indexer_budget / RAT;
    const int IH    = cfg.indexer_n_heads, IHD = cfg.indexer_head_dim;
    const int IKV   = cfg.indexer_kv_heads;
    const int NBLK  = q4_blocks_cap > 0 ? q4_blocks_cap : 1;
    const int EXPW  = q4_expand_w > 0 ? q4_expand_w : 1;
    const int NH    = (cfg.ngram_size - 1) * cfg.heads_per_ngram;
    const int PHD   = NH > 0 ? cfg.ple_embed_dim / NH : 0;
    const int SLEN  = (cfg.ple_conv_kernel - 1) * cfg.ngram_size;
    const int TK    = std::max(1, cfg.top_k);
    const int MI    = std::max(1, cfg.moe_inter);
    const int E     = std::max(1, cfg.n_experts);

    auto df = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float* hyper  = df(size_t(M) * WIDE);
    float* normed = df(size_t(M) * WIDE);
    float* gate   = df(size_t(M) * WIDE);
    float* lora   = df(size_t(M) * std::max(1, LR));
    float* inj    = df(size_t(M) * HC);
    float* pinj   = df(size_t(M) * HC);
    float* pend   = df(size_t(M) * H);
    float* blk_in = df(size_t(M) * H);
    float* t0     = df(size_t(M) * WX);
    float* t1     = df(size_t(M) * WX);
    float* t2     = df(size_t(M) * WX);
    float* t3     = df(size_t(M) * WX);
    float* attn   = df(size_t(M) * std::max(QH * HDX, Hv * Dv));
    float* gsplit = df(size_t(M) * std::max(1, QH * HDX));
    float* alpha  = df(size_t(M) * std::max(1, Hv));
    float* beta   = df(size_t(M) * std::max(1, Hv));
    float* mh     = df(size_t(M) * TK * MI);
    float* rlog   = df(size_t(M) * E);
    float* shout  = df(size_t(M) * H);
    sycl_bf16* xb = sycl::malloc_device<sycl_bf16>(size_t(M) * WX, q);
    int8_t* a8    = sycl::malloc_device<int8_t>(size_t(M) * WX, q);
    float*  a8s   = df(size_t(M));
    int32_t* dtok = sycl::malloc_device<int32_t>(size_t(M), q);
    int32_t* rex  = sycl::malloc_device<int32_t>(size_t(M) * TK, q);
    float*   rwt  = df(size_t(M) * TK);
    // QSA
    float*   ixqk = df(size_t(M) * std::max(1, (IH + IKV) * IHD));
    float*   ixq  = df(size_t(M) * std::max(1, IH * IHD));
    float*   ixk  = df(size_t(M) * std::max(1, IHD));
    float*   lgt  = df(size_t(M) * NBLK);
    int32_t* bsel = sycl::malloc_device<int32_t>(size_t(M) * std::max(1, BTK), q);
    int32_t* idx  = sycl::malloc_device<int32_t>(size_t(M) * EXPW, q);
    int32_t* vis  = sycl::malloc_device<int32_t>(size_t(M), q);
    int32_t* sql  = sycl::malloc_device<int32_t>(size_t(M), q);
    int32_t* qps  = sycl::malloc_device<int32_t>(size_t(M), q);
    // PLE
    int64_t* pids = sycl::malloc_device<int64_t>(size_t(M) * std::max(1, NH), q);
    float*   pemb = df(size_t(M) * std::max(1, cfg.ple_embed_dim));
    float*   pkey = df(size_t(M) * WIDE);
    float*   pval = df(size_t(M) * H);
    float*   pgat = df(size_t(M) * WIDE);
    float*   pcnv = df(size_t(SLEN + M) * WIDE);

    std::vector<void*> mem = {(void*)hyper,(void*)normed,(void*)gate,(void*)lora,
        (void*)inj,(void*)pinj,(void*)pend,(void*)blk_in,(void*)t0,(void*)t1,
        (void*)t2,(void*)t3,(void*)attn,(void*)gsplit,(void*)alpha,(void*)beta,
        (void*)mh,(void*)rlog,(void*)shout,(void*)xb,(void*)a8,(void*)a8s,
        (void*)dtok,(void*)rex,(void*)rwt,(void*)ixqk,(void*)ixq,(void*)ixk,
        (void*)lgt,
        (void*)bsel,(void*)idx,(void*)vis,(void*)sql,(void*)qps,(void*)pids,
        (void*)pemb,(void*)pkey,(void*)pval,(void*)pgat,(void*)pcnv};
    auto cleanup = [&]() { for (void* p : mem) if (p) sycl::free(p, q); };
    for (void* p : mem) if (!p) { cleanup(); return false; }

    const std::vector<sycl::event> none{};
    const float eps = cfg.rms_eps;
    const bool noxmx = std::getenv("GRIMOIRE_BATCHED_PREFILL_NOXMX") != nullptr &&
                       !device_can_matrix(q);

    Xe2DenseMXFP4 q4_dense_f32 = load_xe2_dense_mxfp4_f32();
    Xe2DenseW4A8  q4_w4a8 = w4a8_enabled()
        ? load_xe2_dense_w4a8(M <= 16 ? "grimoire_xe2_dense_w4a8_f32_m16"
                                      : "grimoire_xe2_dense_w4a8_f32")
        : nullptr;

    bool ok = true;
    // The same tuned dispatch prefill_sandwich() uses, and for the same
    // reasons: rule 1 (a converted weight's MXFP4 payload is FREED, so
    // the pointer and not the format decides), rule 4 (a W4A8 tile is
    // 256 wide in N and does not clamp) and rule 5 (a short prompt takes
    // the 16-row tile, not the 128-row one).
    auto mm = [&](const DevQuant& w, const float* x, float* y) {
        if (!ok) return;
        if (w.w.N <= 0) { ok = false; return; }
        if (noxmx) { launch_gemm_batched(q, w.w, x, y, M, none); return; }
        if (w.has_i4() && q4_w4a8 && a8 && a8s && (w.w.N % 256) == 0) {
            launch_quantize_rows_int8(q, x, a8, a8s, M, w.w.K, none);
            q4_w4a8(&q, a8, w.i4, w.i4s, a8s, y, M, w.w.N, w.w.K);
            return;
        }
        if (!w.w.payload) {
            for (int r = 0; r < M; ++r)
                gemv_any(w, x + size_t(r) * w.w.K, y + size_t(r) * w.w.N, none);
            return;
        }
        launch_f32_to_bf16(q, x, xb, size_t(M) * w.w.K, none);
        if (q4_dense_f32 && w.w.fmt == Fmt::MXFP4) {
            q4_dense_f32(&q, xb, w.w.payload,
                         static_cast<const unsigned char*>(w.w.scales),
                         y, M, w.w.N, w.w.K);
            return;
        }
        launch_gemm_xmx(q, w.w, xb, y, M, none);
    };

    q.memcpy(dtok, tokens.data(), size_t(M) * sizeof(int32_t));
    // The n-gram hash reads the request's own token history.
    if(seqb) for(int r=0;r<M;++r)
        q.memcpy(q4_tok_base+size_t(seqb->slot[r])*max_seq+seqb->pos[r],
                 tokens.data()+r,sizeof(int32_t));
    else q.memcpy(q4_tok + start_pos, tokens.data(), size_t(M) * sizeof(int32_t));
    launch_embed_batched(q, embed, dtok, t0, M, H, none);
    // hidden = embed(ids).repeat(1, hc_count): the SAME row in every
    // stream, not a projection.
    for (int c = 0; c < HC; ++c)
        launch_copy_rows_strided(q, t0, hyper + int64_t(c) * H, M, H, WIDE,
                                 H, none);

    bool pending = false;
    auto hc_mix = [&](const LayerDev::HCDev& hc, const float* pblk,
                      const float* pin, float* inj_out, float* block_out) {
        if (!ok) return;
        if (pblk) launch_hc_combine(q, hyper, pin, pblk, hyper, M, HC, H, none);
        launch_hc_norm(q, hyper, hc.norm, normed, M, HC, H, eps, none);
        mm(hc.down, normed, lora);
        launch_hc_silu(q, lora, M * LR, HC, none);
        mm(hc.up, lora, gate);
        launch_hc_gated_mean(q, gate, normed, block_out, M, HC, H, none);
        if (inj_out && hc.inject.w.N) mm(hc.inject, normed, inj_out);
    };

    for (int i = 0; i < cfg.n_layers && ok; ++i) {
        LayerDev& d = L[i];

        if (d.ple) {
            if (pending) {
                launch_hc_combine(q, hyper, pinj, pend, hyper, M, HC, H, none);
                pending = false;
            }
            for(int r=0;r<(seqb?M:1);++r)
                launch_ple_ngram_ids(q,seqb?q4_tok_base+size_t(seqb->slot[r])*max_seq:q4_tok,
                    pids+size_t(r)*NH,seqb?seqb->pos[r]:start_pos,seqb?1:M,d.ple_mul,
                    d.ple_size,d.ple_off,cfg.ngram_size-1,cfg.heads_per_ngram,NH,
                    cfg.eos_token_id,none);
            launch_ple_embed_gather(q, d.ple_table, d.ple_fp8, d.ple_scale,
                                    pids, pemb, M, NH, PHD, d.ple_rows, none);
            mm(d.ple_key,   pemb, pkey);
            mm(d.ple_value, pemb, pval);
            if (!ok) break;
            // PLE history is per conversation; projections remain batched.
            for(int r=0;r<(seqb?M:1);++r) {
                const int count=seqb?1:M;
                float* history=seqb?d.ple_hist_base+size_t(seqb->slot[r])*
                    std::max(1,SLEN)*WIDE:d.ple_hist;
                launch_ple_gate(q,pkey+int64_t(r)*WIDE,pval+int64_t(r)*H,
                    hyper+int64_t(r)*WIDE,d.ple_nk,d.ple_nq,d.ple_nc,
                    pgat,pcnv+int64_t(SLEN)*WIDE,count,HC,H,eps,none);
                if(SLEN>0) q.memcpy(pcnv,history,size_t(SLEN)*WIDE*sizeof(float));
                launch_ple_conv(q,pcnv,pgat,hyper+int64_t(r)*WIDE,d.ple_cw,
                    hyper+int64_t(r)*WIDE,SLEN,count,WIDE,cfg.ple_conv_kernel,
                    cfg.ngram_size,none);
                if(SLEN>0) q.memcpy(history,pcnv+int64_t(count)*WIDE,
                                    size_t(SLEN)*WIDE*sizeof(float));
            }
        }

        hc_mix(d.hc_attn, pending ? pend : nullptr, pending ? pinj : nullptr,
               inj, blk_in);
        pending = false;
        if (!ok) break;

        if (d.kind == LayerKind::LINEAR_ATTN) {
            mm(d.la_qkv, blk_in, t0);
            if (!ok) break;
            for(int r=0;r<(seqb?M:1);++r) {
                ConvParams cp{t0+int64_t(r)*qkv_ch,d.la_conv,
                    seqb?d.conv_base+size_t(seqb->slot[r])*d.conv_slot:d.conv_ring,
                    nullptr,qkv_ch,cfg.conv_kernel};
                launch_causal_conv1d_split_prefill(q,cp,seqb?1:M,
                    t1+int64_t(r)*Hk*Dk,t2+int64_t(r)*Hk*Dk,
                    t3+int64_t(r)*Hv*Dv,nullptr,Hk*Dk,Hv*Dv);
            }
            launch_l2norm_heads(q, t1, M * Hk, Dk, none);
            launch_l2norm_heads(q, t2, M * Hk, Dk, none);
            mm(d.la_ab, blk_in, t0);
            if (!ok) break;
            launch_deltanet_gates_batched(q, t0, d.la_Alog, d.la_dtb,
                                          alpha, beta, M, Hv);
            if (M <= 16) {
                for (int t = 0; t < M; ++t) {
                    DeltaNetParams sp{};
                    sp.q = t1 + size_t(t) * Hk * Dk;
                    sp.k = t2 + size_t(t) * Hk * Dk;
                    sp.v = t3 + size_t(t) * Hv * Dv;
                    sp.a = alpha + size_t(t) * Hv;
                    sp.beta = beta + size_t(t) * Hv;
                    sp.state = seqb?d.dn_base+size_t(seqb->slot[t])*d.dn_slot:d.dn_state;
                    sp.out = attn + size_t(t) * Hv * Dv;
                    sp.n_heads = Hv; sp.k_dim = Dk; sp.v_dim = Dv;
                    sp.n_k_heads = Hk;
                    launch_deltanet_step(q, sp, none);
                }
            } else {
                DeltaNetPrefillParams dp{t1, t2, t3, alpha, beta, d.dn_state,
                                         attn, Hv, Dk, Dv, M, Hk};
                launch_deltanet_prefill(q, dp);
            }
            mm(d.la_z, blk_in, t3);
            if (!ok) break;
            launch_rmsnorm_gate_silu(q, attn, t3, d.la_norm, M * Hv, Dv,
                                     eps, none);
            mm(d.la_out, attn, pend);
        } else {
            const int HD = d.head_dim, KVH = d.kv_heads;
            const bool gated = (d.q_proj.output_rows() == 2 * QH * HD);
            mm(d.q_proj, blk_in, t0);
            if (!ok) break;
            float* qv = t0;
            if (gated) {
                launch_split_qgate_batched(q, t0, t1, gsplit, M, QH, HD, none);
                qv = t1;
            }
            mm(d.k_proj, blk_in, t2);
            mm(d.v_proj, blk_in, t3);
            if (!ok) break;
            int nconv_groups = 1; float nconv_offset = 1.0f;
            get_norm_convention(&nconv_groups, &nconv_offset);
            for(int r=0;r<(seqb?M:1);++r) {
                const int count=seqb?1:M, position=seqb?seqb->pos[r]:start_pos;
                auto* kc=seqb?d.k_base+size_t(seqb->slot[r])*d.kv_slot:d.k_cache;
                auto* vc=seqb?d.v_base+size_t(seqb->slot[r])*d.kv_slot:d.v_cache;
                launch_qk_norm_rope_batched(q,qv+int64_t(r)*QH*HD,
                    t2+int64_t(r)*KVH*HD,d.q_norm,d.k_norm,count,QH,KVH,HD,
                    position,d.rope_theta,d.partial_rope,eps,none,nconv_offset);
                launch_kv_append_batched(q,t2+int64_t(r)*KVH*HD,t3+int64_t(r)*KVH*HD,
                    kc,vc,count,position,KVH,HD,max_seq,none);
            }

            if (d.qsa) {
                mm(d.ix_qk, blk_in, ixqk);
                if (!ok) break;
                // BOTH halves have to be gathered out first.  The
                // projection emits [q | k] PER TOKEN, so at M > 1 the
                // query heads are strided by the whole projection width
                // -- and launch_rmsnorm_heads, launch_rope_rows and
                // launch_qsa_index_logits all index their heads
                // CONTIGUOUSLY.  At M == 1 the two layouts coincide,
                // which is why decode never saw it.
                launch_copy_rows_strided(q, ixqk, ixq, M,
                                         (IH + IKV) * IHD, IH * IHD,
                                         IH * IHD, none);
                launch_copy_rows_strided(q, ixqk + int64_t(IH) * IHD, ixk, M,
                                         (IH + IKV) * IHD, IHD, IHD, none);
                launch_rmsnorm_heads(q, ixq, d.ix_qn, M * IH, IHD, eps,
                                     true, none);
                for(int r=0;r<(seqb?M:1);++r) {
                    const int count=seqb?1:M, position=seqb?seqb->pos[r]:start_pos;
                    const int end=position+count;
                    const int slot=seqb?seqb->slot[r]:seq_slot;
                    float* raw=d.ix_raw_base+size_t(slot)*max_seq*IHD;
                    float* compressed=d.ix_cmp_base+size_t(slot)*(max_seq/RAT+1)*IHD;
                    auto* kc=d.k_base+size_t(slot)*d.kv_slot;
                    auto* vc=d.v_base+size_t(slot)*d.kv_slot;
                    launch_rope_rows(q,ixq+int64_t(r)*IH*IHD,count,IH,IHD,
                        position,d.rope_theta,d.partial_rope,none);
                    q.memcpy(raw+int64_t(position)*IHD,ixk+int64_t(r)*IHD,
                             size_t(count)*IHD*sizeof(float));
                    const int b0=position/RAT, b1=end/RAT;
                    if(b1>b0) {
                        float* dst=compressed+int64_t(b0)*IHD;
                        launch_qsa_pool_blocks(q,raw,dst,b0,b1-b0,RAT,IHD,none);
                        launch_rmsnorm_heads(q,dst,d.ix_kn,b1-b0,IHD,eps,true,none);
                        launch_qsa_rope_blocks(q,dst,b0,b1-b0,IHD,RAT,
                                              d.rope_theta,d.partial_rope,none);
                    }
                    launch_qsa_row_meta(q,vis,sql,qps,count,position,end,RAT,none);
                    const int nb=std::max(1,b1);
                    launch_qsa_index_logits(q,ixq+int64_t(r)*IH*IHD,compressed,lgt,
                                            count,IH,IHD,nb,vis,none);
                    launch_qsa_topk_blocks(q,lgt,bsel,count,nb,BTK,vis,none);
                    launch_qsa_expand_blocks(q,bsel,idx,count,BTK,RAT,
                                             cfg.indexer_budget,sql,qps,none);
                    launch_qsa_attention(q,qv+int64_t(r)*QH*HD,kc,vc,idx,
                        attn+int64_t(r)*QH*HD,count,QH,KVH,HD,max_seq,EXPW,
                        cfg.attn_softmax_scale(HD),none);
                }
            } else {
                for(int r=0;r<(seqb?M:1);++r) {
                    const int slot=seqb?seqb->slot[r]:seq_slot;
                    launch_dflash2_block_attention(q,qv+int64_t(r)*QH*HD,
                        d.k_base+size_t(slot)*d.kv_slot,d.v_base+size_t(slot)*d.kv_slot,
                        attn+int64_t(r)*QH*HD,seqb?1:M,seqb?seqb->pos[r]:start_pos,
                        QH,KVH,HD,max_seq,0,true,cfg.attn_softmax_scale(HD),none);
                }
            }
            if (gated)
                launch_gate_sigmoid_mul_batched(q, attn, gsplit, M, QH * HD,
                                                none);
            mm(d.o_proj, attn, pend);
        }
        if (!ok) break;

        hc_mix(d.hc_mlp, pend, inj, pinj, blk_in);
        if (!ok) break;

        if (d.moe_layer) {
            mm(d.router, blk_in, rlog);
            if (!ok) break;
            launch_router_topk_batched(q, rlog, M, cfg.n_experts, cfg.top_k,
                                       rex, rwt, true, none);
            launch_moe_gate_up_batched(q, d.moe, rex, blk_in, mh, M);
            launch_moe_down_batched(q, d.moe, rex, rwt, mh, pend, M);
            if (d.sh_gu.w.N) {
                const int SI = d.sh_gu.output_rows() / 2;
                mm(d.sh_gu, blk_in, t0);
                if (!ok) break;
                launch_swiglu_batched(q, t0, t1, M, SI);
                mm(d.sh_down, t1, shout);
                if (!ok) break;
                if (d.has_sh_gate) {
                    mm(d.sh_gate_q, blk_in, t2);
                    if (!ok) break;
                    launch_scale_by_sigmoid_batched(q, shout, t2, M, H, none);
                }
                launch_add(q, pend, shout, int(size_t(M) * H), none);
            }
        } else {
            const int FI = d.sh_gu.output_rows() / 2;
            mm(d.sh_gu, blk_in, t0);
            if (!ok) break;
            launch_swiglu_batched(q, t0, t1, M, FI);
            mm(d.sh_down, t1, pend);
        }
        pending = true;
    }

    if (!ok) { q.wait(); cleanup(); return false; }

    if(seqb) {
        hc_mix(hc_final,pending?pend:nullptr,pending?pinj:nullptr,nullptr,blk_in);
        if(!ok) { q.wait(); cleanup(); return false; }
        next_tokens->resize(size_t(M));
        for(int r=0;r<M;++r) {
            gemv_any(lm_head,blk_in+int64_t(r)*H,s.logits,none);
            launch_argmax(q,s.logits,cfg.vocab,s.d_tok,s.d_val,none);
            q.memcpy(next_tokens->data()+r,s.d_tok,sizeof(int32_t));
        }
        q.wait_and_throw();
        cleanup();
        ++g_batch_decode_steps; g_batch_decode_rows+=M;
        return true;
    }
    // prefill()'s contract includes s.logits for the LAST row: the caller
    // takes the first generated token from it and does not re-run the
    // prompt's final token through forward().  The tail mixer therefore
    // runs on that one row -- consuming the same pending combine decode
    // would have consumed.
    {
        const int64_t off = int64_t(M - 1);
        float* hl = hyper + off * WIDE;
        if (pending)
            launch_hc_combine(q, hl, pinj + off * HC, pend + off * H, hl,
                              1, HC, H, none);
        launch_hc_norm(q, hl, hc_final.norm, normed, 1, HC, H, eps, none);
        gemv_any(hc_final.down, normed, lora, none);
        launch_hc_silu(q, lora, LR, HC, none);
        gemv_any(hc_final.up, lora, gate, none);
        launch_hc_gated_mean(q, gate, normed, s.h2, 1, HC, H, none);
        gemv_any(lm_head, s.h2, s.logits, none);
        q.memcpy(s.h, s.h2, size_t(H) * sizeof(float));
    }
    q.wait_and_throw();
    pos += M;
    set_cursor(pos);
    cleanup();
    ++g_qwen4_exp_batched_prefills;
    return true;
}

bool Grimoire::prefill(const std::vector<int32_t>& tokens,
                       std::vector<int32_t>* next_tokens,
                       const SeqBatch* seqb,
                       bool allow_exact_restore) {
    if(seqb){
        // A batch does not have ONE position, so the engine's cursor says
        // nothing about whether it fits.  Every row is checked against
        // its own sequence instead.
        if(tokens.empty()||int(tokens.size())>kMaxBatchRows)
            throw std::invalid_argument("batch is empty or wider than the engine allows");
        for(size_t r=0;r<tokens.size();++r){
            if(seqb->slot[r]<0||seqb->slot[r]>=n_seq_slots)
                throw std::invalid_argument("batch row names a sequence slot that does not exist");
            if(seqb->pos[r]<0||seqb->pos[r]>=max_seq)
                throw std::out_of_range("batch row is past the context window");
        }
        // Two rows in one slot would append two keys to the same
        // position and then both read the survivor.  Fluent, wrong, and
        // impossible to see in the output.
        for(size_t a=0;a+1<tokens.size();++a)
            for(size_t b=a+1;b<tokens.size();++b)
                if(seqb->slot[a]==seqb->slot[b]) {
                    if(!seqb->verify || seqb->pos[b]-seqb->pos[a]!=int(b-a))
                        throw std::invalid_argument("batch rows share a slot without a contiguous verify block");
                }
        const std::string why=batch_unsupported_reason();
        if(!why.empty()){
            static bool said=false;
            if(!said){said=true;
                std::fprintf(stderr,"    batched decode unavailable: %s\n",why.c_str());}
            return false;
        }
    } else
    if(tokens.empty() || pos<0 || pos>max_seq || tokens.size()>size_t(max_seq-pos))
        throw std::invalid_argument("prefill exceeds context capacity or is empty");
    for(auto t:tokens)if(t<0||t>=cfg.vocab)throw std::invalid_argument("invalid prefill token");
    if(tp_enabled() && !seqb)return false;
    // GRIMOIRE_BATCHED_PREFILL_NOXMX: run the batched path on a device with
    // NO matrix hardware, by routing every mm() through the plain-SYCL
    // launch_gemm_batched() instead of the joint_matrix GEMM (see mm()).
    // Without it the batched path is unreachable off the card, so nothing
    // in it -- including a gemma-4 batched prefill -- can be checked against
    // the sequential path that it has to agree with.
    //
    // Requires the device to actually lack matrix support, so setting it on
    // a B70 does nothing and the Tower path is untouched.  CORRECTNESS ONLY:
    // it is a sub-group GEMV per output row and is far slower than the tile
    // it replaces.  Rule 8 -- no number from this configuration means
    // anything.
    // Requires the device to genuinely lack matrix support.  This used to
    // test the aspect directly, which meant an over-reporting CPU turned
    // the fallback OFF -- exactly on the device that needed it -- and sent
    // gemma-4's batched prefill into launch_gemm_xmx on a JIT that cannot
    // build it.  device_can_matrix() is the single owner of that question.
    const bool noxmx_gemm =
        std::getenv("GRIMOIRE_BATCHED_PREFILL_NOXMX")!=nullptr &&
        !device_can_matrix(q);
    // The batched path is XMX/DPAS from end to end.  On a device with no
    // matrix hardware the runtime does not raise -- it takes the process
    // down with a SIGSEGV somewhere inside the JIT -- so refuse here and
    // let the caller's documented sequential fallback run instead.  A B70
    // always has the aspect, so this never fires on the Tower; it is what
    // lets the engine be driven on any SYCL device for verification.
    {
        // Gated on is_gpu() FIRST and deliberately.  If this asked only
        // about the matrix aspect and a B70 driver did not report it, the
        // Tower would silently lose batched prefill and run prompts one
        // token at a time -- a ~50x prompt-processing regression hiding
        // behind one stderr line, which is precisely the class of bug
        // this branch has been clearing out.  A B70 is a GPU, so on the
        // card this predicate is false and behaviour is exactly as
        // before; the escape hatch exists only for a non-GPU device.
        //
        // AND THE ASPECT IS NOT ENOUGH ON A CPU (measured 2026-09-16).
        // This container was rescheduled onto a different host mid-session
        // -- Xeon @ 2.80GHz to Xeon @ 2.10GHz -- and the OpenCL CPU
        // runtime on the second one ADVERTISES ext_intel_matrix.  So
        // no_matrix went false, batched prefill switched itself on for the
        // first time ever on a CPU device, and the joint_matrix kernels
        // reached a JIT that cannot compile them:
        //
        //   test_k2_e2e         SIGSEGV, with the Intel runtime's own
        //                       "PLEASE submit a bug report" line
        //   test_gemma4_prefill MISMATCH (batched vs sequential)
        //   test_spec_e2e       32 failures, batched verify now live
        //
        // Four gates red, nothing in the engine changed, and every symptom
        // pointed at whatever had been committed most recently.  The
        // banner is what gave it away: it read "prefill batched" where
        // every earlier run on a CPU said "SEQUENTIAL fallback".
        //
        // The aspect answers "does this device CLAIM matrix support".  The
        // question is "can it RUN a joint_matrix kernel", and a CPU that
        // advertises the aspect and then crashes its own JIT answers yes
        // to the first and no to the second.  So require a GPU as well:
        // the aspect alone may enable it on a real GPU that reports it,
        // but never on a CPU.  A B70 is a GPU and is unaffected.
        //
        // GRIMOIRE_TRUST_MATRIX_ASPECT=1 restores the old predicate for
        // anyone who wants to test a CPU device that really can.
        const bool no_matrix = !device_can_matrix(q);
        if (no_matrix && !noxmx_gemm) {
            static bool said = false;
            if (!said) {
                said = true;
                std::fprintf(stderr, "    not a GPU and no matrix hardware: "
                    "batched prefill disabled, falling back to sequential\n");
            }
            return false;
        }
    }
    if (cfg.is_muse) {
        if (seqb) return prefill_sandwich(tokens, next_tokens, false, seqb);
        if(std::getenv("GRIMOIRE_MUSE_SEQUENTIAL_PREFILL"))return false;
        return prefill_muse(tokens,next_tokens,allow_exact_restore);
    }
    // gemma-4 has its own batched prefill: the loop below is the Qwen
    // residual graph -- attention output added raw, normalised on the way
    // INTO the FFN -- and gemma-4 is a sandwich, so running a prompt
    // through it would contradict forward_gemma4() token for token while
    // still producing fluent text.  prefill_sandwich() is that graph
    // batched; it declines a verify batch and PP/TP, each of which falls
    // back to sequential decode.
    // ON by default since bin/test_gemma4_prefill went green: batched
    // prefill is token-identical to sequential decode at bf16, fp8_e4m3,
    // int8 and mxfp4, and at head_dim 512 as well as the small fixture.
    // It took three gate runs to get there -- a missing logits tail, a
    // GeGLU call that aliased its input, and a qk-norm convention that
    // defaults to Qwen's -- and every one of those was FLUENT when wrong,
    // so the escape hatch stays: GRIMOIRE_GEMMA4_SEQUENTIAL_PREFILL=1
    // forces the sequential path for an A/B on the card.
    if (cfg.is_gemma4) {
        if (std::getenv("GRIMOIRE_GEMMA4_SEQUENTIAL_PREFILL")) return false;
        return prefill_sandwich(tokens, next_tokens, allow_exact_restore, seqb);
    }
    // Qwen4-Exp: batched prefill below is the Qwen residual graph and
    // this model's is the hyper-connection one, so running a prompt
    // through it would contradict forward_qwen4_exp() token for token and
    // still emit fluent text.  prefill_qwen4_exp() is that graph batched.
    if (cfg.is_qwen4_exp) {
        if (std::getenv("GRIMOIRE_QWEN4EXP_SEQUENTIAL_PREFILL")) return false;
        return prefill_qwen4_exp(tokens, next_tokens, seqb);
    }
    const int M = int(tokens.size());
    if (M <= 0) return false;
    // NOT under seqb (external audit, 2026-09-21).  Every row's own
    // position was already checked against max_seq individually at the
    // top of this function -- that is what a batch HAS instead of one
    // cursor.  This check compares the engine's single scalar `pos`
    // (which a batched call does not advance and may be stale from
    // whatever single-sequence prefill last ran) plus M, where M here
    // counts ROWS -- one token each, from DIFFERENT conversations -- not
    // consecutive tokens appended to that scalar's conversation.  Four
    // rows sitting at position 126 of a 128-token window each have one
    // token of room; `pos + 4 > 128` refused all of them anyway,
    // whatever `pos` happened to be.
    if (!seqb && pos + M > max_seq) return false;
    const int start_pos = pos;
    if (!seqb && !next_tokens && start_pos == 0 && allow_exact_restore &&
        restore_prefix(tokens)) return true;

    const int H = cfg.hidden, Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;
    int W = mtp.ok ? 2*H : H;
    for (const auto& d : L) {
        const DevQuant* ws[] = {&d.la_qkv,&d.la_z,&d.la_out,&d.la_ab,&d.la_all,&d.q_proj,&d.k_proj,
            &d.v_proj,&d.o_proj,&d.router,&d.sh_gu,&d.sh_down,&d.sh_gate_q};
        for (auto* w : ws) { W = std::max(W, w->output_rows()); W = std::max(W, w->w.K); }
    }
    auto df = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float *bh=df(size_t(M)*H), *bn=df(size_t(M)*H), *r0=df(size_t(M)*H), *r1=df(size_t(M)*H);
    float *t0=df(size_t(M)*W), *t1=df(size_t(M)*W), *t2=df(size_t(M)*W),
          *t3=df(size_t(M)*W), *t4=df(size_t(M)*W);
    float* la_fused=df(size_t(M)*12352);
    sycl_bf16* xb=sycl::malloc_device<sycl_bf16>(size_t(M)*W,q);
    sycl_bf16* bn_bf=sycl::malloc_device<sycl_bf16>(size_t(M)*H,q);
    // W4A8 activation staging: int8 rows plus one dequant scale per row.
    int8_t* a8  = w4a8_enabled()?sycl::malloc_device<int8_t>(size_t(M)*std::max(H,W),q):nullptr;
    float*  a8s = w4a8_enabled()?sycl::malloc_device<float>(size_t(M),q):nullptr;
    int32_t* dtok=sycl::malloc_device<int32_t>(M,q);
    float* batch_logits = next_tokens
        ? sycl::malloc_device<float>(size_t(M) * cfg.vocab, q) : nullptr;
    // Dense checkpoints legitimately have top_k == 0.  USM zero-byte
    // allocation returns null and used to make their prefill fail before the
    // first kernel, even though these placeholders are never consumed.
    const int alloc_top_k=std::max(1,cfg.top_k);
    int32_t* rex=sycl::malloc_device<int32_t>(size_t(M)*alloc_top_k,q);
    float* rwt=df(size_t(M)*alloc_top_k);
    float* rlog=df(size_t(M)*std::max(1,cfg.n_experts));
    float* mh=df(size_t(M)*std::max(1,cfg.top_k*cfg.moe_inter));
    // vLLM pads every chunked-GDN input to M + (chunk_size - 1) and zero-fills
    // it, because the kernel reads a whole 64-token tile from the final chunk:
    // max tile base is floor((M-1)/64)*64, so it touches up to row M+62.  With
    // only M rows that tile runs past the end (and for the head-major gates,
    // into the next head), which is why synthetic tokens survived and real text
    // hung.  See csrc/xpu/gdn_attn/gdn_attn_interface.cpp.
    const size_t gdn_tokens=size_t(M)+63;
    // Same trap as alloc_top_k above, missed one line below the comment
    // that describes it: a model with NO linear-attention layers has
    // Hv == 0, the zero-byte USM allocation returns null, the null sweep
    // below reads that as an allocation failure, and the ENTIRE batched
    // prefill is abandoned for the token-at-a-time fallback.  Nothing
    // fails and nothing is wrong in the output -- prompt processing is
    // just quietly ~50x slower, behind one line on stderr.  That is every
    // pure-attention checkpoint, K2 included, not some corner case.
    const size_t gdn_elems=std::max<size_t>(1, gdn_tokens*size_t(Hv));
    float* alpha=df(gdn_elems); float* beta=df(gdn_elems);
    const int R=M*std::max(1,cfg.top_k);
    const int I=std::max(1,cfg.moe_inter);
    const int E=std::max(1,cfg.n_experts);
    Xe2GroupedW4A16 xe2_grouped=load_xe2_grouped();
    Xe2DenseMXFP4 xe2_dense_mxfp4=load_xe2_dense_mxfp4();
    Xe2DenseMXFP4 xe2_dense_mxfp4_f32=load_xe2_dense_mxfp4_f32();
    Xe2DenseW4A8 xe2_w4a8_bf16_big=w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_f32_big =w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_bf16_m8=w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16_m8"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_f32_m8 =w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m8"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_bf16_m8n128=w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16_m8n128"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_f32_m8n128 =w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m8n128"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_bf16_m16=w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16_m16"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_f32_m16 =w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m16"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_bf16_m16n128=w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_bf16_m16n128"):nullptr;
    Xe2DenseW4A8 xe2_w4a8_f32_m16n128 =w4a8_enabled()?
        load_xe2_dense_w4a8("grimoire_xe2_dense_w4a8_f32_m16n128"):nullptr;
    // A 16-row tile is weight-bound at small M; the 128-row tile wastes most
    // of its A loads on padding.  Pick by M once, here.
    const bool tiny_batch = (M > 4 && M <= 8) &&
        xe2_w4a8_f32_m8 && xe2_w4a8_bf16_m8;
    const bool small_batch = (M <= 16) && xe2_w4a8_f32_m16 && xe2_w4a8_bf16_m16;
    const bool use_n128 = std::getenv("GRIMOIRE_W4A8_N128") != nullptr;
    Xe2DenseW4A8 xe2_w4a8_bf16 = use_n128 && M>4 && M<=8 && xe2_w4a8_bf16_m8n128
        ? xe2_w4a8_bf16_m8n128
        : use_n128 && M<=4 && xe2_w4a8_bf16_m16n128 ? xe2_w4a8_bf16_m16n128
        : tiny_batch?xe2_w4a8_bf16_m8:
        (small_batch?xe2_w4a8_bf16_m16:xe2_w4a8_bf16_big);
    Xe2DenseW4A8 xe2_w4a8_f32  = use_n128 && M>4 && M<=8 && xe2_w4a8_f32_m8n128
        ? xe2_w4a8_f32_m8n128
        : use_n128 && M<=4 && xe2_w4a8_f32_m16n128 ? xe2_w4a8_f32_m16n128
        : tiny_batch?xe2_w4a8_f32_m8:
        (small_batch?xe2_w4a8_f32_m16:xe2_w4a8_f32_big);
    Xe2GroupedMXFP4 xe2_grouped_mxfp4_big=load_xe2_grouped_mxfp4();
    // The production grouped kernel is a 64-ROW tile.  In a speculative
    // verifier the few rows spread over ~22 distinct experts per layer, so
    // each expert gets one or two rows of that tile and the rest is padding:
    // Ornith's routed MoE cost 7.08 ms at M=4 against a ~2.8 ms roofline.
    Xe2GroupedMXFP4 xe2_grouped_mxfp4_m8 =
        load_xe2_grouped_sym("grimoire_xe2_grouped_mxfp4_bf16_m8");
    Xe2GroupedMXFP4 xe2_grouped_mxfp4_m16 =
        load_xe2_grouped_sym("grimoire_xe2_grouped_mxfp4_bf16_m16");
    Xe2GroupedMXFP4 xe2_grouped_mxfp4_full =
        load_xe2_grouped_sym("grimoire_xe2_grouped_mxfp4_bf16_full");
    Xe2GroupedMXFP4 xe2_grouped_mxfp4 = xe2_grouped_mxfp4_big;
    // Mirror vLLM's Xe2 dispatch exactly.  Its threshold is average routed
    // rows per expert (Total_M / E), not the original token batch M.
    const int moe_avg_m=cfg.is_moe() ? (R/cfg.n_experts) : 0;
    if(moe_avg_m<=4 && xe2_grouped_mxfp4_m8)
        xe2_grouped_mxfp4=xe2_grouped_mxfp4_m8;
    else if(moe_avg_m<=8 && xe2_grouped_mxfp4_m16)
        xe2_grouped_mxfp4=xe2_grouped_mxfp4_m16;
    else if(moe_avg_m>128 && xe2_grouped_mxfp4_full)
        xe2_grouped_mxfp4=xe2_grouped_mxfp4_full;
    Xe2FusedGateUpMXFP4 xe2_fused_gate_up=load_xe2_fused_gate_up_mxfp4();
    Xe2ChunkPrefill xe2_attention=load_xe2_chunk_prefill();
    load_bestla(cfg.n_layers);
    Xe2ChunkGdn xe2_gdn=load_xe2_chunk_gdn();
    Xe2ChunkGdnRaw xe2_gdn_raw=load_xe2_chunk_gdn_raw();
    OneDnnW4Api od=load_onednn_w4();
    const bool parallel_prefill=!tp_enabled() && std::getenv("GRIMOIRE_PARALLEL_PREFILL") && xe2_dense_mxfp4;
    const bool parallel_shared=!tp_enabled() && std::getenv("GRIMOIRE_PARALLEL_SHARED") && xe2_dense_mxfp4;
    const bool need_aux=parallel_prefill||parallel_shared;
    const bool norm_bf_only=!tp_enabled() && M>=32&&xe2_dense_mxfp4&&xe2_dense_mxfp4_f32&&
        (!cfg.is_moe()||xe2_grouped_mxfp4)&&!need_aux;
    // Deferred gathering leaves the routed result permuted in moe_res and
    // makes the NEXT layer's input norm un-permute it
    // (launch_rmsnorm_moe_residual_batched reads moe_res/pinv/rwt).  That
    // is only valid if the previous layer actually routed.  cfg.is_moe()
    // is a property of the MODEL: K2 is a MoE model whose first layers are
    // DENSE, so with this on, layer 0 writes r0, layer 1 un-permutes
    // routing buffers the dense layer never filled, and layer 0's real
    // output is dropped -- with stale route indices free to read anywhere.
    // The fused routine also normalises over the whole hidden width with
    // (1 + weight), which is not K2's grouped, direct-weight convention,
    // so even an all-routed K2 stage would be numerically wrong.
    //
    // Require every layer this process executes to be routed.  tools/nrun.sh
    // sets GRIMOIRE_DEFER_MOE_GATHER=1, so this is a reachable launcher
    // configuration, not a dormant helper.
    bool all_layers_routed=cfg.is_moe();
    {
        const int lb=pp_enabled()?pp_begin:0, le=pp_enabled()?pp_end:cfg.n_layers;
        for(int li=lb;li<le&&all_layers_routed;++li)
            if(!L[size_t(li)].moe_layer)all_layers_routed=false;
    }
    const bool defer_moe_gather=std::getenv("GRIMOIRE_DEFER_MOE_GATHER")&&
        cfg.is_moe()&&!cfg.is_k2&&all_layers_routed&&
        M>=32&&xe2_grouped_mxfp4&&norm_bf_only;
    // Dense Qwen has no routed experts (R=M, I=1) but its fused MLP projection
    // is much wider than H.  Size bridge scratch from every real projection,
    // not from MoE geometry, or the native dense kernel writes past the end.
    // xperm also backs the padded GDN q/k/v triple (each gdn_tokens rows), so
    // it must cover that as well as the MoE permute and the widest projection.
    const size_t gdn_qkv_elems=gdn_tokens*(2*size_t(Hk)*Dk+size_t(Hv)*Dv);
    // The grouped MoE GEMM rounds EVERY expert's row count up to the tile
    // height: xe2_grouped_raw_launcher.hpp:31 computes
    //     groups += ((rows[e] + tm - 1) / tm) * ...
    // with tm = 64 for p64x128_prod.  The final tile of the LAST expert
    // therefore reads up to tm-1 rows past that expert's data, which is past
    // the end of these buffers when they are sized at exactly R rows.  Whether
    // that faults depends only on what the allocator happened to map next,
    // which is why a 4096-token prefill survived and a 3782-token one took the
    // compute engine down with DEVICE_LOST (MEASURED 2026-09-06 on Ornith,
    // reproduced from the CLI, so not a server-path problem).  Dense models
    // never touch these buffers, which is why Qwen was never affected.
    // One tile of slack costs ~1 MB and removes the whole class of overrun.
    constexpr size_t kMoeTilePad = 128;   // >= tm of every grouped policy
    const size_t Rpad = size_t(R) + kMoeTilePad;
    // The bf16 attention path packs q|k|v back to back into xperm:
    // M*(n_heads*head_dim) + 2*M*(kv_heads*head_dim) elements.  W above is
    // the widest SINGLE projection, so M*W bounds each of those three, not
    // their SUM.  Size it from the sum directly, per layer -- on gemma-4 a
    // sliding layer and a full-attention layer do not even have the same
    // width.
    size_t attn_stage_elems=0;
    for(const auto& ld:L)
        attn_stage_elems=std::max(attn_stage_elems,
            size_t(M)*(size_t(cfg.n_heads)+2*size_t(ld.kv_heads))*
            size_t(ld.head_dim));
    const size_t bridge_elems=std::max(std::max(std::max(Rpad*std::max(H,I),
                                                size_t(M)*W),gdn_qkv_elems),
                                       attn_stage_elems);
    sycl_bf16* xperm=sycl::malloc_device<sycl_bf16>(bridge_elems,q);
    sycl_bf16* grouped_out=(xe2_grouped||xe2_grouped_mxfp4||xe2_dense_mxfp4||xe2_attention||xe2_gdn||od)
        ? sycl::malloc_device<sycl_bf16>(std::max(std::max(Rpad*std::max(H,2*I),
              size_t(M)*W),gdn_tokens*size_t(Hv)*Dv),q) : nullptr;
    sycl_bf16* moe_res=defer_moe_gather?sycl::malloc_device<sycl_bf16>(Rpad*H,q):nullptr;
    float* yperm=df(Rpad*H);
    int32_t* ptoken=sycl::malloc_device<int32_t>(R,q);
    int32_t* pinv=sycl::malloc_device<int32_t>(R,q);
    int32_t* grouped_rows=(xe2_grouped||xe2_grouped_mxfp4) ? sycl::malloc_shared<int32_t>(E,q) : nullptr;
    int32_t* grouped_atomic=(xe2_grouped||xe2_grouped_mxfp4) ? sycl::malloc_device<int32_t>(1,q) : nullptr;
    float* aux0=need_aux?df(size_t(M)*W):nullptr;
    float* aux1=need_aux?df(size_t(M)*W):nullptr;
    sycl_bf16* aux_xb=need_aux?sycl::malloc_device<sycl_bf16>(size_t(M)*W,q):nullptr;
    sycl_bf16* aux_out=need_aux?sycl::malloc_device<sycl_bf16>(size_t(M)*W,q):nullptr;
    std::vector<void*> mem={bh,bn,r0,r1,t0,t1,t2,t3,t4,la_fused,xb,bn_bf,dtok,rex,rwt,rlog,mh,alpha,beta,
        xperm,yperm,ptoken,pinv};
    if (next_tokens) mem.push_back(batch_logits);
    if(a8){mem.push_back(a8);mem.push_back(a8s);}
    if(grouped_out)mem.push_back(grouped_out);
    if(defer_moe_gather)mem.push_back(moe_res);
    if(need_aux){mem.push_back(aux0);mem.push_back(aux1);mem.push_back(aux_xb);mem.push_back(aux_out);}
    if(xe2_grouped||xe2_grouped_mxfp4){mem.push_back(grouped_rows);mem.push_back(grouped_atomic);}
    const size_t gdn_pitch=gdn_tokens;
    sycl_bf16* gdn_a=xe2_gdn_raw?sycl::malloc_device<sycl_bf16>(size_t(Hv)*gdn_pitch*64,q):nullptr;
    sycl_bf16* gdn_w=xe2_gdn_raw?sycl::malloc_device<sycl_bf16>(size_t(Hv)*gdn_pitch*Dk,q):nullptr;
    sycl_bf16* gdn_u=xe2_gdn_raw?sycl::malloc_device<sycl_bf16>(size_t(Hv)*gdn_pitch*Dv,q):nullptr;
    if(xe2_gdn_raw){mem.push_back(gdn_a);mem.push_back(gdn_w);mem.push_back(gdn_u);}
    int8_t* od_zp=od ? sycl::malloc_device<int8_t>(1,q) : nullptr;
    if(od_zp){const int8_t z=8;q.memcpy(od_zp,&z,1);mem.push_back(od_zp);}
    for (size_t mi=0;mi<mem.size();++mi) if (!mem[mi]) {
        // MUST match the `mem` initialiser above element for element.
        // It did not: bn_bf was missing, so every name from index 11 on
        // named the PREVIOUS buffer -- a null `alpha` was reported as
        // "beta", which sends you looking at the wrong allocation.
        static const char* const base_names[]={"bh","bn","r0","r1","t0","t1","t2","t3","t4",
            "la_fused","xb","bn_bf","dtok","rex","rwt","rlog","mh","alpha","beta","xperm",
            "yperm","ptoken","pinv"};
        static_assert(sizeof(base_names)/sizeof(base_names[0]) == 23,
                      "base_names must name every entry of the mem vector");
        const char* name=mi<sizeof(base_names)/sizeof(base_names[0])?base_names[mi]:"optional";
        std::fprintf(stderr,"    prefill allocation failed: %s (M=%d H=%d W=%d R=%d I=%d)\n",
                     name,M,H,W,R,I);
        for(void* z:mem) if(z) sycl::free(z,q); return false;
    }
    q.memcpy(dtok,tokens.data(),size_t(M)*sizeof(int32_t));
    if (pp_enabled() && pp_rank > 0) {
        if (!pp_recv_hidden(bh, size_t(M) * H)) {
            std::fprintf(stderr,"PP rank %d: batched hidden receive failed\n",pp_rank);
            for(void* z:mem) if(z) sycl::free(z,q);
            return false;
        }
        if (!pp_recv_taps(start_pos, M)) {
            std::fprintf(stderr,"PP rank %d: batched tap receive failed\n",pp_rank);
            for(void* z:mem) if(z) sycl::free(z,q);
            return false;
        }
    } else {
        if(!embed_rows(embed,dtok,bh,M)) throw std::runtime_error("batch embedding failed");
    }
    const int hcu[2]={0,M};
    if((xe2_attention||xe2_gdn||xe2_gdn_raw) && M>=2)q.memcpy(dtok,hcu,sizeof(hcu));
    q.memset(r0,0,size_t(M)*H*sizeof(float)); q.memset(r1,0,size_t(M)*H*sizeof(float));
    // The chunked GDN kernel works in whole 64-token chunks.  When M is not a
    // multiple of 64 the final chunk is partial, but the kernel still reads the
    // full tile, so any uninitialised tail of this scratch enters the delta-rule
    // matrix inverse as garbage -- frequently NaN/Inf bit patterns, after which
    // the solve never converges and the host spins on a batch that can never
    // retire.  Measured: M=128 passes, M=140 hangs.  Every benchmark size used
    // so far (1024, 4096) happened to be a multiple of 64, which is why this
    // stayed hidden while real prompts of arbitrary length hung.
    // Cost is one pass over ~85 MB at M=4096, about 0.15 ms.
    if(xe2_gdn_raw){
        q.memset(gdn_a,0,size_t(Hv)*gdn_pitch*64*sizeof(sycl_bf16));
        q.memset(gdn_w,0,size_t(Hv)*gdn_pitch*Dk*sizeof(sycl_bf16));
        q.memset(gdn_u,0,size_t(Hv)*gdn_pitch*Dv*sizeof(sycl_bf16));
        q.memset(alpha,0,gdn_tokens*Hv*sizeof(float));
        q.memset(beta,0,gdn_tokens*Hv*sizeof(float));
    }
    const bool prefill_graph = !pp_enabled() && !tp_enabled() &&
        std::getenv("GRIMOIRE_PREFILL_GRAPH") != nullptr;
    const bool profile_prefill = std::getenv("GRIMOIRE_PROFILE_PREFILL") != nullptr;
    if(profile_prefill && M>=64)
        std::fprintf(stderr,
            "    WARNING: GRIMOIRE_PROFILE_PREFILL at M=%d retains ~%d live\n"
            "    sycl::events and forces a profiling queue.  The Level Zero\n"
            "    adapter then stops batching commands and the host spins at\n"
            "    ~200%%%% CPU with the GPU idle -- this does not complete.\n"
            "    Use GRIMOIRE_TIME_LAYER=all (or =<n>) instead.\n",
            M,int(cfg.n_layers)*11);
    std::vector<std::pair<sycl::event,const char*>> pp_marks;
    // Host-side region timing.  GRIMOIRE_PROFILE_PREFILL retains one live
    // sycl::event per region (~700 at 64 layers) and forces a profiling-
    // enabled queue, which makes the Level Zero adapter emit host-visible
    // timestamped events and stop batching commands into reusable command
    // lists.  Past M=32 that degenerates into a userspace spin -- two threads
    // at 100% for minutes with the GPU idle, which is what stalled every
    // large profiled run in this project's history.  This path instead drains
    // the queue and reads the host clock, so it retains nothing and needs no
    // profiling queue.
    //   GRIMOIRE_TIME_LAYER=<n>    time only layer n (lowest perturbation)
    //   GRIMOIRE_TIME_LAYER=all    time every layer, print per-region sums
    int time_layer=-1; bool time_all=false;
    if(const char* v=std::getenv("GRIMOIRE_TIME_LAYER")){
        if(std::strcmp(v,"all")==0){time_all=true;} else {time_layer=std::atoi(v);}
    }
    const bool host_time=time_all||time_layer>=0;
    int cur_layer=-1; bool tl_active=false;
    std::chrono::high_resolution_clock::time_point tl_prev;
    std::map<std::string,double> tl_sums;
    std::vector<std::string> tl_order;
    auto pp_mark=[&](const char* completed_region){
        if(host_time){
            if(!tl_active) return;
            if(!(time_all||cur_layer==time_layer)) return;
            q.wait();
            const auto now=std::chrono::high_resolution_clock::now();
            const double ms=std::chrono::duration<double,std::milli>(now-tl_prev).count();
            if(tl_sums.find(completed_region)==tl_sums.end())
                tl_order.push_back(completed_region);
            tl_sums[completed_region]+=ms;
            if(!time_all)
                std::fprintf(stderr,"      [layer %d] %-28s %9.3f ms\n",
                             cur_layer,completed_region,ms);
            tl_prev=now;
            return;
        }
        if(prefill_graph || !profile_prefill)return;
        pp_marks.emplace_back(q.submit([&](sycl::handler& h){
            h.parallel_for(sycl::range<1>(1),[=](sycl::id<1>){});
        }),completed_region);
    };
    pp_mark("setup");
    struct OdPlan { int n,k; void* p; void* scratch; };
    std::vector<OdPlan> od_plans;
    // mm() prefers the FP32-output dense kernel for every projection, but only
    // the BF16-output kernel was ever autotuned (tools/autotune_b70_dense_qwen
    // is BF16-only; the f32 tuner was built and never run).  This flag routes
    // mm() through the tuned BF16 kernel plus an explicit convert so the two
    // epilogues can be compared on the full model.
    const bool no_f32_dense=std::getenv("GRIMOIRE_NO_F32_DENSE")!=nullptr;
    const bool exact_verify = tp_enabled() || (next_tokens &&
        std::getenv("GRIMOIRE_MTP_EXACT_VERIFY") != nullptr);
    const float* a8_cached_src = nullptr;
    const sycl_bf16* a8_cached_bf = nullptr;
    int a8_cached_k = 0;
    auto norm_rows=[&](float* h,const float* r0p,const float* r1p,
                       const bf16_t* weight,float* out){
        sycl::event ev;
        for(int r=0;r<M;++r){
            float* hr=h+int64_t(r)*H;
            float* yr=out+int64_t(r)*H;
            const float* a=r0p?r0p+int64_t(r)*H:nullptr;
            const float* b=r1p?r1p+int64_t(r)*H:nullptr;
            if(b) ev=launch_rmsnorm_residual2(q,hr,a,b,weight,yr,H,cfg.rms_eps,{});
            else ev=launch_rmsnorm_residual(q,hr,a,weight,yr,H,cfg.rms_eps,{});
        }
        return ev;
    };
    // WHY A VERIFY BATCH COSTS MORE THAN A DECODE STEP, and the fix.
    //
    // The short circuit below sent EVERY weight of a speculative verify batch
    // to the scalar FP32 GEMV.  That kernel streams the weight once, which is
    // why it looks right, but its FMA count is linear in M -- an extra
    // verified row is an extra full-model FP32 multiply-accumulate pass.
    // MEASURED 2026-09-06 (Qwen3.8-27B, 5993-token prompt, dense FFN region,
    // 64 layers, GRIMOIRE_TIME_LAYER=all):
    //     M=2 24.10 ms   M=3 29.99 ms   M=4 36.14 ms
    // exactly linear: 12.07 ms intercept (the weight sweep, ~470 GB/s) plus
    // 6.02 ms per extra row.  ~9 TFLOP/s of FP32 FMA, about half of what the
    // card can do in FP32 -- so this is not a bad GEMV, it is the wrong
    // ENGINE.  vLLM never pays it because its int4 linears run on the
    // systolic array, where 4 rows cost what 1 row costs.
    //
    // Ruled out first, so nobody repeats them: split-K width in the verify
    // attention (no change, see the note at launch_flash_decode_batched); the
    // int8 activation cache (GRIMOIRE_W4A8_NO_CACHE=1 reproduced the loss
    // byte for byte); rows-per-sub-group (B70_BATCH_RPS=8 moved the slope the
    // WRONG way, 6.02 -> 6.88, which is what killed the theory that the
    // re-read activation tile was the cost).
    //
    // The engine that works is w4a16: int4 weights, BF16 activations.  The
    // int8-activation W4A8 tile is also flat in M and also fast, but int8
    // activations cost draft acceptance -- 2.58 -> 1.88 committed/step,
    // 41/55 -> 30/61 accepts -- so it is never used for verify.  BF16
    // activations lose nothing: the accepted-draft counts, the rollbacks and
    // the emitted token stream are IDENTICAL to the FP32 GEMV path.
    // MEASURED, same prompt, 128 new tokens, k=3:
    //     FP32 GEMV verify   79.0 ms/round   35.7 tok/s
    //     w4a16   verify     55.2 ms/round   49.5 tok/s   (+39%)
    //     both: 42 steps, 3.12 committed/step, 89/103 accepts, 14 rollbacks
    //
    // Falls back to the exact GEMV, never to int8, if the oneDNN s4 bridge
    // is missing -- bridges fail silently in this project and a silent fall
    // back to int8 would look like a random acceptance regression.
    const bool verify_a16 = exact_verify &&
        !std::getenv("GRIMOIRE_VERIFY_EXACT_GEMV") && bool(load_onednn_s4());
    auto mm=[&](const DevQuant& w,const float* x,float* y){
        if(tp_enabled() && w.tp_sharded()) { gemm_tp(w,x,y,M); return; }
        if(exact_verify && M<=4 && !verify_a16){
            if(std::getenv("GRIMOIRE_VERIFY_MAP")){
                static std::set<std::pair<int,int>> seen;
                if(seen.insert({w.w.N,w.w.K}).second)
                    std::fprintf(stderr,
                        "      verify mm  N=%-6d K=%-6d i4=%d payload=%d\n",
                        w.w.N,w.w.K,int(w.has_i4()),int(w.w.payload!=nullptr));
            }
            if(w.has_i4()) {
                // One weight stream, M independent activation rows.  This is
                // the speculative-verification analogue of decode GEMV: it
                // avoids both activation requantization and loading the same
                // 4-bit matrix once per candidate.
                launch_gemv_int4sym_batch(q,w.i4,w.i4s,x,y,w.w.N,w.w.K,M,{});
            } else {
                for(int r=0;r<M;++r)
                    launch_gemv(q,w.w,x+size_t(r)*w.w.K,
                                y+size_t(r)*w.w.N,{});
            }
            return;
        }
        // Converted weights have no MXFP4 payload left, so they MUST take the
        // W4A8 path.  It is also the faster one: measured with the M16 tile,
        // la_qkv 2.78 ms vs 3.44 for the GEMV, and flat in M.
        // Small N at small M: a 128x256 tile producing 96 output rows is all
        // padding.  la_ab (N=96) cost 0.65 ms PER LAYER that way -- 31 ms of a
        // 71 ms four-token batch.  The decode GEMV does the same work in
        // 6.25 us per layer, so just run it once per row.
        // Small N at small M must use the GEMV, not a GEMM tile: N=1024 gives
        // 1024/256 = FOUR work-groups on a 32-Xe-core card.  The old guard
        // tested w.w.payload, which W4A8 FREES -- so converting a weight
        // silently pushed it onto the 4-work-group GEMM and cost more than
        // the conversion saved.  That is why speculation was slower than no
        // speculation on Ornith.
        // N GATE. Verify-sized batches stream the weight ONCE per chunk in the
        // batched decode GEMV -- the same kernel family that gives M=1 decode
        // its ~465 GB/s. The hardcoded N<=2048 sent every LARGE weight to the
        // prefill-tuned xe2_w4a8 GEMM instead, and the FFN at N=17408 is 9.1 of
        // the 14.5 GB a token reads. Measured 2026-09-05: that GEMM runs the
        // 4-row verify pass at 270 GB/s (54.2 ms/round, flat in M from 2 to 6)
        // because its tiling buys compute reuse a 4-row pass cannot use.
        // Tunable so the crossover can be swept without a rebuild.
        // EXPERIMENTAL, opt-in, UNTESTED as of 2026-09-05 -- see the kernel
        // comment in gemv_decode.cpp. Tried first and returns early so the
        // flag being unset leaves every path below bit-for-bit unchanged.
        static const bool nored = std::getenv("GRIMOIRE_GEMV_BATCH_NORED") != nullptr;
        if (nored && M<=16 && w.has_i4() && (w.w.K % 16)==0) {
            for(int r=0;r<M;r+=4){
                const int mb=std::min(4,M-r);
                launch_gemv_int4sym_batch_nored(q,w.i4,w.i4s,
                    x+size_t(r)*w.w.K,y+size_t(r)*w.w.N,
                    w.w.N,w.w.K,mb,{});
            }
            return;
        }
        static const int batch_max_n = [] {
            const char* e = std::getenv("GRIMOIRE_GEMV_BATCH_MAX_N");
            const int v = e ? std::atoi(e) : 2048;
            return v > 0 ? v : 2048;
        }();
        if(M<=16 && w.w.N<=batch_max_n){
            if(w.has_i4()){
                // BATCHED GEMV: weights loaded once per chunk, M dot products
                // against them.  Looping the single-row GEMV instead reloads
                // the weights M times -- that made M=4 18.7 -> 21.2 ms and
                // M=11 33.0 -> 44.3.  The batch kernel tops out at 4 rows, so
                // chunk it.
                for(int r=0;r<M;r+=4){
                    const int mb=std::min(4,M-r);
                    launch_gemv_int4sym_batch(q,w.i4,w.i4s,
                        x+size_t(r)*w.w.K,y+size_t(r)*w.w.N,
                        w.w.N,w.w.K,mb,{});
                }
                return;
            }
            if(w.w.payload){
                for(int r=0;r<M;++r)
                    launch_gemv(q,w.w,x+size_t(r)*w.w.K,y+size_t(r)*w.w.N,{});
                return;
            }
        }
        // ONEDNN W4A8 -- vLLM's own path, ported byte-for-byte in
        // xe2_onednn_bridge.cpp. vLLM routes every linear layer through
        // oneDNN at every batch size; GRIMOIRE sends verify batches to the
        // prefill-tuned xe2_w4a8 tile instead, measured at ~227 GB/s against
        // vLLM's ~500 on the identical workload with identical acceptance.
        // Opt-in via GRIMOIRE_ONEDNN_W4A8=1 so the proven path stays default.
        // ONEDNN W4A16 -- what vLLM actually runs for this model.
        // Its only int4 GEMMs are csrc/xpu/onednn/int4_gemm_w4a16.h and
        // int4_gemm_w4a8.h; there is no custom SYCL int4 GEMM in the tree.
        // The checkpoint is GPTQ-Int4 with 16-bit activations, so the live
        // path is w4a16. Its attribute setup is reproduced exactly in
        // SPlan (xe2_onednn_bridge.cpp): weight scales mask (1<<0)+(1<<1)
        // with {group_size,1} groups and set_fpmath_mode(bf16, true).
        // The one required difference is s4 instead of u4+zp=8, because
        // GRIMOIRE stores int4 sign-extended.
        //
        // GRIMOIRE instead sends verify batches to grimoire_xe2_dense_w4a8_f32
        // (its own kernel, int8 activations). MEASURED 2026-09-05 against the
        // live vLLM, identical model and prompt, equal ~2.5 tok/round:
        //     GRIMOIRE verify 61 ms device  (graph replay, so not dispatch)
        //     vLLM     round  29 ms
        // Opt-in via GRIMOIRE_ONEDNN_W4A16=1.
        // On by default for a verify batch (see the note on verify_a16 at the
        // top of mm); still opt-in for every other batch shape.
        static const bool od_a16_on = std::getenv("GRIMOIRE_ONEDNN_W4A16") != nullptr;
        if((od_a16_on || verify_a16) && w.has_i4() && xb && grouped_out
           && (w.w.K % 128)==0){
            static OneDnnS4Api oda = load_onednn_s4();
            if(oda){
                constexpr int GS = 128;
                const int N = w.w.N, K = w.w.K, G = K / GS;
                struct A16Ent { void* plan; void* scratch; sycl_bf16* wsc; int m; };
                static std::map<const float*, A16Ent> a16_plans;
                auto it = a16_plans.find(w.i4s);
                if(it == a16_plans.end() || it->second.m != M){
                    if(it != a16_plans.end()){
                        if(it->second.plan) oda.destroy(it->second.plan);
                        if(it->second.scratch) sycl::free(it->second.scratch,q);
                        if(it->second.wsc) sycl::free(it->second.wsc,q);
                        a16_plans.erase(it);
                    }
                    void* pl = oda.create(&q, M, N, K, GS, 1);
                    sycl_bf16* ts = nullptr; void* sc = nullptr;
                    if(pl){
                        const size_t sb = oda.scratch_size(pl);
                        sc = sb ? sycl::malloc_device<uint8_t>(sb,q) : nullptr;
                        ts = sycl::malloc_device<sycl_bf16>(size_t(G)*N,q);
                        if(ts){
                            const float* src = w.i4s;
                            q.parallel_for(sycl::range<2>(size_t(G),size_t(N)),
                                [=](sycl::id<2> id){
                                    ts[id[0]*size_t(N)+id[1]] =
                                        sycl_bf16(src[id[1]*size_t(G)+id[0]]);
                                }).wait();
                        }
                    }
                    it = a16_plans.emplace(w.i4s, A16Ent{pl,sc,ts,M}).first;
                }
                if(it->second.plan && it->second.wsc){
                    launch_f32_to_bf16(q,x,xb,size_t(M)*size_t(K));
                    oda.execute(it->second.plan, xb, w.i4, it->second.wsc,
                                grouped_out, it->second.scratch);
                    launch_bf16_to_f32(q,grouped_out,y,size_t(M)*size_t(N));
                    return;
                }
            }
        }
        // The w4a16 plan can fail to build (bridge absent, unsupported shape).
        // A verify batch must land back on the EXACT GEMV when that happens,
        // not on the int8 tile below, which is fast but costs acceptance.
        if(verify_a16 && M<=kSpecBatch){
            if(w.has_i4())
                launch_gemv_int4sym_batch(q,w.i4,w.i4s,x,y,w.w.N,w.w.K,M,{});
            else
                for(int r=0;r<M;++r)
                    launch_gemv(q,w.w,x+size_t(r)*w.w.K,y+size_t(r)*w.w.N,{});
            return;
        }
        if(w.has_i4() && xe2_w4a8_f32 && a8 && a8s){
            static const bool dbg=std::getenv("GRIMOIRE_W4A8_DEBUG")!=nullptr;
            if(dbg){std::printf("    [mm] N=%d K=%d M=%d quantize...",
                                w.w.N,w.w.K,M);std::fflush(stdout);}
            // The cache is keyed on the SOURCE POINTER. Verify reuses the
            // t0..t4 scratch buffers, so a different tensor can land at the
            // same address with the same K and silently reuse stale int8
            // rows. GRIMOIRE_W4A8_NO_CACHE=1 disables the reuse to test
            // whether that -- rather than int8 precision -- is what collapses
            // MTP acceptance to 0 when W4A8 is on without EXACT_VERIFY.
            static const bool a8_no_cache =
                std::getenv("GRIMOIRE_W4A8_NO_CACHE") != nullptr;
            if (a8_no_cache || a8_cached_src != x || a8_cached_k != w.w.K) {
                launch_quantize_rows_int8(q,x,a8,a8s,M,w.w.K,{});
                a8_cached_src = a8_no_cache ? nullptr : x;
                a8_cached_k = w.w.K;
                a8_cached_bf = nullptr;
            }
            if(dbg){q.wait();std::printf(" ok, gemm...");std::fflush(stdout);}
            xe2_w4a8_f32(&q,a8,w.i4,w.i4s,a8s,y,M,w.w.N,w.w.K);
            if(dbg){q.wait();std::printf(" ok\n");std::fflush(stdout);}
            return;
        }
        const sycl_bf16* x_bf=bn_bf;
        if(x!=bn){launch_f32_to_bf16(q,x,xb,size_t(M)*w.w.K);x_bf=xb;}
        if(!no_f32_dense && xe2_dense_mxfp4_f32 && w.w.fmt==Fmt::MXFP4){
            xe2_dense_mxfp4_f32(&q,x_bf,w.w.payload,
                             static_cast<const unsigned char*>(w.w.scales),y,
                             M,w.w.N,w.w.K);
            return;
        }
        if(xe2_dense_mxfp4 && grouped_out && w.w.fmt==Fmt::MXFP4){
            xe2_dense_mxfp4(&q,x_bf,w.w.payload,
                             static_cast<const unsigned char*>(w.w.scales),grouped_out,
                             M,w.w.N,w.w.K);
            launch_bf16_to_f32(q,grouped_out,y,size_t(M)*w.w.N);
            return;
        }
        if(od && od_zp && grouped_out && w.od_w4){
            auto it=std::find_if(od_plans.begin(),od_plans.end(),[&](const OdPlan& p){
                return p.n==w.w.N && p.k==w.w.K;});
            if(it==od_plans.end()){
                void* p=od.create(&q,M,w.w.N,w.w.K,kInt4Group,1);
                if(p){size_t bytes=od.scratch_size(p);void* scratch=bytes?sycl::malloc_device<uint8_t>(bytes,q):nullptr;
                    od_plans.push_back({w.w.N,w.w.K,p,scratch});it=od_plans.end()-1;}
            }
            if(it!=od_plans.end()){
                od.execute(it->p,x_bf,w.payload,w.od_scales,od_zp,grouped_out,it->scratch);
                launch_bf16_to_f32(q,grouped_out,y,size_t(M)*w.w.N);
                return;
            }
        }
        // NO-XMX FALLBACK, opt-in, for correctness testing off the card.
        // launch_gemm_xmx() is joint_matrix from end to end, so on a device
        // without matrix hardware the whole batched prefill is unreachable --
        // which is why DAY-ONE lists the batched path as untestable off the
        // Tower, and why a gemma-4 batched prefill had nowhere to be checked.
        // launch_gemm_batched() is the plain-SYCL batched GEMM that already
        // exists in prefill.cpp; it computes the same product from the f32
        // activations and runs anywhere.
        //
        // This is CORRECTNESS ONLY and it is SLOW -- it is a sub-group GEMV
        // per output row, not a tile.  It is gated on the env var AND on the
        // device actually lacking matrix support, so a B70 never reaches it
        // and the Tower path stays byte-identical.  Rule 8: nothing timed
        // here means anything.
        if(noxmx_gemm){ launch_gemm_batched(q,w.w,x,y,M); return; }
        launch_gemm_xmx(q,w.w,x_bf,y,M);
    };
    // bf16-activation twins of mm().  Several fast paths call the dense MXFP4
    // GEMM directly instead of going through mm(), and a converted weight has
    // no MXFP4 payload left -- touching it is a DEVICE_LOST, which is exactly
    // how o_proj took the device down.
    auto mmb=[&](const DevQuant& w,const sycl_bf16* x_bf,float* y){
        a8_cached_src=nullptr;
        if(w.has_i4() && xe2_w4a8_f32 && a8 && a8s){
            if(a8_cached_bf!=x_bf){
                launch_quantize_rows_int8_bf16(q,x_bf,a8,a8s,M,w.w.K,{});
                a8_cached_bf=x_bf;
            }
            xe2_w4a8_f32(&q,a8,w.i4,w.i4s,a8s,y,M,w.w.N,w.w.K);
            return;
        }
        xe2_dense_mxfp4_f32(&q,x_bf,w.w.payload,
            static_cast<const unsigned char*>(w.w.scales),y,M,w.w.N,w.w.K);
    };
    auto mmbb=[&](const DevQuant& w,const sycl_bf16* x_bf,sycl_bf16* y){
        a8_cached_src=nullptr;
        if(w.has_i4() && xe2_w4a8_bf16 && a8 && a8s){
            if(a8_cached_bf!=x_bf){
                launch_quantize_rows_int8_bf16(q,x_bf,a8,a8s,M,w.w.K,{});
                a8_cached_bf=x_bf;
            }
            xe2_w4a8_bf16(&q,a8,w.i4,w.i4s,a8s,y,M,w.w.N,w.w.K);
            return;
        }
        xe2_dense_mxfp4(&q,x_bf,w.w.payload,
            static_cast<const unsigned char*>(w.w.scales),y,M,w.w.N,w.w.K);
    };
    auto wait_on=[](sycl::queue& target,const sycl::event& dependency){
        return target.submit([&](sycl::handler& h){
            h.depends_on(dependency);
            h.single_task([=](){});
        });
    };
    auto mm_aux=[&](const DevQuant& w,const float* x,float* y){
        launch_f32_to_bf16(q_aux,x,aux_xb,size_t(M)*w.w.K);
        xe2_dense_mxfp4(&q_aux,aux_xb,w.w.payload,
            static_cast<const unsigned char*>(w.w.scales),aux_out,
            M,w.w.N,w.w.K);
        return launch_bf16_to_f32(q_aux,aux_out,y,size_t(M)*w.w.N);
    };
    auto mlp_bf16=[&](const DevQuant& gu,const DevQuant& down,float* y,int layer=-1){
        // Small N at small M belongs on the GEMV, not a 256-wide GEMM tile.
        // Ornith's shared expert is gate_up [1024,2048] / down [2048,512]:
        // through a GEMM that is 4 and 8 work-groups on a 32-Xe-core card,
        // and it measured 2.29 ms against the decode path's 0.78 for the SAME
        // single token.  Decline here and let the caller's mm() route it.
        if(M<=16 && gu.w.N<=2048 && down.w.N<=2048) return false;
        a8_cached_src=nullptr;
        if(exact_verify) return false;
        // BesTLA prefill path: gate and up as two separate GEMMs (it has no fused
        // 2*inter weight), SwiGLU over the two halves of grouped_out, then down.
        if(g_bestla_ready && layer>=0 && M>=g_bestla_min_m && grouped_out && xb){
            const int inter=gu.w.N/2;
            // BesTLA returns its own tensors; use them in place rather than copying.
            void *pg=nullptr,*pu=nullptr,*pd=nullptr;
            if(!g_bestla_ffn(&q,layer,0,bn_bf,&pg,M) &&
               !g_bestla_ffn(&q,layer,1,bn_bf,&pu,M) && pg && pu){
                launch_swiglu_bf16_split(q,static_cast<const sycl_bf16*>(pg),
                                         static_cast<const sycl_bf16*>(pu),xb,M,inter);
                if(!g_bestla_ffn(&q,layer,2,xb,&pd,M) && pd){
                    launch_bf16_to_f32(q,static_cast<const sycl_bf16*>(pd),y,
                                       size_t(M)*down.w.N);
                    return true;
                }
            }
            // any failure falls through to the cutlass path below
        }
        // W4A8: int8 activations x symmetric int4 weights on the native s8xs4
        // DPAS.  Prefill only -- decode never comes through here, and its GEMV
        // is ~4x faster than any GEMM at M=1.
        if(xe2_w4a8_bf16&&xe2_w4a8_f32&&grouped_out&&xb&&a8&&layer>=0&&
           L[layer].sh_gu_i4&&L[layer].sh_dn_i4){
            const LayerDev& ld=L[layer];
            const int inter=gu.w.N/2;
            static const bool dbg=std::getenv("GRIMOIRE_W4A8_DEBUG")!=nullptr;
            static bool once=false;
            if(dbg&&!once){once=true;
                std::printf("\n  [w4a8] M=%d  gu N=%d K=%d  down N=%d K=%d  W=%d\n",
                    M,gu.w.N,gu.w.K,down.w.N,down.w.K,W);
                std::printf("  [w4a8] a8 elems=%zu need gu=%zu down=%zu\n",
                    size_t(M)*std::max(H,W),size_t(M)*gu.w.K,size_t(M)*down.w.K);
                std::fflush(stdout);}
            auto step=[&](const char* what){ if(!dbg)return; q.wait();
                std::printf("  [w4a8] ok: %s\n",what); std::fflush(stdout); };
            if(a8_cached_bf!=bn_bf){
                launch_quantize_rows_int8_bf16(q,bn_bf,a8,a8s,M,gu.w.K,{});
                a8_cached_bf=bn_bf;
            }
            step("quantize A for gate_up");
            xe2_w4a8_bf16(&q,a8,ld.sh_gu_i4,ld.sh_gu_ws,a8s,grouped_out,
                          M,gu.w.N,gu.w.K);
            step("gate_up GEMM");
            pp_mark("  FFN gate_up GEMM");
            launch_swiglu_bf16_quant(q,grouped_out,xb,a8,a8s,M,inter);
            a8_cached_bf=xb;
            step("swiglu");
            pp_mark("  FFN swiglu");
            step("quantize A for down");
            xe2_w4a8_f32(&q,a8,ld.sh_dn_i4,ld.sh_dn_ws,a8s,y,
                         M,down.w.N,down.w.K);
            step("down GEMM");
            pp_mark("  FFN down GEMM");
            return true;
        }
        // A CONVERTED weight still reports fmt == MXFP4 -- only its payload is
        // freed -- so the format check alone does not protect this fallback.
        // Test the pointer, or this faults the device.
        if(!xe2_dense_mxfp4||!xe2_dense_mxfp4_f32||!grouped_out||gu.w.fmt!=Fmt::MXFP4||
           down.w.fmt!=Fmt::MXFP4||!gu.payload||!down.payload)return false;
        const int inter=gu.w.N/2;
        xe2_dense_mxfp4(&q,bn_bf,gu.w.payload,
            static_cast<const unsigned char*>(gu.w.scales),grouped_out,
            M,gu.w.N,gu.w.K);
        pp_mark("  FFN gate_up GEMM");
        launch_swiglu_bf16(q,grouped_out,xb,M,inter);
        pp_mark("  FFN swiglu");
        xe2_dense_mxfp4_f32(&q,xb,down.w.payload,
            static_cast<const unsigned char*>(down.w.scales),y,
            M,down.w.N,down.w.K);
        pp_mark("  FFN down GEMM");
        return true;
    };
    std::unique_ptr<sycl_ext::command_graph<sycl_ext::graph_state::modifiable>> pg;
    if(prefill_graph){
        q.wait();
        pg=std::make_unique<sycl_ext::command_graph<sycl_ext::graph_state::modifiable>>(
            q.get_context(),q.get_device());
        pg->begin_recording(q);
    }
    int prefill_layer_begin=pp_enabled()?pp_begin:0;
    int prefill_layer_limit=pp_enabled()?pp_end:cfg.n_layers;
    if(const char* v=std::getenv("GRIMOIRE_PREFILL_LAYER_LIMIT"))
        prefill_layer_limit=std::max(1,std::min(cfg.n_layers,std::atoi(v)));
    const bool prefill_host_progress=
        std::getenv("GRIMOIRE_PREFILL_HOST_PROGRESS") != nullptr;
    // MoVA routing table, read back once per sparse layer (K2 only).
    std::vector<int32_t> mova_idx_host;
    std::vector<float>   mova_wt_host;
    int raw_gdn_layer_limit=cfg.n_layers;
    if(const char* v=std::getenv("GRIMOIRE_RAW_GDN_LAYER_LIMIT"))
        raw_gdn_layer_limit=std::max(0,std::min(cfg.n_layers,std::atoi(v)));
    // ALLOCATED IS NOT THE SAME AS EXISTING (rule 7b, again).  This used
    // to read next_tokens && M <= kSpecBatch, on the assumption that a
    // caller asking for per-row tokens is always a speculative verify and
    // therefore always has a drafter -- so spec_hidden_steps was always
    // allocated.  A batched decode asks for per-row tokens with no
    // drafter at all, and the memcpy below took a null pointer straight
    // into the runtime.  Test the buffer, not the caller's shape.
    const bool capture_spec =
        next_tokens && M <= kSpecBatch && spec_hidden_steps && (!seqb || seqb->verify);
    const bool spec_route_diag = capture_spec &&
        std::getenv("GRIMOIRE_MTP_ROUTE_DIAG") != nullptr;
    size_t spec_route_total = 0, spec_route_unique = 0;
    int spec_route_layers = 0;
    std::vector<size_t> spec_doff(L.size(), size_t(-1));
    std::vector<size_t> spec_xoff(L.size(), size_t(-1));
    std::vector<size_t> spec_coff(L.size(), size_t(-1));
    if (capture_spec) {
        size_t ds = 0, xs = 0, cs = 0;
        const size_t dn_n = size_t(Hv) * Dv * Dk;
        for (size_t li = 0; li < L.size(); ++li) {
            if (L[li].dn_state) { spec_doff[li] = ds; ds += dn_n; }
            if (L[li].conv_ring) {
                spec_coff[li] = cs; cs += size_t(qkv_ch)*(cfg.conv_kernel-1);
                spec_xoff[li] = xs;
                xs += size_t(kSpecBatch) * qkv_ch;
            }
        }
    }
    for(int li=prefill_layer_begin;li<prefill_layer_limit;++li){
        a8_cached_src=nullptr;
        a8_cached_bf=nullptr;
        cur_layer=li;
        if(host_time&&(time_all||li==time_layer)){
            q.wait();
            tl_prev=std::chrono::high_resolution_clock::now();
            tl_active=true;
        }
        LayerDev& d=L[li];
        sycl::event input_bf_ready;
        if(defer_moe_gather&&li>prefill_layer_begin)
            input_bf_ready=launch_rmsnorm_moe_residual_batched(q,bh,moe_res,pinv,
                rwt,r1,d.in_norm,nullptr,bn_bf,M,cfg.top_k,H,cfg.rms_eps);
        else if(exact_verify)
            input_bf_ready=norm_rows(bh,r0,r1,d.in_norm,bn);
        else input_bf_ready=launch_rmsnorm_residual_batched(
            q,bh,r0,r1,d.in_norm,norm_bf_only?nullptr:bn,M,H,cfg.rms_eps,bn_bf);
        if(exact_verify && debug && li==probe_layer)
            probe("L0 in_norm",bn,H);
        pp_mark("input norm");
        // Match vLLM's target_layer_id + 1 convention.  bh is the completed
        // residual stream after the previous layer. Preserve every row, not
        // just the final token, because context K/V covers the accepted span.
        if(dflash2.target_aux){
            for(size_t tap=0;tap<dflash2.target_layers.size();++tap){
                if(dflash2.target_layers[tap]+1==li){
                    if(seqb) {
                        for(int r=0;r<M;++r)
                            launch_dflash_store_tap(q,bh+int64_t(r)*H,
                                draft_slots[size_t(seqb->slot[r])].aux,1,H,
                                int(dflash2.target_layers.size()),seqb->pos[r],int(tap),{},
                                cfg.is_muse);
                    } else launch_dflash_store_tap(q,bh,dflash2.target_aux,M,H,
                        int(dflash2.target_layers.size()),start_pos,int(tap),{},cfg.is_muse);
                    break;
                }
            }
        }
        if(d.kind==LayerKind::LINEAR_ATTN){
            const int qs=Hk*Dk, vs=Hv*Dv, ch=d.la_qkv.output_rows();
            // The fused projection is a separately quantized concat of qkv,
            // z and gates. It is fast for prompt prefill but is not numerically
            // identical to decode's three projection weights; repeated M=1
            // verification with it drifts into repetition. Exact speculative
            // verification must consume the same weights as decode.
            const bool fused_in=!exact_verify&&d.la_all.payload&&d.la_all.w.N==12352;
            const bool native_rec=(xe2_gdn_raw||xe2_gdn)&&pos==0&&M>=64&&
                (!xe2_gdn_raw||li<raw_gdn_layer_limit);
            const bool bf_dn_qkv=std::getenv("GRIMOIRE_BF16_DN_QKV")&&
                native_rec&&!fused_in&&xe2_dense_mxfp4&&
                d.la_qkv.w.fmt==Fmt::MXFP4;
            const size_t native_qe=gdn_tokens*size_t(Hk)*Dk;
            sycl_bf16* native_fq=native_rec?xperm:nullptr;
            sycl_bf16* native_fk=native_rec?xperm+native_qe:nullptr;
            sycl_bf16* native_fv=native_rec?xperm+2*native_qe:nullptr;
            sycl::event aux_ab_ready,aux_z_ready;
            // RULE 1: GRIMOIRE_W4A8 FREES the MXFP4 payload of a converted
            // weight, and mm_aux() dereferences w.w.payload directly on
            // q_aux instead of routing through mm().  A converted weight
            // still reports fmt==MXFP4, so only the POINTER is a safe test.
            // la_z is 256-aligned and IS converted; falling back to the
            // serial mm() path costs the aux overlap and keeps the card on
            // the bus.  parallel_dn is re-tested at both consumption sites,
            // so narrowing it here keeps all three in agreement.
            const bool parallel_dn=parallel_prefill&&!fused_in&&
                d.la_ab.w.payload&&d.la_z.w.payload;
            if(parallel_dn){
                wait_on(q_aux,input_bf_ready);
                aux_ab_ready=mm_aux(d.la_ab,bn,aux0);
                aux_z_ready=mm_aux(d.la_z,bn,aux1);
            }
            if(fused_in){mm(d.la_all,bn,la_fused);
                launch_split_dn_fused_projections(q,la_fused,t0,mh,rlog,M);
                pp_mark("DN fused qkv+z+gates");}
            else if(bf_dn_qkv){
                // This was the LAST direct-payload site left unrouted, and it
                // is gated on GRIMOIRE_BF16_DN_QKV -- which Ornith sets and
                // Qwen does not.  So Qwen never took it while Ornith did, and
                // with W4A8 the la_qkv payload is freed: dereferencing it took
                // GPU1 off the PCI bus (forcewake 0xFFFFFFFF).
                mmbb(d.la_qkv,bn_bf,grouped_out);
                pp_mark("DN qkv projection");
            }else {mm(d.la_qkv,bn,t0);pp_mark("DN qkv projection");}
            if (capture_spec)
                q.memcpy(spec_conv_inputs + spec_xoff[li], t0,
                         size_t(M) * ch * sizeof(float));
            if(seqb){
                // ONE ROW AT A TIME, each over its OWN ring.  The ring is
                // this layer's whole memory of the conversation -- the
                // last kernel-1 inputs -- so rows sharing one would each
                // convolve over the others' tokens.  Fluent, wrong, and
                // nothing in the output says so.
                //
                // tokens=1 makes the same kernel do exactly what decode
                // does: read the ring for all K-1 history taps, then
                // shift this token in.  Calling the batched routine with
                // one row rather than writing a second one is what keeps
                // the two from drifting apart.
                for(int r=0;r<M;++r){
                    float* ring=d.conv_base+size_t(seqb->slot[r])*d.conv_slot;
                    ConvParams cp{t0+int64_t(r)*ch,d.la_conv,ring,nullptr,
                                  ch,cfg.conv_kernel};
                    launch_causal_conv1d_split_prefill(q,cp,1,
                        t1+int64_t(r)*qs,t2+int64_t(r)*qs,t3+int64_t(r)*vs,
                        nullptr,qs,vs);
                    if(capture_spec) q.memcpy(batch_conv_steps+size_t(r)*spec_conv_elems+
                        spec_coff[li],ring,d.conv_slot*sizeof(float));
                }
            }else if(bf_dn_qkv)
                launch_causal_conv1d_split_bf16_prefill(q,grouped_out,d.la_conv,
                    d.conv_ring,ch,cfg.conv_kernel,M,native_fq,native_fk,native_fv,
                    qs,vs);
            else{
                ConvParams cp{t0,d.la_conv,d.conv_ring,nullptr,ch,cfg.conv_kernel};
                launch_causal_conv1d_split_prefill(q,cp,M,t1,t2,t3,native_fv,qs,vs);
            }
            pp_mark("DN causal conv + split");
            if(!native_rec){
                launch_l2norm_heads(q,t1,M*Hk,Dk,{});
                launch_l2norm_heads(q,t2,M*Hk,Dk,{});
                pp_mark("DN qk norm");
            }
            float* ab_in=nullptr;
            if(fused_in)ab_in=rlog;
            else if(parallel_dn){wait_on(q,aux_ab_ready);ab_in=aux0;}
            else {mm(d.la_ab,bn,t0);pp_mark("DN gate projection");ab_in=t0;}
            const sycl_bf16* recurrence_bf=nullptr;
            if(native_rec){
                launch_deltanet_native_gates(q,ab_in,alpha,beta,M,Hv,
                                             int64_t(gdn_tokens));
                q.memset(native_fq+size_t(M)*qs,0,size_t(63)*qs*sizeof(sycl_bf16));
                q.memset(native_fk+size_t(M)*qs,0,size_t(63)*qs*sizeof(sycl_bf16));
                q.memset(native_fv+size_t(M)*vs,0,size_t(63)*vs*sizeof(sycl_bf16));
                const size_t qe=size_t(M)*Hk*Dk,ve=size_t(M)*Hv*Dv;
                sycl_bf16* fq=native_fq;sycl_bf16* fk=native_fk;sycl_bf16* fv=native_fv;
                if(bf_dn_qkv)launch_l2norm_heads_pair_bf16_io(q,fq,fk,M*Hk,Dk);
                else launch_l2norm_heads_pair_bf16(q,t1,t2,fq,fk,M*Hk,Dk);
                pp_mark("DN qk norm");
                launch_bf16_to_f32(
                    q, reinterpret_cast<const sycl_bf16*>(d.la_Alog), t4, Hv);
                q.memset(reinterpret_cast<uint8_t*>(dtok)+2*sizeof(int32_t),0,1);
                auto* has=reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(dtok)+2*sizeof(int32_t));
                if(xe2_gdn_raw)
                    xe2_gdn_raw(&q,grouped_out,fq,fk,fv,gdn_a,gdn_w,gdn_u,beta,alpha,t4,
                        d.la_dtb,d.dn_state,Dv*Dk,dtok,dtok,has,int(gdn_tokens),
                        Hk,Dk,Hv,Dv);
                else
                    xe2_gdn(&q,grouped_out,fq,fk,fv,beta,alpha,t4,d.la_dtb,d.dn_state,
                        M,Hk,Dk,Hv,Dv,dtok,dtok,has);
                if(xe2_dense_mxfp4_f32&&d.la_out.w.fmt==Fmt::MXFP4)
                    recurrence_bf=grouped_out;
                else launch_bf16_to_f32(q,grouped_out,t0,ve);
            }else{
                launch_deltanet_gates_batched(q,ab_in,d.la_Alog,d.la_dtb,alpha,beta,M,Hv);
                // The chunked prefill recurrence costs ~150 us/layer at M=4 --
                // 7.2 ms across 48 layers -- while the DECODE step does the
                // same work in 8.75 us/layer.  The delta rule is sequential in
                // the token dimension anyway, so at small M just run the decode
                // kernel once per token: 4 tokens = 1.7 ms instead of 7.2.
                // This is what makes an MTP verify batch affordable.
                if(seqb||M<=16){
                    for(int t=0;t<M;++t){
                        DeltaNetParams sp{};
                        sp.q     = t1 + size_t(t)*Hk*Dk;
                        sp.k     = t2 + size_t(t)*Hk*Dk;
                        sp.v     = t3 + size_t(t)*Hv*Dv;
                        sp.a     = alpha + size_t(t)*Hv;
                        sp.beta  = beta  + size_t(t)*Hv;
                        // The one line that makes a hybrid model
                        // concurrent.  Rows of a batch are different
                        // conversations, so each advances ITS OWN delta
                        // state; rows of a prompt are one conversation
                        // and advance the bound slot's, as before.
                        sp.state = seqb
                            ? d.dn_base+size_t(seqb->slot[t])*d.dn_slot
                            : d.dn_state;
                        sp.out   = t0 + size_t(t)*Hv*Dv;
                        sp.n_heads = Hv; sp.k_dim = Dk; sp.v_dim = Dv;
                        sp.n_k_heads = Hk;
                        launch_deltanet_step(q,sp,{});
                        if (capture_spec)
                            q.memcpy(spec_dn_steps + size_t(t) * spec_dn_elems +
                                     spec_doff[li], sp.state,
                                     size_t(Hv) * Dv * Dk * sizeof(float));
                    }
                }else{
                    DeltaNetPrefillParams dp{t1,t2,t3,alpha,beta,d.dn_state,t0,Hv,Dk,Dv,M,Hk};
                    launch_deltanet_prefill(q,dp);
                }
            }
            pp_mark("DN recurrence");
            float* z_in=nullptr;
            if(fused_in)z_in=mh;
            else if(parallel_dn){wait_on(q,aux_z_ready);z_in=aux1;}
            else {mm(d.la_z,bn,t3);pp_mark("DN z projection");z_in=t3;}
            if(recurrence_bf){
                launch_rmsnorm_gate_silu_bf16_io(
                    q,recurrence_bf,z_in,d.la_norm,xb,M*Hv,Dv,cfg.rms_eps);
                mmb(d.la_out,xb,r0);
            }else{
                launch_rmsnorm_gate_silu(q,t0,z_in,d.la_norm,M*Hv,Dv,cfg.rms_eps,{});
                mm(d.la_out,t0,r0);
            }
            pp_mark("DN norm + output projection");
        }else{
            const int QD=d.q_proj.output_rows();
            const bool gated=QD==2*cfg.n_heads*d.head_dim;
            const bool bfqkv=std::getenv("GRIMOIRE_BF16_QKV")&&xe2_attention&&
                xe2_dense_mxfp4&&xe2_dense_mxfp4_f32&&pos==0&&M>=32&&
                d.q_proj.w.fmt==Fmt::MXFP4&&d.k_proj.w.fmt==Fmt::MXFP4&&
                d.v_proj.w.fmt==Fmt::MXFP4&&d.o_proj.w.fmt==Fmt::MXFP4&&
                // a MoVA layer has no v_proj at all; it must take the
                // routed path below, not this fused bf16 one.
                !d.k2_sparse &&
                // launch_qk_norm_rope_bf16_batched only knows partial_rope.
                // A proportional-RoPE layer taking this path would rotate
                // with the wrong frequencies and the wrong pairing, and
                // nothing downstream would notice.
                !d.rope_proportional;
            if(bfqkv){
                const size_t qe=size_t(M)*cfg.n_heads*d.head_dim;
                const size_t ke=size_t(M)*d.kv_heads*d.head_dim;
                sycl_bf16*fq=xperm,*fk=fq+qe,*fv=fk+ke;
                mmbb(d.q_proj,bn_bf,grouped_out);
                if(gated)launch_split_qgate_bf16(q,grouped_out,fq,t2,M,
                    cfg.n_heads,d.head_dim);
                else q.memcpy(fq,grouped_out,qe*sizeof(sycl_bf16));
                mmbb(d.k_proj,bn_bf,fk);
                mmbb(d.v_proj,bn_bf,fv);
                launch_qk_norm_rope_bf16_batched(q,fq,fk,d.q_norm,d.k_norm,M,
                    cfg.n_heads,d.kv_heads,d.head_dim,pos,d.rope_theta,
                    d.partial_rope,cfg.rms_eps);
                launch_kv_append_bf16_batched(q,fk,fv,d.k_cache,d.v_cache,M,pos,
                    d.kv_heads,d.head_dim,max_seq);
                xe2_attention(&q,fq,fk,fv,grouped_out,M,M,cfg.n_heads,
                    d.kv_heads,d.head_dim,dtok,dtok,
                    cfg.attn_softmax_scale(d.head_dim),true);
                const sycl_bf16*o_in=grouped_out;
                if(gated){launch_gate_sigmoid_mul_bf16_io(q,grouped_out,t2,xb,qe);o_in=xb;}
                mmb(d.o_proj,o_in,r0);
            }else{
            mm(d.q_proj,bn,t0);
            pp_mark("attn q proj");
            float* qv=t0;
            if(gated){launch_split_qgate_batched(q,t0,t1,t2,M,cfg.n_heads,d.head_dim);qv=t1;}
            mm(d.k_proj,bn,t3);
            if (d.k2_sparse) {
                // One readback for the whole batch instead of two stalls
                // per token.  The MoE routing scratch is reused: MoVA runs
                // in the attention half and the FFN recomputes its routing
                // before using it -- see mova_value_batched.  If that
                // scratch is not wide enough for the MoVA expert count,
                // fall back rather than overrun it.
                const int NV = d.kv_heads * d.head_dim;
                if (cfg.mova_experts <= std::max(1, cfg.n_experts) &&
                    cfg.mova_top_k <= alloc_top_k) {
                    mova_value_batched(d, bn, t4, M, rlog, rex, rwt,
                                       mova_idx_host, mova_wt_host);
                } else {
                    for (int m = 0; m < M; ++m)
                        mova_value_m1(d, bn + size_t(m) * cfg.hidden,
                                      t4 + size_t(m) * NV, {});
                    q.wait();
                }
            } else mm(d.v_proj,bn,t4);
            pp_mark("attn kv proj");
            if(seqb){
                // ONE ROW AT A TIME, because each row sits at its own
                // position in its own conversation.  The SAME kernels the
                // batched arm uses, called with tokens=1: a row rotated by
                // a different routine from the one the rest of the engine
                // uses would be wrong in a way that is fluent, and calling
                // the batched kernel with one row cannot drift from it.
                //
                // Only the attention section is per-row.  Everything
                // before and after -- every projection, the FFN, the
                // router, the head -- stays batched, and that is where
                // the weights are read.
                const int QH=cfg.n_heads*d.head_dim, KH=d.kv_heads*d.head_dim;
                for(int r=0;r<M;++r){
                    const int P=seqb->pos[r];
                    uint8_t* kc=d.k_base+size_t(seqb->slot[r])*d.kv_slot;
                    uint8_t* vc=d.v_base+size_t(seqb->slot[r])*d.kv_slot;
                    float* qr=qv+int64_t(r)*QH;
                    float* kr=t3+int64_t(r)*KH;
                    float* vr=t4+int64_t(r)*KH;
                    if(d.rope_proportional)
                        launch_qk_norm_rope_proportional_batched(q,qr,kr,d.q_norm,
                            d.k_norm,1,cfg.n_heads,d.kv_heads,d.head_dim,P,
                            d.rope_theta,d.partial_rope,cfg.rms_eps,{},1.0f,
                            d.rope_factor);
                    else
                        launch_qk_norm_rope_batched(q,qr,kr,d.q_norm,d.k_norm,1,
                            cfg.n_heads,d.kv_heads,d.head_dim,P,d.rope_theta,
                            d.partial_rope,cfg.rms_eps);
                    launch_kv_append_batched(q,kr,vr,kc,vc,1,P,d.kv_heads,
                        d.head_dim,max_seq);
                }
            }else{
            if(d.rope_proportional)
                launch_qk_norm_rope_proportional_batched(q,qv,t3,d.q_norm,
                    d.k_norm,M,cfg.n_heads,d.kv_heads,d.head_dim,pos,
                    d.rope_theta,d.partial_rope,cfg.rms_eps,{},1.0f,
                    d.rope_factor);
            else
                launch_qk_norm_rope_batched(q,qv,t3,d.q_norm,d.k_norm,M,cfg.n_heads,
                    d.kv_heads,d.head_dim,pos,d.rope_theta,d.partial_rope,cfg.rms_eps);
            launch_kv_append_batched(q,t3,t4,d.k_cache,d.v_cache,M,pos,d.kv_heads,
                d.head_dim,max_seq);
            }
            pp_mark("attn rope + kv append");
            const sycl_bf16* attention_bf=nullptr;
            if(seqb){
                // Each row attends to ITS OWN conversation and to nothing
                // else.  This is the one place a batched decode differs
                // from a batched prefill in kind rather than in index: a
                // prefill's rows are consecutive and see each other, these
                // rows are strangers.  Any kernel that let row r read row
                // r-1's keys would produce fluent text from the wrong
                // conversation.
                //
                // Attention is per-sequence work: batching cannot make a
                // row read fewer of its own keys, so looping here costs M
                // launches and no extra bytes.  The saving this path is
                // for is in the weights, which the batch already shares.
                const int QH=cfg.n_heads*d.head_dim;
                // pos[r] + 1, which is set_cursor()'s convention: the
                // count INCLUDES the entry this row just appended, so a
                // token attends to itself exactly as it does in
                // single-token decode.  Getting this off by one is the
                // difference between a model that reads its own last
                // token and one that does not, and both are fluent.
                {
                    std::vector<int32_t> lens(size_t(M), 0);
                    for(int r=0;r<M;++r)lens[size_t(r)]=seqb->pos[r]+1;
                    q.memcpy(dtok,lens.data(),size_t(M)*sizeof(int32_t)).wait();
                }
                for(int r=0;r<M;++r){
                    AttnParams ap{};
                    ap.q=qv+int64_t(r)*QH;
                    ap.k_cache=d.k_base+size_t(seqb->slot[r])*d.kv_slot;
                    ap.v_cache=d.v_base+size_t(seqb->slot[r])*d.kv_slot;
                    ap.out=t3+int64_t(r)*QH;
                    ap.seq_len=seqb->pos[r]+1;ap.seq_cap=max_seq;
                    ap.head_dim=d.head_dim;ap.num_heads=cfg.n_heads;
                    ap.num_kv_heads=d.kv_heads;
                    ap.softmax_scale=cfg.attn_softmax_scale(d.head_dim);
                    ap.partials=s.part;ap.part_m=s.pm;ap.part_l=s.pl;
                    ap.splits=GRAPH_SPLITS;ap.d_seq_len=dtok+r;
                    launch_flash_decode(q,ap,{});
                    launch_flash_merge(q,ap,{});
                }
            }else if(xe2_attention && pos==0 && M>=32){
                const size_t qe=size_t(M)*cfg.n_heads*d.head_dim;
                const size_t ke=size_t(M)*d.kv_heads*d.head_dim;
                sycl_bf16* fq=xperm; sycl_bf16* fk=fq+qe; sycl_bf16* fv=fk+ke;
                launch_f32_to_bf16(q,qv,fq,qe);
                launch_f32_to_bf16(q,t3,fk,ke);
                launch_f32_to_bf16(q,t4,fv,ke);
                xe2_attention(&q,fq,fk,fv,grouped_out,M,M,cfg.n_heads,
                    d.kv_heads,d.head_dim,dtok,dtok,
                    cfg.attn_softmax_scale(d.head_dim),true);
                if(xe2_dense_mxfp4_f32&&d.o_proj.w.fmt==Fmt::MXFP4)
                    attention_bf=grouped_out;
                else launch_bf16_to_f32(q,grouped_out,t3,qe);
            }else{
                if(next_tokens && M<=kSpecBatch &&
                   !std::getenv("GRIMOIRE_LEGACY_VERIFY_ATTN")){
                    // Split-K must track context depth here for the same
                    // reason it does in single-token decode: at 4778 tokens
                    // GRAPH_SPLITS(8) left verify at 60 ms/round -- 82% of
                    // speculative decode time -- because each split walked
                    // ~600 keys serially. The workspace is sized for
                    // MAX_SPLITS, and the merge skips empty splits.
                    // Measured: widening verify split-K gave no gain (verify is
                    // a 4-token weight-bound forward pass, 57 vs 60 ms/round)
                    // and wider splits cost draft acceptance through extra
                    // partial-softmax merge error. Keep the proven floor.
                    // RE-TESTED 2026-09-06 and still no gain, now with the
                    // depth-scaled split count the single-token path uses
                    // (attention.cpp decode_splits, 32 keys/split).  At 6k
                    // context that raises the verify split count from 8 to
                    // MAX_SPLITS(128); the "attn flash" region measured
                    // 20.58 ms at 8 splits and 20.45 ms at 128, and end-to-end
                    // tg32 moved 28.6 -> 28.8.  This kernel is not
                    // split-K-bound.  Keep the proven floor.
                    const int vsplits = GRAPH_SPLITS;
                    launch_flash_decode_batched(q,qv,d.k_cache,d.v_cache,t3,
                        M,pos,cfg.n_heads,d.kv_heads,d.head_dim,max_seq,
                        cfg.attn_softmax_scale(d.head_dim),s.part,s.pm,s.pl,
                        vsplits,{});
                }else if(exact_verify){
                        const int start=pos;
                        q.submit([&](sycl::handler& h){
                            h.parallel_for(sycl::range<1>(size_t(M)),[=](sycl::id<1> id){
                                // Match the live decode convention exactly. The
                                // device length is the number of cache entries
                                // visible to this query; row r sees preceding
                                // speculative rows, but not its own KV entry.
                                dtok[id[0]]=start+int(id[0]);
                            });
                        });
                        for(int r=0;r<M;++r){
                            AttnParams ap{};
                            ap.q=qv+int64_t(r)*cfg.n_heads*d.head_dim;
                            ap.k_cache=d.k_cache;ap.v_cache=d.v_cache;
                            ap.out=t3+int64_t(r)*cfg.n_heads*d.head_dim;
                            ap.seq_len=pos+r+1;ap.seq_cap=max_seq;
                            ap.head_dim=d.head_dim;ap.num_heads=cfg.n_heads;
                            ap.num_kv_heads=d.kv_heads;
                            ap.softmax_scale=cfg.attn_softmax_scale(d.head_dim);
                            ap.partials=s.part;ap.part_m=s.pm;ap.part_l=s.pl;
                            ap.splits=GRAPH_SPLITS;ap.d_seq_len=dtok+r;
                            launch_flash_decode(q,ap,{});
                            launch_flash_merge(q,ap,{});
                        }
                }else launch_flash_prefill(q,qv,d.k_cache,d.v_cache,t3,M,
                    next_tokens ? pos - 1 : pos,cfg.n_heads,
                    d.kv_heads,d.head_dim,max_seq,cfg.attn_softmax_scale(d.head_dim));
            }
            pp_mark("attn flash");
            if(attention_bf){
                const sycl_bf16* o_in=attention_bf;
                if(gated){launch_gate_sigmoid_mul_bf16_io(q,attention_bf,t2,xb,
                    size_t(M)*cfg.n_heads*d.head_dim);o_in=xb;}
                mmb(d.o_proj,o_in,r0);
            }else{
                if(gated) launch_gate_sigmoid_mul(q,t3,t2,M*cfg.n_heads*d.head_dim,{});
                mm(d.o_proj,t3,r0);
            }
            }
            pp_mark("full attention");
        }
        const bool fused_ffn_quant=!exact_verify&&!d.moe_layer&&a8&&a8s&&
            xe2_w4a8_bf16&&d.sh_gu_i4&&d.sh_dn_i4;
        auto post_bf_ready=exact_verify
            ? norm_rows(bh,r0,nullptr,d.post_norm,bn)
            : fused_ffn_quant
                ? launch_rmsnorm_residual_batched_quant(q,bh,r0,nullptr,d.post_norm,
                    norm_bf_only?nullptr:bn,bn_bf,a8,a8s,M,H,cfg.rms_eps)
                : launch_rmsnorm_residual_batched(
                    q,bh,r0,nullptr,d.post_norm,norm_bf_only?nullptr:bn,M,H,cfg.rms_eps,bn_bf);
        a8_cached_src=nullptr;
        a8_cached_bf=fused_ffn_quant?bn_bf:nullptr;
        if(d.moe_layer){
            sycl::event shared_ready;
            // RULE 1: as parallel_dn above -- mm_aux() reads sh_gu/sh_down's
            // MXFP4 payload, which W4A8 frees.  This MUST stay a single
            // boolean: the wait below re-tests the same condition, and
            // skipping the projection while still waiting on a default-
            // constructed event would leave r1 unwritten (silent garbage).
            const bool shared_aux=(parallel_prefill||parallel_shared)&&
                d.sh_gu.w.payload&&d.sh_down.w.payload;
            if(shared_aux){
                wait_on(q_aux,post_bf_ready);
                mm_aux(d.sh_gu,bn,aux0);
                launch_swiglu_batched(q_aux,aux0,aux1,M,d.sh_gu.w.N/2);
                shared_ready=mm_aux(d.sh_down,aux1,r1);
                if(d.has_sh_gate){
                    mm_aux(d.sh_gate_q,bn,aux0);
                    shared_ready=launch_scale_by_sigmoid_batched(
                        q_aux,r1,aux0,M,H);
                }
            }
            // the bf16 router kernel softmaxes the top-k and has no bias
            // input, so it cannot express K2's routing.
            const bool bf16_router=!tp_enabled() && !cfg.is_k2&&std::getenv("GRIMOIRE_BF16_ROUTER")&&
                xe2_dense_mxfp4&&d.router.w.fmt==Fmt::MXFP4&&d.router.payload;
            if(bf16_router){
                xe2_dense_mxfp4(&q,bn_bf,d.router.w.payload,
                    static_cast<const unsigned char*>(d.router.w.scales),grouped_out,
                    M,d.router.w.N,d.router.w.K);
                launch_router_topk_bf16_batched(q,grouped_out,M,cfg.n_experts,
                    cfg.top_k,rex,rwt,true);
            }else{
                mm(d.router,bn,rlog);
                // K2: sigmoid scores, bias steers SELECTION ONLY, plain
                // sum-normalise, then router_scaling_factor.  The generic
                // kernel ranks raw logits and softmaxes the top-k -- a
                // different mixture entirely.
                if (cfg.is_k2)
                    launch_router_topk_k2(q, rlog, d.router_bias, M,
                        cfg.n_experts, cfg.top_k, rex, rwt,
                        cfg.norm_topk_prob, cfg.router_scale);
                else
                    launch_router_topk_batched(q,rlog,M,cfg.n_experts,cfg.top_k,rex,rwt,true);
            }
            if (spec_route_diag) {
                std::vector<int32_t> routes(size_t(M) * cfg.top_k);
                q.memcpy(routes.data(), rex,
                         routes.size() * sizeof(int32_t)).wait();
                std::sort(routes.begin(), routes.end());
                const size_t unique = size_t(std::distance(
                    routes.begin(), std::unique(routes.begin(), routes.end())));
                spec_route_total += routes.size();
                spec_route_unique += unique;
                ++spec_route_layers;
            }
            pp_mark("post norm + route");
            // A bad route turns the remap's SLM count into an arbitrary write,
            // and the resulting fault is only reported at the following grouped
            // GEMM.  Allow the real-prompt failure to be bisected before either
            // kernel can consume the table.  The check is intentionally opt-in:
            // it synchronizes and copies M*top_k integers to the host.
            if(const char* check=std::getenv("GRIMOIRE_MOE_ROUTE_CHECK")){
                const int check_layer=std::atoi(check);
                if(check_layer==li){
                    std::vector<int32_t> hroutes(static_cast<size_t>(R));
                    q.memcpy(hroutes.data(),rex,size_t(R)*sizeof(int32_t)).wait();
                    std::vector<int32_t> hcount(size_t(cfg.n_experts),0);
                    int bad=0,duplicate=0,max_rows=0,active=0;
                    for(int t=0;t<M;++t){
                        for(int s=0;s<cfg.top_k;++s){
                            const int e=hroutes[size_t(t)*cfg.top_k+s];
                            if(e<0||e>=cfg.n_experts){++bad;continue;}
                            ++hcount[size_t(e)];
                            for(int p=0;p<s;++p)
                                if(hroutes[size_t(t)*cfg.top_k+p]==e){
                                    ++duplicate;break;
                                }
                        }
                    }
                    for(int n:hcount){if(n){++active;max_rows=std::max(max_rows,n);}}
                    std::fprintf(stderr,
                        "    MoE route check layer %d: rows=%d active=%d max=%d bad=%d duplicate=%d\n",
                        li,R,active,max_rows,bad,duplicate);
                    std::fflush(stderr);
                    if(bad)
                        return false;
                }
            }
            // device_can_matrix, NOT M alone.  Everything under this
            // branch ends in launch_gemm_xmx -- directly, not through
            // mm() -- and launch_gemm_xmx is joint_matrix from end to
            // end.  mm() has had the no-matrix fallback for a while;
            // these two call sites bypass mm() entirely and never got
            // it, so on a device without matrix hardware a MoE prompt of
            // 32 tokens or more reached a JIT that cannot compile the
            // kernel.  The symptom was a SIGSEGV inside Intel's runtime
            // about half the time, or an argmax over a buffer nothing
            // wrote -- "engine returned an invalid token".  Never on a
            // B70, which is why it survived: it is the off-card
            // verification path that was broken, so MoE batched prefill
            // was the one thing that could not be checked here.
            //
            // Rule 18, again: the first guard found is not the last.
            //   grep -n 'launch_gemm_xmx' src/grimoire.cpp
            // is the whole check, and it is what found these two.
            //
            // Below 32 rows, and now on any device that cannot run the
            // tile, the plain-SYCL launch_moe_*_batched pair does the
            // same arithmetic and runs anywhere.  A B70 can, so the
            // Tower path is unchanged.
            if(M>=32 && device_can_matrix(q)){
                if(xe2_grouped_mxfp4 && d.moe.gate_up.fmt==Fmt::MXFP4){
                    launch_moe_remap_bf16_top8(q,bn_bf,rex,xperm,grouped_rows,
                                                ptoken,pinv,M,H,cfg.n_experts);
                    pp_mark("MoE device remap");
                    if(const char* check=std::getenv("GRIMOIRE_MOE_ROUTE_CHECK")){
                        const int check_layer=std::atoi(check);
                        if(check_layer==li){
                            std::vector<int32_t> hroutes(static_cast<size_t>(R));
                            std::vector<int32_t> hrows(static_cast<size_t>(cfg.n_experts));
                            std::vector<int32_t> htoken(static_cast<size_t>(R));
                            std::vector<int32_t> hinv(static_cast<size_t>(R));
                            q.memcpy(hroutes.data(),rex,size_t(R)*sizeof(int32_t));
                            q.memcpy(hrows.data(),grouped_rows,
                                     size_t(cfg.n_experts)*sizeof(int32_t));
                            q.memcpy(htoken.data(),ptoken,size_t(R)*sizeof(int32_t));
                            q.memcpy(hinv.data(),pinv,size_t(R)*sizeof(int32_t)).wait();
                            std::vector<int32_t> expected(size_t(cfg.n_experts),0);
                            for(int e:hroutes) if(e>=0&&e<cfg.n_experts)
                                ++expected[size_t(e)];
                            std::vector<unsigned char> seen(size_t(R),0);
                            int row_mismatch=0,bad_perm=0,duplicate_perm=0;
                            int inverse_mismatch=0,sum_rows=0,offset=0;
                            for(int e=0;e<cfg.n_experts;++e){
                                if(hrows[size_t(e)]!=expected[size_t(e)])
                                    ++row_mismatch;
                                sum_rows+=hrows[size_t(e)];
                                for(int p=offset;p<offset+hrows[size_t(e)]&&p<R;++p){
                                    const int t=htoken[size_t(p)];
                                    if(t<0||t>=M) ++bad_perm;
                                }
                                offset+=hrows[size_t(e)];
                            }
                            for(int r=0;r<R;++r){
                                const int p=hinv[size_t(r)];
                                if(p<0||p>=R){++bad_perm;continue;}
                                if(seen[size_t(p)]) ++duplicate_perm;
                                seen[size_t(p)]=1;
                                if(htoken[size_t(p)]!=r/cfg.top_k)
                                    ++inverse_mismatch;
                            }
                            const int missing_perm=static_cast<int>(std::count(
                                seen.begin(),seen.end(),static_cast<unsigned char>(0)));
                            std::fprintf(stderr,
                                "    MoE remap check layer %d: sum=%d row_mismatch=%d "
                                "bad_perm=%d duplicate_perm=%d missing_perm=%d inverse_mismatch=%d\n",
                                li,sum_rows,row_mismatch,bad_perm,duplicate_perm,
                                missing_perm,inverse_mismatch);
                            std::fflush(stderr);
                            if(row_mismatch||sum_rows!=R||bad_perm||duplicate_perm||
                               missing_perm||inverse_mismatch||
                               std::getenv("GRIMOIRE_MOE_ROUTE_CHECK_STOP"))
                                return false;
                        }
                    }
                    q.memset(grouped_atomic,0,sizeof(int32_t));
                    sycl_bf16* moe_act=xperm;
                    sycl_bf16* moe_down_out=grouped_out;
                    if(xe2_fused_gate_up){
                      xe2_fused_gate_up(&q,xperm,d.gu_pack,d.gu_scale,grouped_out,I,H,
                                       grouped_rows,cfg.n_experts,grouped_atomic);
                      moe_act=grouped_out; moe_down_out=xperm;
                      pp_mark("MoE fused gate+up+SwiGLU");
                    }else{
                      xe2_grouped_mxfp4(&q,xperm,d.gu_pack,d.gu_scale,grouped_out,
                                  2*I,H,grouped_rows,nullptr,cfg.n_experts,grouped_atomic);
                      pp_mark("MoE gate+up GEMM");
                      launch_swiglu_bf16(q,grouped_out,xperm,R,I);
                      pp_mark("MoE SwiGLU");
                    }
                    q.memset(grouped_atomic,0,sizeof(int32_t));
                    if(defer_moe_gather)moe_down_out=moe_res;
                    xe2_grouped_mxfp4(&q,moe_act,d.dn_pack,d.dn_scale,moe_down_out,
                                H,I,grouped_rows,nullptr,cfg.n_experts,grouped_atomic);
                    pp_mark("MoE down GEMM");
                    if(!defer_moe_gather)
                        launch_moe_unpermute_bf16(q,moe_down_out,pinv,rwt,r0,
                                                  M,cfg.top_k,H);
                    pp_mark("MoE gather");
                }else{
                    std::vector<int32_t> hex(R), hp(R), hi(R), count(cfg.n_experts,0), off(cfg.n_experts+1,0);
                    q.memcpy(hex.data(),rex,size_t(R)*sizeof(int32_t)).wait();
                    for(int x:hex) if(x>=0&&x<cfg.n_experts) ++count[x];
                    for(int e=0;e<cfg.n_experts;++e) off[e+1]=off[e]+count[e];
                    std::vector<int> cur=off;
                    for(int r=0;r<R;++r){int e=hex[r];int p=cur[e]++;hp[p]=r/cfg.top_k;hi[r]=p;}
                    q.memcpy(ptoken,hp.data(),size_t(R)*sizeof(int32_t));
                    q.memcpy(pinv,hi.data(),size_t(R)*sizeof(int32_t));
                    launch_permute_rows_bf16(q,bn,ptoken,xperm,R,H);
                    if(xe2_grouped && d.xe2_signed_int4){
                    std::copy(count.begin(),count.end(),grouped_rows);
                    xe2_grouped(&q,xperm,d.gu_pack,d.gu_scale,grouped_out,
                                2*I,H,grouped_rows,nullptr,cfg.n_experts,
                                kInt4Group,grouped_atomic);
                    launch_swiglu_bf16(q,grouped_out,xperm,R,I);
                    xe2_grouped(&q,xperm,d.dn_pack,d.dn_scale,grouped_out,
                                H,I,grouped_rows,nullptr,cfg.n_experts,
                                kInt4Group,grouped_atomic);
                    launch_moe_unpermute_bf16(q,grouped_out,pinv,rwt,r0,
                                              M,cfg.top_k,H);
                    }else{
                    auto sub=[&](const QuantWeight& w,int row0,int n){
                        return slice_quant_rows(w,row0,n);
                    };
                    for(int e=0;e<cfg.n_experts;++e) if(count[e]){
                        auto w=sub(d.moe.gate_up,e*2*I,2*I);
                        launch_gemm_xmx(q,w,xperm+int64_t(off[e])*H,t0+int64_t(off[e])*2*I,count[e]);
                    }
                    launch_swiglu_batched(q,t0,mh,R,I);
                    launch_f32_to_bf16(q,mh,xperm,size_t(R)*I);
                    for(int e=0;e<cfg.n_experts;++e) if(count[e]){
                        auto w=sub(d.moe.down,e*H,H);
                        launch_gemm_xmx(q,w,xperm+int64_t(off[e])*I,yperm+int64_t(off[e])*H,count[e]);
                    }
                    launch_moe_unpermute(q,yperm,pinv,rwt,r0,M,cfg.top_k,H);
                    }
                }
            }else{
                if(tp_enabled()) {
                    const int begin=d.expert_begin, end=begin+d.expert_count;
                    q.parallel_for(sycl::range<1>(size_t(R)),[=](sycl::id<1> ix) {
                        const int g=rex[ix[0]];
                        const bool own=g>=begin && g<end;
                        rex[ix[0]]=own?g-begin:-1;
                        if(!own) rwt[ix[0]]=0.0f;
                    });
                }
                launch_moe_gate_up_batched(q,d.moe,rex,bn,mh,M);
                launch_moe_down_batched(q,d.moe,rex,rwt,mh,r0,M);
            }
            if(tp_enabled() && !tp_allreduce_sum(r0,M*H))
                throw std::runtime_error("TP batched MoE reduction failed");
            pp_mark("routed MoE");
            if(shared_aux) wait_on(q,shared_ready);
            else {
                if(!mlp_bf16(d.sh_gu,d.sh_down,r1,li)){
                    const int SI=d.sh_gu.output_rows()/2;
                    mm(d.sh_gu,bn,t0);launch_swiglu_batched(q,t0,t1,M,SI);mm(d.sh_down,t1,r1);
                }
                if(d.has_sh_gate){mm(d.sh_gate_q,bn,t2);launch_scale_by_sigmoid_batched(q,r1,t2,M,H);}
            }
            pp_mark("shared expert");
        }else{
            if(!mlp_bf16(d.sh_gu,d.sh_down,r0,li)){
                const int FI=d.sh_gu.output_rows()/2;
                mm(d.sh_gu,bn,t0); launch_swiglu_batched(q,t0,t1,M,FI); mm(d.sh_down,t1,r0);
            }
            pp_mark("dense FFN");
            q.memset(r1,0,size_t(M)*H*sizeof(float));
        }
        if(prefill_host_progress){
            q.wait_and_throw();
            std::fprintf(stderr,"    prefill host progress: layer %d/%d complete\n",
                li+1,prefill_layer_limit);
            std::fflush(stderr);
        }
    }
    if (pp_enabled() && pp_rank < pp_world-1) {
        // Fold the last early-layer FFN output into the residual stream before
        // sending the complete MxH boundary tensor to the late-stage rank.
        if (defer_moe_gather)
            launch_moe_unpermute_bf16(q,moe_res,pinv,rwt,r0,
                                      M,cfg.top_k,H);
        launch_add(q,bh,r0,M*H,{});
        launch_add(q,bh,r1,M*H,{});
        if (!pp_send_hidden(bh, size_t(M) * H)) {
            std::fprintf(stderr,"PP rank %d: batched hidden send failed\n",pp_rank);
            for(void* z:mem) if(z) sycl::free(z,q);
            return false;
        }
        if (!pp_send_taps(start_pos, M)) {
            std::fprintf(stderr,"PP rank %d: batched tap send failed\n",pp_rank);
            for(void* z:mem) if(z) sycl::free(z,q);
            return false;
        }
    } else {
      if(defer_moe_gather)
          launch_rmsnorm_moe_residual_batched(q,bh,moe_res,pinv,rwt,r1,fnorm,bn,
              nullptr,M,cfg.top_k,H,cfg.rms_eps);
      else if(exact_verify) norm_rows(bh,r0,r1,fnorm,bn);
      else launch_rmsnorm_residual_batched(q,bh,r0,r1,fnorm,bn,M,H,cfg.rms_eps);
      if(prefill_host_progress){q.wait_and_throw();
          std::fprintf(stderr,"    prefill stage: final norm done\n");std::fflush(stderr);}
      if (next_tokens) {
        // Verification needs the main-model choice after every candidate,
        // not only after the last row. Keep the reductions and copies on the
        // in-order queue, then return all token ids in one host transfer.
        if (tp_enabled()) {
            mm(lm_head,bn,batch_logits);
        } else if (lm_head.has_i4()) {
            if (!exact_verify && xe2_w4a8_f32 && a8 && a8s) {
                launch_quantize_rows_int8(q, bn, a8, a8s, M, H, {});
                xe2_w4a8_f32(&q, a8, lm_head.i4, lm_head.i4s, a8s,
                    batch_logits, M, lm_head.w.N, lm_head.w.K);
            } else {
                launch_gemv_int4sym_batch(q, lm_head.i4, lm_head.i4s, bn,
                    batch_logits, lm_head.w.N, lm_head.w.K, M, {});
            }
        } else if (lm_head.w.fmt == Fmt::MXFP4 && lm_head.w.payload) {
            // Verification is a matrix multiplication, not M independent
            // decode GEMVs.  Load the large vocabulary matrix once per batch.
            mm(lm_head, bn, batch_logits);
        }
        for (int r = 0; r < M; ++r) {
            float* row = batch_logits + int64_t(r) * cfg.vocab;
            if (!tp_enabled() && !lm_head.has_i4() && lm_head.w.fmt != Fmt::MXFP4)
                launch_gemv(q, lm_head.w, bn + int64_t(r) * H, row, {});
            launch_argmax(q, row, cfg.vocab, s.d_tok, s.d_val, {});
            q.memcpy(dtok + r, s.d_tok, sizeof(int32_t));
        }
      } else {
          gemv_any(lm_head,bn+int64_t(M-1)*H,s.logits,{});
      }
      if(prefill_host_progress){q.wait_and_throw();
          std::fprintf(stderr,"    prefill stage: lm_head gemv done\n");std::fflush(stderr);}
      pp_mark("final norm + logits");
    }
    if(prefill_graph){
        const auto graph_build0=std::chrono::high_resolution_clock::now();
        pg->end_recording(q);
        auto exec=pg->finalize();
        const auto graph_replay0=std::chrono::high_resolution_clock::now();
        q.ext_oneapi_graph(exec).wait();
        if(std::getenv("GRIMOIRE_PREFILL_GRAPH_TIME")){
            const auto graph_done=std::chrono::high_resolution_clock::now();
            const double build_ms=std::chrono::duration<double,std::milli>(
                graph_replay0-graph_build0).count();
            const double replay_ms=std::chrono::duration<double,std::milli>(
                graph_done-graph_replay0).count();
            std::printf("    prefill graph M=%d: finalize %.3f ms, replay %.3f ms\n",
                        M,build_ms,replay_ms);
        }
    }
    // mtp_draft consumes the unnormalised hidden state of the last processed
    // token. Sequential forward() leaves it in s.h; the batched path must do
    // the same so speculation can chain across verify steps.
    // BRACES.  Without them the memcpy was conditional and the flag was
    // not, so a verify batch wider than kSpecBatch marked a buffer valid
    // that nothing had written -- the exact bug the flag exists to stop,
    // and on a B70 (where batched prefill works) it is reachable.
    if (capture_spec) {
        q.memcpy(spec_hidden_steps, bh, size_t(M) * H * sizeof(float));
        spec_hidden_valid = true;
    }
    if (seqb) {
        // NOTHING about the engine's single cursor is meaningful here.
        // The rows belong to M different conversations, so there is no
        // "the last token" whose hidden state s.h could hold and no
        // position to advance to; the caller tracks a position per
        // sequence.  Writing either would leave a plausible value that
        // the next single-sequence call would believe.
        ++g_batch_decode_steps;
        g_batch_decode_rows += M;
    } else {
        q.memcpy(s.h, bh + int64_t(M-1) * H, size_t(H) * sizeof(float));
        pos+=M; set_cursor(pos);
    }
    if (next_tokens) {
        next_tokens->resize(M);
        // dtok is only computed where the head runs.  Under PP that is
        // the last stage; the earlier stages leave the buffer alone and
        // take the answer from the backward hop below.
        if (!pp_enabled() || pp_rank == pp_world - 1)
            q.memcpy(next_tokens->data(), dtok, size_t(M) * sizeof(int32_t));
    }
    q.wait();
    if (next_tokens && pp_enabled() && !pp_sync_tokens(*next_tokens)) {
        std::fprintf(stderr, "PP rank %d: verified-token sync failed\n", pp_rank);
        for(void* z:mem) if(z) sycl::free(z,q);
        return false;
    }

    // Populate MTP context in one batch using the same format-aware matrix
    // path as target prefill. Verification replaces draft-conditioned K/V with
    // target-conditioned K/V, including after partial acceptance. Rejected
    // future rows are outside the cursor and will be overwritten next round.
    if(mtp.ok && !seqb) {
        std::vector<int32_t> shifted(size_t(M),0);
        for(int i=0;i<M;++i)
            shifted[i]=next_tokens?(*next_tokens)[i]:(i+1<M?tokens[size_t(i)+1]:0);
        q.memcpy(dtok,shifted.data(),size_t(M)*sizeof(int32_t));
        launch_embed_batched(q,embed,dtok,r0,M,H,{});
        launch_rmsnorm_residual_batched(q,r0,nullptr,nullptr,mtp.pre_e,bn,M,H,cfg.rms_eps);
        launch_rmsnorm_residual_batched(q,bh,nullptr,nullptr,mtp.pre_h,r1,M,H,cfg.rms_eps);
        q.parallel_for(sycl::range<2>(M,H),[=](sycl::id<2> id) {
            const size_t row=id[0],col=id[1];
            t0[row*2*H+col]=bn[row*H+col];
            t0[row*2*H+H+col]=r1[row*H+col];
        });
        mm(mtp.fc,t0,t1);
        launch_rmsnorm_residual_batched(q,t1,nullptr,nullptr,mtp.L.in_norm,r0,M,H,cfg.rms_eps);
        mm(mtp.L.k_proj,r0,t0);mm(mtp.L.v_proj,r0,t2);
        launch_qk_norm_rope_batched(q,nullptr,t0,nullptr,mtp.L.k_norm,M,0,
            mtp.L.kv_heads,mtp.L.head_dim,start_pos,mtp.L.rope_theta,
            mtp.L.partial_rope,cfg.rms_eps);
        launch_kv_append_batched(q,t0,t2,mtp.L.k_cache,mtp.L.v_cache,M,start_pos,
            mtp.L.kv_heads,mtp.L.head_dim,max_seq);
        set_cursor(pos);q.wait_and_throw();
    }

    if (!next_tokens && start_pos == 0) save_prefix(tokens);
    if(host_time&&!tl_sums.empty()){
        double tot=0; for(const auto& kv:tl_sums) tot+=kv.second;
        std::printf("    host region budget (%d tokens, %s):\n",M,
                    time_all?"all layers":"single layer");
        for(const auto& name:tl_order)
            std::printf("      %-28s %9.3f ms  %5.1f%%\n",name.c_str(),
                        tl_sums[name],100.0*tl_sums[name]/tot);
        std::printf("      %-28s %9.3f ms\n","TOTAL (timed regions)",tot);
    }
    if(pp_marks.size()>1){
        std::map<std::string,double> sums;
        for(size_t i=1;i<pp_marks.size();++i){
            const uint64_t a=pp_marks[i-1].first.get_profiling_info<sycl::info::event_profiling::command_end>();
            const uint64_t b=pp_marks[i].first.get_profiling_info<sycl::info::event_profiling::command_start>();
            sums[pp_marks[i].second]+=double(b-a)*1e-6;
        }
        std::printf("    device prefill breakdown (%d tokens):\n",M);
        for(const auto& kv:sums)std::printf("      %-24s %9.3f ms\n",kv.first.c_str(),kv.second);
    }
    if (spec_route_diag && spec_route_layers) {
        std::printf("    MTP route overlap: %zu unique / %zu selections "
                    "across %d layers (%.2fx reuse)\n",
                    spec_route_unique, spec_route_total, spec_route_layers,
                    spec_route_unique ? double(spec_route_total) /
                        double(spec_route_unique) : 0.0);
    }
    for(auto& p:od_plans){if(p.scratch)sycl::free(p.scratch,q);od.destroy(p.p);}
    for(void* p:mem) sycl::free(p,q);
    if(prefill_host_progress){
        std::fprintf(stderr,"    prefill stage: returning\n");std::fflush(stderr);}
    return true;
}

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// Record the decode step once. Every kernel launch in forward() becomes
// a node in the graph; replaying it submits the whole 40-layer sequence
// as a single command list, which removes the per-launch submission cost
// that dominated the step (measured: ~800 launches, ~40% of the budget).
// ---------------------------------------------------------------------
bool Grimoire::build_graph() {
    // Socket send/receive is deliberately outside SYCL graph capture.
    if (pp_enabled() || tp_enabled()) return false;
    if (dag) return false;
    // Qwen4-Exp's decode is NOT capture-safe, and the way it fails is
    // the worst kind: capture bakes every host-side argument into the
    // recorded node, and forward_qwen4_exp() has several.  The token
    // itself reaches launch_embed by value (the other paths read
    // s.d_tok while recording); the PLE layer writes q4_tok[pos] with a
    // captured pos; the QSA indexer ropes at a captured position, picks
    // its compressed-key block from pos/ratio, and stores its raw key at
    // a captured offset.  A replayed graph would decode position 0 with
    // token 1 forever -- fluently.
    //
    // Refusing is the honest fix until someone moves those onto the
    // device cursor AND has a number saying it is worth it; nobody has
    // measured this model on a B70 at all yet (rule 8).
    if (cfg.is_qwen4_exp) return false;
    // Saved OUTSIDE the try: capture moves the live sequence state before
    // recording, and the catch has to be able to put it back.  Declared in
    // the try they were out of scope exactly where they were needed.
    int32_t saved_pos = 0, saved_seq = 0;
    q.memcpy(&saved_pos, s.d_pos, sizeof(int32_t)).wait();
    q.memcpy(&saved_seq, s.d_seq_len, sizeof(int32_t)).wait();
    const int host_pos = pos;
    try {
        sycl_ext::command_graph<sycl_ext::graph_state::modifiable>
            g(q.get_context(), q.get_device());

        // Record against a scratch position so the recording itself does
        // not advance the real sequence state.  build_graph() is called
        // AFTER prefill in the generate path, so the live sequence state
        // must survive capture untouched -- on the failure path too.
        const int32_t zero = 0;
        q.memcpy(s.d_pos, &zero, sizeof(int32_t)).wait();
        const int32_t one = 1;
        q.memcpy(s.d_seq_len, &one, sizeof(int32_t)).wait();

        recording = true;
        g.begin_recording(q);
        forward(1);
        g.end_recording(q);
        recording = false;

        q.memcpy(s.d_pos, &saved_pos, sizeof(int32_t)).wait();
        q.memcpy(s.d_seq_len, &saved_seq, sizeof(int32_t)).wait();
        pos = host_pos;

        gexec = std::make_unique<
            sycl_ext::command_graph<sycl_ext::graph_state::executable>>(g.finalize());
        graph_ok = true;
        return true;
    } catch (const sycl::exception& e) {
        recording = false;
        graph_ok  = false;
        // Capture MOVED the live sequence state before recording: device
        // position, device length and the host counter were all set to a
        // scratch value.  The success path puts them back; this one did
        // not, so a throw anywhere inside forward() left the next real
        // decode starting from the scratch position -- silently, and only
        // on runtimes where capture fails at all.
        q.memcpy(s.d_pos, &saved_pos, sizeof(int32_t)).wait();
        q.memcpy(s.d_seq_len, &saved_seq, sizeof(int32_t)).wait();
        pos = host_pos;
        std::printf("  graph capture unavailable (%s); using direct submission\n",
                    e.what());
        std::fflush(stdout);
        return false;
    }
}

const float* Grimoire::step() {
    if(pos<0 || pos>=max_seq)throw std::out_of_range("context capacity exhausted");
    if (graph_ok) {
        q.ext_oneapi_graph(*gexec).wait();
        ++pos;
        return s.logits;
    }
    return forward(1);
}

} // namespace b70

// =====================================================================
//  Text generation
// =====================================================================
#include "b70/tokenizer.hpp"

namespace b70 {

int grimoire_generate(const std::string& dir, Fmt proj_fmt, int max_seq,
                      const std::string& prompt, int n_predict) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    Tokenizer tk;std::string err;
    if(!tk.load(dir,err)){std::fprintf(stderr,"tokenizer: %s\n",err.c_str());return 1;}
    Grimoire e;
    if(!grimoire_load(e,dir,proj_fmt,max_seq,err)) {
        std::fprintf(stderr,"load: %s\n",err.c_str());e.release();return 1;
    }
    try {
        const auto ids=tk.encode(tk.apply_chat_template(prompt));
        std::vector<int32_t> out;
        ResponseDecoder decoder(tk,tk.special_id("<|begin_of_text|>")>=0);
        auto emit=[](const std::string& piece,bool) {
            return std::fwrite(piece.data(),1,piece.size(),stdout)==piece.size();
        };
        FinishReason reason;
        const auto start=std::chrono::steady_clock::now();
        int n=grimoire_serve_generate(e,ids,n_predict,tk.eos(),out,tk.special_id("<|eot|>"),
            [&](int32_t t){return decoder.push(t,emit);},&reason);
        decoder.finish(emit);
        std::printf("\n");
        std::fprintf(stderr,"prompt=%zu generated=%d finish=%s elapsed=%.3fs\n",ids.size(),n,
            finish_reason_name(reason),std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count());
        e.release();return 0;
    }catch(const std::exception& ex) {
        std::fprintf(stderr,"generation failed: %s\n",ex.what());e.release();return 1;
    }
}

// =====================================================================
//  grimoire-server support -- keep one Grimoire resident across requests
// =====================================================================
// The server (tools/grimoire_server.cpp) only needs an opaque handle plus
// these three calls; it never sees the Grimoire struct definition.
Grimoire* grimoire_new() { return new Grimoire(); }

// How many prefix slots currently hold a conversation.  Exposed for the
// gate, which needs to assert an invariant no token stream can show:
// that N turns of one conversation occupy ONE slot.  Everything about
// slot bookkeeping is silent -- a conversation that took a fresh slot
// per turn would still answer correctly, right up to the point where it
// had evicted every other agent.
int grimoire_prefix_slots_used(Grimoire& e) {
    int n = 0;
    for (const auto& c : e.prefix_slots) if (c.valid) ++n;
    return n;
}

bool grimoire_load(Grimoire& e, const std::string& dir, Fmt proj_fmt,
                    int max_seq, std::string& err) {
    UploadOptions opt;
    opt.lm_head_fmt = proj_fmt;
    opt.quantize_lm_head = (proj_fmt != Fmt::BF16);
    opt.max_seq = max_seq;
    return e.build(dir, opt, err);
}

// Persistent OpenAI server generation. Speculative decode and the direct
// greedy fallback share the same resident engine state.
int grimoire_serve_generate(Grimoire& e, const std::vector<int32_t>& prompt_ids,
                             int n_predict, int eos_id, std::vector<int32_t>& out_ids,
                             int eot_id, const std::function<bool(int32_t)>& on_token,
                             FinishReason* finish) {
    GenerationOptions o;
    o.max_tokens=n_predict; o.eos=eos_id; o.eot=eot_id;
    // Under PP the drafter lives on the LAST stage only, so an earlier
    // stage cannot answer this from its own state -- it takes the answer
    // from the connect-time handshake.  Every stage must agree or one
    // would run the speculative loop while another ran the plain one, and
    // they would deadlock on the next collective.
    o.dflash=(e.pp_enabled()?e.pp_dflash>0:e.dflash2.ok) &&
             e.spec_verify_available();
    const char* depth=std::getenv("GRIMOIRE_MTP_K");
    // Under PP the WIDTH must come from the handshake too, not from this
    // rank's own dflash_block_rows().  pp_dflash already sizes the block
    // each stage RECEIVES; if the depth it verifies came from a local
    // GRIMOIRE_DFLASH_M the two would disagree, the hidden/tap messages
    // would be different lengths, and the socket would desynchronise --
    // with no error where it happened.
    o.draft_depth=o.dflash
        ?(e.pp_enabled()?e.pp_dflash-1:e.dflash_block_rows()-1)
        :std::clamp(depth?std::atoi(depth):3,0,15);
    // spec_active() is the PIPELINE's answer, not this rank's: under PP
    // only the last stage holds the head, and if the ranks disagreed
    // here they would run different decode loops and deadlock.
    o.mtp=e.spec_active() && o.draft_depth>0 && e.spec_verify_available();
    const char* graph=std::getenv("GRIMOIRE_DECODE_GRAPH");
    o.graph=graph && std::atoi(graph)!=0;
    // GRIMOIRE_SPEC_STATS=1 prints accepted-per-step per request.  This is
    // the measurement an A/B needs and tok/s cannot give: a change that
    // makes the draft cheaper and a change that makes it more accurate both
    // move tok/s, and only one of them moves this.  Off by default so a
    // server does not print a line per completion.
    static const bool spec_stats=[]{ const char* s=std::getenv("GRIMOIRE_SPEC_STATS");
        return s && *s && std::atoi(s)!=0; }();
    SpecStats stats;
    if(spec_stats && (o.dflash||o.mtp)) o.stats=&stats;
    FinishReason reason=FinishReason::Length;
    try {
        const int n=generate_tokens(e,prompt_ids,o,out_ids,on_token,reason);
        if(finish)*finish=reason;
        if(o.stats && stats.steps)
            std::printf("  spec: %s depth %d -- %.2f accepted/step "
                        "(%lld of %lld drafted, %.1f%%) over %lld steps\n",
                        o.dflash?(e.dflash2.v2?"DFlash2":"DFlash"):"MTP",
                        o.draft_depth, stats.per_step(), stats.accepted,
                        stats.drafted, 100.0*stats.rate(), stats.steps);
        return n;
    } catch(...) {
        // A failed verifier may have advanced recurrent state. A subsequent
        // request must start clean; never return a successful empty response.
        e.sync();e.reset();throw;
    }
}

// Several conversations, answered together.
//
// The prompts are read one at a time -- prefill is already batched over
// a prompt's own tokens, and two prompts of different lengths share
// nothing -- and then every sequence is STEPPED together, one token per
// sequence per pass.  That second half is the whole point: a decode step
// reads every active weight to produce one token and reads the same
// weights to produce eight, so eight conversations cost about what one
// costs.  Serving them one after another pays that bill eight times.
//
// Sequences finish at different times.  A row whose sequence has
// stopped is simply not in the next batch, so the batch narrows as the
// easy answers land; it is never padded with dead rows, which would
// spend real weight traffic on tokens nobody asked for.
//
// Falls back to answering the requests one at a time -- by the ordinary
// serial path -- whenever the engine cannot batch, so a caller never has
// to ask first.  The reason is printed once, by name.
int grimoire_serve_generate_batch(Grimoire& e,
        const std::vector<std::vector<int32_t>>& prompts,
        int n_predict, int eos_id,
        std::vector<std::vector<int32_t>>& out_ids, int eot_id) {
    out_ids.assign(prompts.size(), {});
    if (prompts.empty()) return 0;
    const std::string why = e.batch_unsupported_reason();
    if (!why.empty() || int(prompts.size()) > Grimoire::kMaxBatchRows ||
        int(prompts.size()) > e.n_seq_slots) {
        if (!why.empty()) {
            static bool said = false;
            if (!said) { said = true;
                std::fprintf(stderr, "    batched decode unavailable: %s -- "
                             "answering one at a time\n", why.c_str()); }
        }
        int n = 0;
        for (size_t i = 0; i < prompts.size(); ++i) {
            FinishReason r{};
            grimoire_serve_generate(e, prompts[i], n_predict, eos_id,
                                    out_ids[i], eot_id, {}, &r);
            ++n;
        }
        return n;
    }

    struct Row {
        size_t idx;       // which request
        int slot;         // its KV rows
        int pos;          // tokens already in the cache
        int next;         // the token to feed on the coming step
        bool live;
        // This row's OWN budget, from generation_budget() -- which is
        // min(n_predict, max_seq - prompt_len), NOT n_predict.  Rows in
        // one batch have different prompt lengths, so they have
        // different budgets, and a single shared cap cannot express
        // that.  See the note at the stop condition below.
        int budget;
    };
    std::vector<Row> rows;
    std::vector<bool> busy(size_t(e.n_seq_slots), false);
    struct InvalidateOnFailure {
        Grimoire& engine;
        std::vector<bool>& busy;
        bool completed = false;
        ~InvalidateOnFailure() {
            if (!completed)
                for (size_t i=0; i<busy.size() && i<engine.prefix_slots.size(); ++i)
                    if (busy[i]) engine.prefix_slots[i].valid=false;
        }
    } invalidator{e, busy};
    const int budget_cap = e.max_seq;
    // Read each prompt into its own slot.  reset() is what clears a slot
    // and claims it; binding after it is what puts this sequence's rows
    // under the live pointers for the prefill that follows.
    for (size_t i = 0; i < prompts.size(); ++i) {
        const int budget = generation_budget(prompts[i], n_predict,
                                             budget_cap, e.cfg.vocab);
        if (budget <= 0) continue;
        const int slot = e.admit_sequence(prompts[i], busy);
        busy[size_t(slot)] = true;
        rows.push_back({i, slot, e.pos, e.argmax_token(), true, budget});
    }

    auto stop = [&](int t) {
        return (eos_id >= 0 && t == eos_id) || (eot_id >= 0 && t == eot_id);
    };
    // The first token of every reply is already decided by the prompt's
    // own last row, exactly as it is in the serial path.
    for (auto& r : rows) {
        if (stop(r.next)) { r.live = false; continue; }
        out_ids[r.idx].push_back(r.next);
    }

    // EACH ROW'S OWN BUDGET, NOT n_predict (2026-09-22).  This used to be
    // `cap = n_predict` for every row, with `r.pos >= e.max_seq` as the
    // only context guard -- and that guard is one token too late.  The
    // first reply token is emitted from the prompt's own prefill before
    // the loop starts, so by the time the loop refuses at pos == max_seq
    // it has already emitted a token the serial path never emits: the
    // serial path's budget is min(n_predict, max_seq - prompt_len), which
    // counts the prompt, so a 126-token prompt in a 128 window yields 2
    // tokens, while this loop yielded 3.  Every token was individually
    // CORRECT, which is why only an equality gate against serial decode
    // could see it.  generation_budget() is already called per row above;
    // it was computed and then discarded.  Rows also differ from each
    // other here -- 124/125/126-token prompts have budgets 4/3/2 -- so
    // one shared cap cannot be right for more than one of them.
    for (;;) {
        std::vector<int32_t> toks;
        std::vector<int>     slots, poss;
        std::vector<size_t>  which;
        for (size_t k = 0; k < rows.size(); ++k) {
            Row& r = rows[k];
            if (!r.live) continue;
            if (int(out_ids[r.idx].size()) >= r.budget || r.pos >= e.max_seq) {
                r.live = false; continue;
            }
            toks.push_back(int32_t(r.next));
            slots.push_back(r.slot);
            poss.push_back(r.pos);
            which.push_back(k);
        }
        if (toks.empty()) break;
        std::vector<std::vector<int32_t>> blocks;
        std::vector<int> consumed(toks.size(),1);
        if(e.speculative_batch()) {
            std::vector<int> remaining;
            // Per row, so a near-context row cannot have a drafter
            // propose past its own budget: decode_spec_batch() bounds
            // draft depth by `remaining - 1`.
            for(size_t k:which)
                remaining.push_back(rows[k].budget-int(out_ids[rows[k].idx].size()));
            e.decode_spec_batch(toks,slots,poss,remaining,blocks,consumed);
        } else {
            std::vector<int32_t> got;
            if(!e.decode_batch(toks,slots,poss,got) || got.size()!=toks.size())
                throw std::runtime_error("batched decode step failed");
            for(int32_t token:got) blocks.push_back({token});
        }
        for(size_t j=0;j<which.size();++j) {
            Row& r=rows[which[j]];
            r.pos+=consumed[j];
            for(int32_t token:blocks[j]) {
                if(token<0 || token>=e.cfg.vocab)
                    throw std::runtime_error("batched decode returned invalid token");
                if(stop(token)) { r.live=false; break; }
                out_ids[r.idx].push_back(token); r.next=token;
                if(int(out_ids[r.idx].size())>=r.budget) { r.live=false; break; }
            }
        }
    }
    // The engine's single cursor means nothing after a batch (see
    // prefill).  Leave it empty rather than leave the last row's numbers
    // looking like a finished single-sequence request.
    for (const auto& r : rows)
        e.cache_sequence(r.slot, r.pos, prompts[r.idx], out_ids[r.idx]);
    e.sync();
    e.graph_ok = false;
    e.pos = 0;
    invalidator.completed = true;
    return int(prompts.size());
}

// =====================================================================
//  A resident batching scheduler.
//
//  The server used to hold a mutex for the whole of a request, so a
//  second caller waited for the first to FINISH -- not for the card, for
//  the reply.  For agentic work that is the wrong unit of sharing
//  entirely: eight agents each want a few hundred tokens and each pays
//  the full weight traffic of every step alone.
//
//  Here the engine belongs to one thread.  Request threads hand it a
//  prompt and wait for tokens; that thread admits what it has room for,
//  reads each new prompt into its own sequence slot, and then steps
//  every live request TOGETHER.  A request that arrives mid-flight joins
//  at the next admission rather than at the end of a queue, and one that
//  finishes leaves without disturbing the rest.
//
//  When the engine cannot batch this model or this device (see
//  batch_unsupported_reason) the width is one and each request is served
//  by the ordinary path, speculation included -- byte for byte what the
//  server did before.  The point is that nothing has to ASK: the caller
//  submits either way.
// =====================================================================
namespace {
struct SchedJob {
    // Set by the submitting thread, read by the scheduler.
    std::vector<int32_t> prompt;
    int budget = 0, eos = -1, eot = -1;
    // Scheduler-owned.
    int slot = -1, pos = 0, next = -1;
    // Shared, under mu.
    std::mutex mu;
    std::condition_variable cv;
    std::vector<int32_t> ready;     // produced, in order
    size_t taken = 0;               // how many the caller has consumed
    bool done = false;
    bool cancelled = false;         // caller hung up or said stop
    std::string error;
    FinishReason reason = FinishReason::Length;
};
} // namespace

struct GrimoireScheduler {
    Grimoire& e;
    int width;
    bool batchable;
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<SchedJob>> pending;
    bool stopping = false;
    std::thread th;

    GrimoireScheduler(Grimoire& eng, int w)
        : e(eng), width(w), batchable(eng.batch_unsupported_reason().empty()) {}

    void submit(const std::shared_ptr<SchedJob>& j) {
        { std::lock_guard<std::mutex> l(m); pending.push_back(j); }
        cv.notify_one();
    }
    void stop() {
        { std::lock_guard<std::mutex> l(m); stopping = true; }
        cv.notify_all();
        if (th.joinable()) th.join();
    }
    // Hand a token to the waiting caller.  Returns false once the caller
    // has gone away, which is how a disconnected client stops costing
    // the card anything.
    static bool push(const std::shared_ptr<SchedJob>& j, int32_t t) {
        std::lock_guard<std::mutex> l(j->mu);
        if (j->cancelled) return false;
        j->ready.push_back(t);
        j->cv.notify_all();
        return true;
    }
    static void finish(const std::shared_ptr<SchedJob>& j, FinishReason r,
                       std::string err = {}) {
        { std::lock_guard<std::mutex> l(j->mu);
          j->reason = r; j->error = std::move(err); j->done = true; }
        j->cv.notify_all();
    }
    static bool cancelled(const std::shared_ptr<SchedJob>& j) {
        std::lock_guard<std::mutex> l(j->mu);
        return j->cancelled;
    }
    void run();
};

void GrimoireScheduler::run() {
    std::vector<std::shared_ptr<SchedJob>> active;
    std::vector<bool> slot_busy(size_t(std::max(1, e.n_seq_slots)), false);
    auto retire = [&](const std::shared_ptr<SchedJob>& j, FinishReason why,
                      bool cache = true, const std::string& err = std::string{}) {
        if (j->slot >= 0) {
            if (cache) {
                try {
                    std::vector<int32_t> reply;
                    { std::lock_guard<std::mutex> lock(j->mu); reply=j->ready; }
                    e.cache_sequence(j->slot, j->pos, j->prompt, reply);
                } catch (const std::exception& ex) {
                    if (size_t(j->slot)<e.prefix_slots.size())
                        e.prefix_slots[size_t(j->slot)].valid=false;
                    std::fprintf(stderr, "prefix snapshot skipped: %s\n", ex.what());
                }
            } else if (size_t(j->slot)<e.prefix_slots.size()) {
                e.prefix_slots[size_t(j->slot)].valid=false;
            }
            slot_busy[size_t(j->slot)] = false;
        }
        finish(j, why, err);
    };
    for (;;) {
        // ---- admit ---------------------------------------------------
        std::vector<std::shared_ptr<SchedJob>> taking;
        {
            std::unique_lock<std::mutex> l(m);
            if (active.empty() && pending.empty()) {
                if (stopping) return;
                cv.wait(l, [&]{ return stopping || !pending.empty(); });
                if (stopping && pending.empty()) return;
            }
            while (int(active.size() + taking.size()) < width && !pending.empty()) {
                taking.push_back(pending.front());
                pending.pop_front();
            }
        }
        for (auto& j : taking) {
            if (cancelled(j)) { finish(j, FinishReason::Cancelled); continue; }
            int slot = -1;
            for (size_t i = 0; i < slot_busy.size(); ++i)
                if (!slot_busy[i]) { slot = int(i); break; }
            if (slot < 0) {   // no room after all; put it back
                std::lock_guard<std::mutex> l(m);
                pending.push_front(j);
                continue;
            }
            try {
                if (!batchable) {
                    // One at a time, by the ordinary path.  Speculation,
                    // the prefix cache and the graph all still apply --
                    // this is the server exactly as it was.
                    //
                    // Under PP this is also where the other stages learn
                    // what to run.  They are sitting in
                    // grimoire_pp_worker_loop waiting for a request; send
                    // it BEFORE generating, because the moment this stage
                    // starts its prefill it will try to hand them a
                    // hidden state they are not yet expecting.
                    if (!grimoire_pp_broadcast_request(e, j->prompt, j->budget,
                                                       j->eos, j->eot)) {
                        finish(j, FinishReason::Length,
                               "pipeline request broadcast failed");
                        continue;
                    }
                    std::vector<int32_t> out;
                    FinishReason r = FinishReason::Length;
                    // UNDER PP, RANK 0 MUST NOT STOP EARLY (external
                    // audit F2, 2026-09-21).  Every rank runs this same
                    // generate_tokens() loop in lockstep -- every step
                    // is a forward()/argmax_token() round trip over the
                    // PP sockets, matched call for call across ranks --
                    // and that lockstep IS the whole protocol.  A
                    // disconnected streaming client is known only to
                    // rank 0; the other ranks have no channel to learn
                    // "stop here too" mid-flight, so if rank 0's own
                    // emit() returns false on cancellation and breaks
                    // its loop early, the other ranks keep decoding
                    // toward the ORIGINAL budget/EOS.  Rank 0 then moves
                    // on to the NEXT request and broadcasts a fresh
                    // PPRequest header down a pipe whose other end is
                    // still mid-token-exchange for this one -- the
                    // header lands in the wrong protocol phase and can
                    // be read as model data, block, or corrupt the next
                    // request.
                    //
                    // So under PP, ignore push()'s "keep going" answer
                    // for the purpose of continuing this loop -- EOS and
                    // budget exhaustion are unaffected (they are
                    // identical on every rank by construction, since
                    // every rank was broadcast the same request) and
                    // still stop the loop normally, in lockstep, exactly
                    // where every other rank also stops. push() itself
                    // is still called and still suppresses delivery once
                    // j->cancelled is set -- the disconnected client
                    // receives nothing further -- this only keeps rank
                    // 0's STEP COUNT matched to its peers. Off PP,
                    // cancellation still frees the card immediately,
                    // which is correct and unchanged there.
                    const bool pp = e.pp_enabled() || e.tp_enabled();
                    grimoire_serve_generate(e, j->prompt, j->budget, j->eos, out,
                        j->eot, [&](int32_t t){
                            const bool kept = push(j, t);
                            return pp ? true : kept;
                        }, &r);
                    finish(j, r);
                    continue;
                }
                slot = e.admit_sequence(j->prompt, slot_busy);
                j->slot = slot;
                j->pos  = e.pos;
                j->next = e.argmax_token();
                slot_busy[size_t(slot)] = true;
                const bool stop = (j->eos >= 0 && j->next == j->eos) ||
                                  (j->eot >= 0 && j->next == j->eot);
                if (stop) {
                    retire(j, FinishReason::Stop);
                } else if (!push(j, int32_t(j->next))) {
                    retire(j, FinishReason::Cancelled);
                } else if (j->budget <= 1) {
                    retire(j, FinishReason::Length);
                } else {
                    active.push_back(j);
                }
            } catch (const std::exception& ex) {
                retire(j, FinishReason::Length, false, ex.what());
            }
        }
        if (active.empty()) continue;

        // ---- step every live request together ------------------------
        std::vector<int32_t> toks; std::vector<int> slots, poss;
        std::vector<size_t> which;
        for (size_t k = 0; k < active.size(); ++k) {
            auto& j = active[k];
            if (cancelled(j)) continue;
            toks.push_back(int32_t(j->next));
            slots.push_back(j->slot);
            poss.push_back(j->pos);
            which.push_back(k);
        }
        std::vector<int32_t> got;
        std::vector<std::vector<int32_t>> blocks;
        std::vector<int> consumed(toks.size(),1);
        bool failed = false; std::string err;
        if (!toks.empty()) {
            try {
                if (e.pp_enabled() || e.tp_enabled()) {
                    Grimoire::PPRequest request;
                    request.kind=3; request.prompt=toks;
                    request.slots.assign(slots.begin(),slots.end());
                    request.positions.assign(poss.begin(),poss.end());
                    if (!e.pp_send_request(request))
                        throw std::runtime_error("pipeline batch broadcast failed");
                }
                if(e.speculative_batch()) {
                    std::vector<int> remaining;
                    for(size_t k:which) {
                        const auto& job=active[k];
                        std::lock_guard<std::mutex> lock(job->mu);
                        remaining.push_back(job->budget-int(job->ready.size()));
                    }
                    e.decode_spec_batch(toks,slots,poss,remaining,blocks,consumed);
                } else {
                    if(!e.decode_batch(toks,slots,poss,got) || got.size()!=toks.size()) {
                        failed=true; err="batched decode step failed";
                    } else for(int32_t token:got) blocks.push_back({token});
                }
            } catch (const std::exception& ex) { failed = true; err = ex.what(); }
        }
        std::vector<std::shared_ptr<SchedJob>> keep;
        for (size_t k = 0; k < active.size(); ++k) {
            auto& j = active[k];
            // A row that was cancelled, or that the step never covered,
            // is retired here rather than silently carried.
            size_t at = which.size();
            for (size_t i = 0; i < which.size(); ++i) if (which[i] == k) { at = i; break; }
            if (failed || at == which.size()) {
                retire(j, cancelled(j) ? FinishReason::Cancelled
                                       : FinishReason::Length, !failed, err);
                continue;
            }
            j->pos+=consumed[at];
            bool done=false;
            for(int32_t token:blocks[at]) {
                const bool stop=(j->eos>=0 && token==j->eos) || (j->eot>=0 && token==j->eot);
                size_t n=0;
                { std::lock_guard<std::mutex> lock(j->mu); n=j->ready.size(); }
                if(token<0 || token>=e.cfg.vocab) {
                    retire(j,FinishReason::Length,false,"batched decode returned invalid token");
                    done=true;
                } else if(stop) { retire(j,FinishReason::Stop); done=true; }
                else if(!push(j,token)) { retire(j,FinishReason::Cancelled); done=true; }
                else if(int(n)+1>=j->budget) { retire(j,FinishReason::Length); done=true; }
                else j->next=token;
                if(done) break;
            }
            if(!done && j->pos>=e.max_seq) { retire(j,FinishReason::Length); done=true; }
            if(!done) keep.push_back(j);
        }
        active.swap(keep);
    }
}

GrimoireScheduler* grimoire_scheduler_new(Grimoire& e, int width) {
    e.serving_control = true;
    const int cap = std::min({width > 0 ? width : 1,
                              Grimoire::kMaxBatchRows,
                              std::max(1, e.n_seq_slots)});
    const std::string why = e.batch_unsupported_reason();
    auto* sc = new GrimoireScheduler(e, why.empty() ? cap : 1);
    std::fprintf(stderr, "  scheduler: %s\n",
        why.empty()
            ? ("batching up to " + std::to_string(sc->width) +
               " requests per step").c_str()
            : ("one request at a time -- " + why).c_str());
    sc->th = std::thread([sc]{ sc->run(); });
    return sc;
}

// How many requests this scheduler will step together.  The server
// reports it on /v1/models, where it used to print the literal 1.
int grimoire_scheduler_width(GrimoireScheduler& sc) { return sc.width; }

void grimoire_scheduler_delete(GrimoireScheduler* sc) {
    if (!sc) return;
    sc->stop();
    delete sc;
}

int grimoire_scheduler_generate(GrimoireScheduler& sc,
        const std::vector<int32_t>& prompt, int n_predict, int eos_id,
        int eot_id, std::vector<int32_t>& out,
        const std::function<bool(int32_t)>& on_token, FinishReason* finish) {
    out.clear();
    auto j = std::make_shared<SchedJob>();
    j->prompt = prompt;
    // Thrown HERE, on the caller's thread, so a bad request is a 400 from
    // the handler that made it rather than a scheduler error with no
    // request attached.
    j->budget = generation_budget(prompt, n_predict, sc.e.max_seq, sc.e.cfg.vocab);
    j->eos = eos_id; j->eot = eot_id;
    if (j->budget == 0) { if (finish) *finish = FinishReason::Length; return 0; }
    sc.submit(j);
    std::unique_lock<std::mutex> l(j->mu);
    for (;;) {
        j->cv.wait(l, [&]{ return j->done || j->taken < j->ready.size(); });
        while (j->taken < j->ready.size()) {
            const int32_t t = j->ready[j->taken++];
            l.unlock();
            out.push_back(t);
            bool keep_going = true;
            if (on_token) keep_going = on_token(t);
            l.lock();
            if (!keep_going) {
                j->cancelled = true;
                // Wait for the scheduler to let the slot go, or the next
                // request could be admitted into a slot still in use.
                j->cv.wait(l, [&]{ return j->done; });
                if (finish) *finish = FinishReason::Cancelled;
                return int(out.size());
            }
        }
        if (j->done && j->taken >= j->ready.size()) break;
    }
    const std::string err = j->error;
    const FinishReason r = j->reason;
    l.unlock();
    if (!err.empty()) throw std::runtime_error(err);
    if (finish) *finish = r;
    return int(out.size());
}

// What every pipeline stage except the first runs instead of listening.
//
// It is deliberately the SAME call the first stage makes: identical
// arguments produce identical control flow, and the per-token messages
// the stages already exchange then line up by construction.  A worker
// that ran its own loop would have to re-derive every branch rank 0
// takes -- whether speculation is on, how wide a draft is, when a stop
// token ends the request -- and one disagreement deadlocks the pipe.
//
// Returns when rank 0 closes the socket or sends a shutdown, which is
// what makes `docker stop` on the front end bring the workers down too
// instead of leaving them holding a card.
bool grimoire_is_pp_worker(Grimoire& e) {
    return (e.pp_enabled() && e.pp_rank > 0) || (e.tp_enabled() && e.tp_rank > 0);
}

// The front end tells the rest of the pipeline what it is about to run.
//
// Exposed rather than inlined in the scheduler so the gate drives THIS
// function and not a copy of it: a protocol whose two ends are written
// twice is a protocol with two chances to disagree, and a disagreement
// here is a deadlock, not an error.
//
// A no-op when there is no pipeline, so a caller does not have to ask.
bool grimoire_pp_broadcast_request(Grimoire& e,
        const std::vector<int32_t>& prompt, int n_predict,
        int eos_id, int eot_id) {
    if ((!e.pp_enabled() && !e.tp_enabled()) || e.comm_rank()!=0) return true;
    Grimoire::PPRequest req;
    req.prompt = prompt; req.budget = n_predict;
    req.eos = eos_id; req.eot = eot_id;
    return e.pp_send_request(req);
}

// Bring the workers down with the front end.  Without it they sit on a
// read() holding a card until something kills them, which on a box where
// the next run wants that card is the difference between a restart and a
// power cycle.
void grimoire_pp_shutdown(Grimoire& e) {
    if ((!e.pp_enabled() && !e.tp_enabled()) || e.comm_rank()!=0) return;
    Grimoire::PPRequest req;
    req.shutdown = true;
    (void)e.pp_send_request(req);
}

void grimoire_pp_worker_loop(Grimoire& e) {
    std::fprintf(stderr, "  PP rank %d: worker ready, waiting for requests\n",
                 e.pp_rank);
    for (;;) {
        Grimoire::PPRequest req;
        if (!e.pp_recv_request(req)) {
            std::fprintf(stderr, "  PP rank %d: front end closed the pipe\n",
                         e.pp_rank);
            return;
        }
        if (req.shutdown) {
            std::fprintf(stderr, "  PP rank %d: shutdown\n", e.pp_rank);
            return;
        }
        std::vector<int32_t> out;
        FinishReason r = FinishReason::Length;
        try {
            if (req.kind==4) {
                e.cache_sequence(req.budget,req.eos,req.prompt,{});
                continue;
            }
            if (req.kind==2) {
                // Rank 0 is the sole owner of live request membership.
                // The selected slot is explicit; workers never select a cache hit.
                std::vector<bool> busy(size_t(e.n_seq_slots),true);
                busy[size_t(req.budget)]=false;
                if(e.admit_sequence(req.prompt,busy)!=req.budget)
                    throw std::runtime_error("pipeline admission slot mismatch");
                (void)e.argmax_token(); // match rank 0's backward token hop
                continue;
            }
            if (req.kind==3) {
                const std::vector<int> slots(req.slots.begin(),req.slots.end());
                const std::vector<int> positions(req.positions.begin(),req.positions.end());
                if(!e.decode_batch(req.prompt,slots,positions,out))
                    throw std::runtime_error("pipeline batch step failed");
                continue;
            }
            grimoire_serve_generate(e, req.prompt, req.budget, req.eos, out,
                                    req.eot, {}, &r);
        } catch (const std::exception& ex) {
            // A stage that threw has left the pipe mid-message; there is
            // no way to resynchronise a byte stream after that, so say so
            // and stop rather than answer the next request from the
            // middle of this one's data.
            std::fprintf(stderr, "  PP rank %d: request failed: %s\n",
                         e.pp_rank, ex.what());
            return;
        }
    }
}

void grimoire_delete(Grimoire* e) { if(e){e->release();delete e;} }

} // namespace b70

namespace b70 {

// ---------------------------------------------------------------------
// Where does a token's device time actually go?
//
// Each region below is the wall-clock gap on the DEVICE between the end
// of one marker kernel and the start of the next -- kernel execution and
// inter-kernel gaps together. The marker column is what the measurement
// itself costs, so a region is only meaningful when it is well above it.
// ---------------------------------------------------------------------
void Grimoire::dump_timeline() {
    if (tl.size() < 2) return;
    using sycl::info::event_profiling::command_start;
    using sycl::info::event_profiling::command_end;
    auto st = [&](size_t i) { return tl[i].first.get_profiling_info<command_start>(); };
    auto en = [&](size_t i) { return tl[i].first.get_profiling_info<command_end>(); };

    std::printf("\n  ---- device timeline, one token ----\n");
    std::printf("  %-22s %10s %10s\n", "region", "us", "marker us");

    double region_total = 0, marker_total = 0, layer_total = 0;
    std::map<std::string, double> by_layer_kind;
    for (size_t i = 1; i < tl.size(); ++i) {
        const double us  = double(st(i) - en(i - 1)) / 1000.0;
        const double mus = double(en(i) - st(i)) / 1000.0;
        region_total += us;
        marker_total += mus;
        const std::string& tag = tl[i].second;
        if (tag.size() > 1 && tag[0] == 'L' && std::isdigit(uint8_t(tag[1]))) {
            layer_total += us;
            by_layer_kind[tag.substr(4)] += us;
        }
        std::printf("  %-22s %10.3f %10.3f\n", tag.c_str(), us, mus);
    }
    const double span = double(en(tl.size() - 1) - st(0)) / 1000.0;
    std::printf("  %-22s %10.3f %10.3f\n", "TOTAL", region_total, marker_total);
    std::printf("  device span (incl markers)      %10.3f us\n", span);
    for (auto& kv : by_layer_kind)
        std::printf("  sum of '%s' layer marks        %10.3f us\n",
                    kv.first.c_str(), kv.second);
    std::printf("  (layer marks only, no tail)     %10.3f us\n", layer_total);
    std::fflush(stdout);
}

void Grimoire::probe(const char* tag, const float* p, int n) {
    if (!debug) return;
    launch_probe(q, p, n, probe_buf, {});
    q.wait();
    float h4[4];
    q.memcpy(h4, probe_buf, 4 * sizeof(float)).wait();
    std::printf("    %-22s rms %12.4g  max %12.4g  nan %.0f  inf %.0f\n",
                tag, h4[0], h4[1], h4[2], h4[3]);
    // RMS/max is too weak for differential tracing: two paths can have the
    // same distribution while differing element-by-element.  In the opt-in
    // exact probe mode, print a deterministic bitwise fingerprint and the
    // first four values.  This is deliberately host-side and debug-only.
    if (std::getenv("GRIMOIRE_PROBE_EXACT")) {
        std::vector<float> host(static_cast<size_t>(n));
        q.memcpy(host.data(), p, size_t(n) * sizeof(float)).wait();
        uint64_t hash = 1469598103934665603ull;
        for (float value : host) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            hash ^= bits;
            hash *= 1099511628211ull;
        }
        std::printf("      bits %016llx  first", (unsigned long long)hash);
        for (int i = 0; i < std::min(n, 4); ++i)
            std::printf(" % .8g", host[size_t(i)]);
        std::printf("\n");
    }
    std::fflush(stdout);
}

} // namespace b70
