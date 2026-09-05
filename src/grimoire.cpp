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
#include "b70/qwen35.hpp"
#include "b70/gptq.hpp"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <memory>
#include <functional>
#include <type_traits>
#include <dlfcn.h>

namespace sycl_ext = sycl::ext::oneapi::experimental;
#include <cstdio>
#include <map>
#include <array>
#include <cctype>
#include <cstring>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>
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
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
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
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
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
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
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
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
        auto fn=reinterpret_cast<Xe2DenseW4A8>(dlsym(handle,sym));
        if(fn)return fn;
        dlclose(handle);handle=nullptr;}
    return nullptr;
}

static Xe2GroupedMXFP4 load_xe2_grouped_sym(const char* sym) {
    void* handle=nullptr;
    const char* env=std::getenv("GRIMOIRE_XE2_GROUPED_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_xe2_grouped.so",
        "/work/src/libgrimoire_xe2_grouped.so","/bridge/libgrimoire_xe2_grouped.so"};
    for(const char* path:paths){if(!path||!*path)continue;
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
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
        handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
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
      handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
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
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
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
        h=dlopen(pth,RTLD_NOW|RTLD_GLOBAL); if(h)break; }
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
            void* fh=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!fh)continue;
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
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
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
    for(const char* path:paths){if(!path||!*path)continue;handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
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
    for(const char* path:paths){if(!path||!*path)continue;handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);
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
        handle=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!handle)continue;
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
        if(!path||!*path)continue; handle=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
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
        if(!path||!*path)continue; handle=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
        api.create=reinterpret_cast<decltype(api.create)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_create"));
        api.scratch_size=reinterpret_cast<decltype(api.scratch_size)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_scratch_size"));
        api.execute=reinterpret_cast<decltype(api.execute)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_execute"));
        api.destroy=reinterpret_cast<decltype(api.destroy)>(dlsym(handle,"grimoire_onednn_mxfp4_w4a16_destroy"));
        if(api)break; api={}; dlclose(handle); handle=nullptr;
    }
    if(!api)std::fprintf(stderr,"  oneDNN MXFP4 W4A16 unavailable\n");
    return api;
}

static OneDnnW4Api load_onednn_w4() {
    static OneDnnW4Api api{}; static bool attempted=false; static void* handle=nullptr;
    if(attempted)return api; attempted=true;
    const char* env=std::getenv("GRIMOIRE_ONEDNN_BRIDGE");
    const char* paths[]={env,"src/libgrimoire_onednn.so","/work/src/libgrimoire_onednn.so",
        "/bridge/libgrimoire_onednn.so"};
    for(const char* path:paths){
        if(!path||!*path)continue; handle=dlopen(path,RTLD_NOW|RTLD_LOCAL); if(!handle)continue;
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
sycl::event launch_rmsnorm_heads(sycl::queue&, float*, const bf16_t*, int, int,
                                 float, bool, const std::vector<sycl::event>&);
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
            if (!ck.shards[r.shard]->read_f32(r.t, f32.data(), rerr)) {
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

bool read_matrix_f32(const Qwen35Model& ck, const TensorRef& r,
                     float* dst, std::string& err) {
    // Native artifacts keep the precision-critical 2-D weights (the delta-rule
    // gates, the MoE router, the shared-expert gate) at RAW BF16.  They have no
    // safetensors shard behind them, so read them straight out of the native
    // payload instead of indexing ck.shards.
    if (r.native) {
        if (r.native->encoding != uint32_t(NativeEncoding::RAW)) {
            err = "packed native tensor requested as a float matrix"; return false;
        }
        const int64_t n = r.t.numel();
        const auto* p = static_cast<const uint8_t*>(
            ck.native_model->payload(*r.native)) + r.native_payload_offset;
        if (r.native->source_dtype == uint32_t(STDtype::BF16)) {
            for (int64_t i = 0; i < n; ++i)
                dst[i] = bf16_to_f32(bf16_t{uint16_t(uint16_t(p[2*i]) |
                                                    (uint16_t(p[2*i+1]) << 8))});
            return true;
        }
        if (r.native->source_dtype == uint32_t(STDtype::F32)) {
            std::memcpy(dst, p, size_t(n) * sizeof(float));
            return true;
        }
        err = "unsupported native RAW dtype for a float matrix"; return false;
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
    if(!ck.shards[r.shard]->read_f32(r.t,f32.data(),err)){
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

    // compressed-tensors weight_packed -> direct MXFP4 upload (no re-quant).
    // r carries the weight_scale in scales_shard/scales_t; both are already
    // in GRIMOIRE's MXFP4 layout (E2M1 payload + E8M0 group-32 scales).
    if (r.t.name.find("weight_packed") != std::string::npos && !r.native &&
        (fmt == Fmt::MXFP4)) {
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

    // A caller asking for BF16 means the tensor is precision-critical.  If the
    // artifact stores it packed anyway it predates the converter's exclusion
    // list, and the precision is already gone -- say so instead of silently
    // honouring the file over the caller.
    if(fmt==Fmt::BF16 && r.native &&
       r.native->encoding==uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2))
        std::printf("\n  WARNING: %s must stay BF16 but this artifact packs it "
                    "as MXFP4 -- reconvert the model.\n", what);
    if(r.native && r.native->encoding==
       uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2)){
        const size_t pb=size_t(N)*K/2,sb=size_t(N)*K/kMXBlock;
        const auto* p=static_cast<const uint8_t*>(ck.native_model->payload(*r.native))+
                      r.native_payload_offset;
        const auto* s=static_cast<const uint8_t*>(ck.native_model->scales(*r.native))+
                      r.native_scale_offset;
        d.payload=dev_copy<uint8_t>(q,p,pb);d.scales=dev_copy<uint8_t>(q,s,sb);
        d.w=QuantWeight{Fmt::MXFP4,N,K,d.payload,d.scales,nullptr,
                        int64_t(K/2),K/kMXBlock};
        if(!d.payload||!d.scales)*ok=false;
        return d;
    }

    std::string rerr;
    PackedWeight p;
    if (r.compressed_int4 && use == Fmt::INT4) {
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

    const int blk = (fmt == Fmt::INT4) ? kInt4Group
                  : (fmt == Fmt::MXFP4 || fmt == Fmt::MXFP8) ? kMXBlock : 1;
    Fmt use = fmt;
    if (blk > 1 && (K % blk) != 0) use = Fmt::BF16;

    const bool native_mx = ra.native && rb.native &&
      ra.native->encoding==uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2) &&
      rb.native->encoding==uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2);
    if(fmt==Fmt::BF16 && native_mx)
        std::printf("\n  WARNING: %s must stay BF16 but this artifact packs it "
                    "as MXFP4 -- reconvert the model.\n", what);
    if(native_mx){
        const size_t pba=size_t(Na)*K/2,pbb=size_t(Nb)*K/2;
        const size_t sba=size_t(Na)*K/kMXBlock,sbb=size_t(Nb)*K/kMXBlock;
        std::vector<uint8_t> hp(pba+pbb),hs(sba+sbb);
        std::memcpy(hp.data(),static_cast<const uint8_t*>(ck.native_model->payload(*ra.native))+
                    ra.native_payload_offset,pba);
        std::memcpy(hp.data()+pba,static_cast<const uint8_t*>(ck.native_model->payload(*rb.native))+
                    rb.native_payload_offset,pbb);
        std::memcpy(hs.data(),static_cast<const uint8_t*>(ck.native_model->scales(*ra.native))+
                    ra.native_scale_offset,sba);
        std::memcpy(hs.data()+sba,static_cast<const uint8_t*>(ck.native_model->scales(*rb.native))+
                    rb.native_scale_offset,sbb);
        d.payload=dev_copy<uint8_t>(q,hp.data(),hp.size());
        d.scales=dev_copy<uint8_t>(q,hs.data(),hs.size());
        d.w=QuantWeight{Fmt::MXFP4,N,K,d.payload,d.scales,nullptr,
                        int64_t(K/2),K/kMXBlock};
        if(!d.payload||!d.scales)*ok=false;
        return d;
    }

    // compressed-tensors weight_packed (HF safetensors): same MXFP4 layout,
    // read via read_raw and concatenate gate|up along N.
    const bool hf_packed = !ra.native && !rb.native &&
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
    if (ra.compressed_int4 && rb.compressed_int4 && use == Fmt::INT4) {
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
        if (std::getenv("GRIMOIRE_DAG"))
            return sycl::property_list{};
        if (std::getenv("GRIMOIRE_TIMELINE") ||
            std::getenv("GRIMOIRE_PROFILE_PREFILL"))
            return {sycl::property::queue::in_order(),
                    sycl::property::queue::enable_profiling()};
        return {sycl::property::queue::in_order()};
    }
    // Match vLLM's XPU worker model: ZE_AFFINITY_MASK keeps the full 0,1
    // visibility list, while each child process selects device local_rank.
    static sycl::device rank_device() {
        std::vector<sycl::device> b70;
        for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("B70") != std::string::npos)
                b70.push_back(d);
        const char* e = std::getenv("GRIMOIRE_PP_RANK");
        if (!e || !*e) e = std::getenv("GRIMOIRE_TP_RANK");
        const int rank = e && *e ? std::atoi(e) : 0;
        if (rank >= 0 && rank < int(b70.size())) return b70[size_t(rank)];
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
    int tp_rank = []{ const char* e=std::getenv("GRIMOIRE_TP_RANK");
        return e&&*e?std::atoi(e):-1; }();
    int pp_begin = 0, pp_end = 0;
    int pp_fd = -1;
    std::string pp_socket;
    bool pp_enabled() const { return pp_rank==0||pp_rank==1; }
    bool tp_enabled() const { return tp_rank==0||tp_rank==1; }
    int comm_rank() const { return pp_enabled()?pp_rank:tp_rank; }
    bool pp_write_all(const void* data,size_t bytes) {
        const uint8_t* p=static_cast<const uint8_t*>(data);
        while(bytes){const ssize_t n=::send(pp_fd,p,bytes,MSG_NOSIGNAL);
            if(n>0){p+=n;bytes-=size_t(n);continue;}
            if(n<0&&errno==EINTR)continue;return false;} return true;
    }
    bool pp_read_all(void* data,size_t bytes) {
        uint8_t* p=static_cast<uint8_t*>(data);
        while(bytes){const ssize_t n=::recv(pp_fd,p,bytes,0);
            if(n>0){p+=n;bytes-=size_t(n);continue;}
            if(n<0&&errno==EINTR)continue;return false;} return true;
    }
    bool pp_connect(std::string& err);
    bool pp_send_hidden(const float* dev,size_t elems);
    bool pp_recv_hidden(float* dev,size_t elems);
    int  pp_sync_token(int token);
    bool tp_allgather(float* dev, int elems, int begin, int count);
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
        uint8_t *gu_pack = nullptr, *gu_scale = nullptr, *gu_zero = nullptr;
        uint8_t *dn_pack = nullptr, *dn_scale = nullptr, *dn_zero = nullptr;
        bool xe2_signed_int4 = false;

        // W4A8 prefill copies of the FFN weights: symmetric int4 g128 plus
        // one f32 scale per (row, group).  Additive -- the MXFP4 originals
        // stay resident because decode's GEMV is far faster at M=1.
        uint8_t *sh_gu_i4 = nullptr, *sh_dn_i4 = nullptr;
        float   *sh_gu_ws = nullptr, *sh_dn_ws = nullptr;

        float *dn_state = nullptr, *conv_ring = nullptr;
        uint8_t *k_cache = nullptr, *v_cache = nullptr;
        sycl::half *k_cache_f16 = nullptr, *v_cache_f16 = nullptr;
        bool muse_sliding = false;
    };
    std::vector<LayerDev> L;

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
    struct PrefixLayerCache {
        float *dn = nullptr, *conv = nullptr;
        uint8_t *k = nullptr, *v = nullptr;
    };
    struct PrefixCache {
        bool valid = false;
        std::vector<int32_t> tokens;
        std::vector<PrefixLayerCache> layers;
        float *hidden = nullptr, *logits = nullptr;
    } prefix_cache;
    static bool prefix_cache_enabled() {
        const char* e = std::getenv("GRIMOIRE_PREFIX_CACHE");
        return e && *e && std::atoi(e) != 0;
    }
    bool restore_prefix(const std::vector<int32_t>& tokens);
    bool save_prefix(const std::vector<int32_t>& tokens);

    bf16_t*  embed = nullptr;
    bf16_t*  fnorm = nullptr;
    sycl::half* fnorm_f16 = nullptr;
    DevQuant lm_head;

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
            bool sliding=true;
        };
        bool ok=false, v2=false;
        DevQuant fc, selector_hidden;
        DevQuant fused_context_kv;
        DevQuant shared_embed_f16, shared_lm_head_f16, draft_lm_head_i4;
        bf16_t *hidden_norm=nullptr, *norm=nullptr;
        sycl::half *hidden_norm_f16=nullptr, *norm_f16=nullptr;
        bf16_t *predecessor=nullptr, *successor=nullptr;
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
        int32_t *block_table=nullptr, *cu_q=nullptr, *cu_k=nullptr;
        int32_t *seqused_k=nullptr;
        // DFlash2 dynamic grouped convolution.  Geometry is derived from the
        // artifact, not the config: base_kernel is [2,taps,hidden] and
        // kernel_projection is [2*taps*groups, hidden].
        float *conv_delta=nullptr, *conv_scratch=nullptr;
        bool fp16_draft=true;   // drafter weights uploaded as FP16
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
        int block_size=64, num_blocks=0;
        float rope_theta=0.0f;
        std::vector<Layer> layers;
        std::vector<int> target_layers;
    } dflash2;
    // Speculative verification advances every recurrent layer optimistically.
    // Keep one device-side checkpoint so a rejection can restore the exact
    // pre-verify state and replay only the accepted prefix. Attention KV
    // entries do not need copying: replay overwrites the speculative slots.
    float* spec_dn_state = nullptr;
    float* spec_conv_ring = nullptr;
    float* spec_dn_steps = nullptr;
    float* spec_conv_inputs = nullptr;
    float* spec_hidden_steps = nullptr;
    size_t spec_dn_elems = 0, spec_conv_elems = 0;
    size_t spec_conv_input_elems = 0;
    static constexpr int kSpecBatch = 16;
    static bool mtp_enabled() {
        static const bool v = []{ const char* e = std::getenv("GRIMOIRE_MTP");
            return e && *e && std::atoi(e) != 0; }();
        return v;
    }

    Scratch s{};
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
    bool  dag = std::getenv("GRIMOIRE_DAG") != nullptr;
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
    void sync() { q.wait(); }     // forward() no longer drains; callers that
                                  // time a region must end it with this.

    // Any decode GEMV.  When a weight has been converted its MXFP4 payload is
    // gone, so every decode call site must come through here.
    sycl::event gemv_any(const DevQuant& dq, const float* x, float* y,
                         const std::vector<sycl::event>& deps) {
        if (tp_enabled() && dq.w.N >= 2 && (dq.w.N % 2) == 0) {
            const int half = dq.w.N / 2;
            const int begin = tp_rank * half;
            sycl::event ev;
            if (dq.has_i4()) {
                const size_t prow = size_t(dq.w.K) / 2;
                const size_t srow = size_t(dq.w.K) / 128;
                ev = launch_gemv_int4sym(q, dq.i4 + size_t(begin) * prow,
                    dq.i4s + size_t(begin) * srow, x, y + begin,
                    half, dq.w.K, deps);
            } else {
                QuantWeight w = dq.w;
                w.N = half;
                w.payload = dq.w.payload + int64_t(begin) * dq.w.row_bytes;
                if (dq.w.scales) {
                    const size_t ss = dq.w.fmt == Fmt::INT4 ? sizeof(bf16_t) : 1;
                    w.scales = static_cast<const uint8_t*>(dq.w.scales) +
                        int64_t(begin) * dq.w.row_scales * ss;
                }
                if (dq.w.zeros)
                    w.zeros = dq.w.zeros + int64_t(begin) * dq.w.row_scales;
                ev = launch_gemv(q, w, x, y + begin, deps);
            }
            ev.wait();
            if (!tp_allgather(y, dq.w.N, begin, half))
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
        static OneDnnW4Api api = load_onednn_w4();
        if (!api) return false;

        auto it = i4_plans.find(dq.i4s);
        if (it == i4_plans.end()) {
            void* pl = api.create(&q, 1, N, K, GS, 1);
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
            if (!i4_zp) {
                i4_zp = sycl::malloc_device<int8_t>(1, q);
                const int8_t z = 8; q.memcpy(i4_zp, &z, 1).wait();
            }
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
                    i4_zp, mx_o, it->second.scratch);
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

    bool build_graph();
    const float* step();          // one token: graph replay if available

    bool build(const std::string& dir, const UploadOptions& opt, std::string& err);
    void reset();
    const float* forward(int token);      // returns device logits
    const float* forward_muse(int token); // Muse Glimmer dense sandwich path
    bf16_t* muse_zero = nullptr;           // zeroed weight -> scaleless (1+0) norm
    sycl::half* muse_zero_f16 = nullptr;   // same, for the FP16 activation path
    const float* forward_dag(int token);  // out-of-order queue + true dependencies
    bool prefill(const std::vector<int32_t>& tokens,
                 std::vector<int32_t>* next_tokens = nullptr);
    bool prefill_muse(const std::vector<int32_t>& tokens,
                      std::vector<int32_t>* next_tokens = nullptr);
    void snapshot_recurrent();
    void restore_recurrent(int saved_pos);
    void commit_spec_prefix(int saved_pos, int accepted);
    // MTP draft: given the hidden state of the token just processed and the
    // token the main model chose next, predict the token AFTER that.
    // from_mtp_hidden: chained drafts feed the head its OWN previous hidden
    // state instead of the main model's h_t, which is how depth > 1 works.
    int  mtp_draft(int next_token, int position, bool from_mtp_hidden = false);
    bool dflash_draft(int bonus_token, int position,
                      std::vector<int32_t>& draft_tokens,
                      bool context_only = false);
    int          argmax_token();
    void release();
};

// ---------------------------------------------------------------------
bool Grimoire::pp_connect(std::string& err) {
    const char* env = std::getenv("GRIMOIRE_PP_SOCKET");
    if ((!env || !*env) && tp_enabled()) env = std::getenv("GRIMOIRE_TP_SOCKET");
    pp_socket = env && *env ? env :
        (tp_enabled() ? "/tmp/grimoire-tp.sock" : "/tmp/grimoire-pp.sock");
    if (pp_socket.size() >= sizeof(sockaddr_un::sun_path)) {
        err = "GRIMOIRE_PP_SOCKET path is too long";
        return false;
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, pp_socket.c_str(), sizeof(addr.sun_path) - 1);
    if (comm_rank() == 1) {
        const int listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (listener < 0) { err = "PP socket() failed"; return false; }
        ::unlink(pp_socket.c_str());
        if (::bind(listener, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
            ::listen(listener, 1) < 0) {
            err = std::string("PP bind/listen failed: ") + std::strerror(errno);
            ::close(listener); return false;
        }
        std::printf("  %s rank 1: waiting for rank 0 on %s\n",
                    tp_enabled() ? "TP" : "PP", pp_socket.c_str());
        std::fflush(stdout);
        do { pp_fd = ::accept(listener, nullptr, nullptr); } while (pp_fd < 0 && errno == EINTR);
        ::close(listener);
    } else {
        pp_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (pp_fd < 0) { err = "PP socket() failed"; return false; }
        bool connected = false;
        for (int attempt = 0; attempt < 6000; ++attempt) {
            if (::connect(pp_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0) {
                connected = true;
                break;
            }
            if (errno != ENOENT && errno != ECONNREFUSED) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!connected) {
            const int saved_errno = errno;
            ::close(pp_fd); pp_fd = -1;
            err = std::string("PP connection failed: ") + std::strerror(saved_errno);
            return false;
        }
    }
    if (pp_fd < 0) {
        err = std::string("PP connection failed: ") + std::strerror(errno);
        return false;
    }
    std::printf("  %s rank %d: connected\n",
                tp_enabled() ? "TP" : "PP", comm_rank());
    std::fflush(stdout);
    return true;
}

bool Grimoire::pp_send_hidden(const float* dev, size_t elems) {
    if (elems > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(elems, q);
        pipe_host_elems = pipe_host ? elems : 0;
    }
    if (!pipe_host) return false;
    q.memcpy(pipe_host, dev, elems * sizeof(float)).wait();
    return pp_write_all(pipe_host, elems * sizeof(float));
}

bool Grimoire::pp_recv_hidden(float* dev, size_t elems) {
    if (elems > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(elems, q);
        pipe_host_elems = pipe_host ? elems : 0;
    }
    if (!pipe_host) return false;
    if (!pp_read_all(pipe_host, elems * sizeof(float))) return false;
    q.memcpy(dev, pipe_host, elems * sizeof(float)).wait();
    return true;
}

int Grimoire::pp_sync_token(int token) {
    int32_t wire = int32_t(token);
    if (comm_rank() == 1) {
        if (!pp_write_all(&wire, sizeof(wire))) return -1;
    } else if (!pp_read_all(&wire, sizeof(wire))) {
        return -1;
    }
    return int(wire);
}

bool Grimoire::tp_allgather(float* dev, int elems, int begin, int count) {
    if (!tp_enabled() || pp_fd < 0) return false;
    if (size_t(elems) > pipe_host_elems) {
        if (pipe_host) sycl::free(pipe_host, q);
        pipe_host = sycl::malloc_host<float>(size_t(elems), q);
        pipe_host_elems = pipe_host ? size_t(elems) : 0;
    }
    if (!pipe_host) return false;
    q.memcpy(pipe_host + begin, dev + begin, size_t(count) * sizeof(float)).wait();
    const int peer_begin = tp_rank == 0 ? count : 0;
    if (tp_rank == 0) {
        if (!pp_write_all(pipe_host + begin, size_t(count) * sizeof(float)) ||
            !pp_read_all(pipe_host + peer_begin, size_t(count) * sizeof(float)))
            return false;
    } else {
        if (!pp_read_all(pipe_host + peer_begin, size_t(count) * sizeof(float)) ||
            !pp_write_all(pipe_host + begin, size_t(count) * sizeof(float)))
            return false;
    }
    q.memcpy(dev + peer_begin, pipe_host + peer_begin,
             size_t(count) * sizeof(float)).wait();
    return true;
}

// ---------------------------------------------------------------------
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
            std::printf("  pipeline: 2x B70 on one shared context\n");
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
    max_seq = opt.max_seq;

    if (pp_enabled() && tp_enabled()) {
        err = "GRIMOIRE_PP_RANK and GRIMOIRE_TP_RANK are mutually exclusive";
        return false;
    }
    if (tp_enabled())
        std::printf("  multiprocess TP rank %d: output-row projection shards\n",
                    tp_rank);

    if ((pp_enabled() || tp_enabled()) && pipeline_enabled()) {
        err = "multiprocess rank mode and single-process GRIMOIRE_PIPELINE are mutually exclusive";
        return false;
    }
    if (pp_enabled()) {
        const char* split_env = std::getenv("GRIMOIRE_PP_SPLIT");
        // GPU 0 is the faster/direct card: give it 60% of the transformer
        // blocks. Ornith has 40 layers, hence the production default 24,16.
        const int split = split_env && *split_env ? std::atoi(split_env)
                                                   : (cfg.n_layers * 3) / 5;
        if (split < 1 || split >= cfg.n_layers) {
            err = "GRIMOIRE_PP_SPLIT must be between 1 and n_layers-1";
            return false;
        }
        pp_begin = pp_rank == 0 ? 0 : split;
        pp_end   = pp_rank == 0 ? split : cfg.n_layers;
        std::printf("  multiprocess PP rank %d: layers [%d,%d)\n",
                    pp_rank, pp_begin, pp_end);
    }

    const int H = cfg.hidden;
    const int Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;

    size_t bytes = 0;
    bool ok = true;
    auto acct = [&](size_t b) { bytes += b; };

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
    embed = dev_copy_t<bf16_t>(q, ck, ck.embed, "embed_tokens", &ok);
    if (!ok) { err = "embed upload failed"; return false; }
    std::printf("ok\n");
    acct(ck.bytes(ck.embed));

    std::printf("  final_norm    ... ");
    std::fflush(stdout);
    fnorm = dev_copy_t<bf16_t>(q, ck, ck.final_norm, "model.norm.weight", &ok);
    if(cfg.is_muse)
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
    if (ck.lm_head.ok() && ck.lm_head.t.shape.size() == 2) {
        const int V = int(ck.lm_head.t.shape[0]);
        if (!preserve_muse_lm_head && opt.quantize_lm_head &&
            opt.lm_head_fmt != Fmt::BF16) {
            lm_head = quantize_upload_t(q, ck, ck.lm_head, opt.lm_head_fmt, "lm_head", &ok);
            acct(size_t(double(V) * H * bits_per_elem(opt.lm_head_fmt) / 8.0));
        } else {
            lm_head.payload = dev_copy_t<uint8_t>(q, ck, ck.lm_head, "lm_head", &ok);
            lm_head.w = QuantWeight{Fmt::BF16, V, H, lm_head.payload, nullptr, nullptr,
                                    int64_t(H) * 2, 0};
            acct(size_t(V) * H * 2);
        }
    }

    std::printf("ok\n");
    std::fflush(stdout);

    // ---- layers
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

        d.in_norm   = dev_copy_t<bf16_t>(lq, ck, src.input_norm, "input_layernorm", &ok);
        d.post_norm = dev_copy_t<bf16_t>(lq, ck, src.post_attn_norm, "post_attention_layernorm", &ok);
        if(cfg.is_muse){
            d.in_norm_f16=upload_f16_vector_t(lq,ck,src.input_norm,
                                               "input_layernorm.fp16",&ok);
            d.post_norm_f16=upload_f16_vector_t(lq,ck,src.post_attn_norm,
                                                 "post_attention_layernorm.fp16",&ok);
        }

        if (d.kind == LayerKind::LINEAR_ATTN) {
            // The three big ones. 67 MB/layer in bf16 -> ~18 MB at int4.
            d.la_qkv = quantize_upload_t(lq, ck, src.la_in_qkv, PF, "la.in_proj_qkv", &ok);
            d.la_z   = quantize_upload_t(lq, ck, src.la_in_z,   PF, "la.in_proj_z",   &ok);
            d.la_out = quantize_upload_t(lq, ck, src.la_out,    PF, "la.out_proj",    &ok);
            acct(size_t(d.la_qkv.w.bytes() + d.la_z.w.bytes() + d.la_out.w.bytes()
                      + d.la_ab.w.bytes()));

            // small tensors stay bf16
            // a and b are both [Hv][H] and feed the same gate kernel;
            // concatenating them into [2*Hv][H] at load turns two tiny
            // GEMV launches into one for every linear layer.
            d.la_ab = concat_upload_t(lq, ck, src.la_in_a, src.la_in_b,
                                      Fmt::BF16, "la.in_proj_ab", &ok);
            // Experimental only: the single wider GEMM measured slower on B70
            // than the three specialized shapes.  Do not spend VRAM on the
            // concatenated copy in the production path.
            if(PF==Fmt::MXFP4 && std::getenv("GRIMOIRE_FUSE_DN_PROJECTIONS"))
                d.la_all=concat4_native_mxfp4_t(lq,ck,src.la_in_qkv,src.la_in_z,
                    src.la_in_a,src.la_in_b,"la.in_proj_qkv_z_ab",&ok);
            d.la_conv = dev_copy_t<bf16_t>(lq, ck, src.la_conv1d, "la.conv1d", &ok);
            d.la_Alog = dev_copy_t<bf16_t>(lq, ck, src.la_A_log, "la.A_log", &ok);
            d.la_dtb  = dev_copy_t<bf16_t>(lq, ck, src.la_dt_bias, "la.dt_bias", &ok);
            d.la_norm = dev_copy_t<bf16_t>(lq, ck, src.la_norm, "la.norm", &ok);

            // recurrent state: constant in context length
            d.dn_state  = sycl::malloc_device<float>(size_t(Hv) * Dv * Dk, lq);
            const int conv_ch = src.la_conv1d.ok() && !src.la_conv1d.t.shape.empty()
                              ? int(src.la_conv1d.t.shape[0]) : qkv_ch;
            const int conv_k  = src.la_conv1d.ok() && src.la_conv1d.t.shape.size() >= 3
                              ? int(src.la_conv1d.t.shape[2]) : cfg.conv_kernel;
            d.conv_ring = sycl::malloc_device<float>(size_t(conv_ch) * (conv_k - 1), lq);
            acct(size_t(Hv) * Dv * Dk * 4 + size_t(conv_ch) * (conv_k - 1) * 4);
        } else {
            d.q_proj = quantize_upload_t(lq, ck, src.q_proj, PF, "self_attn.q_proj", &ok);
            d.k_proj = quantize_upload_t(lq, ck, src.k_proj, PF, "self_attn.k_proj", &ok);
            d.v_proj = quantize_upload_t(lq, ck, src.v_proj, PF, "self_attn.v_proj", &ok);
            if(cfg.is_muse&&PF==Fmt::INT4){
                d.qkv_proj=concat_upload_many_int4_t(lq,ck,
                    {src.q_proj,src.k_proj,src.v_proj},"self_attn.qkv_proj",&ok);
            }
            d.o_proj = quantize_upload_t(lq, ck, src.o_proj, PF, "self_attn.o_proj", &ok);
            // Per-head q/k RMSNorm, applied before RoPE. These were
            // resolved from the checkpoint but never uploaded or used.
            if (src.q_norm.ok())
                d.q_norm = dev_copy_t<bf16_t>(lq, ck, src.q_norm, "self_attn.q_norm", &ok);
            if (src.k_norm.ok())
                d.k_norm = dev_copy_t<bf16_t>(lq, ck, src.k_norm, "self_attn.k_norm", &ok);
            if (cfg.is_muse) {
                d.pre_ff_norm  = dev_copy_t<bf16_t>(lq, ck, src.pre_ff_norm,  "mlp.pre_ff_norm",  &ok);
                d.post_ff_norm = dev_copy_t<bf16_t>(lq, ck, src.post_ff_norm, "mlp.post_ff_norm", &ok);
                d.pre_ff_norm_f16=upload_f16_vector_t(lq,ck,src.pre_ff_norm,
                                                       "mlp.pre_ff_norm.fp16",&ok);
                d.post_ff_norm_f16=upload_f16_vector_t(lq,ck,src.post_ff_norm,
                                                        "mlp.post_ff_norm.fp16",&ok);
                if (src.attn_gate.ok())
                    d.o_gate = quantize_upload_t(lq, ck, src.attn_gate, PF, "self_attn.gate_proj", &ok);
            }
            acct(size_t(d.q_proj.w.bytes() + d.k_proj.w.bytes()
                      + d.v_proj.w.bytes() + d.qkv_proj.w.bytes()
                      + d.o_proj.w.bytes()));

            // FP8 E4M3 KV: K is D-major for coalesced scoring, V is D-minor.
            d.k_cache = sycl::malloc_device<uint8_t>(size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq, lq);
            d.v_cache = sycl::malloc_device<uint8_t>(size_t(cfg.n_kv_heads) * max_seq * cfg.head_dim, lq);
            acct(2 * size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq);
            if(cfg.is_muse){
                // Match Fusion's XPU FlashAttention KV-cache group.
                constexpr int block_size=64;
                const int blocks=(max_seq+block_size-1)/block_size;
                const size_t elems=size_t(blocks)*block_size*cfg.n_kv_heads*
                    cfg.head_dim;
                d.k_cache_f16=sycl::malloc_device<sycl::half>(elems,lq);
                d.v_cache_f16=sycl::malloc_device<sycl::half>(elems,lq);
                if(!d.k_cache_f16||!d.v_cache_f16)ok=false;
                acct(2*elems*sizeof(sycl::half));
            }
        }

        // ---- FFN ------------------------------------------------------
        if (cfg.is_moe()) {
            const int I = cfg.moe_inter, E = cfg.n_experts;
            (void)E;
            d.router = quantize_upload_t(lq, ck, src.router, Fmt::BF16, "mlp.gate", &ok);

            // Experts: concatenate gate and up into one [E][2I][H] block
            // and copy the packed bytes verbatim. No dequantize, no
            // requantize -- the on-disk layout IS the kernel layout.
            const Fmt EF = (!src.e_gate_p.empty() && src.e_gate_p[0].gptq)
                         ? Fmt::INT4 : Fmt::MXFP4;
            const size_t gu_row  = bytes_per_row(EF, H);
            const size_t gu_srow = scales_per_row(EF, H) * (EF == Fmt::INT4 ? sizeof(bf16_t) : 1);
            const size_t dn_row  = bytes_per_row(EF, I);
            const size_t dn_srow = scales_per_row(EF, I) * (EF == Fmt::INT4 ? sizeof(bf16_t) : 1);
            const size_t gu_zrow = EF == Fmt::INT4 ? scales_per_row(EF, H) : 0;
            const size_t dn_zrow = EF == Fmt::INT4 ? scales_per_row(EF, I) : 0;

            d.gu_pack  = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_row, lq);
            d.gu_scale = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_srow, lq);
            d.dn_pack  = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_row, lq);
            d.dn_scale = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_srow, lq);
            if (gu_zrow) d.gu_zero = sycl::malloc_device<uint8_t>(size_t(E) * 2 * I * gu_zrow, lq);
            if (dn_zrow) d.dn_zero = sycl::malloc_device<uint8_t>(size_t(E) * H * dn_zrow, lq);

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
                    const bool packed = !r.gptq && !r.row_scaled &&
                                        r.t.name.find("weight_packed") != std::string::npos;
                    if (packed) return ck.read_raw(r, payload, rr);
                    if(r.native && r.native->encoding==
                       uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2)){
                        const size_t pb=size_t(N)*row_bytes;
                        const size_t sb=size_t(N)*scale_bytes;
                        std::memcpy(payload,static_cast<const uint8_t*>(
                          ck.native_model->payload(*r.native))+r.native_payload_offset,pb);
                        std::memcpy(scales,static_cast<const uint8_t*>(
                          ck.native_model->scales(*r.native))+r.native_scale_offset,sb);
                        return true;
                    }
                    PackedWeight pw;
                    if (r.gptq) {
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
                    std::memcpy(scales, pw.scales_raw.data(), pw.scales_raw.size());
                    if (zero_bytes) {
                        if (pw.zeros.size() != size_t(N) * zero_bytes) {
                            rr = "expert zero layout mismatch"; return false;
                        }
                        std::memcpy(zeros, pw.zeros.data(), pw.zeros.size());
                    }
                    return true;
                };
                for (int e = 0; e < E; ++e) {
                    const size_t goff = size_t(e) * 2 * I;
                    if (!src.e_gate_p[e].ok() || !src.e_up_p[e].ok() ||
                        !src.e_down_p[e].ok()) {
                        std::printf("\n  missing expert %d in layer %d\n", e, i);
                        err = "expert tensors incomplete";
                        return false;
                    }
                    std::string rr;
                    const bool direct = !src.e_gate_p[e].gptq && !src.e_gate_p[e].row_scaled &&
                                        src.e_gate_p[e].t.name.find("weight_packed") != std::string::npos;
                    bool rok;
                    if (direct) {
                        rok = ck.read_raw(src.e_gate_p[e], h_gu.data() + goff * gu_row, rr)
                           && ck.read_raw(src.e_up_p[e], h_gu.data() + (goff + I) * gu_row, rr)
                           && ck.read_raw(src.e_gate_s[e], h_gs.data() + goff * gu_srow, rr)
                           && ck.read_raw(src.e_up_s[e], h_gs.data() + (goff + I) * gu_srow, rr)
                           && ck.read_raw(src.e_down_p[e], h_dn.data() + size_t(e) * H * dn_row, rr)
                           && ck.read_raw(src.e_down_s[e], h_ds.data() + size_t(e) * H * dn_srow, rr);
                    } else {
                        rok = stage_expert(src.e_gate_p[e], h_gu.data() + goff * gu_row,
                                           h_gs.data() + goff * gu_srow, gu_zrow ? h_gz.data() + goff * gu_zrow : nullptr,
                                           I, H, gu_row, gu_srow, gu_zrow, rr)
                           && stage_expert(src.e_up_p[e], h_gu.data() + (goff + I) * gu_row,
                                           h_gs.data() + (goff + I) * gu_srow, gu_zrow ? h_gz.data() + (goff + I) * gu_zrow : nullptr,
                                           I, H, gu_row, gu_srow, gu_zrow, rr)
                           && stage_expert(src.e_down_p[e], h_dn.data() + size_t(e) * H * dn_row,
                                           h_ds.data() + size_t(e) * H * dn_srow, dn_zrow ? h_dz.data() + size_t(e) * H * dn_zrow : nullptr,
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
                q.memcpy(d.gu_scale, h_gs.data(), h_gs.size());
                lq.memcpy(d.dn_pack,  h_dn.data(), h_dn.size());
                q.memcpy(d.dn_scale, h_ds.data(), h_ds.size());
                if (d.gu_zero) q.memcpy(d.gu_zero, h_gz.data(), h_gz.size());
                if (d.dn_zero) q.memcpy(d.dn_zero, h_dz.data(), h_dz.size());
                q.wait();
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
            if (src.sh_gate_w.ok()) {
                d.sh_gate_q = quantize_upload_t(lq, ck, src.sh_gate_w, Fmt::BF16, "shared_expert_gate", &ok);
                d.has_sh_gate = true;
            }
            acct(size_t(d.sh_gu.w.bytes() + d.sh_down.w.bytes()));
        } else {
            d.sh_gu   = concat_upload_t(lq, ck, src.sh_gate, src.sh_up, PF,
                                        "mlp.gate_up", &ok);
            d.sh_down = quantize_upload_t(lq, ck, src.sh_down, PF, "mlp.down_proj", &ok);
            acct(size_t(d.sh_gu.w.bytes() + d.sh_down.w.bytes()));
        }

        // ---- W4A8 weights (opt-in) -----------------------------------
        // Symmetric int4 g128 copies of the two FFN matrices, converted on
        // device from the MXFP4 originals.  Opt-in because it costs ~9 GB of
        // VRAM on top of the model: the MXFP4 weights stay because decode's
        // GEMV beats any GEMM at M=1.
        if (w4a8_enabled() && ok) {
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

    // Verification needs one vocabulary projection per candidate.  Convert
    // the head to the same symmetric-int4 representation as the W4A8 model
    // projections so the verifier can stream it once for all rows.  Decode
    // also consumes this representation through gemv_any(), so there is no
    // second copy and no decode/verify weight mismatch.
    if (w4a8_enabled() && lm_head.w.fmt == Fmt::MXFP4 && lm_head.payload &&
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
    if (mtp_enabled()) {
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
        const TensorRef t_e0 = ref("mtp.layers.0.mlp.experts.0.gate_proj.weight");
        const bool mtp_moe = cfg.is_moe() && t_router.ok() && t_e0.ok();
        if (!t_fc.ok() || !t_q.ok() || (!t_g.ok() && !mtp_moe)) {
            std::printf("not present in this checkpoint -- MTP disabled\n");
        } else {
            bool mok = true;
            // The MTP head's format follows what the checkpoint actually
            // stores. b70_compile_model keeps mtp.* RAW BF16 (a 4-bit draft
            // head accepts 0-23% of its drafts vs 43-73% at BF16), but older
            // artifacts packed it to MXFP4 and a packed native tensor cannot
            // be read back as a float matrix. So: RAW -> BF16, packed -> PF.
            auto mtp_fmt = [&](const TensorRef& r) {
                return (r.native && r.native->encoding ==
                        uint32_t(NativeEncoding::RAW)) ? Fmt::BF16 : PF;
            };
            mtp.fc      = quantize_upload_t(q, ck, t_fc, mtp_fmt(t_fc), "mtp.fc", &mok);
            mtp.pre_h   = dev_copy_t<bf16_t>(q, ck, t_preh, "mtp.pre_h", &mok);
            mtp.pre_e   = dev_copy_t<bf16_t>(q, ck, t_pree, "mtp.pre_e", &mok);
            mtp.norm    = dev_copy_t<bf16_t>(q, ck, t_nrm,  "mtp.norm",  &mok);
            LayerDev& m = mtp.L;
            m.kind      = LayerKind::FULL_ATTN;
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
                // Ornith's MTP decoder layer is MoE, just like every target
                // layer: 256 routed experts (top-8) plus an always-on shared
                // expert. model-v2.b70 already stores all 768 expert matrices
                // in the exact expert-major MXFP4 layout consumed by the
                // normal decode kernels, so assemble one contiguous layer
                // without dequantizing or requantizing anything.
                const int H = cfg.hidden, I = cfg.moe_inter, E = cfg.n_experts;
                const Fmt EF = Fmt::MXFP4;
                const size_t gu_row = bytes_per_row(EF, H);
                const size_t gu_srow = scales_per_row(EF, H);
                const size_t dn_row = bytes_per_row(EF, I);
                const size_t dn_srow = scales_per_row(EF, I);
                const size_t gu_rows = size_t(E) * 2 * I;
                const size_t dn_rows = size_t(E) * H;

                m.router = quantize_upload_t(q, ck, t_router, Fmt::BF16,
                                             "mtp.mlp.gate", &mok);
                m.gu_pack = sycl::malloc_device<uint8_t>(gu_rows * gu_row, q);
                m.gu_scale = sycl::malloc_device<uint8_t>(gu_rows * gu_srow, q);
                m.dn_pack = sycl::malloc_device<uint8_t>(dn_rows * dn_row, q);
                m.dn_scale = sycl::malloc_device<uint8_t>(dn_rows * dn_srow, q);
                if (!m.gu_pack || !m.gu_scale || !m.dn_pack || !m.dn_scale)
                    mok = false;

                auto copy_native = [&](const TensorRef& r, uint8_t* payload,
                                       uint8_t* scales, size_t pb, size_t sb,
                                       const char* what) {
                    if (!mok) return;
                    if (!r.ok() || !r.native ||
                        r.native->encoding !=
                            uint32_t(NativeEncoding::MXFP4_GRIMOIRE_XE2)) {
                        std::printf("\n  %s is not native MXFP4\n", what);
                        mok = false;
                        return;
                    }
                    q.memcpy(payload,
                        static_cast<const uint8_t*>(ck.native_model->payload(*r.native))
                            + r.native_payload_offset, pb);
                    q.memcpy(scales,
                        static_cast<const uint8_t*>(ck.native_model->scales(*r.native))
                            + r.native_scale_offset, sb);
                };
                for (int e = 0; e < E && mok; ++e) {
                    const std::string p = "mtp.layers.0.mlp.experts." +
                                          std::to_string(e) + ".";
                    const TensorRef eg = ref((p + "gate_proj.weight").c_str());
                    const TensorRef eu = ref((p + "up_proj.weight").c_str());
                    const TensorRef ed = ref((p + "down_proj.weight").c_str());
                    const size_t goff = size_t(e) * 2 * I;
                    copy_native(eg, m.gu_pack + goff * gu_row,
                        m.gu_scale + goff * gu_srow,
                        size_t(I) * gu_row, size_t(I) * gu_srow,
                        "mtp expert gate");
                    copy_native(eu, m.gu_pack + (goff + I) * gu_row,
                        m.gu_scale + (goff + I) * gu_srow,
                        size_t(I) * gu_row, size_t(I) * gu_srow,
                        "mtp expert up");
                    copy_native(ed, m.dn_pack + size_t(e) * H * dn_row,
                        m.dn_scale + size_t(e) * H * dn_srow,
                        size_t(H) * dn_row, size_t(H) * dn_srow,
                        "mtp expert down");
                }
                q.wait();

                m.moe.cfg.hidden = H; m.moe.cfg.inter = I;
                m.moe.cfg.num_experts = E; m.moe.cfg.top_k = cfg.top_k;
                m.moe.cfg.shared_inter = cfg.shared_inter;
                m.moe.gate_up = QuantWeight{EF, E * 2 * I, H,
                    m.gu_pack, m.gu_scale, nullptr, int64_t(gu_row),
                    scales_per_row(EF, H)};
                m.moe.down = QuantWeight{EF, E * H, I,
                    m.dn_pack, m.dn_scale, nullptr, int64_t(dn_row),
                    scales_per_row(EF, I)};

                const TensorRef t_sg = ref(
                    "mtp.layers.0.mlp.shared_expert.gate_proj.weight");
                const TensorRef t_su = ref(
                    "mtp.layers.0.mlp.shared_expert.up_proj.weight");
                const TensorRef t_sd = ref(
                    "mtp.layers.0.mlp.shared_expert.down_proj.weight");
                const TensorRef t_sgw = ref(
                    "mtp.layers.0.mlp.shared_expert_gate.weight");
                m.sh_gu = concat_upload_t(q, ck, t_sg, t_su, PF,
                                          "mtp.shared.gate_up", &mok);
                m.sh_down = quantize_upload_t(q, ck, t_sd, PF,
                                              "mtp.shared.down", &mok);
                m.sh_gate_q = quantize_upload_t(q, ck, t_sgw, Fmt::BF16,
                                                "mtp.shared.gate", &mok);
                m.has_sh_gate = t_sgw.ok();
                mtp_ffn_bytes = gu_rows * (gu_row + gu_srow)
                              + dn_rows * (dn_row + dn_srow)
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
                              size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq, q);
            m.v_cache   = sycl::malloc_device<uint8_t>(
                              size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq, q);
            const size_t mb = mtp.fc.w.bytes() + m.q_proj.w.bytes() + m.k_proj.w.bytes()
                            + m.v_proj.w.bytes() + m.o_proj.w.bytes()
                            + mtp_ffn_bytes
                            + size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq * 2;
            acct(mb);
            mtp.ok = mok;
            std::printf("%s (%.2f GiB)\n", mok ? "ok" : "FAILED",
                        double(mb) / 1073741824.0);
        }
    }

    // ---- DFlash sidecar weights -------------------------------------
    const char* dpath=std::getenv("GRIMOIRE_DFLASH_MODEL");
    if(!dpath||!*dpath)dpath=std::getenv("GRIMOIRE_DFLASH2_MODEL");
    if (dpath && *dpath) {
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
        dflash2.q_heads=int(dr("layers.0.self_attn.q_proj.weight").t.shape[0])/
                           dflash2.head_dim;
        dflash2.kv_heads=int(dr("layers.0.self_attn.k_proj.weight").t.shape[0])/
                            dflash2.head_dim;
        dflash2.inter=int(dr("layers.0.mlp.gate_proj.weight").t.shape[0]);
        if(cfg.is_muse){
            dflash2.layers.resize(5);
            dflash2.target_layers={1,13,25,37,49};
            dflash2.mask_token=201818;
            dflash2.rope_theta=500000.0f;
            dflash2.sliding_window=2048;
            dflash2.shared_embed_f16=upload_f16_t(
                q,ck,ck.embed,"dflash2.shared_embed",&dok);
            dflash2.shared_lm_head_f16=upload_f16_t(
                q,ck,ck.lm_head,"dflash2.shared_lm_head",&dok);
            db+=dflash2.shared_embed_f16.w.bytes();
            db+=dflash2.shared_lm_head_f16.w.bytes();
        }else{
            // Non-Muse DFlash/DFlash2 geometry (layer count, target-layer
            // taps, mask token, rope theta, sliding window) is specific to
            // the DRAFT checkpoint, not the target model. Every published
            // DFlash2 checkpoint (Ornith's, Qwen's) ships a config.json with
            // this exact shape, so read it instead of hardcoding one
            // model's numbers -- a fixed 6-layer/40-target-layer draft
            // silently truncated Qwen's 5-layer/64-target-layer one to 6
            // layers and read past its last real layer.
            dflash2.layers.resize(6);
            dflash2.target_layers={1,6,11,16,22,27,32,37};
            dflash2.mask_token=248077;
            dflash2.rope_theta=10000000.0f;
            dflash2.sliding_window=4096;
            const std::string dcfg_path=std::string(dpath)+"/config.json";
            std::ifstream dcfg_f(dcfg_path);
            if(dcfg_f){
                std::string dcfg((std::istreambuf_iterator<char>(dcfg_f)),
                                 std::istreambuf_iterator<char>());
                auto find_int=[&](const char* key,long dflt)->long{
                    const std::string pat=std::string("\"")+key+"\":";
                    const size_t p0=dcfg.find(pat);
                    if(p0==std::string::npos)return dflt;
                    size_t p1=p0+pat.size();
                    while(p1<dcfg.size()&&(dcfg[p1]==' '||dcfg[p1]=='\n'))++p1;
                    if(p1>=dcfg.size()||dcfg[p1]=='n')return dflt;
                    return std::strtol(dcfg.c_str()+p1,nullptr,10);
                };
                auto find_int_arr=[&](const char* key)->std::vector<int>{
                    std::vector<int> out;
                    const std::string pat=std::string("\"")+key+"\":";
                    size_t p0=dcfg.find(pat);
                    if(p0==std::string::npos)return out;
                    p0=dcfg.find('[',p0);
                    if(p0==std::string::npos)return out;
                    const size_t p1=dcfg.find(']',p0);
                    if(p1==std::string::npos)return out;
                    const char* c=dcfg.c_str()+p0+1;
                    const char* end=dcfg.c_str()+p1;
                    while(c<end){
                        char* nx=nullptr;
                        const long v=std::strtol(c,&nx,10);
                        if(nx==c)break;
                        out.push_back(int(v));c=nx;
                        while(c<end&&(*c==','||*c==' '||*c=='\n'))++c;
                    }
                    return out;
                };
                const long nl=find_int("num_hidden_layers",6);
                if(nl>0)dflash2.layers.resize(size_t(nl));
                const std::vector<int> tl=find_int_arr("target_layer_ids");
                if(!tl.empty())dflash2.target_layers=tl;
                dflash2.mask_token=int(find_int("mask_token_id",dflash2.mask_token));
                dflash2.rope_theta=float(find_int("rope_theta",
                    long(dflash2.rope_theta)));
                dflash2.sliding_window=int(find_int("sliding_window",
                    dflash2.sliding_window));
            }
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
        const Fmt dflash_fmt=muse_draft_q?Fmt::MXFP4
            :(cfg.is_muse?Fmt::BF16:(dflash2.v2?PF:Fmt::BF16));
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
        if(cfg.is_muse){
            dflash2.hidden_norm_f16=hload(hn_name,"dflash.hidden_norm");
            dflash2.norm_f16=hload("norm.weight","dflash2.norm");
        }else{
            dflash2.hidden_norm=bload(hn_name,"dflash.hidden_norm");
            dflash2.norm=bload("norm.weight","dflash2.norm");
        }
        if(dflash2.v2){
            dflash2.selector_hidden=qload(
                "candidate_selector.hidden_projection.weight","dflash2.selector_hidden");
            dflash2.predecessor=bload(
                "candidate_selector.predecessor_codebook","dflash2.predecessor");
            dflash2.successor=bload(
                "candidate_selector.successor_codebook","dflash2.successor");
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
            // Original DFlash uses sliding attention for layers 0..4 and
            // full attention for layer 5. DFlash2's six layers are sliding.
            d.sliding=cfg.is_muse||dflash2.v2||i<5;
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
                // BUG-COMPATIBILITY WITH FUSION, NOT MODEL SEMANTICS.
                // vLLM builds _k_norm_weights as a [num_layers, head_dim]
                // stack and hands it to ops.rms_norm, whose weight must be
                // [head_dim]. The kernel therefore reads only the first
                // head_dim values and applies LAYER 0's k_norm to every draft
                // layer in the context precompute, despite the comment there
                // claiming the weight is selected per layer. Verified against
                // the running reference: the effective weight recovered from
                // Fusion's own pre-RoPE context K is identical for all five
                // layers (pairwise cos 1.0000, rms 1.08547) and equals
                // layers.0.self_attn.k_norm.weight, while the checkpoint's
                // five k_norm tensors genuinely differ (rms 1.085, 1.349,
                // 0.895, 1.381, 0.955). Applying the per-layer weights here -
                // the model-faithful thing - puts Grimoire's context K at
                // cos 0.93-0.97 against Fusion; layer 0's weight for all
                // raises every layer to ~0.99.
                // The draft QUERY path is unaffected and keeps its correct
                // per-layer d.k_norm_f16, which already matches Fusion.
                for(size_t i=0;i<dflash2.layers.size();++i)
                    q.memcpy(dflash2.k_norm_all_f16+i*dflash2.head_dim,
                        dflash2.layers[0].k_norm_f16,
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
                (dflash2.draft_lm_head_i4.has_i4()?
                    dflash2.draft_lm_head_i4.w.N:cfg.vocab);
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
                dflash2.block_table=sycl::malloc_device<int32_t>(dflash2.num_blocks,q);
                dflash2.cu_q=sycl::malloc_device<int32_t>(2,q);
                dflash2.cu_k=sycl::malloc_device<int32_t>(2,q);
                dflash2.seqused_k=sycl::malloc_device<int32_t>(1,q);
                db+=size_t(dflash2.num_blocks+5)*sizeof(int32_t);
                if(!dflash2.q_f16||!dflash2.k_f16||!dflash2.v_f16||
                   !dflash2.attn_f16||!dflash2.linear_in_f16||
                   !dflash2.linear_out_f16||!dflash2.block_table||!dflash2.cu_q||
                   !dflash2.cu_k||!dflash2.seqused_k)dok=false;
                if(dok){
                    std::vector<int32_t> blocks(size_t(dflash2.num_blocks));
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
                const int VW=std::max({cfg.hidden,cfg.n_heads*cfg.head_dim,
                    cfg.n_kv_heads*cfg.head_dim,2*cfg.dense_inter,cfg.vocab});
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
        acct(db);
        std::printf("ok (%s, %.2f GiB device, target taps + draft KV ready)\n",
                    dflash2.v2?"DFlash2DraftModel":"DFlashDraftModel",
                    double(db)/1073741824.0);
    }

    if (mtp.ok) {
        mtp.cat   = sycl::malloc_device<float>(size_t(cfg.hidden) * 2, q);
        mtp.x     = sycl::malloc_device<float>(size_t(cfg.hidden), q);
        mtp.h2    = sycl::malloc_device<float>(size_t(cfg.hidden), q);
        mtp.resid = sycl::malloc_device<float>(size_t(cfg.hidden), q);
    }

    if (mtp.ok || dflash2.ok) {
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

    std::printf("\n  scratch buffers ... ");
    std::fflush(stdout);

    // ---- scratch ------------------------------------------------------
    const int TK = cfg.is_moe() ? cfg.top_k : 1;
    const int SI = cfg.is_moe() ? cfg.shared_inter : cfg.dense_inter;
    s.h       = sycl::malloc_device<float>(H, q);
    s.h2      = sycl::malloc_device<float>(H, q);
    s.resid   = sycl::malloc_device<float>(H, q);
    // qkv scratch serves both paths; the full-attn q_proj is larger
    // when attn_output_gate doubles its rows.
    int qkv_max = qkv_ch;
    for (const auto& dl : L) {
        if (dl.la_qkv.w.N > qkv_max) qkv_max = dl.la_qkv.w.N;
        if (dl.q_proj.w.N > qkv_max) qkv_max = dl.q_proj.w.N;
    }
    s.qkv     = sycl::malloc_device<float>(qkv_max, q);
    int aux_max = Hv * Dv;
    for (const auto& dl : L) {
        if (dl.la_z.w.N   > aux_max) aux_max = dl.la_z.w.N;
        if (dl.k_proj.w.N > aux_max) aux_max = dl.k_proj.w.N;
        if (dl.v_proj.w.N > aux_max) aux_max = dl.v_proj.w.N;
    }
    s.zbuf    = sycl::malloc_device<float>(aux_max, q);
    s.abuf    = sycl::malloc_device<float>(aux_max * 2, q);  // a|b concatenated
    s.bbuf    = sycl::malloc_device<float>(aux_max, q);
    // attn_out is shared by the DeltaNet path (Hv*Dv) and the full
    // attention path (n_heads*head_dim). With head_dim 256 the latter is
    // 4096, equal to the former only by coincidence -- size it from both.
    {
        int ao = std::max(H, Hv * Dv);
        ao = std::max(ao, cfg.n_heads * cfg.head_dim);
        s.attn_out = sycl::malloc_device<float>(ao, q);
    }
    s.moe_h   = sycl::malloc_device<float>(size_t(TK) * SI + SI, q);
    s.moe_y   = sycl::malloc_device<float>(H, q);
    s.logits  = sycl::malloc_device<float>(cfg.vocab, q);
    s.rlogits = sycl::malloc_device<float>(cfg.is_moe() ? cfg.n_experts : 1, q);
    s.d_expert= sycl::malloc_device<int32_t>(TK, q);
    s.d_weight= sycl::malloc_device<float>(TK, q);
    s.qsplit  = sycl::malloc_device<float>(size_t(cfg.n_heads) * cfg.head_dim, q);
    s.gsplit  = sycl::malloc_device<float>(size_t(cfg.n_heads) * cfg.head_dim, q);
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
    s.part    = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS * cfg.head_dim, q);
    s.pm      = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS, q);
    s.pl      = sycl::malloc_device<float>(size_t(kSpecBatch) * cfg.n_heads * MAX_SPLITS, q);
    q.wait();

    if (pp_enabled() || tp_enabled()) {
        pipe_host = sycl::malloc_host<float>(size_t(H), q);
        if (!pipe_host) { err = "multiprocess host staging allocation failed"; return false; }
        pipe_host_elems = size_t(H);
        if (!pp_connect(err)) return false;
    }

    std::printf("ok\n  zeroing recurrent state ... ");
    std::fflush(stdout);
    reset();
    std::printf("ok\n");
    vram_gb = double(bytes) / 1073741824.0;
    load_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - t0).count();
    return true;
}

// ---------------------------------------------------------------------
// A new sequence must start from a zeroed recurrent state. The DeltaNet
// state carries the ENTIRE history in a fixed buffer, so forgetting this
// silently conditions the next conversation on the previous one -- with
// no error and no obvious symptom.
// ---------------------------------------------------------------------
void Grimoire::reset() {
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const int qkv_ch = 2 * cfg.lin_k_heads * cfg.lin_k_dim + Hv * Dv;
    for (auto& d : L) {
        if (d.dn_state)
            q.memset(d.dn_state, 0, size_t(Hv) * Dv * Dk * sizeof(float));
        if (d.conv_ring)
            q.memset(d.conv_ring, 0, size_t(qkv_ch) * (cfg.conv_kernel - 1) * sizeof(float));
    }
    const int32_t z = 0, one = 1;
    if (s.d_pos)     q.memcpy(s.d_pos, &z, sizeof(int32_t));
    if (s.d_seq_len) q.memcpy(s.d_seq_len, &one, sizeof(int32_t));
    q.wait();
    dag_tail.clear();
    dflash2.context_pos = 0;
    pos = 0;
}

bool Grimoire::save_prefix(const std::vector<int32_t>& tokens) {
    if (!prefix_cache_enabled() || tokens.empty()) return false;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const size_t dn_bytes = size_t(Hv) * Dv * Dk * sizeof(float);
    const size_t conv_bytes = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
        Hv * Dv) * (cfg.conv_kernel - 1) * sizeof(float);
    const size_t kv_bytes = size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq;
    if (prefix_cache.layers.empty()) {
        prefix_cache.layers.resize(L.size());
        for (size_t i = 0; i < L.size(); ++i) {
            auto& c = prefix_cache.layers[i];
            const auto& d = L[i];
            if (d.dn_state) c.dn = sycl::malloc_device<float>(dn_bytes / sizeof(float), q);
            if (d.conv_ring) c.conv = sycl::malloc_device<float>(conv_bytes / sizeof(float), q);
            if (d.k_cache) c.k = sycl::malloc_device<uint8_t>(kv_bytes, q);
            if (d.v_cache) c.v = sycl::malloc_device<uint8_t>(kv_bytes, q);
            if ((d.dn_state && !c.dn) || (d.conv_ring && !c.conv) ||
                (d.k_cache && !c.k) || (d.v_cache && !c.v)) {
                std::fprintf(stderr, "  prefix cache allocation failed\n");
                return false;
            }
        }
        prefix_cache.hidden = sycl::malloc_device<float>(cfg.hidden, q);
        prefix_cache.logits = sycl::malloc_device<float>(cfg.vocab, q);
        if (!prefix_cache.hidden || !prefix_cache.logits) return false;
    }
    for (size_t i = 0; i < L.size(); ++i) {
        const auto& d = L[i]; auto& c = prefix_cache.layers[i];
        if (d.dn_state) q.memcpy(c.dn, d.dn_state, dn_bytes);
        if (d.conv_ring) q.memcpy(c.conv, d.conv_ring, conv_bytes);
        if (d.k_cache) q.memcpy(c.k, d.k_cache, kv_bytes);
        if (d.v_cache) q.memcpy(c.v, d.v_cache, kv_bytes);
    }
    q.memcpy(prefix_cache.hidden, s.h, size_t(cfg.hidden) * sizeof(float));
    q.memcpy(prefix_cache.logits, s.logits, size_t(cfg.vocab) * sizeof(float)).wait();
    prefix_cache.tokens = tokens;
    prefix_cache.valid = true;
    return true;
}

bool Grimoire::restore_prefix(const std::vector<int32_t>& tokens) {
    if (!prefix_cache_enabled() || !prefix_cache.valid ||
        prefix_cache.tokens != tokens) return false;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim, Dk = cfg.lin_k_dim;
    const size_t dn_bytes = size_t(Hv) * Dv * Dk * sizeof(float);
    const size_t conv_bytes = size_t(2 * cfg.lin_k_heads * cfg.lin_k_dim +
        Hv * Dv) * (cfg.conv_kernel - 1) * sizeof(float);
    const size_t kv_bytes = size_t(cfg.n_kv_heads) * cfg.head_dim * max_seq;
    for (size_t i = 0; i < L.size(); ++i) {
        auto& d = L[i]; const auto& c = prefix_cache.layers[i];
        if (d.dn_state) q.memcpy(d.dn_state, c.dn, dn_bytes);
        if (d.conv_ring) q.memcpy(d.conv_ring, c.conv, conv_bytes);
        if (d.k_cache) q.memcpy(d.k_cache, c.k, kv_bytes);
        if (d.v_cache) q.memcpy(d.v_cache, c.v, kv_bytes);
    }
    q.memcpy(s.h, prefix_cache.hidden, size_t(cfg.hidden) * sizeof(float));
    q.memcpy(s.logits, prefix_cache.logits, size_t(cfg.vocab) * sizeof(float));
    pos = int(tokens.size());
    q.memcpy(s.d_pos, &pos, sizeof(int32_t));
    q.memcpy(s.d_seq_len, &pos, sizeof(int32_t)).wait();
    std::printf("  prefix cache HIT: %zu tokens\n", tokens.size());
    return true;
}

void Grimoire::snapshot_recurrent() {
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
    q.memcpy(s.d_pos, &saved_pos, sizeof(int32_t));
    q.memcpy(s.d_seq_len, &saved_pos, sizeof(int32_t));
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
    q.memcpy(s.h, spec_hidden_steps + size_t(step) * cfg.hidden,
             size_t(cfg.hidden) * sizeof(float));
    pos = saved_pos + accepted;
    q.memcpy(s.d_pos, &pos, sizeof(int32_t));
    q.memcpy(s.d_seq_len, &pos, sizeof(int32_t));
}

void Grimoire::release() {
    if(dflash2.fc_plan){
        auto od=load_onednn_bf16();
        if(od)od.destroy(dflash2.fc_plan);
        dflash2.fc_plan=nullptr;
    }
    if (pp_fd >= 0) { ::close(pp_fd); pp_fd = -1; }
    if (comm_rank() == 1 && !pp_socket.empty()) ::unlink(pp_socket.c_str());
    if (pipe_host) { sycl::free(pipe_host, q); pipe_host = nullptr; }
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
        for (void* p : {(void*)d.in_norm, (void*)d.post_norm,
                        (void*)d.in_norm_f16, (void*)d.post_norm_f16,
                        (void*)d.pre_ff_norm_f16, (void*)d.post_ff_norm_f16,
                        (void*)d.la_conv, (void*)d.la_Alog, (void*)d.la_dtb,
                        (void*)d.la_norm,  (void*)d.q_norm, (void*)d.k_norm, (void*)d.gu_pack,
                        (void*)d.gu_scale, (void*)d.dn_pack, (void*)d.dn_scale,
                        (void*)d.gu_zero, (void*)d.dn_zero,
                        (void*)d.dn_state, (void*)d.conv_ring,
                        (void*)d.k_cache, (void*)d.v_cache,
                        (void*)d.k_cache_f16, (void*)d.v_cache_f16})
            if (p) sycl::free(p, q);
    }
    if (mtp.ok) {
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
    dflash2.fused_context_kv.release(q);
    dflash2.shared_embed_f16.release(q);
    dflash2.shared_lm_head_f16.release(q);
    dflash2.draft_lm_head_i4.release(q);
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
    lm_head.release(q);
    if (embed) sycl::free(embed, q);
    if (fnorm) sycl::free(fnorm, q);
    if (fnorm_f16) sycl::free(fnorm_f16, q);
    if (spec_dn_state) sycl::free(spec_dn_state, q);
    if (spec_conv_ring) sycl::free(spec_conv_ring, q);
    if (spec_dn_steps) sycl::free(spec_dn_steps, q);
    if (spec_conv_inputs) sycl::free(spec_conv_inputs, q);
    if (spec_hidden_steps) sycl::free(spec_hidden_steps, q);
    for (auto& c : prefix_cache.layers)
        for (void* p : {(void*)c.dn, (void*)c.conv, (void*)c.k, (void*)c.v})
            if (p) sycl::free(p, q);
    if (prefix_cache.hidden) sycl::free(prefix_cache.hidden, q);
    if (prefix_cache.logits) sycl::free(prefix_cache.logits, q);
    prefix_cache = {};
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
            const bool gated = QD == 2 * cfg.n_heads * cfg.head_dim;
            sycl::event e_qp = gemv_any(d.q_proj, s.h2, s.qkv, deps({e_in}));
            sycl::event e_qready = e_qp;
            if (gated)
                e_qready = launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                    cfg.n_heads, cfg.head_dim, deps({e_qp}));
            float* qvec = gated ? s.qsplit : s.qkv;

            sycl::event e_kp = gemv_any(d.k_proj, s.h2, s.zbuf,
                deps({(dag_mask & 2) ? e_in : e_qready}));
            sycl::event e_vp = gemv_any(d.v_proj, s.h2, s.bbuf,
                deps({(dag_mask & 2) ? e_in : e_kp}));
            sycl::event e_qn = e_qready;
            if (d.q_norm)
                e_qn = launch_rmsnorm_heads(q, qvec, d.q_norm, cfg.n_heads,
                    cfg.head_dim, cfg.rms_eps, true,
                    deps({(dag_mask & 2) ? e_qready : e_vp}));
            sycl::event e_kn = e_kp;
            if (d.k_norm)
                e_kn = launch_rmsnorm_heads(q, s.zbuf, d.k_norm, cfg.n_kv_heads,
                    cfg.head_dim, cfg.rms_eps, true,
                    deps({(dag_mask & 2) ? e_kp : e_qn}));
            sycl::event e_qr = launch_rope_dev(q, qvec, cfg.n_heads, cfg.head_dim,
                s.d_pos, cfg.rope_theta, cfg.partial_rope,
                deps({(dag_mask & 2) ? e_qn : e_kn}));
            sycl::event e_kr = launch_rope_dev(q, s.zbuf, cfg.n_kv_heads, cfg.head_dim,
                s.d_pos, cfg.rope_theta, cfg.partial_rope,
                deps({(dag_mask & 2) ? e_kn : e_qr}));
            sycl::event e_kv = launch_kv_append_dev(q, s.zbuf, s.bbuf,
                d.k_cache, d.v_cache, s.d_pos, cfg.n_kv_heads, cfg.head_dim,
                max_seq, deps({e_kr, e_vp}));

            AttnParams ap{};
            ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
            ap.out = s.attn_out; ap.seq_len = pos + 1; ap.seq_cap = max_seq;
            ap.head_dim = cfg.head_dim; ap.num_heads = cfg.n_heads;
            ap.num_kv_heads = cfg.n_kv_heads;
            ap.softmax_scale = 1.0f / std::sqrt(float(cfg.head_dim));
            ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
            ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
            sycl::event e_fd = launch_flash_decode(q, ap, deps({e_qr, e_kv}));
            sycl::event e_fm = launch_flash_merge(q, ap, deps({e_fd}));
            sycl::event e_gate = e_fm;
            if (gated)
                e_gate = launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit,
                    cfg.n_heads * cfg.head_dim, deps({e_fm, e_qready}));
            e_attn = gemv_any(d.o_proj, s.attn_out, s.moe_y, deps({e_gate}));
        }

        e_h = launch_rmsnorm_residual(q, s.h, s.moe_y, d.post_norm,
                                      s.h2, H, cfg.rms_eps, deps({e_attn}));
        if (cfg.is_moe()) {
            sycl::event e_router = gemv_any(d.router, s.h2, s.rlogits, deps({e_h}));
            sycl::event e_top = launch_router_topk(q, s.rlogits, cfg.n_experts,
                cfg.top_k, s.d_expert, s.d_weight, true, deps({e_router}));
            sycl::event e_gu = launch_moe_gate_up(
                q, d.moe, s.d_expert, s.h2, s.moe_h, deps({e_top}));
            sycl::event e_routed = launch_moe_down(
                q, d.moe, s.d_expert, s.d_weight, s.moe_h, s.moe_y, deps({e_gu}));

            sycl::event e_shgu = gemv_any(d.sh_gu, s.h2, s.sh_g,
                deps({(dag_mask & 4) ? e_h : e_routed}));
            const int SI = d.sh_gu.w.N / 2;
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
            const int FI = d.sh_gu.w.N / 2;
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
    const int H = cfg.hidden, HD = cfg.head_dim;
    const int QH = cfg.n_heads, KVH = cfg.n_kv_heads;
    const int QW = QH * HD;                 // attention output width
    const std::vector<sycl::event> none{};
    const float eps = cfg.rms_eps;
    const float sm_scale = cfg.query_prescale / std::sqrt(float(HD));

    // embed, then SCALELESS RMSNorm on the token embedding (Muse: no sqrt(H)).
    launch_embed(q, embed, token, s.h2, H, none);
    launch_rmsnorm_residual(q, s.h2, nullptr, muse_zero, s.h, H, eps, none);

    for (int i = 0; i < cfg.n_layers; ++i) {
        LayerDev& d = L[i];
        if (dflash2.ok) {
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
        gemv_any(d.v_proj, s.h2, s.bbuf, none);
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
        launch_flash_decode(q, ap, none);
        launch_flash_merge(q, ap, none);
        // per-head sigmoid attention output gate (separate projection).
        gemv_any(d.o_gate, s.h2, s.gsplit, none);
        launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit, QW, none);
        gemv_any(d.o_proj, s.attn_out, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
        // --- feed-forward block (sandwich: pre_ff -> mlp -> post_ff -> +res)
        launch_rmsnorm_residual(q, s.h, nullptr, d.pre_ff_norm, s.h2, H, eps, none);
        const int I = d.sh_gu.w.N / 2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        launch_swiglu(q, s.sh_g, s.sh_g + I, s.sh_g, I, none);
        gemv_any(d.sh_down, s.sh_g, s.moe_y, none);
        launch_rmsnorm_residual_batched(q, s.moe_y, nullptr, nullptr,
            d.post_ff_norm, s.sh_out, 1, H, cfg.post_norm_eps, nullptr, none);
        launch_add(q, s.h, s.sh_out, H, none);
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

const float* Grimoire::forward(int token) {
    if (cfg.is_muse) return forward_muse(token);
    if (dag) return forward_dag(token);
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
    if (pp_enabled() && pp_rank == 1) {
        if (!pp_recv_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank 1: hidden receive failed\n");
            return nullptr;
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
        if (dflash2.ok) {
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
            const int qkv_ch = d.la_qkv.w.N;         // real shape, not config math
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
            const int QD = d.q_proj.w.N;
            const int KD = d.k_proj.w.N;
            // With attn_output_gate the projection emits [q | gate] per
            // head, so only half its rows are queries. Feeding all of
            // them to attention treats gate values as queries.
            const bool gated = (QD == 2 * cfg.n_heads * cfg.head_dim);
            gemv_any(d.q_proj, s.h2, s.qkv, none);
            if (gated)
                launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                                   cfg.n_heads, cfg.head_dim, none);
            float* qvec = gated ? s.qsplit : s.qkv;
            MK("  q gemv + split");
            gemv_any(d.k_proj, s.h2, s.zbuf, none);
            gemv_any(d.v_proj, s.h2, s.bbuf, none);
            MK("  k+v gemv");

            const int qheads = cfg.n_heads;
            // q_norm / k_norm come BEFORE RoPE in the reference:
            //   q = q_norm(q.view(heads, head_dim))
            //   k = k_norm(k_proj(x).view(heads, head_dim))
            //   q, k = apply_rotary_pos_emb(q, k, cos, sin)
            // Both are Qwen3_5MoeRMSNorm, so zero-centered.
            if ((fusion_mask & 2) && d.q_norm && d.k_norm) {
                launch_qk_norm_rope(q, qvec, s.zbuf, d.q_norm, d.k_norm,
                    qheads, cfg.n_kv_heads, cfg.head_dim, s.d_pos,
                    cfg.rope_theta, cfg.partial_rope, cfg.rms_eps, none);
            } else {
                if (d.q_norm)
                    launch_rmsnorm_heads(q, qvec, d.q_norm, qheads, cfg.head_dim,
                                         cfg.rms_eps, true, none);
                if (d.k_norm)
                    launch_rmsnorm_heads(q, s.zbuf, d.k_norm, cfg.n_kv_heads,
                                         cfg.head_dim, cfg.rms_eps, true, none);
                launch_rope_dev(q, qvec, qheads, cfg.head_dim, s.d_pos,
                                cfg.rope_theta, cfg.partial_rope, none);
                launch_rope_dev(q, s.zbuf, cfg.n_kv_heads, cfg.head_dim, s.d_pos,
                                cfg.rope_theta, cfg.partial_rope, none);
            }
            MK("  q/k norm + rope");
            launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                                 s.d_pos, cfg.n_kv_heads, cfg.head_dim,
                                 max_seq, none);
            MK("  kv_append");

            AttnParams ap{};
            ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
            ap.out = s.attn_out;
            ap.seq_len = pos + 1; ap.seq_cap = max_seq;
            ap.head_dim = cfg.head_dim; ap.num_heads = qheads;
            ap.num_kv_heads = cfg.n_kv_heads;
            ap.softmax_scale = 1.0f / std::sqrt(float(cfg.head_dim));
            ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
            // FIXED split count so the launch geometry never changes and
            // the graph stays valid; the kernel reads the live length.
            ap.splits    = GRAPH_SPLITS;
            ap.d_seq_len = s.d_seq_len;
            if (i == probe_layer) probe("FA v", s.bbuf, cfg.n_kv_heads * cfg.head_dim);
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
                    s.attn_out, 1, pos + 1 + bdelta, qheads, cfg.n_kv_heads,
                    cfg.head_dim, max_seq, ap.softmax_scale,
                    s.part, s.pm, s.pl, vs, {});
                MK("  flash_decode_batched");
            } else {
            launch_flash_decode(q, ap, none);
            MK("  flash_decode");
            launch_flash_merge(q, ap, none);
            MK("  flash_merge");
            }
            if (i == probe_layer) probe("FA attn core", s.attn_out, qheads * cfg.head_dim);

            // apply the output gate before projecting back
            if (gated)
                launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit,
                                        cfg.n_heads * cfg.head_dim, none);
            if (i == probe_layer) probe("FA after gate", s.attn_out, qheads * cfg.head_dim);
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

        if (cfg.is_moe()) {
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
            launch_moe_gate_up(q, d.moe, s.d_expert, s.h2, s.moe_h, none);
            MK("  moe_gate_up");
            if (i == probe_layer) probe("L0 moe_h (gate_up)", s.moe_h, cfg.top_k * I);
            launch_moe_down(q, d.moe, s.d_expert, s.d_weight, s.moe_h, s.moe_y, none);
            MK("  moe_down");
            if (i == probe_layer) probe("L0 moe routed", s.moe_y, H);

            // always-on shared expert, added to the routed result
            const int SI = d.sh_gu.w.N / 2;
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
            const int FI = d.sh_gu.w.N / 2;
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
    if (pp_enabled() && pp_rank == 0) {
        if (fusion_mask & 4) {
            launch_add(q, s.h, s.moe_y, H, none);
            launch_add(q, s.h, s.sh_out, H, none);
        } else {
            launch_add(q, s.h, s.moe_y, H, none);
        }
        if (!pp_send_hidden(s.h, size_t(H))) {
            std::fprintf(stderr, "PP rank 0: hidden send failed\n");
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
int Grimoire::mtp_draft(int next_token, int position, bool from_mtp_hidden) {
    if (!mtp.ok) return -1;
    const int H = cfg.hidden;
    const std::vector<sycl::event> none{};
    LayerDev& d = mtp.L;

    // position for this draft: token t+1 sits at `position`
    q.memcpy(s.d_pos, &position, sizeof(int32_t));
    const int32_t seq = position + 1;
    q.memcpy(s.d_seq_len, &seq, sizeof(int32_t));

    // cat order.  DeepSeek/Qwen MTP is fc([norm(embedding) ; norm(hidden)]),
    // i.e. EMBEDDING FIRST.  Hidden-first measured 0/159 acceptance, which is
    // what a wrong concat looks like -- not a weak head.
    // residual == nullptr, so s.h is read and NOT modified.
    static const bool hid_first = std::getenv("GRIMOIRE_MTP_HID_FIRST") != nullptr;
    float* p_emb = hid_first ? mtp.cat + H : mtp.cat;
    float* p_hid = hid_first ? mtp.cat     : mtp.cat + H;
    // mtp.x is read here and only overwritten by the fc gemv below, so a
    // chained draft can use it in place as its hidden input.
    const float* hsrc = from_mtp_hidden ? mtp.x : s.h;
    launch_rmsnorm_residual(q, const_cast<float*>(hsrc), nullptr, mtp.pre_h,
                            p_hid, H, cfg.rms_eps, none);
    launch_embed(q, embed, next_token, mtp.resid, H, none);
    launch_rmsnorm_residual(q, mtp.resid, nullptr, mtp.pre_e, p_emb,
                            H, cfg.rms_eps, none);
    launch_gemv(q, mtp.fc.w, mtp.cat, mtp.x, none);      // [H] = [H][2H] x [2H]

    // ---- one full-attention decoder layer over mtp.x --------------------
    q.memset(s.moe_y, 0, size_t(H) * sizeof(float));
    launch_rmsnorm_residual(q, mtp.x, nullptr, d.in_norm, s.h2, H, cfg.rms_eps, none);

    const int QD = d.q_proj.w.N;
    const bool gated = (QD == 2 * cfg.n_heads * cfg.head_dim);
    gemv_any(d.q_proj, s.h2, s.qkv, none);
    if (gated)
        launch_split_qgate(q, s.qkv, s.qsplit, s.gsplit,
                           cfg.n_heads, cfg.head_dim, none);
    float* qvec = gated ? s.qsplit : s.qkv;
    gemv_any(d.k_proj, s.h2, s.zbuf, none);
    gemv_any(d.v_proj, s.h2, s.bbuf, none);

    if ((fusion_mask & 2) && d.q_norm && d.k_norm) {
        launch_qk_norm_rope(q, qvec, s.zbuf, d.q_norm, d.k_norm,
            cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, s.d_pos,
            cfg.rope_theta, cfg.partial_rope, cfg.rms_eps, none);
    } else {
        if (d.q_norm)
            launch_rmsnorm_heads(q, qvec, d.q_norm, cfg.n_heads, cfg.head_dim,
                                 cfg.rms_eps, true, none);
        if (d.k_norm)
            launch_rmsnorm_heads(q, s.zbuf, d.k_norm, cfg.n_kv_heads,
                                 cfg.head_dim, cfg.rms_eps, true, none);
        launch_rope_dev(q, qvec, cfg.n_heads, cfg.head_dim, s.d_pos,
                        cfg.rope_theta, cfg.partial_rope, none);
        launch_rope_dev(q, s.zbuf, cfg.n_kv_heads, cfg.head_dim, s.d_pos,
                        cfg.rope_theta, cfg.partial_rope, none);
    }
    launch_kv_append_dev(q, s.zbuf, s.bbuf, d.k_cache, d.v_cache,
                         s.d_pos, cfg.n_kv_heads, cfg.head_dim, max_seq, none);

    AttnParams ap{};
    ap.q = qvec; ap.k_cache = d.k_cache; ap.v_cache = d.v_cache;
    ap.out = s.attn_out;
    ap.seq_len = position + 1; ap.seq_cap = max_seq;
    ap.head_dim = cfg.head_dim; ap.num_heads = cfg.n_heads;
    ap.num_kv_heads = cfg.n_kv_heads;
    ap.softmax_scale = 1.0f / std::sqrt(float(cfg.head_dim));
    ap.partials = s.part; ap.part_m = s.pm; ap.part_l = s.pl;
    ap.splits = GRAPH_SPLITS; ap.d_seq_len = s.d_seq_len;
    launch_flash_decode(q, ap, none);
    launch_flash_merge(q, ap, none);
    if (gated)
        launch_gate_sigmoid_mul(q, s.attn_out, s.gsplit,
                                cfg.n_heads * cfg.head_dim, none);
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

        const int SI = d.sh_gu.w.N / 2;
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
        const int FI = d.sh_gu.w.N / 2;
        gemv_any(d.sh_gu, s.h2, s.sh_g, none);
        launch_swiglu(q, s.sh_g, s.sh_g + FI, s.sh_g, FI, none);
        gemv_any(d.sh_down, s.sh_g, s.moe_y, none);
    }

    // ---- final norm + the model's own lm_head ---------------------------
    launch_rmsnorm_residual(q, mtp.x, s.moe_y, mtp.norm, s.h2, H, cfg.rms_eps, none);
    static const int draft_vocab = [] {
        const char* v = std::getenv("GRIMOIRE_MTP_DRAFT_VOCAB");
        return v && *v ? std::atoi(v) : 0;
    }();
    const int dv = draft_vocab > 0 ? std::min(draft_vocab, cfg.vocab) : cfg.vocab;
    if (dv < cfg.vocab && lm_head.has_i4())
        launch_gemv_int4sym(q, lm_head.i4, lm_head.i4s, s.h2, s.logits,
                            dv, H, none);
    else
        gemv_any(lm_head, s.h2, s.logits, none);
    launch_argmax(q, s.logits, dv, s.d_tok, s.d_val, none);
    int32_t tok = 0;
    q.memcpy(&tok, s.d_tok, sizeof(int32_t)).wait();
    return int(tok);
}

int Grimoire::argmax_token() {
    const std::vector<sycl::event> none{};
    // Only the late-stage rank owns valid logits. Send its selected token
    // back so rank 0's independent generation loop stays in lockstep.
    if (pp_enabled() && pp_rank == 0) return pp_sync_token(-1);
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
bool Grimoire::dflash_draft(int bonus_token, int position,
                            std::vector<int32_t>& draft_tokens,
                            bool context_only) {
    constexpr int MMAX=16;
    // Ornith's verify slope is ~2.95 ms per extra verified token, so the
    // profitable block is far shorter than the drafter's native 16. Let the
    // depth be tuned; M is the block INCLUDING the bonus row, so M=4 verifies
    // 3 drafts.
    static const int m_env=[]{const char* v=std::getenv("GRIMOIRE_DFLASH_M");
        return v&&*v?std::atoi(v):0;}();
    int M=dflash2.draft_head_rows?8:MMAX;
    if(m_env>=2&&m_env<=MMAX)M=m_env;
    if(!dflash2.ok||position<0||position+M>max_seq){
        static bool once2=false;
        if(!once2){once2=true;std::fprintf(stderr,
            "  DFlash2: entry guard ok=%d pos=%d M=%d max_seq=%d\n",
            int(dflash2.ok),position,M,max_seq);}
        return false;}
    // DFlash2 needs its dynamic convolutions; without them the draft is the
    // wrong function and acceptance collapses silently.
    if(dflash2.v2&&(!dflash2.conv_delta||!dflash2.conv_scratch)){
        static bool once=false;
        if(!once){once=true;std::fprintf(stderr,
            "  DFlash2: conv buffers missing (taps %d groups %d delta %p scratch %p)\n",
            dflash2.conv_taps,dflash2.conv_groups,
            (void*)dflash2.conv_delta,(void*)dflash2.conv_scratch);}
        return false;}
    if(dflash2.v2&&cfg.is_muse)return false;
    const int H=dflash2.hidden,QH=dflash2.q_heads,KVH=dflash2.kv_heads;
    const int HD=dflash2.head_dim,QW=QH*HD,KVW=KVH*HD,I=dflash2.inter;
    const int MASK=dflash2.mask_token;
    const int NT=int(dflash2.target_layers.size());
    const float eps=cfg.is_muse?1.0e-5f:1.0e-6f;
    const float theta=dflash2.rope_theta;
    const auto fa2_paged=cfg.is_muse?load_xe2_dflash_paged_f16():nullptr;
    if(cfg.is_muse&&!fa2_paged)return false;
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
    static int dflash_dump_call=0;
    const bool dumping=dump_dir&&*dump_dir&&dflash_dump_call==0;
    ++dflash_dump_call;
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
    auto mm=[&](const DevQuant& w,const float* x,float* y,int rows){
        if(cfg.is_muse&&w.fp16){
            const sycl::half* out=mm_f16_raw(w,x,rows);
            launch_f16_to_f32(q,out,y,size_t(rows)*w.w.N,{});
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
            for(auto& d:dflash2.layers){
                mm(d.k,dflash2.normed,dflash2.k,rows);
                mm(d.v,dflash2.normed,dflash2.v,rows);
                launch_qk_norm_rope_batched(q,dflash2.q,dflash2.k,
                    d.q_norm,d.k_norm,rows,QH,KVH,HD,start,theta,1.0f,eps,{},0.0f);
                launch_kv_append_batched(q,dflash2.k,dflash2.v,
                    d.k_cache,d.v_cache,rows,start,KVH,HD,max_seq);
            }
        }
        checkpoint("context KV");
        dflash2.context_pos+=rows;
    }
    draft_mark("context ingest");
    if(context_only){
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
    else
        launch_embed_batched(q,embed,dflash2.tokens,dflash2.resid,M,H);
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
            const int window=d.sliding?dflash2.sliding_window-1:-1;
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
                    "num_blocks=%d window=%d causal=0\n",
                    position,M,used,sk,cq[0],cq[1],ck2[0],ck2[1],
                    dflash2.block_size,dflash2.num_blocks,window);
                std::fflush(stderr);
            }
            const int rc=fa2_paged(&q,dflash2.q_f16,d.k_cache_f16,
                d.v_cache_f16,dflash2.attn_f16,M,used,QH,KVH,HD,
                dflash2.block_size,dflash2.num_blocks,dflash2.block_table,
                dflash2.cu_q,dflash2.cu_k,dflash2.seqused_k,
                1.0f/std::sqrt(float(HD)),window,window,false);
            if(rc)return false;
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
                d.sliding?dflash2.sliding_window:0,false,
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
        if(!head_w4)return false;
        launch_quantize_rows_int8(q,dflash2.normed+H,dflash2.a8,dflash2.a8s,
                                  M-1,H,{});
        head_w4(&q,dflash2.a8,dflash2.draft_head_i4,
                dflash2.draft_head_i4s,dflash2.a8s,dflash2.logits,
                M-1,dflash2.draft_head_rows,H);
    }else if(cfg.is_muse&&dflash2.draft_lm_head_i4.has_i4()&&w4){
        const auto& head=dflash2.draft_lm_head_i4;
        launch_quantize_rows_int8(q,dflash2.normed+H,dflash2.a8,dflash2.a8s,
                                  M-1,H,{});
        w4(&q,dflash2.a8,head.i4,head.i4s,dflash2.a8s,dflash2.logits,
           M-1,head.w.N,H);
    }else if(cfg.is_muse){
        mm(dflash2.shared_lm_head_f16,dflash2.normed+H,
           dflash2.logits,M-1);
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
    const int draft_vocab=dflash2.draft_head_rows?
        dflash2.draft_head_rows:cfg.vocab;
    const int draft_stride=dflash2.draft_logits_stride?
        dflash2.draft_logits_stride:draft_vocab;
    for(int r=0;r<M-1;++r){
        launch_argmax(q,dflash2.logits+int64_t(r)*draft_stride,draft_vocab,
                      s.d_tok,s.d_val,{});
        if(dflash2.draft_head_rows){
            const int32_t* map=dflash2.draft_head_token_ids;
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
                            std::vector<int32_t>* next_tokens) {
    const int M=int(tokens.size());
    if(M<=0||pos+M>max_seq)return false;
    const int start_pos=pos;
    if(!next_tokens&&start_pos==0&&restore_prefix(tokens))return true;
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
        if(dflash2.ok){
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
        const int window_left=d.muse_sliding?2047:-1;
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
    if(next_tokens&&M<=kSpecBatch&&spec_hidden_steps)
        launch_f16_to_f32(q,hidden,spec_hidden_steps,size_t(M)*H,{});
    launch_f16_to_f32(q,hidden+int64_t(M-1)*H,s.h,size_t(H),{});
    pos+=M;
    q.memcpy(s.d_pos,&pos,sizeof(int32_t));
    q.memcpy(s.d_seq_len,&pos,sizeof(int32_t));
    if(next_tokens){
        next_tokens->resize(M);
        q.memcpy(next_tokens->data(),outtok,size_t(M)*sizeof(int32_t));
    }
    q.wait_and_throw();
    cleanup();
    if(!next_tokens&&start_pos==0)save_prefix(tokens);
    return true;
}

bool Grimoire::prefill(const std::vector<int32_t>& tokens,
                       std::vector<int32_t>* next_tokens) {
    if (cfg.is_muse) {
        if(std::getenv("GRIMOIRE_MUSE_SEQUENTIAL_PREFILL"))return false;
        return prefill_muse(tokens,next_tokens);
    }
    const int M = int(tokens.size());
    if (M <= 0 || pos + M > max_seq) return false;
    const int start_pos = pos;
    if (!next_tokens && start_pos == 0 && restore_prefix(tokens)) return true;
    if (pp_enabled() && next_tokens) {
        std::fprintf(stderr, "multiprocess PP does not yet support verifier prefill\n");
        return false;
    }
    const int H = cfg.hidden, Hk = cfg.lin_k_heads, Dk = cfg.lin_k_dim;
    const int Hv = cfg.lin_v_heads, Dv = cfg.lin_v_dim;
    const int qkv_ch = 2 * Hk * Dk + Hv * Dv;
    int W = H;
    for (const auto& d : L) {
        const DevQuant* ws[] = {&d.la_qkv,&d.la_z,&d.la_out,&d.la_ab,&d.la_all,&d.q_proj,&d.k_proj,
            &d.v_proj,&d.o_proj,&d.router,&d.sh_gu,&d.sh_down,&d.sh_gate_q};
        for (auto* w : ws) { W = std::max(W, w->w.N); W = std::max(W, w->w.K); }
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
    float* alpha=df(gdn_tokens*Hv); float* beta=df(gdn_tokens*Hv);
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
    Xe2GroupedMXFP4 xe2_grouped_mxfp4 = xe2_grouped_mxfp4_big;
    if(M<=8 && xe2_grouped_mxfp4_m8)  xe2_grouped_mxfp4 = xe2_grouped_mxfp4_m8;
    else if(M<=16 && xe2_grouped_mxfp4_m16) xe2_grouped_mxfp4 = xe2_grouped_mxfp4_m16;
    Xe2FusedGateUpMXFP4 xe2_fused_gate_up=load_xe2_fused_gate_up_mxfp4();
    Xe2ChunkPrefill xe2_attention=load_xe2_chunk_prefill();
    load_bestla(cfg.n_layers);
    Xe2ChunkGdn xe2_gdn=load_xe2_chunk_gdn();
    Xe2ChunkGdnRaw xe2_gdn_raw=load_xe2_chunk_gdn_raw();
    OneDnnW4Api od=load_onednn_w4();
    const bool parallel_prefill=std::getenv("GRIMOIRE_PARALLEL_PREFILL") && xe2_dense_mxfp4;
    const bool parallel_shared=std::getenv("GRIMOIRE_PARALLEL_SHARED") && xe2_dense_mxfp4;
    const bool need_aux=parallel_prefill||parallel_shared;
    const bool norm_bf_only=M>=32&&xe2_dense_mxfp4&&xe2_dense_mxfp4_f32&&
        (!cfg.is_moe()||xe2_grouped_mxfp4)&&!need_aux;
    const bool defer_moe_gather=std::getenv("GRIMOIRE_DEFER_MOE_GATHER")&&
        cfg.is_moe()&&M>=32&&xe2_grouped_mxfp4&&norm_bf_only;
    // Dense Qwen has no routed experts (R=M, I=1) but its fused MLP projection
    // is much wider than H.  Size bridge scratch from every real projection,
    // not from MoE geometry, or the native dense kernel writes past the end.
    // xperm also backs the padded GDN q/k/v triple (each gdn_tokens rows), so
    // it must cover that as well as the MoE permute and the widest projection.
    const size_t gdn_qkv_elems=gdn_tokens*(2*size_t(Hk)*Dk+size_t(Hv)*Dv);
    const size_t bridge_elems=std::max(std::max(size_t(R)*std::max(H,I),
                                                size_t(M)*W),gdn_qkv_elems);
    sycl_bf16* xperm=sycl::malloc_device<sycl_bf16>(bridge_elems,q);
    sycl_bf16* grouped_out=(xe2_grouped||xe2_grouped_mxfp4||xe2_dense_mxfp4||xe2_attention||xe2_gdn||od)
        ? sycl::malloc_device<sycl_bf16>(std::max(std::max(size_t(R)*std::max(H,2*I),
              size_t(M)*W),gdn_tokens*size_t(Hv)*Dv),q) : nullptr;
    sycl_bf16* moe_res=defer_moe_gather?sycl::malloc_device<sycl_bf16>(size_t(R)*H,q):nullptr;
    float* yperm=df(size_t(R)*H);
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
        static const char* const base_names[]={"bh","bn","r0","r1","t0","t1","t2","t3","t4",
            "la_fused","xb","dtok","rex","rwt","rlog","mh","alpha","beta","xperm","yperm",
            "ptoken","pinv"};
        const char* name=mi<sizeof(base_names)/sizeof(base_names[0])?base_names[mi]:"optional";
        std::fprintf(stderr,"    prefill allocation failed: %s (M=%d H=%d W=%d R=%d I=%d)\n",
                     name,M,H,W,R,I);
        for(void* z:mem) if(z) sycl::free(z,q); return false;
    }
    q.memcpy(dtok,tokens.data(),size_t(M)*sizeof(int32_t));
    if (pp_enabled() && pp_rank == 1) {
        if (!pp_recv_hidden(bh, size_t(M) * H)) {
            std::fprintf(stderr, "PP rank 1: batched hidden receive failed\n");
            for(void* z:mem) if(z) sycl::free(z,q);
            return false;
        }
    } else {
        launch_embed_batched(q,embed,dtok,bh,M,H);
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
    const bool exact_verify = next_tokens &&
        std::getenv("GRIMOIRE_MTP_EXACT_VERIFY") != nullptr;
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
    auto mm=[&](const DevQuant& w,const float* x,float* y){
        if(exact_verify && M<=4){
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
        if(M<=16 && w.w.N<=2048){
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
    int raw_gdn_layer_limit=cfg.n_layers;
    if(const char* v=std::getenv("GRIMOIRE_RAW_GDN_LAYER_LIMIT"))
        raw_gdn_layer_limit=std::max(0,std::min(cfg.n_layers,std::atoi(v)));
    const bool capture_spec = next_tokens && M <= kSpecBatch;
    const bool spec_route_diag = capture_spec &&
        std::getenv("GRIMOIRE_MTP_ROUTE_DIAG") != nullptr;
    size_t spec_route_total = 0, spec_route_unique = 0;
    int spec_route_layers = 0;
    std::vector<size_t> spec_doff(L.size(), size_t(-1));
    std::vector<size_t> spec_xoff(L.size(), size_t(-1));
    if (capture_spec) {
        size_t ds = 0, xs = 0;
        const size_t dn_n = size_t(Hv) * Dv * Dk;
        for (size_t li = 0; li < L.size(); ++li) {
            if (L[li].dn_state) { spec_doff[li] = ds; ds += dn_n; }
            if (L[li].conv_ring) {
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
        if(dflash2.ok){
            for(size_t tap=0;tap<dflash2.target_layers.size();++tap){
                if(dflash2.target_layers[tap]+1==li){
                    launch_dflash_store_tap(q,bh,dflash2.target_aux,M,H,
                        int(dflash2.target_layers.size()),start_pos,int(tap),{},
                        cfg.is_muse);
                    break;
                }
            }
        }
        if(d.kind==LayerKind::LINEAR_ATTN){
            const int qs=Hk*Dk, vs=Hv*Dv, ch=d.la_qkv.w.N;
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
            const bool parallel_dn=parallel_prefill&&!fused_in;
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
            if(bf_dn_qkv)
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
                if(M<=16){
                    for(int t=0;t<M;++t){
                        DeltaNetParams sp{};
                        sp.q     = t1 + size_t(t)*Hk*Dk;
                        sp.k     = t2 + size_t(t)*Hk*Dk;
                        sp.v     = t3 + size_t(t)*Hv*Dv;
                        sp.a     = alpha + size_t(t)*Hv;
                        sp.beta  = beta  + size_t(t)*Hv;
                        sp.state = d.dn_state;
                        sp.out   = t0 + size_t(t)*Hv*Dv;
                        sp.n_heads = Hv; sp.k_dim = Dk; sp.v_dim = Dv;
                        sp.n_k_heads = Hk;
                        launch_deltanet_step(q,sp,{});
                        if (capture_spec)
                            q.memcpy(spec_dn_steps + size_t(t) * spec_dn_elems +
                                     spec_doff[li], d.dn_state,
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
            const int QD=d.q_proj.w.N;
            const bool gated=QD==2*cfg.n_heads*cfg.head_dim;
            const bool bfqkv=std::getenv("GRIMOIRE_BF16_QKV")&&xe2_attention&&
                xe2_dense_mxfp4&&xe2_dense_mxfp4_f32&&pos==0&&M>=32&&
                d.q_proj.w.fmt==Fmt::MXFP4&&d.k_proj.w.fmt==Fmt::MXFP4&&
                d.v_proj.w.fmt==Fmt::MXFP4&&d.o_proj.w.fmt==Fmt::MXFP4;
            if(bfqkv){
                const size_t qe=size_t(M)*cfg.n_heads*cfg.head_dim;
                const size_t ke=size_t(M)*cfg.n_kv_heads*cfg.head_dim;
                sycl_bf16*fq=xperm,*fk=fq+qe,*fv=fk+ke;
                mmbb(d.q_proj,bn_bf,grouped_out);
                if(gated)launch_split_qgate_bf16(q,grouped_out,fq,t2,M,
                    cfg.n_heads,cfg.head_dim);
                else q.memcpy(fq,grouped_out,qe*sizeof(sycl_bf16));
                mmbb(d.k_proj,bn_bf,fk);
                mmbb(d.v_proj,bn_bf,fv);
                launch_qk_norm_rope_bf16_batched(q,fq,fk,d.q_norm,d.k_norm,M,
                    cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,pos,cfg.rope_theta,
                    cfg.partial_rope,cfg.rms_eps);
                launch_kv_append_bf16_batched(q,fk,fv,d.k_cache,d.v_cache,M,pos,
                    cfg.n_kv_heads,cfg.head_dim,max_seq);
                xe2_attention(&q,fq,fk,fv,grouped_out,M,M,cfg.n_heads,
                    cfg.n_kv_heads,cfg.head_dim,dtok,dtok,
                    1.0f/std::sqrt(float(cfg.head_dim)),true);
                const sycl_bf16*o_in=grouped_out;
                if(gated){launch_gate_sigmoid_mul_bf16_io(q,grouped_out,t2,xb,qe);o_in=xb;}
                mmb(d.o_proj,o_in,r0);
            }else{
            mm(d.q_proj,bn,t0);
            float* qv=t0;
            if(gated){launch_split_qgate_batched(q,t0,t1,t2,M,cfg.n_heads,cfg.head_dim);qv=t1;}
            mm(d.k_proj,bn,t3); mm(d.v_proj,bn,t4);
            launch_qk_norm_rope_batched(q,qv,t3,d.q_norm,d.k_norm,M,cfg.n_heads,
                cfg.n_kv_heads,cfg.head_dim,pos,cfg.rope_theta,cfg.partial_rope,cfg.rms_eps);
            launch_kv_append_batched(q,t3,t4,d.k_cache,d.v_cache,M,pos,cfg.n_kv_heads,
                cfg.head_dim,max_seq);
            const sycl_bf16* attention_bf=nullptr;
            if(xe2_attention && pos==0 && M>=32){
                const size_t qe=size_t(M)*cfg.n_heads*cfg.head_dim;
                const size_t ke=size_t(M)*cfg.n_kv_heads*cfg.head_dim;
                sycl_bf16* fq=xperm; sycl_bf16* fk=fq+qe; sycl_bf16* fv=fk+ke;
                launch_f32_to_bf16(q,qv,fq,qe);
                launch_f32_to_bf16(q,t3,fk,ke);
                launch_f32_to_bf16(q,t4,fv,ke);
                xe2_attention(&q,fq,fk,fv,grouped_out,M,M,cfg.n_heads,
                    cfg.n_kv_heads,cfg.head_dim,dtok,dtok,
                    1.0f/std::sqrt(float(cfg.head_dim)),true);
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
                    const int vsplits = GRAPH_SPLITS;
                    launch_flash_decode_batched(q,qv,d.k_cache,d.v_cache,t3,
                        M,pos,cfg.n_heads,cfg.n_kv_heads,cfg.head_dim,max_seq,
                        1.0f/std::sqrt(float(cfg.head_dim)),s.part,s.pm,s.pl,
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
                            ap.q=qv+int64_t(r)*cfg.n_heads*cfg.head_dim;
                            ap.k_cache=d.k_cache;ap.v_cache=d.v_cache;
                            ap.out=t3+int64_t(r)*cfg.n_heads*cfg.head_dim;
                            ap.seq_len=pos+r+1;ap.seq_cap=max_seq;
                            ap.head_dim=cfg.head_dim;ap.num_heads=cfg.n_heads;
                            ap.num_kv_heads=cfg.n_kv_heads;
                            ap.softmax_scale=1.0f/std::sqrt(float(cfg.head_dim));
                            ap.partials=s.part;ap.part_m=s.pm;ap.part_l=s.pl;
                            ap.splits=GRAPH_SPLITS;ap.d_seq_len=dtok+r;
                            launch_flash_decode(q,ap,{});
                            launch_flash_merge(q,ap,{});
                        }
                }else launch_flash_prefill(q,qv,d.k_cache,d.v_cache,t3,M,
                    next_tokens ? pos - 1 : pos,cfg.n_heads,
                    cfg.n_kv_heads,cfg.head_dim,max_seq,1.0f/std::sqrt(float(cfg.head_dim)));
            }
            if(attention_bf){
                const sycl_bf16* o_in=attention_bf;
                if(gated){launch_gate_sigmoid_mul_bf16_io(q,attention_bf,t2,xb,
                    size_t(M)*cfg.n_heads*cfg.head_dim);o_in=xb;}
                mmb(d.o_proj,o_in,r0);
            }else{
                if(gated) launch_gate_sigmoid_mul(q,t3,t2,M*cfg.n_heads*cfg.head_dim,{});
                mm(d.o_proj,t3,r0);
            }
            }
            pp_mark("full attention");
        }
        const bool fused_ffn_quant=!exact_verify&&!cfg.is_moe()&&a8&&a8s&&
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
        if(cfg.is_moe()){
            sycl::event shared_ready;
            if(parallel_prefill||parallel_shared){
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
            const bool bf16_router=std::getenv("GRIMOIRE_BF16_ROUTER")&&
                xe2_dense_mxfp4&&d.router.w.fmt==Fmt::MXFP4&&d.router.payload;
            if(bf16_router){
                xe2_dense_mxfp4(&q,bn_bf,d.router.w.payload,
                    static_cast<const unsigned char*>(d.router.w.scales),grouped_out,
                    M,d.router.w.N,d.router.w.K);
                launch_router_topk_bf16_batched(q,grouped_out,M,cfg.n_experts,
                    cfg.top_k,rex,rwt,true);
            }else{
                mm(d.router,bn,rlog);
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
            if(M>=32){
                if(xe2_grouped_mxfp4 && d.moe.gate_up.fmt==Fmt::MXFP4){
                    launch_moe_remap_bf16_top8(q,bn_bf,rex,xperm,grouped_rows,
                                                ptoken,pinv,M,H,cfg.n_experts);
                    pp_mark("MoE device remap");
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
                        QuantWeight z=w; z.N=n; z.payload=w.payload+int64_t(row0)*w.row_bytes;
                        const size_t ss=(w.fmt==Fmt::INT4)?sizeof(bf16_t):1;
                        if(w.scales) z.scales=(uint8_t*)w.scales+int64_t(row0)*w.row_scales*ss;
                        if(w.zeros) z.zeros=w.zeros+int64_t(row0)*w.row_scales;
                        return z;
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
                launch_moe_gate_up_batched(q,d.moe,rex,bn,mh,M);
                launch_moe_down_batched(q,d.moe,rex,rwt,mh,r0,M);
            }
            pp_mark("routed MoE");
            if(parallel_prefill||parallel_shared) wait_on(q,shared_ready);
            else {
                if(!mlp_bf16(d.sh_gu,d.sh_down,r1,li)){
                    const int SI=d.sh_gu.w.N/2;
                    mm(d.sh_gu,bn,t0);launch_swiglu_batched(q,t0,t1,M,SI);mm(d.sh_down,t1,r1);
                }
                if(d.has_sh_gate){mm(d.sh_gate_q,bn,t2);launch_scale_by_sigmoid_batched(q,r1,t2,M,H);}
            }
            pp_mark("shared expert");
        }else{
            if(!mlp_bf16(d.sh_gu,d.sh_down,r0,li)){
                const int FI=d.sh_gu.w.N/2;
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
    if (pp_enabled() && pp_rank == 0) {
        // Fold the last early-layer FFN output into the residual stream before
        // sending the complete MxH boundary tensor to the late-stage rank.
        if (defer_moe_gather)
            launch_moe_unpermute_bf16(q,moe_res,pinv,rwt,r0,
                                      M,cfg.top_k,H);
        launch_add(q,bh,r0,M*H,{});
        launch_add(q,bh,r1,M*H,{});
        if (!pp_send_hidden(bh, size_t(M) * H)) {
            std::fprintf(stderr, "PP rank 0: batched hidden send failed\n");
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
        if (lm_head.has_i4()) {
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
            if (!lm_head.has_i4() && lm_head.w.fmt != Fmt::MXFP4)
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
    if (capture_spec)
        q.memcpy(spec_hidden_steps, bh, size_t(M) * H * sizeof(float));
    q.memcpy(s.h, bh + int64_t(M-1) * H, size_t(H) * sizeof(float));
    pos+=M; q.memcpy(s.d_pos,&pos,sizeof(int)); q.memcpy(s.d_seq_len,&pos,sizeof(int));
    if (next_tokens) {
        next_tokens->resize(M);
        q.memcpy(next_tokens->data(), dtok, size_t(M) * sizeof(int32_t));
    }
    q.wait();
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
    try {
        sycl_ext::command_graph<sycl_ext::graph_state::modifiable>
            g(q.get_context(), q.get_device());

        // Record against a scratch position so the recording itself does
        // not advance the real sequence state.  The host-side pos and the
        // device counters are saved and restored: build_graph() is called
        // AFTER prefill in the generate path, so the live sequence state
        // must survive capture untouched.
        int32_t saved_pos = 0, saved_seq = 0;
        q.memcpy(&saved_pos, s.d_pos, sizeof(int32_t)).wait();
        q.memcpy(&saved_seq, s.d_seq_len, sizeof(int32_t)).wait();
        const int host_pos = pos;

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
        std::printf("  graph capture unavailable (%s); using direct submission\n",
                    e.what());
        std::fflush(stdout);
        return false;
    }
}

const float* Grimoire::step() {
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
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    Tokenizer tk;
    std::string err;
    if (!tk.load(dir, err)) { std::printf("tokenizer: %s\n", err.c_str()); return 1; }

    Grimoire e;
    UploadOptions opt;
    opt.lm_head_fmt = proj_fmt;
    opt.quantize_lm_head = (proj_fmt != Fmt::BF16);
    opt.max_seq = max_seq;
    if (!e.build(dir, opt, err)) { std::printf("\nload: %s\n", err.c_str()); return 1; }

    const std::string templated = tk.apply_chat_template(prompt);
    const std::vector<int32_t> ids = tk.encode(templated);
    std::printf("\n  prompt: %zu tokens\n\n", ids.size());
    if(const char* dd=std::getenv("GRIMOIRE_DFLASH_DUMP")){
        std::string fn=std::string(dd)+"/g_00_prompt_ids.txt";
        if(std::FILE* f=std::fopen(fn.c_str(),"w")){
            for(size_t i=0;i<ids.size();++i)
                std::fprintf(f,"%d%s",ids[i],i+1<ids.size()?",":"\n");
            std::fclose(f);
        }
    }

    e.reset();

    // GRIMOIRE_PREFILL_WARMUP=1 runs the prefill once and discards it, then
    // times the second run. oneDNN keys its matmul primitives on (m,n,k), so a
    // cold process JIT-compiles a fresh plan set for every distinct prompt
    // length -- measured at ~181 ms in layer 0 alone (33.8 ms for qkv, 147.0 ms
    // for the FFN down shape). A long-running server pays that once and
    // amortises it; a fresh CLI process pays it on every invocation. Use this
    // when comparing against the Fusion server, which has warm plans.
    if (std::getenv("GRIMOIRE_PREFILL_WARMUP")) {
        if (e.prefill(ids)) {
            e.reset();
        }
    }

    const auto p0 = std::chrono::high_resolution_clock::now();
    if (!e.prefill(ids)) {
        std::printf("batched prefill unavailable, using sequential fallback\n");
        for (size_t i = 0; i < ids.size(); ++i) {
            e.forward(ids[i]);
            if ((i & 63) == 63) e.sync();
        }
    }
    e.sync();                      // pp must time execution, not submission
    if (e.dflash2.ok &&
        (!e.dflash2.v2 || std::getenv("GRIMOIRE_DFLASH2") != nullptr)) {
        std::vector<int32_t> unused;
        if (!e.dflash_draft(0, e.pos, unused, true)) {
            std::fprintf(stderr,
                "\n  DFlash context preparation failed at position %d\n", e.pos);
            e.release();
            return 1;
        }
    }
    const auto p1 = std::chrono::high_resolution_clock::now();
    const double pp_ms = std::chrono::duration<double, std::milli>(p1 - p0).count();

    const bool gen_progress=std::getenv("GRIMOIRE_PREFILL_HOST_PROGRESS")!=nullptr;
    if(gen_progress){std::fprintf(stderr,"    generate: prefill returned, calling argmax\n");
        std::fflush(stderr);}
    int tok = e.argmax_token();
    if(gen_progress){std::fprintf(stderr,"    generate: first token id=%d\n",tok);
        std::fflush(stderr);}
    std::string out;
    // Opt-in device-resident decode. The recorded graph submits all ~800
    // per-token launches as one command list; its embed node reads s.d_tok,
    // which argmax_token() has already written, so replay needs no rebind.
    // Speculative paths call forward() directly and are unaffected.
    const char* graph_env = std::getenv("GRIMOIRE_DECODE_GRAPH");
    if (graph_env && *graph_env && std::atoi(graph_env) != 0) {
        std::printf("  recording decode graph ... ");
        std::fflush(stdout);
        const bool ok = e.build_graph();
        std::printf("%s\n", ok ? "ok" : "unavailable");
        std::fflush(stdout);
    }
    const auto g0 = std::chrono::high_resolution_clock::now();
    int n = 0;
    const bool mtp_measure_only = std::getenv("GRIMOIRE_MTP_MEASURE") != nullptr;
    const bool mtp_spec = Grimoire::mtp_enabled() && e.mtp.ok && !mtp_measure_only;
    const bool dflash_spec = e.dflash2.ok &&
        (!e.dflash2.v2 || std::getenv("GRIMOIRE_DFLASH2") != nullptr);
    if (mtp_spec || dflash_spec) {
        const int configured_k = dflash_spec ? 15 : [] {
            const char* v = std::getenv("GRIMOIRE_MTP_K");
            int k = v && *v ? std::atoi(v) : 3;
            // Exact verification currently preserves decode summation through
            // the four-row GEMV.  The optimized W4A8 path has an eight-row
            // tile and matching recurrent checkpoints, so it can use K=7.
            const int max_k = std::getenv("GRIMOIRE_MTP_EXACT_VERIFY") ? 3 : 7;
            return std::max(0, std::min(max_k, k));
        }();
        int steps = 0, accepted_drafts = 0, attempted_drafts = 0;
        int committed_total = 0, rollbacks = 0;
        const bool profile_spec = std::getenv("GRIMOIRE_MTP_PROFILE") != nullptr;
        double snapshot_ms = 0.0, draft_ms = 0.0, verify_ms = 0.0,
               commit_ms = 0.0;
        bool stop = false;

        while (n < n_predict && !stop) {
            if (tok == tk.eos() || e.pos >= e.max_seq) break;
            const int k = std::min(configured_k, e.max_seq - e.pos - 1);
            const int saved_pos = e.pos;
            auto phase0 = std::chrono::high_resolution_clock::now();
            e.snapshot_recurrent();
            if (profile_spec) {
                e.q.wait();
                auto now = std::chrono::high_resolution_clock::now();
                snapshot_ms += std::chrono::duration<double, std::milli>(now-phase0).count();
                phase0 = now;
            }

            // candidates[0] is the already-known main token. Each MTP call
            // predicts one token farther into the future.
            std::vector<int32_t> candidates;
            candidates.reserve(size_t(k) + 1);
            candidates.push_back(tok);
            if(dflash_spec){
                std::vector<int32_t> block;
                if(!e.dflash_draft(tok,saved_pos,block)){
                    std::fprintf(stderr,"\n  DFlash draft failed at position %d\n",saved_pos);
                    e.release();return 1;
                }
                const int take=std::min(k,int(block.size()));
                candidates.insert(candidates.end(),block.begin(),block.begin()+take);
            }else{
                int draft = tok;
                for (int j = 1; j <= k; ++j) {
                    draft = e.mtp_draft(draft, saved_pos + j - 1, j > 1);
                    if (draft < 0) break;
                    candidates.push_back(draft);
                }
            }
            if (profile_spec) {
                e.q.wait();
                auto now = std::chrono::high_resolution_clock::now();
                draft_ms += std::chrono::duration<double, std::milli>(now-phase0).count();
                phase0 = now;
            }

            std::vector<int32_t> verified;
            if (!e.prefill(candidates, &verified) || verified.size() != candidates.size()) {
                std::fprintf(stderr, "\n  %s verifier failed at position %d\n",
                             dflash_spec?"DFlash":"MTP", saved_pos);
                e.release();
                return 1;
            }
            if (profile_spec) {
                auto now = std::chrono::high_resolution_clock::now();
                verify_ms += std::chrono::duration<double, std::milli>(now-phase0).count();
                phase0 = now;
            }

            if (std::getenv("GRIMOIRE_DFLASH_DUMP")) {
                static int vdump = 0;
                if (vdump++ < 2) {
                    // The committed token each step is verified[0]. If that is
                    // wrong the target's batched verify forward is broken,
                    // independent of draft quality.
                    std::fprintf(stderr, "  verify@%d candidates:", saved_pos);
                    for (size_t i = 0; i < candidates.size(); ++i)
                        std::fprintf(stderr, " %d", candidates[i]);
                    std::fprintf(stderr, "\n  verify@%d verified  :", saved_pos);
                    for (size_t i = 0; i < verified.size(); ++i)
                        std::fprintf(stderr, " %d", verified[i]);
                    std::fprintf(stderr, "\n");
                    std::fflush(stderr);
                }
            }
            int accepted = 1;
            for (; accepted < int(candidates.size()); ++accepted) {
                ++attempted_drafts;
                if (candidates[accepted] != verified[accepted - 1]) break;
                ++accepted_drafts;
            }
            const int next = verified[accepted - 1];

            // The optimistic verifier consumed every candidate. On a partial
            // rejection, restore the exact recurrent state and replay only
            // the prefix whose tokens matched the main model. KV cache slots
            // are overwritten by that replay and therefore need no snapshot.
            if (accepted < int(candidates.size())) {
                ++rollbacks;
                static bool state_checked = false;
                const bool check_state = !state_checked &&
                    std::getenv("GRIMOIRE_MTP_VALIDATE_STATE");
                if (check_state) {
                    state_checked = true;
                    e.commit_spec_prefix(saved_pos, accepted);
                    e.q.wait();
                    std::vector<float> got_dn(e.spec_dn_elems),
                                       got_cv(e.spec_conv_elems),
                                       got_h(size_t(e.cfg.hidden));
                    size_t d0 = 0, c0 = 0;
                    const size_t dn_n = size_t(e.cfg.lin_v_heads) *
                        e.cfg.lin_v_dim * e.cfg.lin_k_dim;
                    const size_t cv_n = size_t(2 * e.cfg.lin_k_heads * e.cfg.lin_k_dim +
                        e.cfg.lin_v_heads * e.cfg.lin_v_dim) * (e.cfg.conv_kernel - 1);
                    for (const auto& ld : e.L) {
                        if (ld.dn_state) { e.q.memcpy(got_dn.data()+d0,ld.dn_state,
                            dn_n*sizeof(float)); d0+=dn_n; }
                        if (ld.conv_ring) { e.q.memcpy(got_cv.data()+c0,ld.conv_ring,
                            cv_n*sizeof(float)); c0+=cv_n; }
                    }
                    e.q.memcpy(got_h.data(),e.s.h,size_t(e.cfg.hidden)*sizeof(float)).wait();

                    e.restore_recurrent(saved_pos);
                    std::vector<int32_t> prefix(candidates.begin(),
                                                candidates.begin() + accepted);
                    // Replay through the same verifier path. Passing no
                    // output vector selects ordinary prompt-prefill kernels,
                    // including fused projections whose summation differs
                    // from exact speculative verification; comparing those
                    // states produced a false parity failure.
                    std::vector<int32_t> replay_verified;
                    e.prefill(prefix, &replay_verified);
                    std::vector<float> ref_dn(e.spec_dn_elems),
                                       ref_cv(e.spec_conv_elems),
                                       ref_h(size_t(e.cfg.hidden));
                    d0=0;c0=0;
                    for (const auto& ld : e.L) {
                        if (ld.dn_state) { e.q.memcpy(ref_dn.data()+d0,ld.dn_state,
                            dn_n*sizeof(float)); d0+=dn_n; }
                        if (ld.conv_ring) { e.q.memcpy(ref_cv.data()+c0,ld.conv_ring,
                            cv_n*sizeof(float)); c0+=cv_n; }
                    }
                    e.q.memcpy(ref_h.data(),e.s.h,size_t(e.cfg.hidden)*sizeof(float)).wait();
                    auto report_diff=[](const char* name,const std::vector<float>& a,
                                        const std::vector<float>& b){
                        double max_abs=0.0, sum=0.0; size_t bad=0;
                        for(size_t i=0;i<a.size();++i){
                            const double d=std::abs(double(a[i])-double(b[i]));
                            max_abs=std::max(max_abs,d);sum+=d;if(d!=0.0)++bad;
                        }
                        std::fprintf(stderr,"\n  MTP state %-8s: %zu/%zu differ, "
                            "max %.7g mean %.7g\n",name,bad,a.size(),max_abs,
                            a.empty()?0.0:sum/a.size());
                    };
                    report_diff("deltanet",got_dn,ref_dn);
                    report_diff("conv",got_cv,ref_cv);
                    report_diff("hidden",got_h,ref_h);
                } else {
                    e.commit_spec_prefix(saved_pos, accepted);
                }
            }
            if (profile_spec) {
                e.q.wait();
                auto now = std::chrono::high_resolution_clock::now();
                commit_ms += std::chrono::duration<double, std::milli>(now-phase0).count();
            }

            ++steps;
            committed_total += accepted;
            for (int i = 0; i < accepted; ++i) {
                const int t = candidates[i];
                if (t == tk.eos() || n >= n_predict) { stop = true; break; }
                const std::string piece = tk.decode_one(t);
                out += piece;
                std::printf("%s", piece.c_str());
                ++n;
            }
            tok = next;
        }

        const auto g1 = std::chrono::high_resolution_clock::now();
        const double tg_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();
        std::printf("\n\n  pp %zu tokens in %.0f ms -> %.1f tok/s\n",
                    ids.size(), pp_ms, 1000.0 * ids.size() / pp_ms);
        std::printf("  %s(k=%d) tg %d tokens in %.0f ms -> %.1f tok/s\n",
                    dflash_spec?"DFlash":"MTP",configured_k, n, tg_ms,
                    n > 0 ? 1000.0 * n / tg_ms : 0.0);
        std::printf("  %s steps %d, %.2f committed/step, draft accepts %d/%d, "
                    "rollbacks %d\n",
                    dflash_spec?"DFlash":"MTP",steps,
                    steps ? double(committed_total) / steps : 0.0,
                    accepted_drafts, attempted_drafts, rollbacks);
        if (profile_spec)
            std::printf("  %s profile snapshot %.1f ms, draft %.1f ms, verify %.1f ms, "
                        "commit %.1f ms\n",dflash_spec?"DFlash":"MTP",
                        snapshot_ms, draft_ms, verify_ms, commit_ms);
        e.release();
        return 0;
    }
    // MTP ACCEPTANCE MEASUREMENT.  No speculation yet: run the draft head
    // every step and check, one step later, whether it called the token the
    // model actually produced.  This is the number that decides whether the
    // verify + rollback machinery is worth building -- measure it before
    // building it.
    const bool mtp_meas = Grimoire::mtp_enabled() && mtp_measure_only;
    int mtp_hit = 0, mtp_tot = 0;
    // A draft made at the end of iteration i is a prediction for the token
    // that arrives at the start of iteration i+2 -- mtp_draft(T) predicts the
    // token AFTER T, and T itself is only emitted next iteration.  So the
    // check needs a two-deep pipeline; comparing one step early reads 0-2%
    // on a head that is actually working.
    // A draft of depth j made at iteration i predicts the token that arrives
    // at iteration i+1+j.  Chained drafts feed the head its own hidden state.
    const int MTPK = [] { const char* v = std::getenv("GRIMOIRE_MTP_K");
        int k = v && *v ? std::atoi(v) : 4; return k < 1 ? 1 : (k > 8 ? 8 : k); }();
    std::vector<std::array<int, 9>> pend(size_t(n_predict) + 16);
    for (auto& a : pend) a.fill(-1);
    std::array<int, 9> hit_d{}, tot_d{};
    for (; n < n_predict; ++n) {
        if (tok == tk.eos()) break;
        if (mtp_meas) {
            for (int j = 1; j <= MTPK; ++j) {
                const int d = pend[size_t(n)][j];
                if (d < 0) continue;
                ++tot_d[j];
                if (d == tok) ++hit_d[j];
                if (j == 1) { ++mtp_tot; if (d == tok) ++mtp_hit; }
            }
        }
        const std::string piece = tk.decode_one(tok);
        out += piece;
        std::printf("%s", piece.c_str());
        // argmax_token() left this token in s.d_tok, which is exactly what
        // the recorded graph's embed node reads, so replay needs no rebind.
        if (e.graph_ok) e.step(); else e.forward(tok);
        tok = e.argmax_token();
        if (mtp_meas) {
            int t = tok;
            for (int j = 1; j <= MTPK; ++j) {
                t = e.mtp_draft(t, e.pos + j - 1, j > 1);
                const size_t slot = size_t(n) + 1 + size_t(j);
                if (t < 0) break;
                if (slot < pend.size()) pend[slot][j] = t;
            }
        }
    }
    const auto g1 = std::chrono::high_resolution_clock::now();
    const double tg_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

    std::printf("\n\n  pp %zu tokens in %.0f ms -> %.1f tok/s\n",
                ids.size(), pp_ms, 1000.0 * ids.size() / pp_ms);
    std::printf("  tg %d tokens in %.0f ms -> %.1f tok/s\n",
                n, tg_ms, n > 0 ? 1000.0 * n / tg_ms : 0.0);
    if (mtp_meas && tot_d[1] > 0) {
        std::printf("\n  MTP acceptance by draft depth (%d chained):\n", MTPK);
        double expected = 1.0, run = 1.0;
        for (int j = 1; j <= MTPK; ++j) {
            if (!tot_d[j]) continue;
            const double pj = double(hit_d[j]) / double(tot_d[j]);
            // a depth-j draft is only usable if every shallower one was right
            run = (j == 1) ? pj : run * (pj > 0 ? pj : 0);
            expected += run;
            std::printf("    depth %d: %4d/%4d = %5.1f%%   cumulative accept %.2f tok/step\n",
                        j, hit_d[j], tot_d[j], 100.0 * pj, expected);
        }
        const double base_ms = tg_ms / std::max(1, n);
        // one verify step (~1.10x a plain step for a small batch) plus the
        // draft heads, against `expected` committed tokens
        const double step_ms = base_ms * 1.10 + double(MTPK) * 1.55;
        std::printf("    -> %.2f tok per %.1f ms step  ->  %.1f tok/s projected\n",
                    expected, step_ms, 1000.0 * expected / step_ms);
    }
    if (mtp_meas && mtp_tot > 0) {
        const double acc = 100.0 * double(mtp_hit) / double(mtp_tot);
        std::printf("  MTP acceptance %d/%d = %.1f%%\n", mtp_hit, mtp_tot, acc);
        // With one draft token: a step commits 1 + acc tokens on average and
        // costs about one verify step plus the draft head.
        std::printf("  -> at %.1f%% and a 1.05x verify step, projected TG ~%.1f tok/s\n",
                    acc, 1000.0 * n / tg_ms * (1.0 + acc / 100.0) / 1.10);
    }
    e.release();
    return 0;
}

// =====================================================================
//  grimoire-server support -- keep one Grimoire resident across requests
// =====================================================================
// The server (tools/grimoire_server.cpp) only needs an opaque handle plus
// these three calls; it never sees the Grimoire struct definition.
Grimoire* grimoire_new() { return new Grimoire(); }

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
                             int eot_id,
                             const std::function<bool(int32_t)>& on_token) {
    auto is_stop = [&](int t) { return t == eos_id || (eot_id >= 0 && t == eot_id); };
    // Streaming clients (llama-benchy, open-webui) need each token as it is
    // produced, not one JSON blob at the end -- without it they cannot measure
    // time-to-first-token and llama-benchy reports "No results collected".
    // Returning false from the callback stops generation (client disconnected).
    bool cancelled = false;
    auto emit = [&](int32_t t) {
        out_ids.push_back(t);
        if (on_token && !on_token(t)) cancelled = true;
    };
    const auto rs_t0 = std::chrono::steady_clock::now();
    e.reset();
    std::fprintf(stderr, "    [reset] %.0f ms\n",
        std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-rs_t0).count());
    std::fflush(stderr);
    const auto pf_t0 = std::chrono::steady_clock::now();
    const bool pf_ok = e.prefill(prompt_ids);
    if (!pf_ok) {
        std::fprintf(stderr, "    [prefill] BATCHED PREFILL FAILED -- falling "
            "back to per-token forward (this is ~100x slower)\n");
        for (size_t i = 0; i < prompt_ids.size(); ++i) {
            e.forward(prompt_ids[i]);
            if ((i & 63) == 63) e.sync();
        }
    }
    e.sync();
    if (e.dflash2.ok &&
        (!e.dflash2.v2 || std::getenv("GRIMOIRE_DFLASH2") != nullptr)) {
        std::vector<int32_t> unused;
        if (!e.dflash_draft(0, e.pos, unused, true)) {
            std::fprintf(stderr,
                "    [prefill] DFlash context preparation failed at position %d\n",
                e.pos);
            return 0;
        }
    }
    {
        const double pf_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pf_t0).count();
        std::fprintf(stderr, "    [prefill] %zu tokens in %.0f ms -> %.1f tok/s\n",
            prompt_ids.size(), pf_ms,
            pf_ms > 0.0 ? 1000.0 * double(prompt_ids.size()) / pf_ms : 0.0);
    }
    const auto am_t0 = std::chrono::steady_clock::now();
    int tok = e.argmax_token();
    std::fprintf(stderr, "    [argmax] %.0f ms\n",
        std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-am_t0).count());
    std::fflush(stderr);
    out_ids.clear();
    out_ids.reserve(size_t(n_predict));
    int n = 0;

    // Speculative decode, same accept/reject contract as the CLI path: draft a
    // block, verify it with one batched forward, accept the matching prefix,
    // and on partial rejection restore the recurrent state to the accepted
    // prefix (commit_spec_prefix) so the next step starts from exact state.
    // Falls through to plain greedy below when no drafter is loaded.
    const bool mtp_spec = Grimoire::mtp_enabled() && e.mtp.ok;
    const bool dflash_spec = e.dflash2.ok &&
        (!e.dflash2.v2 || std::getenv("GRIMOIRE_DFLASH2") != nullptr);
    // The CLI has used this graph-replay path since c957f2f, but the server
    // fallback still submitted every kernel in forward() separately. Build
    // the same reusable graph once for non-speculative server decode. The
    // graph reads the current token from s.d_tok, which argmax_token() wrote.
    const char* graph_env = std::getenv("GRIMOIRE_DECODE_GRAPH");
    if (!mtp_spec && !dflash_spec && graph_env && *graph_env &&
        std::atoi(graph_env) != 0 && !e.graph_ok) {
        const bool ok = e.build_graph();
        std::fprintf(stderr, "    [decode] command graph %s\n",
                     ok ? "ready" : "unavailable; using direct submission");
        std::fflush(stderr);
    }
    // The first token is already known from the prefill argmax. Emitting it
    // before the first speculation round removes a whole draft+verify cycle
    // from time-to-first-token. A benchmark charges that to prefill, which is
    // why reported PP read ~1790 against ~2070 of real prefill compute.
    int skip_first = 0;
    if ((mtp_spec || dflash_spec) && on_token && n < n_predict && !is_stop(tok)) {
        emit(tok);
        ++n;
        skip_first = 1;
    }
    // Acceptance/verify accounting for the server path. GRIMOIRE_MTP_PROFILE
    // only ever instrumented the CLI loop, which is why a slow server decode
    // could never be attributed to acceptance vs verify cost.
    const bool spec_profile = std::getenv("GRIMOIRE_MTP_PROFILE") != nullptr;
    int spec_rounds = 0, spec_drafted = 0, spec_accepted = 0;
    double spec_draft_ms = 0.0, spec_verify_ms = 0.0;
    const auto spec_t0 = std::chrono::steady_clock::now();
    if (mtp_spec || dflash_spec) {
        const int configured_k = dflash_spec ? 15 : [] {
            const char* v = std::getenv("GRIMOIRE_MTP_K");
            int k = v && *v ? std::atoi(v) : 3;
            const int max_k = std::getenv("GRIMOIRE_MTP_EXACT_VERIFY") ? 3 : 7;
            return std::max(0, std::min(max_k, k));
        }();
        bool stop = false;
        while (n < n_predict && !stop) {
            if (is_stop(tok) || e.pos >= e.max_seq) break;
            const int k = std::min(configured_k, e.max_seq - e.pos - 1);
            if (k <= 0) break;
            const int saved_pos = e.pos;
            e.snapshot_recurrent();
            // Match the proven CLI MTP path: profiling drains the checkpoint
            // before starting the timed draft phase.
            if (spec_profile) e.q.wait();

            const auto dr_t0 = std::chrono::steady_clock::now();
            std::vector<int32_t> candidates;
            candidates.reserve(size_t(k) + 1);
            candidates.push_back(tok);
            if (dflash_spec) {
                std::vector<int32_t> block;
                if (!e.dflash_draft(tok, saved_pos, block)) break;
                const int take = std::min(k, int(block.size()));
                candidates.insert(candidates.end(), block.begin(),
                                  block.begin() + take);
            } else {
                int draft = tok;
                for (int j = 1; j <= k; ++j) {
                    draft = e.mtp_draft(draft, saved_pos + j - 1, j > 1);
                    if (draft < 0) break;
                    candidates.push_back(draft);
                }
            }
            if (candidates.size() < 2) break;   // drafter produced nothing

            spec_draft_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - dr_t0).count();

            const auto vf_t0 = std::chrono::steady_clock::now();
            std::vector<int32_t> verified;
            if (!e.prefill(candidates, &verified) ||
                verified.size() != candidates.size()) break;
            spec_verify_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - vf_t0).count();

            int accepted = 1;
            for (; accepted < int(candidates.size()); ++accepted)
                if (candidates[accepted] != verified[accepted - 1]) break;
            ++spec_rounds;
            spec_drafted  += int(candidates.size()) - 1;
            spec_accepted += accepted - 1;
            const int next = verified[accepted - 1];

            if (accepted < int(candidates.size()))
                e.commit_spec_prefix(saved_pos, accepted);

            for (int i = skip_first; i < accepted; ++i) {
                const int t = candidates[i];
                if (is_stop(t) || n >= n_predict) { stop = true; break; }
                emit(t);
                ++n;
                if (cancelled) { stop = true; break; }
            }
            skip_first = 0;
            tok = next;
        }
        if (spec_profile && spec_rounds > 0) {
            const double tot_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - spec_t0).count();
            std::fprintf(stderr,
                "    [spec] %d rounds  accept %d/%d = %.1f%%  %.2f tok/round"
                "  draft %.0f ms  verify %.0f ms  other %.0f ms -> %.1f tok/s\n",
                spec_rounds, spec_accepted, spec_drafted,
                spec_drafted ? 100.0 * spec_accepted / spec_drafted : 0.0,
                double(n) / spec_rounds, spec_draft_ms, spec_verify_ms,
                tot_ms - spec_draft_ms - spec_verify_ms,
                tot_ms > 0.0 ? 1000.0 * n / tot_ms : 0.0);
            std::fflush(stderr);
        }
        return n;
    }

    for (; n < n_predict; ++n) {
        if (is_stop(tok)) break;
        emit(tok);
        if (cancelled) { ++n; break; }
        if (e.graph_ok) e.step(); else e.forward(tok);
        tok = e.argmax_token();
    }
    // Deliberately no e.release() here: the engine stays resident across
    // requests. release() is only correct at process shutdown, which this
    // server does not currently hook (process exit reclaims the GPU context
    // regardless).
    return n;
}

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
