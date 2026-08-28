// GRIMOIRE -> vLLM FlashAttention-2 bridge.
//
// vLLM's fast prefill attention is NOT libattn_kernels_xe_2.so (which GRIMOIRE's
// existing xe2_attention_bridge.cpp calls via cutlass_chunk_prefill_xe2).  It is
// _vllm_fa2_C.abi3.so, registered as the torch op "_vllm_fa2_C::varlen_fwd".
// Profiled 2026-08-25 on Qwen3.8-27B prefill@4096: 2.84 ms/layer vs GRIMOIRE's
// 10.87 ms/layer over 16 full-attention layers.
//
// Called through the dispatcher (boxed) rather than by binding the raw C++ symbol:
// mha_varlen_fwd has 28 parameters and a signature mismatch would corrupt the stack.
// The boxed path matches vLLM's own Python call site argument-for-argument.
#include <torch/all.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/xpu/XPUStream.h>
#include <sycl/sycl.hpp>
#include <cstdio>
#include <optional>
#include <vector>
#include <dlfcn.h>

// The op is registered by _vllm_fa2_C.abi3.so's TORCH_LIBRARY static initialiser,
// which only runs when that library is loaded.  Loading it here (RTLD_GLOBAL so its
// torch symbols resolve) is what makes "_vllm_fa2_C::varlen_fwd" findable.
static bool grimoire_fa2_load_ext() {
    static bool done = [] {
        // _vllm_fa2_C.abi3.so is a CPython extension: it references PyModule_Create2
        // and friends, so libpython must be resolvable (RTLD_GLOBAL) before it loads.
        for (const char* py : {"libpython3.12.so.1.0", "libpython3.12.so",
                               "libpython3.so"}) {
            if (dlopen(py, RTLD_NOW | RTLD_GLOBAL)) break;
        }
        const char* env = std::getenv("GRIMOIRE_FA2_LIB");
        const char* cands[] = {
            env,
            "/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_vllm_fa2_C.abi3.so",
            "_vllm_fa2_C.abi3.so"
        };
        for (const char* c : cands) {
            if (!c || !*c) continue;
            if (dlopen(c, RTLD_NOW | RTLD_GLOBAL)) return true;
        }
        std::fprintf(stderr, "  FA2: could not load _vllm_fa2_C (%s)\n", dlerror());
        return false;
    }();
    return done;
}

extern "C" int grimoire_xe2_fa2_available() {
    grimoire_fa2_load_ext();
    auto h = c10::Dispatcher::singleton().findSchema({"_vllm_fa2_C::varlen_fwd", ""});
    return h.has_value() ? 1 : 0;
}

extern "C" void grimoire_xe2_fa2_prefill_bf16(
    sycl::queue* queue, const void* query, const void* key, const void* value,
    void* output, int q_tokens, int kv_tokens, int q_heads, int kv_heads,
    int head_dim, const int* cu_q, const int* cu_k, float softmax_scale,
    bool causal) {

    grimoire_fa2_load_ext();
    auto handle = c10::Dispatcher::singleton().findSchema({"_vllm_fa2_C::varlen_fwd", ""});
    if (!handle.has_value()) { std::fprintf(stderr, "  FA2 op not registered\n"); return; }

    auto bf = at::TensorOptions().dtype(at::kBFloat16).device(at::kXPU);
    auto ii = at::TensorOptions().dtype(at::kInt).device(at::kXPU);

    at::Tensor q = at::from_blob(const_cast<void*>(query), {q_tokens, q_heads, head_dim}, bf);
    at::Tensor k = at::from_blob(const_cast<void*>(key),   {kv_tokens, kv_heads, head_dim}, bf);
    at::Tensor v = at::from_blob(const_cast<void*>(value), {kv_tokens, kv_heads, head_dim}, bf);
    at::Tensor o = at::from_blob(output, {q_tokens, q_heads, head_dim}, bf);
    at::Tensor cq = at::from_blob(const_cast<int*>(cu_q), {2}, ii);
    at::Tensor ck = at::from_blob(const_cast<int*>(cu_k), {2}, ii);

    // Adopt GRIMOIRE's own queue as torch's current XPU stream for the duration of
    // the call.  Without this the op runs on torch's stream and every invocation needs
    // a queue->wait() before and a stream synchronize() after -- 32 pipeline stalls
    // per forward pass, which cost more than the kernel saves (226 ms vs 173.9 ms
    // in-model, despite being 4x faster standalone).  Adopting the queue removes both
    // syncs: producer, kernel and consumer are all ordered on the same queue.
    c10::xpu::XPUStream ext = c10::xpu::getStreamFromExternal(queue, 0);
    c10::xpu::XPUStream prev = c10::xpu::getCurrentXPUStream(0);
    c10::xpu::setCurrentXPUStream(ext);
    struct StreamRestore {
        c10::xpu::XPUStream s;
        ~StreamRestore(){ c10::xpu::setCurrentXPUStream(s); }
    } restore{prev};

    // Exact schema (28 args), dumped from the registered op:
    //   varlen_fwd(Tensor q, Tensor k, Tensor v, Tensor? out, Tensor cu_seqlens_q,
    //     Tensor cu_seqlens_k, Tensor? seqused_k, Tensor? leftpad_k,
    //     Tensor? block_table, Tensor? alibi_slopes, int max_seqlen_q,
    //     int max_seqlen_k, float p_dropout, Tensor? k_scale, Tensor? v_scale,
    //     float softmax_scale, Tensor? softmax_sink, bool zero_tensors,
    //     bool is_causal, int window_size_left, int window_size_right,
    //     float softcap, bool return_softmax, Generator? gen, int? num_splits,
    //     bool mix_batch, Tensor? splits_per_seq, Tensor? work_list) -> Tensor[]
    std::vector<c10::IValue> stack;
    stack.reserve(28);
    stack.emplace_back(q);                       //  1 q
    stack.emplace_back(k);                       //  2 k
    stack.emplace_back(v);                       //  3 v
    stack.emplace_back(o);                       //  4 out (written in place)
    stack.emplace_back(cq);                      //  5 cu_seqlens_q
    stack.emplace_back(ck);                      //  6 cu_seqlens_k
    stack.emplace_back(c10::IValue());           //  7 seqused_k
    stack.emplace_back(c10::IValue());           //  8 leftpad_k
    stack.emplace_back(c10::IValue());           //  9 block_table
    stack.emplace_back(c10::IValue());           // 10 alibi_slopes
    stack.emplace_back((int64_t)q_tokens);       // 11 max_seqlen_q
    stack.emplace_back((int64_t)kv_tokens);      // 12 max_seqlen_k
    stack.emplace_back(0.0);                     // 13 p_dropout
    stack.emplace_back(c10::IValue());           // 14 k_scale
    stack.emplace_back(c10::IValue());           // 15 v_scale
    stack.emplace_back((double)softmax_scale);   // 16 softmax_scale
    stack.emplace_back(c10::IValue());           // 17 softmax_sink
    stack.emplace_back(false);                   // 18 zero_tensors
    stack.emplace_back(causal);                  // 19 is_causal
    stack.emplace_back((int64_t)-1);             // 20 window_size_left
    stack.emplace_back((int64_t)-1);             // 21 window_size_right
    stack.emplace_back(0.0);                     // 22 softcap
    stack.emplace_back(false);                   // 23 return_softmax
    stack.emplace_back(c10::IValue());           // 24 gen
    stack.emplace_back(c10::IValue());           // 25 num_splits
    stack.emplace_back(false);                   // 26 mix_batch
    stack.emplace_back(c10::IValue());           // 27 splits_per_seq
    stack.emplace_back(c10::IValue());           // 28 work_list

    try {
        handle->callBoxed(&stack);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  FA2 call failed: %s\n", e.what());
        return;
    }
    // No synchronize: the kernel was enqueued on GRIMOIRE's queue, so the caller's
    // subsequent work on that same in-order queue is already correctly ordered.
}
