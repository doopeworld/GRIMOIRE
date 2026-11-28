// b70/lora_dev.hpp -- the device side of an adapter (SYCL). One device allocation holds
// the host blob; per-layer views point into it; one mid scratch is shared across every
// LoRA launch, reused between them because they are submitted in order on the engine's
// queue. Built by Grimoire once the checkpoint loads; freed when the engine is destroyed.
// The host-safe loader lives in b70/lora.hpp -- the two are linked by the same geometry
// and by rounding points (mid formed in fp32, rounded to bf16 once, then B * bf16(mid)
// accumulated into the caller's fp32 buffer).
#pragma once

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

#include "b70/lora.hpp"

namespace b70::lora {

// Mid + delta for one module (defined in the .cpp): the two kernels are submitted
// back to back on the caller's queue through the shared mid scratch.
sycl::event lora_apply(sycl::queue& q, const LoraModule& lm, const float* x, float* out,
                       int out_ld, uint16_t* mid, int m,
                       const std::vector<sycl::event>& deps = {});

struct LoraDevice {
    sycl::queue* q = nullptr;
    uint16_t* dev_blob = nullptr;           // device copy of the host blob
    std::vector<LoraLayer> layers;          // device views into dev_blob
    uint16_t* mid = nullptr;                // [rows][r16] scratch (packed bf16)
    size_t mid_elems = 0;
    int r16 = 0;
    int n_max = 0;
    float scale = 1.f;
    std::string name;

    LoraDevice(sycl::queue& queue, const LoraAdapter& ad, size_t scratch_rows);
    ~LoraDevice();
    LoraDevice(const LoraDevice&) = delete;
    LoraDevice& operator=(const LoraDevice&) = delete;

    // Ensure the mid scratch fits rows (rows * r16 elements). Called by the engine before a
    // prefill; never per decode token. Reallocates with q.wait() in between.
    // Returns true if the scratch was reallocated (the mid pointer changed --
    // anything captured against it, e.g. a command graph, must be re-recorded).
    bool ensure_scratch(size_t rows);

    // Mid + delta for one module, using this device's mid scratch (the caller has already
    // ensured it fits m rows -- ensure_scratch(m) before the first launch of a step).
    sycl::event apply(const LoraModule& lm, const float* x, float* out, int out_ld,
                      int m, const std::vector<sycl::event>& deps = {}) {
        if (!lm.present() || m < 1) return sycl::event{};
        return lora_apply(*q, lm, x, out, out_ld, mid, m, deps);
    }
};

// mid[r][j] = bf16(sum_c bf16(A[j][c]) * x[r][c])   -- m*r16 threads, each loops over k.
sycl::event launch_lora_mid(sycl::queue& q, const LoraModule& lm, const float* x,
                             uint16_t* mid, int m, const std::vector<sycl::event>& deps = {});

// out[r*out_ld + out_off + i] += lm.scale * sum_j bf16(B[i][j]) * bf16(mid[r][j]) -- m*n threads.
sycl::event launch_lora_delta(sycl::queue& q, const LoraModule& lm, const uint16_t* mid,
                              float* out, int out_ld, int m, const std::vector<sycl::event>& deps = {});

// Launch mid + delta for one module. Returns the delta's event. The scratch is shared and
// reused in order across modules -- the engine never issues two LoRA launches on the same
// module out of order, so the reuse is safe.  (Defined in the .cpp.)

}  // namespace b70::lora
