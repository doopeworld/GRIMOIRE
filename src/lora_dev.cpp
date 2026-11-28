// b70/lora_dev.cpp -- the device side of an adapter (b70/lora_dev.hpp). The host blob is
// copied to device and every layer's view is re-pointed at the copy; the scratch is one
// shared mid buffer reused across launches in order.
#include "b70/lora_dev.hpp"

#include <stdexcept>

namespace b70::lora {

LoraDevice::LoraDevice(sycl::queue& queue, const LoraAdapter& ad, size_t scratch_rows) : q(&queue) {
    r16 = ad.r16();
    n_max = ad.n_max();
    scale = ad.scale();
    name = ad.name();
    const size_t bytes = ad.blob_size();
    if (bytes == 0) throw std::runtime_error("lora: empty packed blob");
    dev_blob = sycl::malloc_device<uint16_t>(bytes / 2, queue);
    if (!dev_blob) throw std::bad_alloc();
    queue.memcpy(dev_blob, ad.blob(), bytes).wait();
    const uint16_t* host_base = reinterpret_cast<const uint16_t*>(ad.blob());
    layers.assign(ad.layers(), LoraLayer{});
    auto copy = [&](LoraModule& dst, const LoraModule& src) {
        if (!src.present()) return;
        dst.a = dev_blob + (src.a - host_base);
        dst.b = dev_blob + (src.b - host_base);
        dst.r16 = src.r16;
        dst.k = src.k;
        dst.n = src.n;
        dst.out_off = src.out_off;
        dst.scale = src.scale;
    };
    for (int i = 0; i < ad.layers(); ++i) {
        const LoraLayer& hv = ad.layer(i);
        LoraLayer& dv = layers[size_t(i)];
        copy(dv.q, hv.q);   copy(dv.k, hv.k);   copy(dv.v, hv.v);   copy(dv.o, hv.o);
        copy(dv.gate, hv.gate);   copy(dv.up, hv.up);   copy(dv.down, hv.down);
    }
    ensure_scratch(scratch_rows);
}

LoraDevice::~LoraDevice() {
    if (mid) sycl::free(mid, *q);
    if (dev_blob) sycl::free(dev_blob, *q);
}

bool LoraDevice::ensure_scratch(size_t rows) {
    if (rows * size_t(r16) <= mid_elems) return false;
    if (mid) { q->wait(); sycl::free(mid, *q); }
    mid_elems = rows * size_t(r16);
    mid = sycl::malloc_device<uint16_t>(mid_elems, *q);
    if (!mid) throw std::bad_alloc();
    return true;
}

sycl::event launch_lora_mid(sycl::queue& q, const LoraModule& lm, const float* x,
                             uint16_t* mid, int m, const std::vector<sycl::event>& deps) {
    if (!lm.present() || m < 1) return sycl::event{};
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>({size_t(m), size_t(lm.r16)}),
                       [=](sycl::id<2> id) {
                           const size_t r = size_t(id[0]), j = size_t(id[1]);
                           float mid_val = 0.0f;
                           const uint16_t* a = lm.a + j * lm.k;
                           const float* xr = x + r * lm.k;
                           for (int c = 0; c < lm.k; ++c)
                               mid_val += bf16_to_f32(bf16_t{a[size_t(c)]}) * xr[size_t(c)];
                           mid[r * lm.r16 + j] = f32_to_bf16(mid_val).bits;
                       });
    });
}

sycl::event launch_lora_delta(sycl::queue& q, const LoraModule& lm, const uint16_t* mid,
                              float* out, int out_ld, int m, const std::vector<sycl::event>& deps) {
    if (!lm.present() || m < 1) return sycl::event{};
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>({size_t(m), size_t(lm.n)}),
                       [=](sycl::id<2> id) {
                           const size_t r = size_t(id[0]), i = size_t(id[1]);
                           float d = 0.0f;
                           const uint16_t* b = lm.b + i * lm.r16;
                           const uint16_t* mrow = mid + r * lm.r16;
                           for (int j = 0; j < lm.r16; ++j)
                               d += bf16_to_f32(bf16_t{b[size_t(j)]}) *
                                    bf16_to_f32(bf16_t{mrow[size_t(j)]});
                           out[r * out_ld + lm.out_off + i] += lm.scale * d;
                       });
    });
}

sycl::event lora_apply(sycl::queue& q, const LoraModule& lm, const float* x, float* out,
                        int out_ld, uint16_t* mid, int m,
                        const std::vector<sycl::event>& deps) {
    if (!lm.present() || m < 1) return sycl::event{};
    const auto e1 = launch_lora_mid(q, lm, x, mid, m, deps);
    return launch_lora_delta(q, lm, mid, out, out_ld, m, {e1});
}

}  // namespace b70::lora
