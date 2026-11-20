// SPDX-License-Identifier: GPL-3.0-or-later
#include "bench.hpp"
#include "projection_check.hpp"
#include "layout.hpp"
#include "gemm_xe2_policy.hpp"
#include "grouped_gemm_xe2.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>

namespace qfn_xe2 {
namespace {
using Bf16 = b70::sycl_bf16;
using Clock = std::chrono::steady_clock;
static_assert(sizeof(Bf16) == sizeof(cutlass::bfloat16_t));
// The lane picks its policy from A_avg_M = allocated rows / experts, so each
// instantiation must sit in the branch it is named for.
constexpr int tail_rows = 658, tail_capacity = tail_rows * 10;
static_assert(slots / experts > 128, "20,480 allocated rows pick the M128 policy");
static_assert(tail_capacity / experts > 8 && tail_capacity / experts <= 128,
              "6,580 allocated rows pick the M32 policy");

template <typename T> struct Buffer {
    sycl::queue& queue;
    T* data;
    Buffer(sycl::queue& q, size_t count) : queue(q), data(sycl::malloc_device<T>(count, q)) {
        if (!data) throw std::bad_alloc();
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { sycl::free(data, queue); }
};

// Native USM submission of the unmodified lane MoEGEMM device function.
// Match MoEGEMMLauncher geometry, subgroup/GRF properties and scheduler SLM.
template <typename D, int Cols = N, int Inner = K, typename P = MoE::w4a16_policy,
          int Capacity = slots>
sycl::event launch_lane(sycl::queue& q, const Bf16* x, const uint8_t* packed,
                        const void* scales, D* out, const int32_t* counts, int32_t* atomic) {
    using namespace cute;
    using T = cutlass::bfloat16_t;
    using Out = std::conditional_t<std::is_same_v<D,float>,float,T>;
    // Let the pinned CuTe selector choose a 32-bit D block store; the
    // BF16 arm keeps the lane's explicit 16-bit store atom.
    using Store = std::conditional_t<std::is_same_v<D,float>,void,typename P::GmemTiledCopyD>;
    using MMA = typename MoE::TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, T>>,
                                          Layout<typename P::WGTile>, typename P::SGLayout>::TiledMMA;
    constexpr size_t threads = size(MMA{});
    static_assert(512 % threads == 0);
    using namespace sycl::ext::intel::info::device;
    const auto dev = q.get_device();
    const size_t workers = dev.get_info<gpu_slices>() * dev.get_info<gpu_subslices_per_slice>() * (512 / threads);
    namespace sx = sycl::ext::oneapi::experimental;
    namespace ix = sycl::ext::intel::experimental;
    sx::properties props{sx::sub_group_size<16>, ix::grf_size<256>};
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int32_t, 1> slm(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<3>{{1, workers, threads}, {1, 1, threads}}, props, [=](auto) {
            // Both BF16 types hold identical 16-bit payloads; no scale/A conversion.
            MoE::MoEGEMM<MoE::A_DTYPE::BITS16, MoE::B_DTYPE::INT4, typename P::GmemTiledCopyA,
                         typename P::GmemTiledCopyB, Store, 'R', 'C', 'R'>(
                reinterpret_cast<const T*>(x), packed, static_cast<const T*>(scales),
                static_cast<const T*>(nullptr), reinterpret_cast<Out*>(out), MMA{}, counts,
                experts, group, Cols, Inner, atomic, slm);
        });
    });
}

auto widen_live(sycl::queue& q, const Bf16* src, const int32_t* live_rows, float* dst) {
    return q.parallel_for(sycl::range<1>(size_t(slots)*N), [=](sycl::id<1> i) {
        if (i[0] < size_t(*live_rows)*N) dst[i] = float(src[i]);
    });
}

bool check_apply_scale(sycl::queue& q) {
    // The two observed failures plus exact power-of-two controls. Probe
    // the dequantizer directly, before DPAS or the BF16 output store.
    constexpr std::array<int, 4> codes{5,-7,4,-8};
    constexpr std::array<uint16_t, 4> scales{0x3f23,0x3f0d,0x3f00,0x3e80};
    std::array<uint16_t, 4> input{};
    for (size_t i=0;i<input.size();++i) input[i]=b70::f32_to_bf16(float(codes[i])).bits;
    Buffer<uint16_t> d(q,16);
    auto* result = d.data;
    q.parallel_for(sycl::nd_range<1>{16,16}, [=](sycl::nd_item<1> lane) [[sycl::reqd_sub_group_size(16)]] {
        const size_t i = lane.get_local_linear_id()%input.size();
        auto w = sycl::bit_cast<cutlass::bfloat16_t>(input[i]);
        float s = b70::bf16_to_f32({scales[i]});
        result[lane.get_local_linear_id()] = sycl::bit_cast<uint16_t>(MoE::apply_scale(w,s));
    });
    std::array<uint16_t, 16> observed{};
    q.memcpy(observed.data(),result,sizeof(observed)).wait_and_throw();
    bool pass = true;
    for (size_t j=0;j<input.size();++j) {
        const float raw = float(codes[j])*b70::bf16_to_f32({scales[j]});
        const auto expected = b70::f32_to_bf16(raw).bits;
        size_t bad=0;
        for (size_t i=j;i<observed.size();i+=input.size()) bad+=observed[i]!=expected;
        std::fprintf(stderr,"QFN apply_scale: code=%d scale=0x%04x expected=0x%04x observed=0x%04x trunc=0x%04x bad=%zu/4\n",
                     codes[j],unsigned(scales[j]),unsigned(expected),unsigned(observed[j]),b70::f32_to_bits(raw)>>16,bad);
        pass &= !bad;
    }
    return pass;
}

double ms(const sycl::event& e) {
    using namespace sycl::info::event_profiling;
    return double(e.get_profiling_info<command_end>() - e.get_profiling_info<command_start>()) * 1.e-6;
}
}

void round_n64_gate_up(sycl::queue& q, const int32_t* live_rows, float* out) {
    q.parallel_for(sycl::range<1>(size_t(slots)*N), [=](sycl::id<1> i) {
        if (i[0] < size_t(*live_rows)*N) out[i] = float(Bf16(out[i]));
    });
}

void launch_signed_gate_up(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                           const int32_t* counts, const int32_t* live_rows, Bf16* scratch,
                           int32_t* atomic, float* out, bool fp32_output) {
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts*N || w.K != K ||
        w.row_bytes != K/2 || w.row_scales != K/group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 gate/up needs the qualified signed W4A16 layout and an in-order queue");
    // Caller proves the loader's signed-layout flag, and owns scratch
    // through the in-order consumer. Never fold resident bytes in place.
    q.fill(atomic,int32_t{0},1);
    if (fp32_output) {
        launch_lane(q,x,w.payload,w.scales,out,counts,atomic);
    } else {
        launch_lane(q,x,w.payload,w.scales,scratch,counts,atomic);
        widen_live(q,scratch,live_rows,out);
    }
}

// 658-row tail chunk: same pinned kernel, the lane's A_avg_M <= 128 branch
// (M32/N64/K32) for its 6,580 allocated route rows over 128 experts.
void launch_signed_gate_up_tail(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                                const int32_t* counts, int32_t* atomic, float* out) {
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts * N || w.K != K ||
        w.row_bytes != K / 2 || w.row_scales != K / group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 tail gate/up needs the qualified signed W4A16 layout");
    q.fill(atomic, int32_t{0}, 1);
    launch_lane<float, N, K, MoE::w4a16_policy_m_32, tail_capacity>(
        q, x, w.payload, w.scales, out, counts, atomic);
}

void launch_signed_down_tail(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                             const int32_t* counts, int32_t* atomic, float* out) {
    constexpr int cols = K, inner = N / 2;
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts * cols || w.K != inner ||
        w.row_bytes != inner / 2 || w.row_scales != inner / group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 tail down needs signed W4A16 [128,2560,640]");
    q.fill(atomic, int32_t{0}, 1);
    launch_lane<float, cols, inner, MoE::w4a16_policy_m_32, tail_capacity>(
        q, x, w.payload, w.scales, out, counts, atomic);
}

// The lane's own dispatch rule, applied to a full prefill chunk: with
// ~5,120 live rows over 128 experts, A_avg_M ~ 40 selects the M32/N64 branch
// rather than the M128/N256 one this path has been forcing.
void launch_signed_gate_up_p32(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                               const int32_t* counts, int32_t* atomic, float* out) {
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts * N || w.K != K ||
        w.row_bytes != K / 2 || w.row_scales != K / group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 gate/up p32 needs the qualified signed W4A16 layout");
    q.fill(atomic, int32_t{0}, 1);
    launch_lane<float, N, K, MoE::w4a16_policy_m_32, slots>(q, x, w.payload, w.scales, out, counts, atomic);
}

void launch_signed_down_p32(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                            const int32_t* counts, int32_t* atomic, float* out) {
    constexpr int cols = K, inner = N / 2;
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts * cols || w.K != inner ||
        w.row_bytes != inner / 2 || w.row_scales != inner / group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 down p32 needs signed W4A16 [128,2560,640]");
    q.fill(atomic, int32_t{0}, 1);
    launch_lane<float, cols, inner, MoE::w4a16_policy_m_32, slots>(q, x, w.payload, w.scales, out, counts, atomic);
}

void launch_signed_down(sycl::queue& q, const b70::QuantWeight& w, const Bf16* x,
                        const int32_t* counts, int32_t* atomic, float* out) {
    constexpr int cols = K, inner = N / 2;
    static_assert(cols % 256 == 0 && inner % group == 0);
    if (!q.is_in_order() || w.fmt != b70::Fmt::INT4 || w.N != experts * cols || w.K != inner ||
        w.row_bytes != inner / 2 || w.row_scales != inner / group || !w.payload || !w.scales)
        throw std::runtime_error("Xe2 down needs signed W4A16 [128,2560,640] and an in-order queue");
    q.fill(atomic, int32_t{0}, 1);
    launch_lane<float, cols, inner>(q, x, w.payload, w.scales, out, counts, atomic);
}

bool compare_gate_up(sycl::queue& main, const b70::QuantWeight& w, int layer, int rank,
                     const Bf16* x, const int32_t* counts, const int32_t* offsets,
                     const b70::QfnMoeTile* tiles, const int32_t* live_tiles, int max_tiles,
                     std::span<const HotRow> oracle_rows, bool fp32_output) {
    if (w.fmt != b70::Fmt::INT4 || w.N != experts * N || w.K != K || w.row_bytes != K / 2 ||
        w.row_scales != K / group || !w.payload || !w.zeros || !w.scales)
        throw std::runtime_error("Xe2 bench requires the native QFN GPTQ W4A16 gate/up layout");
    main.wait_and_throw();
    sycl::queue q(main.get_context(), main.get_device(),
                  {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    int live = 0;
    q.memcpy(&live, offsets + experts, sizeof(live)).wait_and_throw();
    if (live <= 0 || live > slots) throw std::runtime_error("invalid Xe2 benchmark live routes");
    std::vector<uint8_t> zeros(size_t(w.N) * w.row_scales);
    q.memcpy(zeros.data(), w.zeros, zeros.size()).wait_and_throw();
    // The native loader may already have folded a symmetric GPTQ tensor.
    const bool signed_payload = std::all_of(zeros.begin(), zeros.end(), [](uint8_t z) { return z == 0xff; });
    if (!signed_payload && !std::all_of(zeros.begin(), zeros.end(), [](uint8_t z) { return z == 8; })) {
        const auto [lo, hi] = std::minmax_element(zeros.begin(), zeros.end());
        throw std::runtime_error("Xe2 cannot apply runtime zero points " + std::to_string(*lo) + ".." + std::to_string(*hi));
    }

    const size_t words = size_t(w.N) * w.row_bytes / sizeof(uint32_t);
    const size_t values = size_t(live) * N;
    Buffer<uint32_t> folded(q, signed_payload ? 1 : words);
    const uint8_t* packed = signed_payload ? w.payload : reinterpret_cast<const uint8_t*>(folded.data);
    // As in the lane, keep D allocated to routed-slot capacity, not
    // sum(counts). Xe2's block operations must see the same storage extent.
    constexpr size_t capacity = size_t(slots) * N;
    Buffer<Bf16> bf(q, capacity);
    Buffer<float> out(q, capacity);
    Buffer<int32_t> atomic(q, 1);
    int tile_count = 0;
    std::array<int32_t, experts> host_counts{};
    std::array<int32_t, experts+1> host_offsets{};
    q.memcpy(&tile_count, live_tiles, sizeof(tile_count));
    q.memcpy(host_counts.data(), counts, sizeof(host_counts));
    q.memcpy(host_offsets.data(), offsets, sizeof(host_offsets)).wait_and_throw();
    if (tile_count < 1 || tile_count > max_tiles) throw std::runtime_error("invalid Xe2 tile count");
    std::vector<b70::QfnMoeTile> host_tiles(tile_count);
    q.memcpy(host_tiles.data(), tiles, host_tiles.size()*sizeof(host_tiles[0])).wait_and_throw();
    auto extents = layer0_allocations(max_tiles, !signed_payload);
    check_layout(host_counts, host_offsets, host_tiles, extents, !signed_payload);
    auto fold = [&] {
        const auto* source = reinterpret_cast<const uint32_t*>(w.payload);
        auto* dst = folded.data;
        return q.parallel_for(sycl::range<1>(words), [=](sycl::id<1> i) { dst[i] = fold_int4(source[i]); });
    };
    auto control = [&] {
        return b70::launch_qfn_int4_grouped(q, w, N, x, out.data, tiles, max_tiles, {}, false, live_tiles);
    };
    const auto* live_rows = offsets + experts;
    auto xe2 = [&] {
        auto clear = q.fill(atomic.data, int32_t{0}, 1);
        if (fp32_output) {
            auto gemm = launch_lane(q,x,packed,w.scales,out.data,counts,atomic.data);
            return std::array{clear,gemm,gemm}; // FP32 writes directly to the consumer buffer
        }
        auto gemm = launch_lane(q,x,packed,w.scales,bf.data,counts,atomic.data);
        return std::array{clear,gemm,widen_live(q,bf.data,live_rows,out.data)};
    };

    // Gate before timing. Poison outputs to make partial writes fail closed.
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> a(values), b(values);
    q.fill(out.data, poison, values);
    control();
    q.memcpy(a.data(), out.data, values * sizeof(float)).wait_and_throw();
    if (!signed_payload) fold();
    if (fp32_output) q.fill(out.data,poison,values);
    else q.fill(bf.data,Bf16(poison),values);
    xe2();
    q.memcpy(b.data(), out.data, values * sizeof(float)).wait_and_throw();
    if (!oracle_rows.empty()) {
        if (oracle_rows.size() != size_t(live)) throw std::runtime_error("one-hot oracle row count differs");
        bool passed = check_apply_scale(q);
        for (int half = 0; half < 2; ++half) {
            const char* projection = half ? "up" : "gate";
            size_t n64_bad = 0, xe2_bad = 0, trunc_bad = 0, exact_product_bad = 0, negative_bad = 0, positive_bad = 0;
            for (int row = 0; row < live; ++row) {
                const auto hot = oracle_rows[row];
                for (int n = half*N/2; n < (half+1)*N/2; ++n) {
                    const size_t i = size_t(row)*N+n;
                    const float raw = hot_product(hot,n);
                    const auto rounded = b70::f32_to_bf16(raw);
                    const float want = b70::bf16_to_f32(rounded);
                    const float trunc = b70::bits_to_f32(b70::f32_to_bits(raw) & 0xffff0000u);
                    if (a[i] != want) {
                        if (!n64_bad) std::fprintf(stderr,
                            "QFN one-hot first wrong: arm=n64 projection=%s expert=%d row=%d k=%d group=%d col=%d code=%d scale=0x%04x expected=0x%04x observed=0x%08x\n",
                            projection, hot.expert, row, hot.k, hot.k/group, n-half*N/2, hot_code(hot.expert,n,hot.k),
                            unsigned(hot_scale(hot.expert,n,hot.k/group)), unsigned(rounded.bits), b70::f32_to_bits(a[i]));
                        ++n64_bad;
                    }
                    if (b[i] != want) {
                        if (!xe2_bad) std::fprintf(stderr,
                            "QFN one-hot first wrong: arm=xe2 projection=%s expert=%d row=%d k=%d group=%d col=%d code=%d scale=0x%04x expected=0x%04x observed=0x%04x trunc=0x%04x\n",
                            projection, hot.expert, row, hot.k, hot.k/group, n-half*N/2, hot_code(hot.expert,n,hot.k),
                            unsigned(hot_scale(hot.expert,n,hot.k/group)), unsigned(rounded.bits),
                            b70::f32_to_bits(b[i])>>16, b70::f32_to_bits(trunc)>>16);
                        ++xe2_bad;
                        exact_product_bad += raw == want;
                        negative_bad += raw < 0;
                        positive_bad += raw > 0;
                    }
                    trunc_bad += b[i] != trunc;
                }
            }
            std::fprintf(stderr,
                "QFN one-hot check: projection=%s values=%zu n64_bad=%zu xe2_bad=%zu xe2_vs_trunc_bad=%zu xe2_exact_product_bad=%zu xe2_negative_bad=%zu xe2_positive_bad=%zu\n",
                projection, size_t(live)*N/2, n64_bad, xe2_bad, trunc_bad, exact_product_bad, negative_bad, positive_bad);
            passed &= !n64_bad && !xe2_bad;
        }
        return passed;
    }
    size_t outside = 0, exact_bf16 = 0, exact_fp32 = 0, trunc_bf16 = 0, nonfinite = 0;
    double square = 0, norm = 0, max_abs = 0;
    for (size_t i = 0; i < values; ++i) {
        const double delta = double(a[i]) - b[i];
        const bool equal = b70::f32_to_bits(a[i]) == b70::f32_to_bits(b[i]);
        exact_fp32 += equal;
        outside += fp32_output ? (!equal || !std::isfinite(a[i]) || !std::isfinite(b[i])) : !within_bf16_roundoff(a[i], b[i]);
        nonfinite += !std::isfinite(a[i]) || !std::isfinite(b[i]);
        exact_bf16 += float(Bf16(a[i])) == b[i];
        trunc_bf16 += b70::bits_to_f32(b70::f32_to_bits(a[i]) & 0xffff0000u) == b[i];
        max_abs = std::max(max_abs, std::abs(delta));
        square += delta * delta;
        norm += double(a[i]) * a[i];
    }
    std::fprintf(stderr, "QFN Xe2 projection: rank=%d layer=%d live=%d values=%zu outside=%zu exact_bf16=%zu max_abs=%.8g rel_rms=%.8g layout=%s trunc_bf16=%zu output=%s exact_fp32=%zu nonfinite=%zu\n",
                 rank, layer, live, values, outside, exact_bf16, max_abs, std::sqrt(square / norm),
                 signed_payload ? "resident-s4" : "u4-z8", trunc_bf16,
                 fp32_output ? "fp32" : "bf16", exact_fp32, nonfinite);
    if (outside) return false;

    // Both Xe2 arms widen D to the existing FP32 consumer. 'staged' folds
    // weights outside the call (one-time), 'per-call' charges the full fold.
    // Paired samples alternate direction. CPU wall includes all submissions.
    const std::array<const char*, 3> names{"n64", signed_payload ? (fp32_output ? "xe2-fp32" : "xe2-direct") : "xe2-staged", "xe2-per-call"};
    const int num_arms = signed_payload ? 2 : 3;
    for (int rep = -2; rep < 8; ++rep) for (int i = 0; i < num_arms; ++i) {
        const int arm = (rep % 2 == 0) ? i : num_arms - 1 - i;
        const auto start = Clock::now();
        sycl::event conversion;
        if (arm == 2) conversion = fold();
        auto events = arm ? xe2() : std::array{sycl::event{}, control(), sycl::event{}};
        events[arm ? 2 : 1].wait_and_throw();
        const double wall = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        if (rep >= 0) std::fprintf(stderr,
            "QFN Xe2 timing: rank=%d layer=%d pair=%d arm=%s wall_ms=%.6f fold_ms=%.6f clear_ms=%.6f gemm_ms=%.6f widen_ms=%.6f\n",
            rank, layer, rep, names[arm], wall, arm == 2 ? ms(conversion) : 0., arm ? ms(events[0]) : 0.,
            ms(events[1]), arm && !fp32_output ? ms(events[2]) : 0.);
    }
    return true;
}
#ifndef QFN_XE2_SINGLE_ONLY   // DecodeM8 needs prefill kernels the probe omits
namespace {
constexpr int decode_top_k = 10, decode_inter = N / 2, decode_hidden = K;
using DecodePolicy = MoE::w4a16_policy_m_8;

struct ErrorStats {
    double error_sq = 0., reference_sq = 0., max_abs = 0., peak = 0.;
    bool finite = true;
    void add(float a, float b) {
        if (!std::isfinite(a) || !std::isfinite(b)) { finite = false; return; }
        const double d = double(a) - b;
        error_sq += d*d;
        reference_sq += double(b)*b;
        max_abs = std::max(max_abs, std::abs(d));
        peak = std::max(peak, double(std::abs(b)));
    }
    double rms() const { return reference_sq == 0. && error_sq == 0. ? 0. : std::sqrt(error_sq / reference_sq); }
    // These are numerical sanity bounds, not a parity waiver. The model must
    // also pass the independent all-rank frozen-ID gate before timing.
    bool acceptable() const { return finite && rms() <= .015 && max_abs <= .04 * peak; }
};
}

struct DecodeM8::Scratch {
    sycl::queue q;
    Buffer<Bf16> x, h;
    Buffer<int32_t> counts, inverse, live, counter;
    Buffer<float> gu, down, result;
    explicit Scratch(sycl::queue& input_queue) :
        q(input_queue), x(q, decode_top_k * decode_hidden), h(q, decode_top_k * decode_inter),
        counts(q, experts), inverse(q, decode_top_k), live(q, 1), counter(q, 1),
        gu(q, decode_top_k * N), down(q, decode_top_k * decode_hidden), result(q, decode_hidden) {}
};

DecodeM8::DecodeM8(sycl::queue& q) : scratch(std::make_unique<Scratch>(q)) {}
DecodeM8::~DecodeM8() { scratch->q.wait(); }

void DecodeM8::run(const b70::MoeLayer& moe, const int32_t* routes, const float* weights,
                   const float* input, float* output, const float* reference_hidden,
                   int rank, int pos, int layer) {
    auto& s = *scratch;
    auto& q = s.q;
    const auto valid = [](const b70::QuantWeight& w, int n, int k) {
        return w.fmt == b70::Fmt::INT4 && w.N == experts*n && w.K == k &&
               w.row_bytes == k/2 && w.row_scales == k/group && w.payload && w.scales;
    };
    if (!q.is_in_order() || moe.cfg.top_k != decode_top_k || moe.cfg.num_experts != experts ||
        moe.cfg.hidden != decode_hidden || moe.cfg.inter != decode_inter ||
        !valid(moe.gate_up, N, K) || !valid(moe.down, decode_hidden, decode_inter))
        throw std::runtime_error("Xe2 decode M8 needs QFN EP4 W4A16 and one in-order queue");
    auto* counts = s.counts.data;
    auto* inv = s.inverse.data;
    auto* live = s.live.data;
    auto* x = s.x.data;
    q.parallel_for(sycl::range<1>(decode_hidden), [=](sycl::id<1> idx) {
        const int j = int(idx[0]);
        if (j < experts) {
            int count = 0;
            for (int i = 0; i < decode_top_k; ++i) count += routes[i] == j;
            counts[j] = count;
        }
        if (j < decode_top_k) {
            const int e = routes[j];
            int index = -1;
            if (e >= 0 && e < experts) {
                index = 0;
                for (int i = 0; i < decode_top_k; ++i)
                    index += routes[i] >= 0 && (routes[i] < e || (routes[i] == e && i < j));
            }
            inv[j] = index;
        }
        int n = 0;
        for (int i = 0; i < decode_top_k; ++i) n += routes[i] >= 0 && routes[i] < experts;
        if (j == 0) *live = n;
        const Bf16 v(input[j]);
        for (int i = 0; i < n; ++i) x[i*decode_hidden + j] = v;
    });
    if (reference_hidden) {
        const float poison = std::numeric_limits<float>::quiet_NaN();
        q.fill(s.gu.data, poison, decode_top_k * N);
        q.fill(s.down.data, poison, decode_top_k * decode_hidden);
    }
    q.fill(s.counter.data, int32_t{0}, 1);
    launch_lane<float, N, K, DecodePolicy>(q, x, moe.gate_up.payload, moe.gate_up.scales,
                                         s.gu.data, counts, s.counter.data);
    b70::launch_swiglu_bf16_out(q, s.gu.data, s.h.data, decode_top_k, decode_inter, {}, live);
    q.fill(s.counter.data, int32_t{0}, 1);
    launch_lane<float, decode_hidden, decode_inter, DecodePolicy>(q, s.h.data, moe.down.payload,
                                          moe.down.scales, s.down.data, counts, s.counter.data);
    b70::launch_moe_unpermute(q, s.down.data, inv, weights, reference_hidden ? s.result.data : output,
                              1, decode_top_k, decode_hidden, {});
    if (!reference_hidden) return;

    std::array<int32_t, decode_top_k> order{};
    std::array<Bf16, decode_top_k * decode_inter> actual_h{};
    std::array<float, decode_top_k * decode_inter> ref_h{};
    std::array<float, decode_hidden> actual_y{}, ref_y{};
    q.memcpy(order.data(), inv, sizeof order);
    q.memcpy(actual_h.data(), s.h.data, sizeof actual_h);
    q.memcpy(ref_h.data(), reference_hidden, sizeof ref_h);
    q.memcpy(actual_y.data(), s.result.data, sizeof actual_y);
    q.memcpy(ref_y.data(), output, sizeof ref_y).wait_and_throw();
    ErrorStats hidden, weighted;
    int n = 0;
    for (int slot = 0; slot < decode_top_k; ++slot) {
        if (order[slot] < 0) continue;
        ++n;
        for (int k = 0; k < decode_inter; ++k)
            hidden.add(float(actual_h[order[slot]*decode_inter+k]), ref_h[slot*decode_inter+k]);
    }
    for (int j = 0; j < decode_hidden; ++j) weighted.add(actual_y[j], ref_y[j]);
    const bool ok = hidden.acceptable() && weighted.acceptable();
    std::fprintf(stderr, "QFN M8 decode replay: rank=%d pos=%d layer=%d live=%d hidden_rms=%.8g hidden_max=%.8g weighted_rms=%.8g weighted_max=%.8g finite=%d pass=%d\n",
                 rank, pos, layer, n, hidden.rms(), hidden.max_abs, weighted.rms(), weighted.max_abs,
                 hidden.finite && weighted.finite, ok);
    if (!ok) throw std::runtime_error("Xe2 M8 real decode replay outside numerical guard");
}
#endif
} // namespace qfn_xe2
