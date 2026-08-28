// =====================================================================
//  test_safetensors.cpp  --  writes real safetensors files, reads them
//  back through the same path the converter uses, and checks the
//  end-to-end HF -> quantized pipeline.
// =====================================================================
#include "b70/safetensors.hpp"
#include "b70/weights.hpp"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include <string>

using namespace b70;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); \
    std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

// ---------------------------------------------------------------------
// Build a safetensors file by hand so the parser is tested against the
// real byte layout, not a mock.
// ---------------------------------------------------------------------
struct Entry {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};

static bool write_safetensors(const std::string& path,
                              const std::vector<Entry>& entries,
                              const std::string& extra_meta = "") {
    std::string hdr = "{";
    if (!extra_meta.empty()) hdr += "\"__metadata__\":{" + extra_meta + "},";
    uint64_t off = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        const Entry& e = entries[i];
        hdr += "\"" + e.name + "\":{\"dtype\":\"" + e.dtype + "\",\"shape\":[";
        for (size_t d = 0; d < e.shape.size(); ++d) {
            if (d) hdr += ",";
            hdr += std::to_string(e.shape[d]);
        }
        hdr += "],\"data_offsets\":[" + std::to_string(off) + ","
             + std::to_string(off + e.bytes.size()) + "]}";
        off += e.bytes.size();
        if (i + 1 < entries.size()) hdr += ",";
    }
    hdr += "}";
    // safetensors pads the header to an 8-byte boundary with spaces
    while (hdr.size() % 8) hdr += ' ';

    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint64_t hlen = hdr.size();
    std::fwrite(&hlen, 8, 1, f);
    std::fwrite(hdr.data(), 1, hdr.size(), f);
    for (const Entry& e : entries) std::fwrite(e.bytes.data(), 1, e.bytes.size(), f);
    std::fclose(f);
    return true;
}

template <typename T>
static std::vector<uint8_t> raw(const std::vector<T>& v) {
    std::vector<uint8_t> b(v.size() * sizeof(T));
    std::memcpy(b.data(), v.data(), b.size());
    return b;
}

static uint16_t f32_to_f16(float f) {
    const uint32_t u = f32_to_bits(f);
    const uint32_t s = (u >> 16) & 0x8000;
    int   e = int((u >> 23) & 0xFF) - 127 + 15;
    uint32_t m = u & 0x7FFFFF;
    if (e <= 0)   return uint16_t(s);
    if (e >= 31)  return uint16_t(s | 0x7C00);
    return uint16_t(s | (uint32_t(e) << 10) | (m >> 13));
}

// ---------------------------------------------------------------------
static void test_parse() {
    std::printf("Header parsing and dtype conversion\n");
    const char* path = "/tmp/b70_test_a.safetensors";

    std::vector<float>    f32 = {1.0f, -2.5f, 3.25f, 0.0f, 1e-8f, 65504.0f};
    std::vector<uint16_t> bf16, f16;
    for (float v : f32) { bf16.push_back(f32_to_bf16(v).bits); f16.push_back(f32_to_f16(v)); }
    std::vector<int8_t>  i8  = {-128, -1, 0, 1, 127, 42};
    std::vector<int32_t> i32 = {-100000, -1, 0, 1, 100000, 7};

    std::vector<Entry> es = {
        {"w.f32",  "F32",  {2, 3}, raw(f32)},
        {"w.bf16", "BF16", {6},    raw(bf16)},
        {"w.f16",  "F16",  {3, 2}, raw(f16)},
        {"w.i8",   "I8",   {6},    raw(i8)},
        {"w.i32",  "I32",  {6},    raw(i32)},
    };
    CHECK(write_safetensors(path, es, "\"format\":\"pt\""), "could not write test file");

    SafeTensors st;
    std::string err;
    CHECK(st.open(path, err), "open failed: %s", err.c_str());
    CHECK(st.tensors().size() == 5, "expected 5 tensors, got %zu", st.tensors().size());
    CHECK(st.metadata().count("format") == 1, "metadata not parsed");
    CHECK(st.metadata().at("format") == "pt", "metadata value wrong");

    const STTensor* t = st.find("w.f32");
    CHECK(t != nullptr, "w.f32 missing");
    if (t) {
        CHECK(t->shape.size() == 2 && t->shape[0] == 2 && t->shape[1] == 3, "shape wrong");
        CHECK(t->numel() == 6, "numel %lld", (long long)t->numel());
        std::vector<float> out(6);
        st.read_f32(*t, out.data(), err);
        for (int i = 0; i < 6; ++i)
            CHECK(out[i] == f32[i], "f32[%d] %g != %g", i, out[i], f32[i]);
    }

    t = st.find("w.bf16");
    if (t) {
        std::vector<float> out(6);
        st.read_f32(*t, out.data(), err);
        for (int i = 0; i < 6; ++i)
            CHECK(out[i] == bf16_to_f32(bf16_t{bf16[i]}), "bf16[%d] mismatch", i);
    }

    t = st.find("w.f16");
    if (t) {
        std::vector<float> out(6);
        st.read_f32(*t, out.data(), err);
        // 1.0, -2.5, 3.25 and 0.0 are exact in fp16
        CHECK(out[0] == 1.0f,  "f16 1.0 -> %g", out[0]);
        CHECK(out[1] == -2.5f, "f16 -2.5 -> %g", out[1]);
        CHECK(out[2] == 3.25f, "f16 3.25 -> %g", out[2]);
        CHECK(out[3] == 0.0f,  "f16 0.0 -> %g", out[3]);
        CHECK(out[5] == 65504.0f, "f16 max -> %g", out[5]);
    }

    t = st.find("w.i8");
    if (t) {
        std::vector<float> out(6);
        st.read_f32(*t, out.data(), err);
        CHECK(out[0] == -128.0f && out[4] == 127.0f, "i8 conversion wrong");
    }
    t = st.find("w.i32");
    if (t) {
        std::vector<float> out(6);
        st.read_f32(*t, out.data(), err);
        CHECK(out[0] == -100000.0f, "i32 conversion wrong: %g", out[0]);
    }
    std::remove(path);
}

// ---------------------------------------------------------------------
// A truncated or lying header must be rejected, not segfault. Anything
// downloaded off the internet gets this treatment.
// ---------------------------------------------------------------------
static void test_malformed() {
    std::printf("Malformed input rejection\n");
    std::string err;

    {   // shape says 100 elements, byte range says 4
        const char* p = "/tmp/b70_bad_shape.safetensors";
        std::string hdr = "{\"x\":{\"dtype\":\"F32\",\"shape\":[100],"
                          "\"data_offsets\":[0,4]}}";
        while (hdr.size() % 8) hdr += ' ';
        FILE* f = std::fopen(p, "wb");
        uint64_t h = hdr.size();
        std::fwrite(&h, 8, 1, f);
        std::fwrite(hdr.data(), 1, hdr.size(), f);
        float v = 1.0f;
        std::fwrite(&v, 4, 1, f);
        std::fclose(f);
        SafeTensors st;
        CHECK(!st.open(p, err), "should reject shape/bytes mismatch");
        std::remove(p);
    }
    {   // offsets past end of file
        const char* p = "/tmp/b70_bad_off.safetensors";
        std::string hdr = "{\"x\":{\"dtype\":\"F32\",\"shape\":[1000000],"
                          "\"data_offsets\":[0,4000000]}}";
        while (hdr.size() % 8) hdr += ' ';
        FILE* f = std::fopen(p, "wb");
        uint64_t h = hdr.size();
        std::fwrite(&h, 8, 1, f);
        std::fwrite(hdr.data(), 1, hdr.size(), f);
        std::fclose(f);
        SafeTensors st;
        CHECK(!st.open(p, err), "should reject out-of-range offsets");
        std::remove(p);
    }
    {   // unknown dtype
        const char* p = "/tmp/b70_bad_dt.safetensors";
        std::string hdr = "{\"x\":{\"dtype\":\"BAD8\",\"shape\":[4],"
                          "\"data_offsets\":[0,4]}}";
        while (hdr.size() % 8) hdr += ' ';
        FILE* f = std::fopen(p, "wb");
        uint64_t h = hdr.size();
        std::fwrite(&h, 8, 1, f);
        std::fwrite(hdr.data(), 1, hdr.size(), f);
        uint32_t z = 0;
        std::fwrite(&z, 4, 1, f);
        std::fclose(f);
        SafeTensors st;
        CHECK(!st.open(p, err), "should reject unknown dtype");
        std::remove(p);
    }
    {   // garbage header length
        const char* p = "/tmp/b70_bad_len.safetensors";
        FILE* f = std::fopen(p, "wb");
        uint64_t h = 0xFFFFFFFFFFULL;
        std::fwrite(&h, 8, 1, f);
        std::fclose(f);
        SafeTensors st;
        CHECK(!st.open(p, err), "should reject bogus header length");
        std::remove(p);
    }
    std::printf("  4 malformed inputs rejected cleanly\n");
}

// ---------------------------------------------------------------------
// The real pipeline: a bf16 checkpoint tensor -> fp32 -> quantized ->
// dequantized, error within the format's budget.
// ---------------------------------------------------------------------
static void test_pipeline() {
    std::printf("\nHF bf16 checkpoint -> quantized, per format\n");
    const int N = 128, K = 512;
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.0f, 0.02f);

    std::vector<uint16_t> bf(size_t(N) * K);
    std::vector<float>    truth(size_t(N) * K);
    for (size_t i = 0; i < bf.size(); ++i) {
        const float v = nd(rng);
        const bf16_t b = f32_to_bf16(v);
        bf[i] = b.bits;
        truth[i] = bf16_to_f32(b);      // what the checkpoint actually stores
    }

    const char* path = "/tmp/b70_pipe.safetensors";
    std::vector<Entry> es = {{"model.layers.0.mlp.gate_proj.weight", "BF16", {N, K}, raw(bf)}};
    write_safetensors(path, es);

    SafeTensors st;
    std::string err;
    CHECK(st.open(path, err), "open: %s", err.c_str());
    const STTensor* t = st.find("model.layers.0.mlp.gate_proj.weight");
    CHECK(t != nullptr, "tensor not found");
    if (!t) return;

    std::vector<float> f32(t->numel());
    st.read_f32(*t, f32.data(), err);
    for (size_t i = 0; i < f32.size(); ++i)
        CHECK(f32[i] == truth[i], "bf16 read mismatch at %zu", i);

    const Fmt fs[] = { Fmt::BF16, Fmt::FP8_E4M3, Fmt::INT8,
                       Fmt::INT4, Fmt::MXFP8, Fmt::MXFP4 };
    for (Fmt f : fs) {
        PackedWeight p = quantize(f32.data(), N, K, f);
        QuantWeight  w = p.view();
        double se = 0, sr = 0;
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) {
                const double a = truth[size_t(n) * K + k], b = w.at(n, k);
                se += (a - b) * (a - b);
                sr += a * a;
            }
        const double rel = std::sqrt(se / sr);
        const double mb = double(p.payload.size() + p.scales_raw.size() + p.zeros.size()) / 1048576.0;
        std::printf("  %-10s %7.3f MiB  rel-RMS %.6f\n", fmt_name(f), mb, rel);
        CHECK(rel < 0.2, "%s error %.4f", fmt_name(f), rel);
    }
    // bf16 -> BF16 must be lossless: the checkpoint is already bf16
    PackedWeight p = quantize(f32.data(), N, K, Fmt::BF16);
    QuantWeight  w = p.view();
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            CHECK(w.at(n, k) == truth[size_t(n) * K + k],
                  "bf16 passthrough must be lossless at (%d,%d)", n, k);
    std::printf("  bf16 passthrough is bit-exact\n");
    std::remove(path);
}

int main() {
    std::printf("=== safetensors ingest tests ===\n\n");
    test_parse();
    test_malformed();
    test_pipeline();
    std::printf("\n%s (%d failures)\n", g_fail ? "FAILED" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
