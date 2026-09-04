#include "b70/native_model.hpp"
#include "b70/qwen35.hpp"
#include "b70/weights.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {
uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

bool seek_pad(FILE* f, uint64_t& pos) {
    const uint64_t next = align_up(pos, b70::kNativeAlignment);
    static const unsigned char zeros[b70::kNativeAlignment]{};
    if (next != pos && std::fwrite(zeros, 1, next - pos, f) != next - pos) return false;
    pos = next;
    return true;
}

bool write_bytes(FILE* f, const void* p, size_t n, uint64_t& pos) {
    if (n && std::fwrite(p, 1, n, f) != n) return false;
    pos += n;
    return true;
}

// Tensors that must survive conversion at full BF16 precision.
//
// The quantization policy below is otherwise purely structural -- any 2-D BF16
// ".weight" -- so norms, A_log and dt_bias are spared only by accident of being
// 1-D and conv1d only by being 3-D.  These four are 2-D and were therefore
// being quantized by rank alone, with no regard for what they feed:
//
//   in_proj_a  -> A_log -> exponentiated into the delta-rule decay `alpha`.
//                 A 4-bit absolute error becomes an exponentially amplified
//                 error in alpha and then compounds along the sequence as
//                 alpha^M: harmless at 32 tokens, destructive at 4096.
//   in_proj_b  -> `beta`, the delta-rule write strength, which also forms the
//                 matrix inverted by the chunked GDN kernel.
//   mlp.gate   -> the MoE router.  4-bit logits can flip which of 256 experts
//                 a token is routed to.
//   shared_expert_gate -> a single row (1xH) feeding a sigmoid.
//
// Every reference quantizer for this architecture (AutoRound, GPTQ) excludes
// in_proj_a/in_proj_b for the same reason, and grimoire.cpp already asks for
// Fmt::BF16 for all four.  Keeping them costs ~47 MB on Qwen 27B and ~51 MB on
// Ornith -- a rounding error against a 15-18 GiB artifact.
bool keeps_bf16(const std::string& name) {
    auto ends_with = [&](const char* suffix) {
        const size_t n = std::strlen(suffix);
        return name.size() >= n && name.compare(name.size() - n, n, suffix) == 0;
    };
    // The MTP draft head. Measured 2026-09-04 on Qwen3.8-27B at a 4096-token
    // prompt with llama-benchy: the head quantized to MXFP4 accepts 0-23% of
    // its drafts (TG 14.8), while the same architecture with a BF16 head
    // accepts 43-73% (TG 28-33, peaking at 47.2). Speculation multiplies the
    // head's error -- a draft is either bit-exact against the target argmax or
    // it is thrown away -- so 4-bit rounding here does not degrade quality
    // gracefully, it destroys acceptance. The head is one small layer: keeping
    // it costs ~0.65 GiB against a 15 GiB artifact and buys back ~2x decode.
    if (name.rfind("mtp.", 0) == 0) return true;
    // ".mlp.gate.weight" must not catch ".mlp.gate_proj.weight".
    return ends_with(".linear_attn.in_proj_a.weight") ||
           ends_with(".linear_attn.in_proj_b.weight") ||
           ends_with(".mlp.gate.weight") ||
           ends_with(".mlp.shared_expert_gate.weight");
}

bool is_matrix_weight(const std::string& name, const b70::TensorRef& r) {
    if (r.t.dtype != b70::STDtype::BF16) return false;
    const bool matrix = r.t.shape.size() == 2;
    const bool fused_experts = r.t.shape.size() == 3 &&
                               name.find(".mlp.experts.") != std::string::npos;
    if (!matrix && !fused_experts) return false;
    if (name.find("embed_tokens") != std::string::npos) return false;
    if (keeps_bf16(name)) return false;
    return fused_experts ||
           (name.size() >= 7 && name.compare(name.size() - 7, 7, ".weight") == 0);
}

}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s BF16_MODEL_DIR OUTPUT.b70\n", argv[0]);
        return 2;
    }
    b70::Qwen35Model model;
    std::string err;
    if (!model.load(argv[1], err, true, true)) {
        std::fprintf(stderr, "load: %s\n", err.c_str()); return 1;
    }
    model.unmap_all();
    FILE* out = std::fopen(argv[2], "wb+");
    if (!out) { std::fprintf(stderr, "open %s: %s\n", argv[2], std::strerror(errno)); return 1; }

    b70::NativeFileHeader hdr{b70::kNativeMagic, b70::kNativeVersion,
        b70::kNativeAlignment, 0, 0, 0, 0x031, 0};
    uint64_t pos = 0;
    if (!write_bytes(out, &hdr, sizeof hdr, pos) || !seek_pad(out, pos)) return 1;
    std::vector<b70::NativeTensorRecord> toc;

    for (const auto& kv : model.index) {
        const std::string& name = kv.first;
        const b70::TensorRef& ref = kv.second;
        b70::NativeTensorRecord rec{};
        std::snprintf(rec.name, sizeof rec.name, "%s", name.c_str());
        rec.source_dtype = uint32_t(ref.t.dtype);
        rec.rank = uint32_t(std::min<size_t>(4, ref.t.shape.size()));
        for (size_t i = 0; i < rec.rank; ++i) rec.shape[i] = ref.t.shape[i];

        if (is_matrix_weight(name, ref)) {
            const int K = int(ref.t.shape.back());
            const int N = int(ref.t.numel() / K);
            std::vector<float> fp(size_t(N) * K);
            if (!model.shards[ref.shard]->read_f32(ref.t, fp.data(), err)) {
                std::fprintf(stderr, "read %s: %s\n", name.c_str(), err.c_str()); return 1;
            }
            auto q = b70::quantize(fp.data(), N, K, b70::Fmt::MXFP4);
            rec.padded_n = N; rec.padded_k = K;
            rec.encoding = uint32_t(b70::NativeEncoding::MXFP4_GRIMOIRE_XE2);
            rec.tile_n = 64; rec.tile_k = 256;
            if (!seek_pad(out, pos)) return 1;
            rec.payload_offset = pos; rec.payload_bytes = q.payload.size();
            if (!write_bytes(out, q.payload.data(), q.payload.size(), pos) || !seek_pad(out, pos)) return 1;
            rec.scales_offset = pos; rec.scales_bytes = q.scales_raw.size();
            if (!write_bytes(out, q.scales_raw.data(), q.scales_raw.size(), pos)) return 1;
        } else {
            std::vector<unsigned char> raw(size_t(ref.t.end - ref.t.begin));
            if (!model.read_raw(ref, raw.data(), err)) {
                std::fprintf(stderr, "read %s: %s\n", name.c_str(), err.c_str()); return 1;
            }
            rec.encoding = uint32_t(b70::NativeEncoding::RAW);
            if (!seek_pad(out, pos)) return 1;
            rec.payload_offset = pos; rec.payload_bytes = raw.size();
            if (!write_bytes(out, raw.data(), raw.size(), pos)) return 1;
        }
        toc.push_back(rec);
        std::printf("\r  %-70.70s", name.c_str()); std::fflush(stdout);
    }
    if (!seek_pad(out, pos)) return 1;
    hdr.toc_offset = pos; hdr.tensor_count = toc.size();
    if (!write_bytes(out, toc.data(), toc.size() * sizeof(toc[0]), pos)) return 1;
    hdr.file_size = pos;
    if (std::fseek(out, 0, SEEK_SET) || std::fwrite(&hdr, 1, sizeof hdr, out) != sizeof hdr ||
        std::fclose(out)) return 1;
    std::printf("\ncompiled %zu tensors -> %s (%.2f GiB)\n", toc.size(), argv[2],
                double(hdr.file_size) / 1073741824.0);
    return 0;
}
