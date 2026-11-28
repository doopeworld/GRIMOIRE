#include "b70/qwen35.hpp"
#include "b70/qwen4_exp.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <unistd.h>

namespace fs = std::filesystem;
using b70::Qwen35Model;

static void require(bool yes, const std::string& message) {
    if (!yes) throw std::runtime_error(message);
}

static void shard(const fs::path& path, const std::string& header, const std::string& data) {
    std::ofstream out(path, std::ios::binary);
    const uint64_t bytes = header.size();
    out.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    out << header << data;
    require(bool(out), "fixture shard write failed");
}

// Real-checkpoint qualification. The dense index supplies hash metadata;
// the external index supplies only table payload and its global scale.
static void compare_ple(const char* original, const char* dense) {
    ::unsetenv("GRIMOIRE_QFN_PLE_SOURCE");
    const auto load = [](Qwen35Model& model, const char* path) {
        std::string error;
        const bool ok = model.load(path, error);
        require(ok, error);
        model.unmap_all();
    };
    Qwen35Model qsa, target;
    load(qsa, original);
    ::setenv("GRIMOIRE_QFN_PLE_SOURCE", original, 1);
    load(target, dense);
    require(qsa.layers.size() == target.layers.size(), "layer counts differ");
    size_t qsa_a = 0, table_rows_checked = 0;
    for (size_t l = 0; l < target.layers.size(); ++l) {
        const auto& a = qsa.layers[l];
        const auto& b = target.layers[l];
        qsa_a += a.qsa;
        require(!b.qsa, "dense target retained a sparse layer");
        require(a.ple == b.ple, "PLE placement differs");
        if (!b.ple) continue;
        auto read = [](const Qwen35Model& model, const b70::TensorRef& ref) {
            require(ref.ok(), "expected indexed PLE tensor");
            std::vector<uint8_t> bytes(ref.t.end - ref.t.begin);
            std::string error;
            require(model.read_raw(ref, bytes.data(), error), "PLE pread failed: " + ref.t.name);
            return bytes;
        };
        auto i64 = [&](const b70::TensorRef& r1, const b70::TensorRef& r2) {
            require(r1.t.dtype == b70::STDtype::I64 && r2.t.dtype == r1.t.dtype,
                    "stored hash dtype");
            const auto raw = read(qsa, r1);
            require(raw == read(target, r2), "stored hash metadata differs: " + r1.t.name);
            std::vector<int64_t> out(raw.size() / sizeof(int64_t));
            std::memcpy(out.data(), raw.data(), raw.size());
            return out;
        };
        const auto mul = i64(a.ple_mul_t, b.ple_mul_t);
        const auto sizes = i64(a.ple_vsz_t, b.ple_vsz_t);
        const auto offsets = i64(a.ple_off_t, b.ple_off_t);
        require(!sizes.empty() && sizes.size() == offsets.size(), "wrong hash head count");
        require(mul.size() == size_t(target.cfg.ngram_size), "wrong hash order");
        require(a.ple_table_scale.t.dtype == b.ple_table_scale.t.dtype &&
                read(qsa, a.ple_table_scale) == read(target, b.ple_table_scale), "global scale differs");
        float scale = 0;
        std::string error;
        require(b.ple_table_scale.t.numel() == 1 && target.shards.at(b.ple_table_scale.shard)->read_f32(
                    b.ple_table_scale.t, &scale, error) && std::isfinite(scale) && scale > 0, "invalid global scale");
        require(!a.ple_shards.empty() && a.ple_shards.size() == b.ple_shards.size(), "PLE shard count differs");
        const auto span = a.ple_shards[0].t.shape.at(0), width = a.ple_shards[0].t.shape.at(1);
        require(span > 0 && width > 0, "empty PLE shard");
        for (size_t s = 0; s < a.ple_shards.size(); ++s) {
            const auto& x = a.ple_shards[s].t;
            const auto& y = b.ple_shards[s].t;
            require(x.name == y.name && x.shape == y.shape && x.dtype == b70::STDtype::F8_E4M3 &&
                    y.dtype == x.dtype, "PLE shard order or layout differs");
        }
        // Include EOS in the middle to exercise history reset as well as the
        // first token's EOS padding. Also probe both sides of every shard edge.
        const int ids[] = {17, 23, target.cfg.eos_token_id, 391, 760, 11};
        std::vector<int64_t> rows(sizes.size());
        auto compare_row = [&](int64_t row) {
            require(row >= 0, "negative hashed row");
            const auto s = size_t(row / span);
            const auto lr = row % span;
            require(s < a.ple_shards.size() && lr < a.ple_shards[s].t.shape.at(0), "hashed row outside table");
            auto x = a.ple_shards[s], y = b.ple_shards[s];
            for (auto* ref : {&x, &y}) {
                ref->t.begin += uint64_t(lr) * width;
                ref->t.end = ref->t.begin + width;
            }
            require(read(qsa, x) == read(target, y), "FP8 lookup differs at row " + std::to_string(row));
            ++table_rows_checked;
        };
        for (int t = 0; t < int(std::size(ids)); ++t) {
            b70::qwen4_exp::ple_ngram_ids(ids[t], ids, t, mul.data(), sizes.data(), offsets.data(),
                target.cfg.ngram_size - 1, target.cfg.heads_per_ngram, int(sizes.size()),
                target.cfg.eos_token_id, rows.data());
            for (auto row : rows) compare_row(row);
        }
        for (size_t s = 0; s < a.ple_shards.size(); ++s) {
            compare_row(int64_t(s) * span);
            compare_row(int64_t(s) * span + a.ple_shards[s].t.shape.at(0) - 1);
        }
        std::printf("layer %zu: %zu FP8 shards, scale %.9g, hash metadata identical\n",
                    l + 1, b.ple_shards.size(), scale);
    }
    require(qsa_a != 0 && table_rows_checked != 0, "expected a QSA source with PLE");
    std::printf("PASS %zu PLE row probes, QSA layers %zu -> 0\n", table_rows_checked, qsa_a);
}

int main(int argc, char** argv) try {
    if (argc == 3) {
        compare_ple(argv[1], argv[2]);
        return 0;
    }
    if (argc == 2) {
        Qwen35Model model;
        std::string error;
        const bool ok = model.load(argv[1], error);
        require(ok, error);
        size_t qsa = 0, ple = 0;
        for (const auto& layer : model.layers) {
            qsa += layer.qsa;
            ple += layer.ple_shards.size();
        }
        std::printf("resolved layers=%zu qsa=%zu PLE shards=%zu tensors=%zu\n",
                    model.layers.size(), qsa, ple, model.index.size());
        return 0;
    }
    char tmp_dir[] = "/tmp/grimoire-qfn-index-XXXXXX";
    require(::mkdtemp(tmp_dir) != nullptr, "mkdtemp failed");
    const fs::path tmp = tmp_dir;
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); }
    } cleanup{tmp};
    const fs::path root = tmp / "models", model = root / "dense", source = root / "table";
    fs::create_directories(model);
    fs::create_directories(source);
    const std::string embed = "model.embed_tokens.weight";
    const std::string indexer = "model.layers.0.self_attn.indexer.q_layernorm.weight";
    const std::string ple = "model.layers.0.ple.ple_embedding.ngram_embedding.";
    std::ofstream(model/"config.json") << R"({"model_type":"qwen4_exp_text","hidden_size":2,"num_hidden_layers":1,"ngram_size":2,"heads_per_ngram":1,"ple_embed_dim":2,"ple_layer_ids":[1],"layer_types":["full_attention"]})";
    shard(root/"trunk.safetensors", "{\""+embed+R"(":{"dtype":"BF16","shape":[1,2],"data_offsets":[0,4]},")"+
        indexer+R"(":{"dtype":"BF16","shape":[1],"data_offsets":[4,6]}})", std::string(6, '\0'));
    fs::create_symlink("../trunk.safetensors", model/"body.safetensors");
    shard(source/"ple.safetensors", "{\""+ple+R"(shard_0.weight":{"dtype":"F8_E4M3","shape":[1,2],"data_offsets":[0,2]},")"+
        ple+R"(weight_scale":{"dtype":"BF16","shape":[],"data_offsets":[2,4]},"sidecar.extra":{"dtype":"U8","shape":[1],"data_offsets":[4,5]}})",
        std::string("\x38\x40\x00\x40\x7b", 5));
    const auto manifest = model/"model.safetensors.index.json";
    std::ofstream(manifest) << "{\"weight_map\":{\"" << embed << "\":\"body.safetensors\"}}";
    const auto table_manifest = source/"model.safetensors.index.json";
    std::ofstream(table_manifest) << "{\"weight_map\":{\"" << ple << "shard_0.weight\":\"ple.safetensors\",\""
                                 << ple << "weight_scale\":\"ple.safetensors\",\"sidecar.extra\":\"ple.safetensors\"}}";
    ::setenv("GRIMOIRE_MODEL_ROOT", root.c_str(), 1);
    ::setenv("GRIMOIRE_QFN_PLE_SOURCE", source.c_str(), 1);
    std::string error;
    auto loads = [&](Qwen35Model& ld) { error.clear(); return ld.load(model.string(), error, true, true); };
    {
        Qwen35Model ld;
        const bool ok = loads(ld);
        require(ok, error);
        require(ld.index.count(indexer) == 0, "unindexed QSA tensor was exposed");
        require(ld.index.count("sidecar.extra") == 0, "PLE source leaked a non-PLE tensor");
        require(ld.index.size() == 3 && ld.shards.size() == 2, "wrong imported roles");
        const auto& row = ld.index.at(ple + "shard_0.weight");
        const auto& scale = ld.index.at(ple + "weight_scale");
        ld.unmap_all();
        uint8_t bytes[2]{};
        float s = 0;
        require(ld.read_raw(row, bytes, error) && bytes[0] == 0x38 && bytes[1] == 0x40, "FP8 PLE bytes via pread");
        require(ld.shards.at(scale.shard)->read_f32(scale.t, &s, error) && s == 2, "external BF16 PLE scale");
    }
    {
        Qwen35Model ld;
        std::ofstream(manifest) << "{\"weight_map\":{\"not.present\":\"body.safetensors\"}}";
        require(!loads(ld) && error.find("not.present") != std::string::npos, "must reject a bad shard membership: " + error);
    }
    {
        Qwen35Model ld;
        std::ofstream(manifest) << "{\"weight_map\":{\"" << embed << "\":\"body.safetensors\",\""
                                << embed << "\":\"body.safetensors\"}}";
        require(!loads(ld) && error.find("index.json") != std::string::npos, "must reject duplicate map entries");
    }
    {
        Qwen35Model ld;
        fs::rename(root/"trunk.safetensors", tmp/"outside.safetensors");
        fs::remove(model/"body.safetensors");
        fs::create_symlink("../../outside.safetensors", model/"body.safetensors");
        std::ofstream(manifest) << "{\"weight_map\":{\"" << embed << "\":\"body.safetensors\"}}";
        require(!loads(ld) && error.find("outside model root") != std::string::npos, "must reject an escaping alias: " + error);
    }
    std::puts("PASS QFN index allowlist, separate PLE role, cross-shard byte/scale reads, and refusals");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
}
