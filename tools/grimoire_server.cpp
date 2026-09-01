// =====================================================================
//  grimoire-server -- OpenAI-compatible HTTP front end
//
//  Launch shape mirrors vLLM's `vllm serve <model> --quantization ...`:
//  one image, per-model config passed as CLI flags, not baked into the
//  container. The model loads ONCE at startup and stays resident; requests
//  are served against that live engine.
//
//  GRIMOIRE's KV cache and generation state are NOT safe for concurrent
//  decode. Requests are served one at a time behind a mutex -- a second
//  request queues rather than corrupting the first's state. That is a real
//  difference from vLLM's continuous batching and should be documented as
//  such, not hidden.
// =====================================================================
#include "b70/formats.hpp"
#include "b70/tokenizer.hpp"
#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

namespace b70 {
struct Grimoire;
Grimoire* grimoire_new();
bool grimoire_load(Grimoire& e, const std::string& dir, Fmt proj_fmt,
                    int max_seq, std::string& err);
// Plain greedy decode (no MTP/DFlash speculation -- see the definition in
// grimoire.cpp for why). Tokenize/detokenize and the chat template stay in
// this file; this call only prefills+decodes already-tokenized ids.
int grimoire_serve_generate(Grimoire& e, const std::vector<int32_t>& prompt_ids,
                             int n_predict, int eos_id,
                             std::vector<int32_t>& out_ids);
}  // namespace b70

// ---- tiny JSON helpers (only what an OpenAI request/response needs) ------
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else out += char(c);
        }
    }
    return out;
}

// Extract a top-level string field "key":"value" (handles \" and \\ escapes,
// not full JSON -- sufficient for the fixed OpenAI request shape this serves).
static bool json_find_string(const std::string& j, const std::string& key,
                              std::string& out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = j.find(needle);
    if (p == std::string::npos) return false;
    p = j.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < j.size() && std::isspace(static_cast<unsigned char>(j[p]))) ++p;
    if (p >= j.size() || j[p] != '"') return false;
    ++p;
    std::string val;
    while (p < j.size() && j[p] != '"') {
        if (j[p] == '\\' && p + 1 < j.size()) {
            char c = j[p + 1];
            if (c == 'n') val += '\n';
            else if (c == 't') val += '\t';
            else if (c == 'r') val += '\r';
            else val += c;
            p += 2;
        } else {
            val += j[p];
            ++p;
        }
    }
    out = val;
    return true;
}

static bool json_find_number(const std::string& j, const std::string& key,
                              double& out) {
    const std::string needle = "\"" + key + "\"";
    size_t p = j.find(needle);
    if (p == std::string::npos) return false;
    p = j.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < j.size() && std::isspace(static_cast<unsigned char>(j[p]))) ++p;
    size_t start = p;
    while (p < j.size() &&
           (std::isdigit(static_cast<unsigned char>(j[p])) || j[p] == '.' ||
            j[p] == '-' || j[p] == '+' || j[p] == 'e' || j[p] == 'E'))
        ++p;
    if (p == start) return false;
    out = std::atof(j.substr(start, p - start).c_str());
    return true;
}

// Extract every "content":"..." string in a "messages":[...] array, in
// order, concatenated as system+user turns already flattened by the caller
// (this server only supports a single system + single user turn per
// request, matching the one apply_chat_template signature the engine has).
static std::vector<std::string> json_find_all_content(const std::string& j) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        p = j.find("\"content\"", p);
        if (p == std::string::npos) break;
        std::string val;
        size_t sub = p;
        std::string tail = j.substr(p);
        std::string one;
        if (json_find_string(tail, "content", one)) out.push_back(one);
        p += 9;
    }
    return out;
}

static std::vector<std::string> json_find_all_role(const std::string& j) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        p = j.find("\"role\"", p);
        if (p == std::string::npos) break;
        std::string tail = j.substr(p);
        std::string one;
        if (json_find_string(tail, "role", one)) out.push_back(one);
        p += 6;
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model_dir, host = "0.0.0.0", dflash_model;
    b70::Fmt proj_fmt = b70::Fmt::INT4;
    int max_seq = 8192, port = 8000;
    bool defer_moe_gather = false, bf16_qkv = false, bf16_dn_qkv = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--proj" && i + 1 < argc) {
            const std::string v = argv[++i];
            if      (v == "int4")  proj_fmt = b70::Fmt::INT4;
            else if (v == "int8")  proj_fmt = b70::Fmt::INT8;
            else if (v == "mxfp4") proj_fmt = b70::Fmt::MXFP4;
            else if (v == "mxfp8") proj_fmt = b70::Fmt::MXFP8;
            else if (v == "fp8" || v == "fp8_e4m3") proj_fmt = b70::Fmt::FP8_E4M3;
            else if (v == "fp8_e5m2") proj_fmt = b70::Fmt::FP8_E5M2;
            else if (v == "bf16")  proj_fmt = b70::Fmt::BF16;
            else { std::fprintf(stderr, "unknown --proj: %s\n", v.c_str()); return 1; }
        }
        else if (a == "--ctx" && i + 1 < argc) max_seq = std::atoi(argv[++i]);
        else if (a == "--port" && i + 1 < argc) port = std::atoi(argv[++i]);
        else if (a == "--host" && i + 1 < argc) host = argv[++i];
        else if (a == "--dflash-model" && i + 1 < argc) dflash_model = argv[++i];
        else if (a == "--defer-moe-gather") defer_moe_gather = true;
        else if (a == "--bf16-qkv") bf16_qkv = true;
        else if (a == "--bf16-dn-qkv") bf16_dn_qkv = true;
        else if (a == "-h" || a == "--help") {
            std::printf(
                "grimoire-server -- OpenAI-compatible front end\n\n"
                "  grimoire-server --model <dir> [--proj FORMAT] [--ctx N]\n"
                "                  [--port N] [--host ADDR]\n"
                "                  [--dflash-model <dir>]         (NOT SUPPORTED YET -- rejected)\n"
                "                  [--defer-moe-gather]           (Ornith only)\n"
                "                  [--bf16-qkv] [--bf16-dn-qkv]   (Ornith only)\n\n"
                "  Do NOT pass --defer-moe-gather/--bf16-qkv/--bf16-dn-qkv for Qwen:\n"
                "  they are MoE-specific and produce degenerate output on a dense model.\n");
            return 0;
        }
    }
    if (model_dir.empty()) {
        std::fprintf(stderr, "need --model <dir>; see --help\n");
        return 1;
    }

    // The engine reads these via getenv() at build/prefill time (see
    // src/grimoire.cpp); setting them here promotes what used to be
    // operator-set shell env into first-class launch flags, so the whole
    // configuration lives on one command line like `vllm serve`.
    // Loading the drafter switches prefill onto the batched prefill_muse path,
    // which expects the speculative verify/rollback loop. This server does
    // plain greedy decode (see grimoire_serve_generate), and mixing the two
    // silently corrupts output from the second token onward -- verified:
    // with --dflash-model the haiku prompt returns
    //   " to impressive - Location: Location: Location"
    // instead of the correct " to=selfWrite a haiku about".
    // Refuse rather than serve garbage. Remove this once speculative decode
    // is wired into the server.
    if (!dflash_model.empty()) {
        std::fprintf(stderr,
            "--dflash-model is not supported yet: this server does plain greedy\n"
            "decode and the drafter's batched prefill path would corrupt output\n"
            "after the first token. Re-run without it.\n");
        return 1;
    }
    if (defer_moe_gather) setenv("GRIMOIRE_DEFER_MOE_GATHER", "1", 1);
    if (bf16_qkv) setenv("GRIMOIRE_BF16_QKV", "1", 1);
    if (bf16_dn_qkv) setenv("GRIMOIRE_BF16_DN_QKV", "1", 1);

    b70::Tokenizer tk;
    std::string err;
    if (!tk.load(model_dir, err)) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }

    std::fprintf(stderr, "loading %s ...\n", model_dir.c_str());
    b70::Grimoire* e = b70::grimoire_new();
    if (!b70::grimoire_load(*e, model_dir, proj_fmt, max_seq, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "ready. listening on %s:%d\n", host.c_str(), port);

    std::mutex engine_mu;  // serializes requests; see file header

    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream j;
        j << "{\"object\":\"list\",\"data\":[{\"id\":\""
          << json_escape(model_dir) << "\",\"object\":\"model\"}]}";
        res.set_content(j.str(), "application/json");
    });

    auto handle_generate = [&](const std::string& prompt_text, int max_tokens,
                                httplib::Response& res, bool chat_shape) {
        std::lock_guard<std::mutex> lock(engine_mu);
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<int32_t> ids = tk.encode(prompt_text);
        std::vector<int32_t> out_ids;
        const int n_predict = max_tokens > 0 ? max_tokens : 128;
        const int completion_tokens = b70::grimoire_serve_generate(
            *e, ids, n_predict, tk.eos(), out_ids);
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::string text = tk.decode(out_ids);
        std::fprintf(stderr, "[req] prompt=%zu completion=%d %.2fs\n",
                     ids.size(), completion_tokens, dt);

        std::ostringstream j;
        const long long now = static_cast<long long>(std::time(nullptr));
        if (chat_shape) {
            j << "{\"id\":\"chatcmpl-grimoire\",\"object\":\"chat.completion\","
              << "\"created\":" << now << ",\"model\":\"" << json_escape(model_dir)
              << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
              << "\"content\":\"" << json_escape(text) << "\"},"
              << "\"finish_reason\":\"stop\"}],"
              << "\"usage\":{\"prompt_tokens\":" << ids.size()
              << ",\"completion_tokens\":" << completion_tokens
              << ",\"total_tokens\":" << (ids.size() + size_t(completion_tokens))
              << "}}";
        } else {
            j << "{\"id\":\"cmpl-grimoire\",\"object\":\"text_completion\","
              << "\"created\":" << now << ",\"model\":\"" << json_escape(model_dir)
              << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
              << "\",\"finish_reason\":\"stop\"}],"
              << "\"usage\":{\"prompt_tokens\":" << ids.size()
              << ",\"completion_tokens\":" << completion_tokens
              << ",\"total_tokens\":" << (ids.size() + size_t(completion_tokens))
              << "}}";
        }
        res.set_content(j.str(), "application/json");
    };

    // open-webui target.
    svr.Post("/v1/chat/completions",
             [&](const httplib::Request& req, httplib::Response& res) {
        std::vector<std::string> roles = json_find_all_role(req.body);
        std::vector<std::string> contents = json_find_all_content(req.body);
        std::string system, user;
        for (size_t i = 0; i < roles.size() && i < contents.size(); ++i) {
            if (roles[i] == "system") system = contents[i];
            else if (roles[i] == "user") user = contents[i];
        }
        if (user.empty() && !contents.empty()) user = contents.back();
        double max_tok = 0;
        json_find_number(req.body, "max_tokens", max_tok);
        const std::string prompt = tk.apply_chat_template(user, system);
        handle_generate(prompt, int(max_tok), res, /*chat_shape=*/true);
    });

    // llama-benchy target: raw prompt, no chat template.
    svr.Post("/v1/completions",
             [&](const httplib::Request& req, httplib::Response& res) {
        std::string prompt;
        json_find_string(req.body, "prompt", prompt);
        double max_tok = 0;
        json_find_number(req.body, "max_tokens", max_tok);
        handle_generate(prompt, int(max_tok), res, /*chat_shape=*/false);
    });

    svr.listen(host, port);
    return 0;
}
