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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <functional>
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
                             std::vector<int32_t>& out_ids, int eot_id,
                             const std::function<bool(int32_t)>& on_token = {});
}  // namespace b70

// Harmony channel extraction. Muse-Glimmer emits its chain of thought in a
// "to=self" channel and the user-facing answer in "to=user"; vLLM strips this
// with --reasoning-parser. Without it open-webui shows the whole monologue.
// Returns the LAST complete to=user segment, or the input unchanged when the
// markers are absent (Qwen/Ornith, which do not use channels).
static std::string harmony_final(const std::string& text) {
    static const std::string kUser = "assistant to=user";
    const size_t last = text.rfind(kUser);
    if (last == std::string::npos) return text;
    size_t b = last + kUser.size();
    // A channel body may begin right after the marker or after a separator.
    while (b < text.size() && (text[b] == ' ' || text[b] == '\n')) ++b;
    // The segment ends at the next channel marker ("assistant to=" / "to=self").
    size_t end = text.size();
    for (const char* m : {"assistant to=", "to=self"}) {
        const size_t p = text.find(m, b);
        if (p != std::string::npos && p < end) end = p;
    }
    std::string out = text.substr(b, end - b);
    // Trim the "assistant" prefix the turn marker leaves behind.
    while (!out.empty() && (out.front() == ' ' || out.front() == '\n')) out.erase(out.begin());
    while (!out.empty() && (out.back() == ' ' || out.back() == '\n')) out.pop_back();
    return out.empty() ? text : out;
}

// ---- tiny JSON helpers (only what an OpenAI request/response needs) ------
static bool json_find_true(const std::string& j, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    size_t p = j.find(needle);
    if (p == std::string::npos) return false;
    p = j.find(':', p + needle.size());
    if (p == std::string::npos) return false;
    ++p;
    while (p < j.size() && std::isspace(static_cast<unsigned char>(j[p]))) ++p;
    return j.compare(p, 4, "true") == 0;
}

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

// Extract message fields in request order. Open WebUI sends string content;
// roles and content are paired below to preserve the entire conversation.
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
                "                  [--dflash-model <dir>]         speculative decode drafter\n"
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
    // which requires the speculative verify/rollback loop. That loop now lives
    // in grimoire_serve_generate (same accept/reject contract as the CLI), so
    // the drafter is safe to load: it is selected by GRIMOIRE_DFLASH_MODEL,
    // which the engine reads during build().
    if (!dflash_model.empty())
        setenv("GRIMOIRE_DFLASH_MODEL", dflash_model.c_str(), 1);
    if (defer_moe_gather) setenv("GRIMOIRE_DEFER_MOE_GATHER", "1", 1);
    if (bf16_qkv) setenv("GRIMOIRE_BF16_QKV", "1", 1);
    if (bf16_dn_qkv) setenv("GRIMOIRE_BF16_DN_QKV", "1", 1);

    b70::Tokenizer tk;
    std::string err;
    if (!tk.load(model_dir, err)) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
        return 1;
    }
    const bool harmony_model = tk.special_id("<|begin_of_text|>") >= 0;

    std::fprintf(stderr, "loading %s ...\n", model_dir.c_str());
    b70::Grimoire* e = b70::grimoire_new();
    if (!b70::grimoire_load(*e, model_dir, proj_fmt, max_seq, err)) {
        std::fprintf(stderr, "load: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "ready. listening on %s:%d\n", host.c_str(), port);

    std::mutex engine_mu;  // serializes requests; see file header

    httplib::Server svr;
    // Without this, any C++ exception inside a handler surfaces as a bare
    // HTTP 500 with an empty body and nothing in the log -- indistinguishable
    // from a hang. Report what actually threw.
    svr.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                 std::exception_ptr ep) {
        std::string what = "unknown exception";
        try { std::rethrow_exception(ep); }
        catch (const std::exception& ex) { what = ex.what(); }
        catch (...) {}
        std::fprintf(stderr, "[error] %s\n", what.c_str());
        std::fflush(stderr);
        res.status = 500;
        res.set_content(std::string("{\"error\":\"") + what + "\"}",
                        "application/json");
    });

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        std::ostringstream j;
        j << "{\"object\":\"list\",\"data\":[{\"id\":\""
          << json_escape(model_dir) << "\",\"object\":\"model\"}]}";
        res.set_content(j.str(), "application/json");
    });

    // OpenAI SSE stream. Raw completions stay token-by-token for llama-benchy.
    // Harmony chat buffers the private to=self channel and starts emitting only
    // after to=user, so Open WebUI never receives chain-of-thought or protocol
    // markers.
    auto handle_stream = [&](const std::string& prompt_text, int max_tokens,
                             httplib::Response& res, bool chat_shape) {
        const std::string mdl = model_dir;
        res.set_chunked_content_provider(
            "text/event-stream",
            [&, prompt_text, max_tokens, mdl, chat_shape](size_t, httplib::DataSink& sink) {
                const auto ttft_t0 = std::chrono::steady_clock::now();
                std::lock_guard<std::mutex> lock(engine_mu);
                const long long now = static_cast<long long>(std::time(nullptr));
                const char* obj = chat_shape ? "chat.completion.chunk"
                                             : "text_completion";
                auto send = [&](const std::string& payload) {
                    const std::string line = "data: " + payload + "\n\n";
                    return sink.write(line.data(), line.size());
                };
                std::ostringstream head;
                head << "{\"id\":\"chatcmpl-grimoire\",\"object\":\"" << obj
                     << "\",\"created\":" << now << ",\"model\":\""
                     << json_escape(mdl) << "\",\"choices\":[{\"index\":0,"
                     << (chat_shape ? "\"delta\":{\"role\":\"assistant\"}"
                                    : "\"text\":\"\"")
                     << ",\"finish_reason\":null}]}";
                // Deliberately NOT sent yet. A streaming client measures
                // time-to-first-token from the first event it receives; if the
                // role frame goes out before prefill runs, TTFT reads ~0 and
                // the prefill columns come back empty (llama-benchy shows no
                // pp row). Emit it together with the first real token instead.
                bool head_sent = false;

                const auto enc_t0 = std::chrono::steady_clock::now();
                std::vector<int32_t> ids = tk.encode(prompt_text);
                const int room = std::max(1, max_seq - int(ids.size()));
                const int n_predict = max_tokens > 0 ? max_tokens
                    : std::min(4096, room);
                std::fprintf(stderr, "    [encode] %.0f ms (%zu tok)\n",
                    std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-enc_t0).count(), ids.size());
                std::fflush(stderr);
                std::vector<int32_t> out_ids;
                std::string harmony_pending;
                bool harmony_user = false;
                const auto t0 = std::chrono::steady_clock::now();
                const int produced = b70::grimoire_serve_generate(
                    *e, ids, n_predict, tk.eos(), out_ids,
                    tk.special_id("<|eot|>"),
                    [&](int32_t tokid) -> bool {
                        std::string piece = tk.decode_one(tokid);
                        if (chat_shape && harmony_model) {
                            harmony_pending += piece;
                            if (!harmony_user) {
                                static const std::string marker_text =
                                    "assistant to=user";
                                const size_t marker = harmony_pending.find(marker_text);
                                if (marker == std::string::npos) return true;
                                harmony_user = true;
                                harmony_pending.erase(0, marker + marker_text.size());
                                while (!harmony_pending.empty() &&
                                       (harmony_pending.front() == ' ' ||
                                        harmony_pending.front() == '\n'))
                                    harmony_pending.erase(harmony_pending.begin());
                            }
                            piece.swap(harmony_pending);
                            if (piece.empty()) return true;
                        }
                        if (chat_shape && !head_sent) {
                            head_sent = true;
                            std::fprintf(stderr,
                                "    [ttft] first token at %.0f ms\n",
                                std::chrono::duration<double,std::milli>(
                                    std::chrono::steady_clock::now()-ttft_t0).count());
                            std::fflush(stderr);
                            if (!send(head.str())) return false;
                        }
                        std::ostringstream c;
                        c << "{\"id\":\"chatcmpl-grimoire\",\"object\":\"" << obj
                          << "\",\"created\":" << now << ",\"model\":\""
                          << json_escape(mdl) << "\",\"choices\":[{\"index\":0,"
                          << (chat_shape
                                ? "\"delta\":{\"content\":\"" + json_escape(piece) + "\"}"
                                : "\"text\":\"" + json_escape(piece) + "\"")
                          << ",\"finish_reason\":null}]}";
                        return send(c.str());
                    });

                if (chat_shape && !head_sent) send(head.str());
                std::ostringstream tail;
                tail << "{\"id\":\"chatcmpl-grimoire\",\"object\":\"" << obj
                     << "\",\"created\":" << now << ",\"model\":\""
                     << json_escape(mdl) << "\",\"choices\":[{\"index\":0,"
                     << (chat_shape ? "\"delta\":{}" : "\"text\":\"\"")
                     << ",\"finish_reason\":\"stop\"}],"
                     << "\"usage\":{\"prompt_tokens\":" << ids.size()
                     << ",\"completion_tokens\":" << produced
                     << ",\"total_tokens\":" << (ids.size() + size_t(produced))
                     << "}}";
                send(tail.str());
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                const double dt = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[req/stream] prompt=%zu completion=%d %.2fs\n",
                             ids.size(), produced, dt);
                sink.done();
                return false;
            });
    };

    auto handle_generate = [&](const std::string& prompt_text, int max_tokens,
                                httplib::Response& res, bool chat_shape) {
        std::lock_guard<std::mutex> lock(engine_mu);
        const auto t0 = std::chrono::steady_clock::now();
        const auto tk0 = std::chrono::steady_clock::now();
        std::vector<int32_t> ids = tk.encode(prompt_text);
        const double enc_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tk0).count();
        std::vector<int32_t> out_ids;
        const int room = std::max(1, max_seq - int(ids.size()));
        const int n_predict = max_tokens > 0 ? max_tokens
            : std::min(4096, room);
        const int completion_tokens = b70::grimoire_serve_generate(
            *e, ids, n_predict, tk.eos(), out_ids, tk.special_id("<|eot|>"));
        const double dt = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        std::string text = tk.decode(out_ids);
        if (chat_shape) text = harmony_final(text);
        std::fprintf(stderr, "[req] prompt=%zu completion=%d %.2fs (encode %.0f ms)\n",
                     ids.size(), completion_tokens, dt, enc_ms);

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
        const std::vector<std::string> roles = json_find_all_role(req.body);
        const std::vector<std::string> contents = json_find_all_content(req.body);
        std::vector<b70::ChatMessage> messages;
        for (size_t i = 0; i < roles.size() && i < contents.size(); ++i)
            if (roles[i] == "system" || roles[i] == "user" ||
                roles[i] == "assistant")
                messages.push_back({roles[i], contents[i]});
        if (messages.empty() && !contents.empty())
            messages.push_back({"user", contents.back()});
        double max_tok = 0;
        if (!json_find_number(req.body, "max_completion_tokens", max_tok))
            json_find_number(req.body, "max_tokens", max_tok);
        const std::string prompt = tk.apply_chat_template(messages);
        if (json_find_true(req.body, "stream"))
            handle_stream(prompt, int(max_tok), res, /*chat_shape=*/true);
        else
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
