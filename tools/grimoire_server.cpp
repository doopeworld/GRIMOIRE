// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

// =====================================================================
//  grimoire-server -- OpenAI-compatible HTTP front end
//
//  Launch shape mirrors vLLM's `vllm serve <model> --quantization ...`:
//  one image, per-model config passed as CLI flags, not baked into the
//  container. The model loads ONCE at startup and stays resident; requests
//  are served against that live engine.
//
//  Requests that overlap in time are STEPPED TOGETHER by a resident
//  scheduler (see grimoire_scheduler_new): one thread owns the engine,
//  each request gets its own sequence slot, and a decode step carries one
//  token for every live request.  A decode step's cost is reading the
//  weights, not producing the token, so this is most of the bill for
//  concurrent work -- and it is exactly the shape agentic use has.
//
//  It was a mutex held for the whole of a request, which meant the second
//  caller waited for the first to FINISH.  Where the engine still cannot
//  batch -- a linear-attention model, TP, PP, a loaded drafter -- the
//  scheduler runs one request at a time by the old path, speculation
//  included, and says so on startup.  Nothing here has to ask which.
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

#include "b70/grimoire_api.hpp"
#include "b70/http_request.hpp"
#include <memory>
#include <random>
#include <cctype>
#include <csignal>
#include <pthread.h>
#include <thread>
using b70::json_escape;

// ---------------------------------------------------------------------
// GRACEFUL STOP (2026-11-10).  This server used to install no handler for
// SIGTERM / SIGINT, so `docker stop` (and the `multi` entrypoint, which
// forwards TERM to every rank) killed each rank process on the spot: GPU
// work still queued, device memory torn down by the kernel instead of by
// Grimoire::release().  The xe driver then had to reset the engines --
// MEASURED: gpu0 CCS x2 + BCS "Timedout job" at a T0 stop on 11-10 19:53,
// both B70s at a TP2 stop on 11-09 13:52.  Killing a process with GPU work
// in flight is the one thing this project must never do to a card.
//
// Now the stop signals are blocked in every thread (the mask is set before
// the SYCL / Level Zero runtime starts any) and one thread takes them with
// sigwait().  Rank 0 then stops accepting HTTP, ends every live generation
// at its next token, lets the scheduler finish the step it is in, sends
// the TP / PP workers their shutdown message, and frees the engine through
// grimoire_delete() -> release(), which drains every queue before it frees
// anything.  A worker rank just keeps following rank 0 until that shutdown
// message (or rank 0's closed pipe) and then takes the same release() path.
// A second signal is ignored on purpose: Docker's own --stop-timeout is the
// backstop, never a kill from inside while the card is busy.
static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_main_done{false};
static std::atomic<httplib::Server*> g_svr{nullptr};

static void block_stop_signals() {
    sigset_t s; sigemptyset(&s); sigaddset(&s, SIGTERM); sigaddset(&s, SIGINT);
    pthread_sigmask(SIG_BLOCK, &s, nullptr);
}

static void start_stop_signal_thread() {
    std::thread([] {
        sigset_t s; sigemptyset(&s); sigaddset(&s, SIGTERM); sigaddset(&s, SIGINT);
        int sig = 0;
        if (sigwait(&s, &sig) != 0) return;
        g_stop.store(true);
        std::fprintf(stderr, "signal %d: stopping gracefully -- no new requests, live generations end at their "
                             "next token, GPU queues drain before memory is freed\n", sig);
        // Call httplib's stop() EXACTLY ONCE, and only once listen() is running:
        // before that it does nothing, and a second call while listen() is still
        // draining a live request trips its assert(svr_sock_ != INVALID_SOCKET)
        // -> abort() with GPU work in flight (MEASURED 11-10 20:21: gpu0 CCS reset).
        while (!g_main_done.load()) {
            httplib::Server* p = g_svr.load();
            if (p && p->is_running()) { p->stop(); break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }).detach();
}

// OpenAI tool_calls array; streaming deltas also carry each call's index.
static std::string tool_calls_json(const std::vector<b70::ParsedToolCall>& calls, bool with_index) {
    static std::mt19937_64 rng{std::random_device{}()};
    std::string out="[";
    for(size_t i=0;i<calls.size();++i) {
        char id[32];std::snprintf(id,sizeof id,"call_%016llx",static_cast<unsigned long long>(rng()));
        if(i)out+=",";
        out+="{"+(with_index?"\"index\":"+std::to_string(i)+",":std::string())+"\"id\":\""+id+
             "\",\"type\":\"function\",\"function\":{\"name\":\""+json_escape(calls[i].name)+
             "\",\"arguments\":\""+json_escape(calls[i].arguments)+"\"}}";
    }
    return out+"]";
}

int main(int argc, char** argv) {
    // Line-buffered stdout: a served process writes to a pipe (docker logs), where
    // full buffering held the engine's printf lines (GRIMOIRE_SPEC_TIME, spec
    // stats) until exit -- and a stopped container never flushed them.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    // Before ANY thread exists (the SYCL / Level Zero runtime starts its own
    // in grimoire_load): every later thread inherits the blocked mask, so a
    // stop signal can only ever be taken by the stop thread below.
    block_stop_signals();
    start_stop_signal_thread();
    // Level Zero sysman reports the card's free VRAM, which sizes prefill
    // chunks (Grimoire::prefill_token_budget); it must be on before SYCL starts.
    setenv("ZES_ENABLE_SYSMAN", "1", 0);
    std::string model_dir, host = "0.0.0.0", dflash_model;
    b70::Fmt proj_fmt = b70::Fmt::INT4;
    bool proj_explicit = false;
    int max_seq = 8192, port = 8000;
    bool defer_moe_gather = false, bf16_qkv = false, bf16_dn_qkv = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if ((a == "--model" || a == "-m") && i + 1 < argc) model_dir = argv[++i];
        else if (a == "--proj" && i + 1 < argc) {
            const std::string v = argv[++i];
            proj_explicit = true;
            if      (v == "int4")  proj_fmt = b70::Fmt::INT4;
            else if (v == "int8")  proj_fmt = b70::Fmt::INT8;
            else if (v == "mxfp4") proj_fmt = b70::Fmt::MXFP4;
            else if (v == "mxfp8") proj_fmt = b70::Fmt::MXFP8;
            else if (v == "fp8" || v == "fp8_e4m3") proj_fmt = b70::Fmt::FP8_E4M3;
            else if (v == "fp8_e5m2") proj_fmt = b70::Fmt::FP8_E5M2;
            else if (v == "bf16")  proj_fmt = b70::Fmt::BF16;
            else if (v == "native") { proj_fmt = b70::Fmt::MXFP4; setenv("GRIMOIRE_GGUF_NATIVE","1",1); }
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
                "  grimoire-server --model <dir_or_gguf> [--proj FORMAT] [--ctx N]\n"
                "                  FORMAT includes native (GGUF Q4_K/Q6_K, MXFP4 base)\n"
                "                  [--port N] [--host ADDR]\n"
                "                  [--dflash-model <dir>]         speculative decode drafter\n"
                "                  [--defer-moe-gather]           (Ornith only)\n"
                "                  [--bf16-qkv] [--bf16-dn-qkv]   (Ornith only)\n\n"
                "  Do NOT pass --defer-moe-gather/--bf16-qkv/--bf16-dn-qkv for Qwen:\n"
                "  they are MoE-specific and produce degenerate output on a dense model.\n");
            return 0;
        }
    }
    const char* gguf_native = std::getenv("GRIMOIRE_GGUF_NATIVE");
    if (gguf_native && std::strcmp(gguf_native, "1") == 0) {
        if (proj_explicit && proj_fmt != b70::Fmt::MXFP4) {
            std::fprintf(stderr, "native GGUF requires --proj native or mxfp4 for other tensors\n");
            return 1;
        }
        proj_fmt = b70::Fmt::MXFP4;
    }
    if (max_seq < 2 || port < 1 || port > 65535) {
        std::fprintf(stderr, "invalid --ctx or --port\n"); return 1;
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

    std::unique_ptr<b70::Grimoire,void(*)(b70::Grimoire*)> owner(e,b70::grimoire_delete);

    // A PIPELINE STAGE THAT IS NOT THE FRONT END DOES NOT SERVE HTTP.
    //
    // This is what makes a two-card server possible at all, and the
    // reason there was not one before is smaller than it looks: under PP
    // every stage runs the same generation loop and they stay in step by
    // exchanging a message per token, but the PROMPT only ever reached
    // rank 0. The CLI never noticed -- every rank is launched with the
    // same -p and reads it off its own command line -- so the gap only
    // existed for a server.
    //
    // The front end forwards each request down the pipe; the workers run
    // it and answer nothing. Both ends of that were added together, and
    // neither is reachable from the CLI, which is untouched.
    if (b70::grimoire_is_pp_worker(*e)) {
        // TP workers follow rank 0 through the same loop now, so say which.
        const char* pp = std::getenv("GRIMOIRE_PP_RANK");
        const char* tp = std::getenv("GRIMOIRE_TP_RANK");
        std::fprintf(stderr, "this is %s rank %d, not the front end: "
                             "serving nothing, following rank 0\n",
                     pp ? "pipeline" : "tensor-parallel",
                     std::atoi(pp ? pp : (tp ? tp : "0")));
        // Returns on rank 0's shutdown message or its closed pipe; the engine is
        // then freed by `owner` -> grimoire_delete() -> release(), queues drained.
        b70::grimoire_pp_worker_loop(*e);
        g_main_done.store(true);
        return 0;
    }
    // A stop that arrived while the model was loading: no warm-up, no listen.
    auto stop_before_serving = [&] {
        std::fprintf(stderr, "stopped before serving: workers told to shut down, engine released\n");
        b70::grimoire_pp_shutdown(*e);
        g_main_done.store(true);
        return 0;
    };
    if (g_stop.load()) return stop_before_serving();
    // oneDNN (on unless GRIMOIRE_ONEDNN=0) compiles its kernels the first time a
    // weight meets a new row class (src/onednn_gemm.cpp): the first request
    // after a start paid ~0.45 s for it (MEASURED 2026-10-25, Qwen3.8-27B-FP8 on
    // two cards, code probe TTFT 1.42 vs 0.97 s off).  Pay it here, before any
    // client: a few prompts of different lengths, one token each, through the
    // same path a request takes (TP / PP workers follow it like any request).
    // GRIMOIRE_SERVER_WARMUP=0 skips it.
    {
        const char* od = std::getenv("GRIMOIRE_ONEDNN");
        const char* wu = std::getenv("GRIMOIRE_SERVER_WARMUP");
        if (!(od && *od && std::atoi(od) == 0) && !(wu && *wu == '0')) {
            const auto t0 = std::chrono::steady_clock::now();
            const std::vector<int32_t> unit =
                tk.encode("The quick brown fox jumps over the lazy dog while the band plays. ");
            int done = 0;
            for (const int len : {73, 333, 777, 2153}) {
                if (unit.empty() || len + 2 >= max_seq || g_stop.load()) break;
                std::vector<int32_t> ids;
                while (int(ids.size()) < len) ids.insert(ids.end(), unit.begin(), unit.end());
                ids.resize(size_t(len));
                const int eot = tk.special_id("<|eot|>");
                if (!b70::grimoire_pp_broadcast_request(*e, ids, 1, tk.eos(), eot)) break;
                std::vector<int32_t> out;
                b70::grimoire_serve_generate(*e, ids, 1, tk.eos(), out, eot);
                ++done;
            }
            std::fprintf(stderr, "oneDNN warm-up: %d prompts in %.1f s\n", done,
                         std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    // How many requests may be in flight together.  The engine caps it at
    // what the KV cache was allocated for (GRIMOIRE_SEQ_SLOTS) and at its
    // own row limit, so asking for more than the box can hold is not an
    // error, just a smaller number.
    if (g_stop.load()) return stop_before_serving();
    const char* bw = std::getenv("GRIMOIRE_MAX_BATCH");
    std::unique_ptr<b70::GrimoireScheduler,void(*)(b70::GrimoireScheduler*)>
        sched(b70::grimoire_scheduler_new(*e, bw && *bw ? std::atoi(bw) : 8),
              b70::grimoire_scheduler_delete);
    httplib::Server svr;
    auto error_json=[](const std::string& message,const char* type) {
        return std::string("{\"error\":{\"message\":\"")+json_escape(message)+
            "\",\"type\":\""+type+"\"}}";
    };
    svr.set_exception_handler([&](const httplib::Request&,httplib::Response& res,std::exception_ptr ep) {
        try{std::rethrow_exception(ep);}
        catch(const std::invalid_argument& ex){res.status=400;res.set_content(error_json(ex.what(),"invalid_request_error"),"application/json");}
        catch(const std::exception& ex){res.status=500;res.set_content(error_json(ex.what(),"engine_error"),"application/json");}
        catch(...){res.status=500;res.set_content(error_json("unknown engine exception","engine_error"),"application/json");}
    });
    svr.Get("/health",[](const httplib::Request&,httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}","application/json");
    });
    svr.Get("/v1/models",[&](const httplib::Request&,httplib::Response& res) {
        res.set_content("{\"object\":\"list\",\"data\":[{\"id\":\""+json_escape(model_dir)+
            "\",\"object\":\"model\"}],\"grimoire\":{\"decoding\":\"greedy,top_k,top_p,temperature\",\"concurrency\":"+
            std::to_string(b70::grimoire_scheduler_width(*sched))+"}}","application/json");
    });
    auto handle=[&](const httplib::Request& req,httplib::Response& res,bool chat) {
        const auto request=b70::parse_completion_request(req.body,chat);
        const std::string prompt=chat?tk.apply_chat_template(request.messages,request.chat):request.prompt;
        const bool think=chat&&!harmony_model&&request.chat.enable_thinking;
        const std::vector<b70::Json> tools=request.tools;
        const auto ids=tk.encode(prompt,true);
        const int budget=b70::generation_budget(ids,request.max_tokens,max_seq,int(tk.vocab_size()));
        const auto sampling=request.sampling;
        const std::string model=json_escape(model_dir);
        const auto request_id=std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        if(request.stream) {
            res.set_chunked_content_provider("text/event-stream",
                [&,ids,budget,chat,model,request_id,think,tools,sampling](size_t,httplib::DataSink& sink) {
                    auto send=[&](const std::string& payload) {
                        const auto line="data: "+payload+"\n\n";return sink.write(line.data(),line.size());
                    };
                    const std::string prefix="{\"id\":\"cmpl-"+request_id+"\",\"object\":\""+
                        (chat?"chat.completion.chunk":"text_completion")+"\",\"model\":\""+model+"\"";
                    b70::ResponseDecoder decoder(tk,chat&&harmony_model,think);
                    if(chat&&!tools.empty())decoder.enable_tools();
                    bool connected=true;
                    auto emit=[&](const std::string& piece,bool reasoning) {
                        const std::string value=chat?"\"delta\":{\""+std::string(reasoning?"reasoning_content":"content")+
                            "\":\""+json_escape(piece)+"\"}":"\"text\":\""+json_escape(piece)+"\"";
                        connected=send(prefix+",\"choices\":[{\"index\":0,"+value+",\"finish_reason\":null}]}");
                        return connected;
                    };
                    try {
                        b70::FinishReason finish;
                        std::vector<int32_t> out;
                        const int n=b70::grimoire_scheduler_generate(*sched,ids,budget,
                            tk.eos(),tk.eos2(),out,
                            [&](int32_t t){return decoder.push(t,emit)&&!g_stop.load();},&finish,sampling);
                        if(connected)connected=decoder.finish(emit);
                        std::string reason=b70::finish_reason_name(finish);
                        if(connected && !decoder.tool_call_text().empty()) {
                            const auto calls=b70::parse_xml_tool_calls(decoder.tool_call_text(),tools);
                            if(calls.empty())connected=emit(decoder.tool_call_text(),false);
                            else {
                                connected=send(prefix+",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":"+
                                    tool_calls_json(calls,true)+"},\"finish_reason\":null}]}");
                                reason="tool_calls";
                            }
                        }
                        if(connected)send(prefix+",\"choices\":[{\"index\":0,"+
                            (chat?"\"delta\":{}":"\"text\":\"\"")+",\"finish_reason\":\""+
                            reason+"\"}],\"usage\":{\"prompt_tokens\":"+
                            std::to_string(ids.size())+",\"completion_tokens\":"+std::to_string(n)+
                            ",\"total_tokens\":"+std::to_string(ids.size()+n)+"}}");
                    } catch(const std::exception& ex) {
                        std::fprintf(stderr,"[stream error] %s\n",ex.what());
                        if(connected)send(error_json(ex.what(),"engine_error"));
                    }
                    if(connected)send("[DONE]");
                    sink.done();return false;
                });
        } else {
            b70::FinishReason finish;
            std::vector<int32_t> out;
            // a stop ends this request at its next token (the scheduler then cancels the job)
            const int n=b70::grimoire_scheduler_generate(*sched,ids,budget,tk.eos(),
                tk.eos2(),out,[](int32_t){return !g_stop.load();},&finish,sampling);
            std::string content,reasoning;
            b70::ResponseDecoder decoder(tk,chat&&harmony_model,think);
            if(chat&&!tools.empty())decoder.enable_tools();
            auto emit=[&](const std::string& p,bool r){(r?reasoning:content)+=p;return true;};
            for(auto t:out)decoder.push(t,emit);
            decoder.finish(emit);
            std::string reason=b70::finish_reason_name(finish), calls_json;
            if(!decoder.tool_call_text().empty()) {
                const auto calls=b70::parse_xml_tool_calls(decoder.tool_call_text(),tools);
                if(calls.empty())content+=decoder.tool_call_text();
                else {
                    calls_json=tool_calls_json(calls,false);reason="tool_calls";
                    while(!content.empty() && std::isspace(static_cast<unsigned char>(content.back())))content.pop_back();
                }
            }
            const std::string content_json=!calls_json.empty()&&content.empty()?"null":"\""+json_escape(content)+"\"";
            const std::string answer=chat?"\"message\":{\"role\":\"assistant\",\"content\":"+content_json+
                (reasoning.empty()?"":",\"reasoning_content\":\""+json_escape(reasoning)+"\"")+
                (calls_json.empty()?"":",\"tool_calls\":"+calls_json)+"}":
                "\"text\":\""+json_escape(content)+"\"";
            res.set_content("{\"id\":\"cmpl-"+request_id+"\",\"object\":\""+(chat?"chat.completion":"text_completion")+
                "\",\"model\":\""+model+"\",\"choices\":[{\"index\":0,"+answer+",\"finish_reason\":\""+
                reason+"\"}],\"usage\":{\"prompt_tokens\":"+std::to_string(ids.size())+
                ",\"completion_tokens\":"+std::to_string(n)+",\"total_tokens\":"+std::to_string(ids.size()+n)+"}}","application/json");
        }
    };
    svr.Post("/v1/chat/completions",[&](const httplib::Request& req,httplib::Response& res){handle(req,res,true);});
    svr.Post("/v1/completions",[&](const httplib::Request& req,httplib::Response& res){handle(req,res,false);});
    g_svr.store(&svr);
    // the stop thread may have fired between the check above and this point
    const bool listened = g_stop.load() ? true : svr.listen(host,port);
    g_svr.store(nullptr);
    if(!listened && !g_stop.load()){
        std::fprintf(stderr,"could not listen on %s:%d\n",host.c_str(),port);
        sched.reset(); b70::grimoire_pp_shutdown(*e); g_main_done.store(true);
        return 1;
    }
    // listen() returned: every connection thread has finished (live generations
    // were cut at their next token).  Ordered teardown, the same as the engine's
    // own tests: the scheduler finishes its step and stops, the TP / PP workers
    // get their shutdown message, then `owner` -> grimoire_delete() -> release()
    // drains every GPU queue before freeing device memory.
    sched.reset();
    b70::grimoire_pp_shutdown(*e);
    std::fprintf(stderr,"scheduler stopped, workers told to shut down; releasing the engine\n");
    g_main_done.store(true);
    return 0;
}
