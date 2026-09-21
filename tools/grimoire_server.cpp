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
using b70::json_escape;

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
        std::fprintf(stderr, "this is pipeline stage %d, not the front end: "
                             "serving nothing, following rank 0\n",
                     std::atoi(std::getenv("GRIMOIRE_PP_RANK") ?
                               std::getenv("GRIMOIRE_PP_RANK") : "0"));
        b70::grimoire_pp_worker_loop(*e);
        return 0;
    }
    // How many requests may be in flight together.  The engine caps it at
    // what the KV cache was allocated for (GRIMOIRE_SEQ_SLOTS) and at its
    // own row limit, so asking for more than the box can hold is not an
    // error, just a smaller number.
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
            "\",\"object\":\"model\"}],\"grimoire\":{\"decoding\":\"greedy\",\"concurrency\":"+
            std::to_string(b70::grimoire_scheduler_width(*sched))+"}}","application/json");
    });
    auto handle=[&](const httplib::Request& req,httplib::Response& res,bool chat) {
        const auto request=b70::parse_completion_request(req.body,chat);
        const std::string prompt=chat?tk.apply_chat_template(request.messages):request.prompt;
        const auto ids=tk.encode(prompt);
        const int budget=b70::generation_budget(ids,request.max_tokens,max_seq,int(tk.vocab_size()));
        const std::string model=json_escape(model_dir);
        const auto request_id=std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        if(request.stream) {
            res.set_chunked_content_provider("text/event-stream",
                [&,ids,budget,chat,model,request_id](size_t,httplib::DataSink& sink) {
                    auto send=[&](const std::string& payload) {
                        const auto line="data: "+payload+"\n\n";return sink.write(line.data(),line.size());
                    };
                    const std::string prefix="{\"id\":\"cmpl-"+request_id+"\",\"object\":\""+
                        (chat?"chat.completion.chunk":"text_completion")+"\",\"model\":\""+model+"\"";
                    b70::ResponseDecoder decoder(tk,chat&&harmony_model);
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
                            tk.eos(),tk.special_id("<|eot|>"),out,
                            [&](int32_t t){return decoder.push(t,emit);},&finish);
                        if(connected)connected=decoder.finish(emit);
                        if(connected)send(prefix+",\"choices\":[{\"index\":0,"+
                            (chat?"\"delta\":{}":"\"text\":\"\"")+",\"finish_reason\":\""+
                            b70::finish_reason_name(finish)+"\"}],\"usage\":{\"prompt_tokens\":"+
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
            const int n=b70::grimoire_scheduler_generate(*sched,ids,budget,tk.eos(),
                tk.special_id("<|eot|>"),out,{},&finish);
            std::string content,reasoning;
            b70::ResponseDecoder decoder(tk,chat&&harmony_model);
            auto emit=[&](const std::string& p,bool r){(r?reasoning:content)+=p;return true;};
            for(auto t:out)decoder.push(t,emit);
            decoder.finish(emit);
            const std::string answer=chat?"\"message\":{\"role\":\"assistant\",\"content\":\""+
                json_escape(content)+"\""+(reasoning.empty()?"":",\"reasoning_content\":\""+json_escape(reasoning)+"\"")+"}":
                "\"text\":\""+json_escape(content)+"\"";
            res.set_content("{\"id\":\"cmpl-"+request_id+"\",\"object\":\""+(chat?"chat.completion":"text_completion")+
                "\",\"model\":\""+model+"\",\"choices\":[{\"index\":0,"+answer+",\"finish_reason\":\""+
                b70::finish_reason_name(finish)+"\"}],\"usage\":{\"prompt_tokens\":"+std::to_string(ids.size())+
                ",\"completion_tokens\":"+std::to_string(n)+",\"total_tokens\":"+std::to_string(ids.size()+n)+"}}","application/json");
        }
    };
    svr.Post("/v1/chat/completions",[&](const httplib::Request& req,httplib::Response& res){handle(req,res,true);});
    svr.Post("/v1/completions",[&](const httplib::Request& req,httplib::Response& res){handle(req,res,false);});
    if(!svr.listen(host,port)){std::fprintf(stderr,"could not listen on %s:%d\n",host.c_str(),port);return 1;}
    return 0;
}
