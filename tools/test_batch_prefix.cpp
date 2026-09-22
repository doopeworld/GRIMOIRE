// Composes actual scheduler batching with prefix resume, shared prompts,
// cancellation and slot reuse. A serial control supplies the expected tokens.
#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"
#include <barrier>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#include <exception>

using namespace b70;
namespace b70 {
extern long g_prefix_tokens_reused_calls;
extern long g_batch_decode_steps;
extern long g_batch_decode_rows;
}
using Replies = std::vector<std::vector<std::vector<int32_t>>>;

static Replies run(const std::string& dir, bool cache, bool concurrent) {
    ::setenv("GRIMOIRE_PREFIX_CACHE", cache ? "1" : "0", 1);
    // prefix_cache_enabled tests presence, so off must really be absent.
    if (!cache) ::unsetenv("GRIMOIRE_PREFIX_CACHE");
    std::string error;
    Grimoire* e=grimoire_new();
    if (!grimoire_load(*e,dir,Fmt::BF16,256,error))
        throw std::runtime_error(error);
    GrimoireScheduler* sc=concurrent ? grimoire_scheduler_new(*e,3) : nullptr;
    Replies result(3,std::vector<std::vector<int32_t>>(3));
    std::vector<std::vector<int32_t>> prompts(3);
    for (int k=0;k<3;++k)
        for (int i=0;i<23;++i) prompts[k].push_back((i*17+3)%120);
    const long reuse0=g_prefix_tokens_reused_calls;
    const long steps0=g_batch_decode_steps, rows0=g_batch_decode_rows;
    std::barrier start(3);
    std::vector<std::exception_ptr> errors(3);
    auto client=[&](int k) {
        for(int turn=0;turn<3;++turn) {
            if(concurrent) start.arrive_and_wait();
            if(errors[k]) continue; // still meet the next turn's barrier
            try {
                auto& reply=result[k][turn];
                const bool cancel=(turn==1 && k==2);
                int emitted=0;
                auto emit=[&](int32_t) { return !cancel || ++emitted<3; };
                if(sc) grimoire_scheduler_generate(*sc,prompts[k],12,-1,-1,reply,emit);
                else grimoire_serve_generate(*e,prompts[k],12,-1,reply,-1,emit);
                prompts[k].insert(prompts[k].end(),reply.begin(),reply.end());
                prompts[k].push_back(40+k);
                prompts[k].push_back(70+turn);
            } catch(...) { errors[k]=std::current_exception(); }
        }
    };
    if(concurrent) {
        std::vector<std::thread> threads;
        for(int k=0;k<3;++k) threads.emplace_back(client,k);
        for(auto& thread:threads) thread.join();
    } else for(int k=0;k<3;++k) client(k);
    if(sc) grimoire_scheduler_delete(sc);
    grimoire_delete(e);
    for(const auto& err:errors) if(err) std::rethrow_exception(err);
    const long reused=g_prefix_tokens_reused_calls-reuse0;
    const long rows=g_batch_decode_rows-rows0, steps=g_batch_decode_steps-steps0;
    std::printf("cache=%d concurrent=%d resumes=%ld rows=%ld steps=%ld\n",
                cache,concurrent,reused,rows,steps);
    if(concurrent && !(rows>steps)) throw std::runtime_error("no multi-row decode");
    if(concurrent && cache && reused<=0) throw std::runtime_error("no prefix resume");
    return result;
}
int main() {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    ::setenv("GRIMOIRE_SEQ_SLOTS","4",1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX","1",1);
    char path[]="/tmp/grimoire-batch-prefix-XXXXXX";
    if(!::mkdtemp(path)) return 2;
    try {
        for(const auto& arch : {mini::dense(),mini::hybrid()}) {
            const auto dir=std::filesystem::path(path)/arch.name;
            mini::write_model(dir,arch);
            const auto reference=run(dir.string(),false,false);
            const auto cold=run(dir.string(),false,true);
            const auto warm=run(dir.string(),true,true);
            if(reference!=cold || reference!=warm)
                throw std::runtime_error("serial/batched/cached token mismatch");
        }
        std::puts("ALL PASS");
        return 0;
    } catch(const std::exception& ex) {
        std::fprintf(stderr,"FAIL: %s\n",ex.what());
        return 1;
    }
}
