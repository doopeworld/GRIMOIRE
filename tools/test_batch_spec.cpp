#include "b70/grimoire_api.hpp"
#include "b70/formats.hpp"
#include "mini_model.hpp"
#include <barrier>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
using namespace b70;
namespace b70 {
extern long g_spec_batch_steps,g_spec_batch_proposals,g_spec_batch_accepted,g_spec_batch_sequences;
}
using Replies=std::vector<std::vector<int32_t>>;
static Grimoire* load(const std::filesystem::path& path) {
    auto* e=grimoire_new(); std::string error;
    if(!grimoire_load(*e,path.string(),Fmt::BF16,128,error)) {
        grimoire_delete(e); throw std::runtime_error(error);
    }
    return e;
}
int main() {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    ::setenv("GRIMOIRE_SEQ_SLOTS","4",1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX","1",1);
    ::setenv("GRIMOIRE_DFLASH_M","4",1);
    ::unsetenv("GRIMOIRE_PREFIX_CACHE");
    char tmp[]="/tmp/grimoire-batch-spec-XXXXXX";
    if(!::mkdtemp(tmp)) return 2;
    Replies prompts(3);
    for(int k=0;k<3;++k) for(int i=0;i<17+k*7;++i)
        prompts[k].push_back((i*17+k*13+7)%120);
    try {
        for(bool hybrid:{false,true}) for(int draft_kind:{0,1,2}) {
            const bool dflash=draft_kind!=0, forced=draft_kind==2;
            const auto dir=std::filesystem::path(tmp)/((hybrid?"hybrid":"dense")+std::string(forced?"-forced":dflash?"-dflash":"-mtp"));
            mini::write_model(dir,hybrid?mini::hybrid(4,!dflash):mini::dense(4,!dflash));
            ::unsetenv("GRIMOIRE_MTP"); ::unsetenv("GRIMOIRE_DFLASH_MODEL");
            auto* baseline=load(dir);
            Replies reference(3);
            for(int k=0;k<3;++k)
                grimoire_serve_generate(*baseline,prompts[k],12,-1,reference[k],-1);
            Replies near_prompts(3), near_reference(3);
            for(int k=0;k<3;++k) {
                for(int i=0;i<124+k;++i) near_prompts[k].push_back((i*17+k*13+7)%120);
                grimoire_serve_generate(*baseline,near_prompts[k],6,-1,near_reference[k],-1);
            }
            grimoire_delete(baseline);
            if(dflash) {
                const auto draft=dir/"draft";
                int best=-1, best_n=0;
                if(forced) for(const auto& reply:reference) for(int token:reply) {
                    int count=0;
                    for(const auto& other:reference) for(int t:other) count+=t==token;
                    if(count>best_n) { best=token; best_n=count; }
                }
                mini::write_model(draft,mini::dflash_draft(2,false,false,best),20260913);
                ::setenv("GRIMOIRE_DFLASH_MODEL",draft.c_str(),1);
            } else ::setenv("GRIMOIRE_MTP","1",1);
            auto* e=load(dir);
            const long step0=g_spec_batch_steps, seq0=g_spec_batch_sequences;
            const long proposed0=g_spec_batch_proposals, accepted0=g_spec_batch_accepted;
            Replies actual;
            grimoire_serve_generate_batch(*e,prompts,12,-1,actual,-1);
            if(actual!=reference) throw std::runtime_error("direct speculative batch differs from ordinary decoding");
            auto* scheduler=grimoire_scheduler_new(*e,3);
            if(grimoire_scheduler_width(*scheduler)!=3)
                throw std::runtime_error("speculative scheduler disabled batching");
            for(int round=0;round<2;++round) {
                Replies replies(3); std::vector<std::exception_ptr> errors(3);
                std::barrier start(3); std::vector<std::thread> clients;
                for(int k=0;k<3;++k) clients.emplace_back([&,k] {
                    start.arrive_and_wait();
                    try {
                        int emitted=0;
                        grimoire_scheduler_generate(*scheduler,prompts[k],12,-1,-1,replies[k],
                            [&](int32_t) { return !(round==0 && k==1) || ++emitted<3; });
                    } catch(...) { errors[k]=std::current_exception(); }
                });
                for(auto& client:clients) client.join();
                for(int k=0;k<3;++k) {
                    if(errors[k]) std::rethrow_exception(errors[k]);
                    auto expected=reference[k];
                    if(round==0 && k==1) expected.resize(3);
                    if(expected!=replies[k]) throw std::runtime_error("scheduled speculative tokens differ");
                }
            }
            grimoire_scheduler_delete(scheduler);
            grimoire_serve_generate_batch(*e,near_prompts,6,-1,actual,-1);
            if(actual!=near_reference)
                throw std::runtime_error("near-context speculative batch differs");
            grimoire_delete(e);
            if(forced && g_spec_batch_accepted==accepted0)
                throw std::runtime_error("forced draft never exercised acceptance");
            std::printf("%s: steps=%ld sequences=%ld proposals=%ld accepted=%ld\n",
                dir.filename().c_str(),g_spec_batch_steps-step0,g_spec_batch_sequences-seq0,
                g_spec_batch_proposals-proposed0,g_spec_batch_accepted-accepted0);
            if(g_spec_batch_sequences-seq0<=g_spec_batch_steps-step0 ||
               g_spec_batch_proposals<=proposed0)
                throw std::runtime_error("batch never verified drafts from overlapping sequences");
        }
        std::puts("ALL PASS");
        return 0;
    } catch(const std::exception& ex) {
        std::fprintf(stderr,"FAIL: %s\n",ex.what());
        return 1;
    }
}
