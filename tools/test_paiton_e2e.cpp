// Inspect the real engine state, not just output token agreement. Link
// with the native objects EXCEPT grimoire.o (the engine is included here).
#include "../src/grimoire.cpp"
#include "mini_model.hpp"
#include <sys/wait.h>
#include <filesystem>
#include <fstream>

using namespace b70;
namespace fs = std::filesystem;
using TestEngine = std::unique_ptr<Grimoire, decltype(&grimoire_delete)>;
static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
static TestEngine load(const fs::path& path, Fmt fmt, bool replay) {
    ::setenv("GRIMOIRE_SPEC_REPLAY", replay ? "1" : "0", 1);
    TestEngine e(grimoire_new(), grimoire_delete);
    std::string err;
    if (!e || !grimoire_load(*e,path.string(),fmt,256,err))
        throw std::runtime_error("load: " + err);
    return e;
}
static std::vector<float> state_of(Grimoire& e) {
    std::vector<float> out;
    auto append = [&](const float* p, size_t n) {
        const size_t at=out.size(); out.resize(at+n);
        e.q.memcpy(out.data()+at,p,n*sizeof(float)).wait_and_throw();
    };
    for (const auto& l : e.L) {
        if (l.dn_state) append(l.dn_state,size_t(e.cfg.lin_v_heads)*e.cfg.lin_v_dim*e.cfg.lin_k_dim);
        if (l.conv_ring) append(l.conv_ring,size_t(2*e.cfg.lin_k_heads*e.cfg.lin_k_dim+
            e.cfg.lin_v_heads*e.cfg.lin_v_dim)*(e.cfg.conv_kernel-1));
    }
    append(e.s.h,e.cfg.hidden);
    require(std::all_of(out.begin(),out.end(),[](float x){return std::isfinite(x);}),"nonfinite engine state");
    return out;
}
static void same_state(Grimoire& a, Grimoire& b) {
    const auto x=state_of(a), y=state_of(b);
    require(a.pos==b.pos && x.size()==y.size() &&
            !std::memcmp(x.data(),y.data(),x.size()*sizeof(float)),"engine rollback state differs");
}
static void rejects(Grimoire& e,int saved,int accepted) {
    const auto before=state_of(e); const int pos=e.pos;
    bool refused=false;
    try { e.commit_spec_prefix(saved,accepted); }
    catch (const std::runtime_error&) { refused=true; }
    require(refused && before==state_of(e) && pos==e.pos,"invalid commit changed state or was accepted");
}
static mini::Arch draft_fixture() {
    auto a=mini::dflash_draft(2);
    a.tensors.push_back({"candidate_selector.hidden_projection.weight",{8,64}});
    a.tensors.push_back({"candidate_selector.predecessor_codebook",{128,8}});
    a.tensors.push_back({"candidate_selector.successor_codebook",{128,8}});
    for (int l=0;l<2;++l) for (const char* conv : {"attention_conv", "mlp_conv"}) {
        const auto p="layers."+std::to_string(l)+"."+conv;
        a.tensors.push_back({p+".base_kernel",{2,3,64}});
        a.tensors.push_back({p+".kernel_projection.weight",{24,64}});
    }
    return a;
}
static int pp_child(const fs::path& root, bool replay, int rank) {
    auto e=load(root/"hybrid",Fmt::BF16,replay);
    require(e->spec_dn_state && e->spec_hidden_steps,"pipeline stage has no rollback checkpoint");
    require(replay ? e->spec_dn_updates && !e->spec_dn_steps
                   : e->spec_dn_steps && !e->spec_dn_updates,"pipeline stage has wrong rollback storage");
    require((rank==1)==e->dflash2.ok && e->pp_dflash==8,"pipeline drafter/width handshake differs");
    // Both ranks own a recurrent layer. Rank zero has NO MTP and NO
    // local DFlash head: this is the allocation-predicate regression.
    for (int i=0;i<41;++i) { e->forward((i*17+3)%128); e->argmax_token(); }
    for (int accepted : {1,3,8}) {
        const int saved=e->pos;
        e->snapshot_recurrent();
        std::vector<int32_t> candidates, verified;
        for(int i=0;i<8;++i)candidates.push_back((saved+i*13)%128);
        require(e->prefill(candidates,&verified),"pipeline batched verify declined");
        require(verified.size()==8,"pipeline verify returned wrong width");
        e->commit_spec_prefix(saved,accepted);
        e->forward(verified[accepted-1]);
        require(e->argmax_token()>=0,"pipeline post-rollback argmax failed");
    }
    const auto state=state_of(*e);
    const auto name=root/(std::string("pp-")+(replay?"replay":"full")+"-"+std::to_string(rank)+".bin");
    std::ofstream out(name,std::ios::binary);
    out.write(reinterpret_cast<const char*>(&e->pos),sizeof(e->pos));
    out.write(reinterpret_cast<const char*>(state.data()),state.size()*sizeof(float));
    require(bool(out),"pipeline state write failed");
    return 0;
}
static void pp_gate(const char* self,const fs::path& root) {
    for(int replay : {0,1}) {
        pid_t children[2];
        for(int rank=0;rank<2;++rank) {
            const std::string r=std::to_string(rank), mode=std::to_string(replay);
            const std::string socket=(root/("pp-"+mode+".sock")).string();
            const std::string logfile=(root/("pp-"+mode+"-"+r+".log")).string();
            children[rank]=::fork();
            require(children[rank]>=0,"fork failed");
            if(children[rank]==0) {
                ::setenv("GRIMOIRE_PP_RANK",r.c_str(),1);
                ::setenv("GRIMOIRE_PP_WORLD_SIZE","2",1);
                ::setenv("GRIMOIRE_PP_SPLIT","2",1);
                ::setenv("GRIMOIRE_PP_SOCKET",socket.c_str(),1);
                std::freopen(logfile.c_str(),"w",stdout);
                std::freopen(logfile.c_str(),"a",stderr);
                ::execl(self,self,"--pp-check",root.c_str(),mode.c_str(),r.c_str(),nullptr);
                ::_exit(127);
            }
        }
        bool ok=true;
        for(pid_t pid : children) { int st=0; require(::waitpid(pid,&st,0)==pid,"waitpid failed");
            ok = ok && WIFEXITED(st) && WEXITSTATUS(st)==0; }
        require(ok,"pipeline child failed (logs retained in fixture directory)");
    }
    auto read=[](const fs::path& p) {
        std::ifstream f(p,std::ios::binary); require(bool(f),"missing pipeline state");
        return std::vector<char>(std::istreambuf_iterator<char>(f),{});
    };
    for(int rank=0;rank<2;++rank) {
        const auto full=read(root/("pp-full-"+std::to_string(rank)+".bin"));
        const auto replay=read(root/("pp-replay-"+std::to_string(rank)+".bin"));
        require(!full.empty() && full==replay,"pipeline replay state differs from full snapshots");
    }
}
int main(int argc,char** argv) {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    ::setenv("GRIMOIRE_DEVICE_ANY","1",1);
    ::setenv("GRIMOIRE_BATCHED_PREFILL_NOXMX","1",1);
    try {
        if(argc==5 && !std::strcmp(argv[1],"--pp-check"))
            return pp_child(argv[2],std::atoi(argv[3])!=0,std::atoi(argv[4]));
        const bool single_process = argc==2 && !std::strcmp(argv[1],"--single-process");
        require(argc==1 || single_process,"usage: test_paiton_e2e [--single-process]");
        // Make the gate independent of a user's serving recipe.
        for(const char* var : {"GRIMOIRE_PP_RANK","GRIMOIRE_TP_RANK","GRIMOIRE_PP_LAYERS",
             "GRIMOIRE_DFLASH_MODEL","GRIMOIRE_DFLASH_DRAFT_BF16","GRIMOIRE_PREFIX_CACHE"}) ::unsetenv(var);
        char temp[]="/tmp/grimoire-paiton-XXXXXX";
        require(::mkdtemp(temp),"mkdtemp failed"); const fs::path root=temp;
        std::printf("Paiton fixture directory: %s\n",temp);
        mini::write_model(root/"hybrid",mini::hybrid(4,true));
        ::setenv("GRIMOIRE_MTP","1",1);
        int rounds=0;
        for(Fmt fmt : {Fmt::BF16,Fmt::FP8_E4M3}) {
            auto full=load(root/"hybrid",fmt,false), replay=load(root/"hybrid",fmt,true);
            require(full->spec_dn_steps && !full->spec_dn_updates &&
                    replay->spec_dn_updates && !replay->spec_dn_steps &&
                    replay->spec_update_elems<replay->spec_dn_elems,"compact storage not active");
            for(int i=0;i<41;++i) { full->forward((i*17+3)%128); replay->forward((i*17+3)%128); }
            same_state(*full,*replay);
            for(auto [width,accepted] : {std::pair<int,int>{2,1},{8,1},{8,3},{8,7},
                                         {8,8},{16,9},{16,15},{16,16}}) {
                const int saved=full->pos;
                full->snapshot_recurrent(); replay->snapshot_recurrent();
                std::vector<int32_t> candidates,va,vb;
                for(int i=0;i<width;++i)candidates.push_back((saved+i*13)%128);
                require(full->prefill(candidates,&va) && replay->prefill(candidates,&vb),"batched verify declined");
                require(va==vb && int(va.size())==width,"verify tokens differ");
                same_state(*full,*replay);
                rejects(*replay,saved+1,accepted); rejects(*replay,saved,0); rejects(*replay,saved,width+1);
                full->commit_spec_prefix(saved,accepted); replay->commit_spec_prefix(saved,accepted);
                same_state(*full,*replay); rejects(*replay,saved,accepted);
                full->forward(va[accepted-1]); replay->forward(vb[accepted-1]);
                require(full->argmax_token()==replay->argmax_token(),"post-rollback token differs");
                same_state(*full,*replay); ++rounds;
            }
            replay->reset(); rejects(*replay,0,1);
        }
        ::unsetenv("GRIMOIRE_MTP");
        mini::write_model(root/"dense",mini::dense(4));
        mini::write_model(root/"draft",draft_fixture());
        ::setenv("GRIMOIRE_DFLASH_MODEL",(root/"draft").c_str(),1);
        ::setenv("GRIMOIRE_DFLASH_M","4",1);
        ::setenv("GRIMOIRE_DFLASH_DRAFT_FORMAT","bf16",1);
        auto narrow=load(root/"dense",Fmt::INT4,true);
        ::setenv("GRIMOIRE_DFLASH_M","8",1);
        ::setenv("GRIMOIRE_DFLASH_DRAFT_FORMAT","fp8",1);
        auto wide=load(root/"dense",Fmt::INT4,true);
        require(narrow->dflash2.fc.w.fmt==Fmt::BF16 && wide->dflash2.fc.w.fmt==Fmt::FP8_E4M3,
                "draft precision inherited target or another engine");
        for(auto* e : {narrow.get(),wide.get()}) {
            const int width=e==narrow.get()?4:8;
            require(e->dflash2.v2 && e->dflash2.selector_ok && e->dflash_block_rows()==width,
                    "DFlash2 selector or per-engine width inactive");
            for(int i=0;i<8;++i)e->forward((i*7+3)%128);
            std::vector<int32_t> proposed;
            require(e->dflash_draft(e->argmax_token(),e->pos,proposed),"DFlash2 draft failed");
            require(int(proposed.size())==width-1,"DFlash2 proposal count differs");
            for(int tok : proposed)require(tok>=0 && tok<128,"DFlash2 invalid token");
        }
        narrow.reset(); wide.reset();
        std::printf("Paiton single-process gate: %d rollback rounds and DFlash2 width/precision isolation PASS\n",rounds);
        // Fresh processes: SYCL worker threads must not be inherited by a
        // fork-only child, and each rank needs its own queue/socket.
        if (single_process)
            std::puts("SKIPPED by --single-process: two-stage PP; run the default gate on a host with Unix sockets");
        else {
            pp_gate(fs::absolute(argv[0]).c_str(),root);
            std::puts("Paiton two-stage PP state parity: PASS");
        }
        fs::remove_all(root);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
