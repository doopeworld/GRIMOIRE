// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/inkling2.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string_view>

using namespace b70::inkling2;

static std::string text(const char* name) {
    std::ifstream in(name,std::ios::binary);
    if(!in) throw std::runtime_error(std::string("cannot read ")+name);
    return {std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>()};
}
static void info(const Config& c,const std::vector<LayerLayout>& layout) {
    const auto plan=c.visits();
    const auto local=std::count_if(plan.begin(),plan.end(),[](auto v){return v.sliding;});
    const auto nv=std::count_if(layout.begin(),layout.end(),[](auto l){return l.gate_up==PackedFormat::NVFP4 && l.down==PackedFormat::NVFP4;});
    const auto mx=std::count_if(layout.begin(),layout.end(),[](auto l){return l.gate_up==PackedFormat::MXFP8 && l.down==PackedFormat::MXFP8;});
    std::cout<<"inkling2_loop text layers="<<c.layers<<" visits="<<plan.size()<<" kv_conv_slots="<<c.cache_slots()
             <<" local_visits="<<local<<" full_visits="<<plan.size()-local<<" nvfp4_layers="<<nv<<" mxfp8_layers="<<mx
             <<" output_vocab="<<c.output_vocab<<" padded_vocab="<<c.vocab<<'\n';
}
static int number(const char* v) {
    int n;
    const auto s=std::string_view(v);
    const auto p=std::from_chars(s.data(),s.data()+s.size(),n);
    if(p.ec!=std::errc{} || p.ptr!=s.data()+s.size()) throw std::invalid_argument("expected a decimal integer");
    return n;
}

int main(int argc,char** argv) try {
    if(argc==2 && std::string_view(argv[1])=="--help") {
        std::cout<<"grimoire-inkling2-reference --inspect-header CONFIG SAFETENSORS_PREFIX\n"
                   "grimoire-inkling2-reference --inspect MODEL_DIR\n"
                   "grimoire-inkling2-reference --tokens MODEL_DIR IDS_FILE CONTEXT NEW_TOKENS [LAST_LOGITS_F32]\n"
                   "Host-only validation, no GPU, tokenizer, images, API, sampling or MTP.\n";
        return 0;
    }
    if(argc==4 && std::string_view(argv[1])=="--inspect-header") {
        const auto c=parse_config(text(argv[2]));
        std::ifstream in(argv[3],std::ios::binary);
        unsigned char len[8];
        if(!in.read(reinterpret_cast<char*>(len),8)) throw std::runtime_error("short safetensors prefix");
        uint64_t sz=0;
        for(int i=0;i<8;++i) sz|=uint64_t(len[i])<<(8*i);
        if(!sz || sz>16*1024*1024) throw std::runtime_error("invalid safetensors header length");
        std::string header(sz,'\0');
        if(!in.read(header.data(),header.size())) throw std::runtime_error("truncated safetensors header");
        info(c,validate_tensors(c,read_tensor_header(header)));
        return 0;
    }
    if(argc==3 && std::string_view(argv[1])=="--inspect") {
        const auto c=Checkpoint::open(argv[2]); info(c.config,c.layout); return 0;
    }
    if((argc==6 || argc==7) && std::string_view(argv[1])=="--tokens") {
        const auto context=number(argv[4]); const auto n=number(argv[5]);
        // Accidental 1M context or long CPU generation is never implicit.
        if(context<1 || context>4096 || n<0 || n>512) throw std::invalid_argument("reference limits: ctx 1–4096, generation 0–512");
        std::ifstream in(argv[3]);
        if(!in) throw std::invalid_argument("cannot read IDs file");
        std::vector<int> ids;
        for(std::string id; in>>id;) {
            ids.push_back(number(id.c_str()));
            if(ids.size()>size_t(context)) throw std::invalid_argument("IDs exceed context");
        }
        if(!in.eof() || ids.empty() || ids.size()+n>size_t(context)) throw std::invalid_argument("invalid ID stream or request budget");
        auto weights=Checkpoint::open(argv[2]);
        for(auto id:ids) if(id<0 || id>=weights.config.output_vocab) throw std::invalid_argument("invalid input ID");
        TextReference model(std::move(weights),context);
        std::vector<float> logits(argc==7?size_t(model.config().output_vocab):0);
        int next=-1;
        for(auto id:ids) next=model.step(id,logits);
        if(argc==7) {
            // Last prompt's pre-generation logits, little-endian host f32, for an independent parity check.
            std::ofstream out(argv[6],std::ios::binary|std::ios::trunc);
            out.write(reinterpret_cast<const char*>(logits.data()),logits.size()*sizeof(float));
            if(!out) throw std::runtime_error("could not write logits");
        }
        std::cout<<"next="<<next<<" generated=";
        for(int i=0;i<n;++i) {
            std::cout<<(i?",":"")<<next<<std::flush;
            if(next==model.config().eos || i+1==n) break;
            next=model.step(next);
        }
        std::cout<<'\n';
        return 0;
    }
    throw std::invalid_argument("bad command; see --help");
} catch(const std::exception& e) {
    std::cerr<<"grimoire-inkling2-reference: "<<e.what()<<'\n';
    return 1;
}
