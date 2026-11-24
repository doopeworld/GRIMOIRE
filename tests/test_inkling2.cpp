// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/inkling2.hpp"
#include "b70/json.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <stdexcept>

using namespace b70::inkling2;

// The device packer's arithmetic is also ordinary C++. Compare every input
// weight against the independent scalar checkpoint reader at the Mini shapes.
template<PackedFormat F>
static void check_pair_layout(int n,int k) {
    constexpr bool nv=F==PackedFormat::NVFP4;
    const int pb=nv?k/2:k, sb=nv?k/16:k/32;
    std::vector<uint8_t> payload(size_t(n)*pb), scales(size_t(n)*sb);
    for(size_t i=0;i<payload.size();++i) {
        const uint8_t code=uint8_t(i*73+(i/pb)*29+i/13);
        payload[i]=!nv && (code&127)==127?uint8_t(code-1):code;
    }
    for(size_t i=0;i<scales.size();++i)
        scales[i]=uint8_t((nv?39:120)+(i*19+i/sb)%13);
    const float amax=2.7f, denom=3.1f;
    const ExpertMatrix matrix{F,n,k,payload.data(),scales.data(),nv?amax/denom:1.f};
    const PackedRows oracle{F,1,n,k,payload.data(),scales.data(),&amax,denom};
    std::vector<uint16_t> packed(size_t(n)*k);
    for(int kp=0;kp<k/2;++kp) for(int col=0;col<n;++col) {
        const auto dword=expert_bf16_pair<F>(matrix,col,kp);
        const auto at=2*vnni_pair_index(n,col,kp);
        packed[at]=uint16_t(dword); packed[at+1]=uint16_t(dword>>16);
    }
    std::vector<float> row(k);
    for(int col=0;col<n;++col) {
        oracle.row(0,col,row);
        for(int j=0;j<k;++j)
            assert(packed[size_t(j/2)*n*2+2*col+j%2]==b70::f32_to_bf16(row[j]).bits);
    }
}

int main(int argc,char** argv) {
    const auto check_bad=[](auto fn){
        try { fn(); }
        catch(const std::exception&) { return; }
        throw std::runtime_error("expected rejection");
    };
    if(argc>=2) {
        std::ifstream in(argv[1]);
        std::string raw(std::istreambuf_iterator<char>(in),{});
        auto c=parse_config(raw);
        assert(c.layers==24 && c.cache_slots()==48 && c.output_vocab==200058 && c.vocab==201024);
        const auto v=c.visits();
        int full=0,revisit=0;
        std::set<int> slots;
        for(auto visit:v) {
            full+=!visit.sliding; revisit+=visit.has_message;
            assert(slots.insert(visit.cache_slot).second);
        }
        assert(v.size()==48 && full==8 && revisit==24);
        assert(v[6].physical==6 && v[6].cache_slot==6 && !v[6].has_message);
        assert(v[18].physical==6 && v[18].cache_slot==24 && v[18].has_message);
        assert(v[30].physical==6 && v[30].cache_slot==36 && v[30].has_message);
        assert(v[42].physical==18 && v[42].cache_slot==18 && !v[42].has_message);
        auto cfg=b70::Json::parse(raw);
        cfg.object["text_config"].object["loop_config"].object["is_varloop"].boolean=true;
        check_bad([&]{parse_config(cfg.dump());});
    }
    // Bias changes selection, but normalized weights use the original sigmoid scores.
    const auto r=route(std::array{0.f,1.f,-1.f},std::array{0.f,0.f,1.f},2,8.f);
    const float a=1.f/(1.f+std::exp(1.f)),b=1.f/(1.f+std::exp(-1.f));
    assert(r.size()==2 && r[0].expert==2 && r[1].expert==1);
    assert(std::abs(r[0].weight-8.f*a/(a+b))<1e-6f);
    assert(std::abs(r[1].weight-8.f*b/(a+b))<1e-6f);
    const float g=6.f,u=-3.f;
    const float expected=4.f*std::tanh(g/4.f)/(1.f+std::exp(-g))*25.f*std::tanh(u/25.f);
    assert(std::abs(swiglu(g,u,4.f,25.f)-expected)<2e-6f);
    // Codes 2 and 11 are +1 and -1.5, per-expert multiplier amax/denom=2/4.
    std::array<uint8_t,16> fp4{}; fp4.fill(0xb2);
    std::array<uint8_t,2> scales{0x40,0x38}; // E4M3 2, 1
    std::array<float,2> amax{2.f,8.f};
    PackedRows nv{PackedFormat::NVFP4,2,1,16,fp4.data(),scales.data(),amax.data(),4.f};
    std::array<float,16> row;
    nv.row(0,0,row); assert(row[0]==1.f && row[1]==-1.5f);
    nv.row(1,0,row); assert(row[0]==2.f && row[1]==-3.f);
    std::array<uint8_t,32> fp8{}; fp8.fill(0x38);
    const uint8_t adjusted=126;
    PackedRows mx{PackedFormat::MXFP8,1,1,32,fp8.data(),&adjusted};
    std::array<float,32> widened;
    mx.row(0,0,widened); assert(widened[0]==.5f && widened[31]==.5f);
    nv.denominator=0;
    check_bad([&]{nv.row(0,0,row);});
    auto one=read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})");
    assert(one.at("x").shape==std::vector<int64_t>{1});
    check_bad([]{read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}})");});
    check_bad([]{read_tensor_header(R"({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}})");});
    for(int k:{2304,1152}) {
        check_pair_layout<PackedFormat::NVFP4>(2304,k);
        check_pair_layout<PackedFormat::MXFP8>(2304,k);
    }
    if(argc>=3) {
        auto c=Checkpoint::open(argv[2]);
        for(int layer:{0,c.config.layers-1}) for(bool down:{false,true}) {
            const auto a=c.expert_matrix(layer,0,down), b=c.expert_matrix(layer,1,down);
            const bool nv=layer==0;
            assert((a.format==PackedFormat::NVFP4)==nv && a.format==b.format);
            assert(a.n==(down?c.config.hidden:2*c.config.inter));
            assert(a.k==(down?c.config.inter:c.config.hidden));
            assert(b.payload-a.payload==a.n*(nv?a.k/2:a.k));
            assert(b.scales-a.scales==a.n*(nv?a.k/16:a.k/32));
            assert(a.outer==(nv?.5f:1.f) && b.outer==1.f);
        }
        check_bad([&]{c.expert_matrix(-1,0,false);});
        check_bad([&]{c.expert_matrix(0,c.config.experts,true);});
        TextReference model(std::move(c),16);
        std::vector<float> first(model.config().output_vocab), after(first.size());
        const int id=model.step(2,first);
        model.step(7);
        model.step(9);
        check_bad([&]{model.step(model.config().output_vocab);});  // padded ID cannot advance the sequence
        assert(model.position()==3);
        model.reset();
        const int again=model.step(2,after);
        assert(model.position()==1 && id==again && first==after);
    }
    std::cout<<"PASS Inkling2 schedule, expert views, Mini-shape VNNI layout, routing, SwiGLU, reset and malformed header rejection\n";
}
