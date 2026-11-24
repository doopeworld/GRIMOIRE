// SPDX-License-Identifier: GPL-3.0-or-later
#include "inkling2_sycl.hpp"
#include "b70/inkling2.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string_view>

using namespace b70::inkling2;
namespace {
constexpr int H=2304, I=1152, M=17;
constexpr float gate_cap=4, up_cap=25;

// Sparse deterministic source data: still the exact Mini matrix sizes and
// row strides, with one nonzero down element per output channel. Use the
// scalar checkpoint reader, not the VNNI packer, for expected weights.
struct Fixture {
    std::vector<uint8_t> data, scale;
    int n, k;
    PackedFormat format;
    float amax, denom=3.1f;
    Fixture(PackedFormat fmt,int n,int k,float a,bool down)
        : data(size_t(n)*payload_row_bytes(fmt,k)), scale(size_t(n)*scale_row_bytes(fmt,k)),
          n(n), k(k), format(fmt), amax(a) {
        const bool nv=fmt==PackedFormat::NVFP4;
        for(int c=0;c<n;++c) {
            for(int s=0;s<(nv?k/16:k/32);++s)
                scale[size_t(c)*(nv?k/16:k/32)+s]=uint8_t((nv?52:125)+(c*5+s*3)%5);
            for(int j=0;j<k;++j) {
                if(down && j!=down_index(c,k)) continue;
                const int tag=(c*3+j*7+j/13)%7;
                const bool negative=(c+j/3)%5<2;
                if(nv)
                    data[(size_t(c)*k+j)/2]|=uint8_t((1+tag+(negative?8:0))<<((j%2)*4));
                else
                    data[size_t(c)*k+j]=uint8_t(0x34+tag+(negative?0x80:0));
            }
        }
    }
    static int down_index(int col,int k) { return (col*19+col/3+17)%k; }
    ExpertMatrix view(const uint8_t* p,const uint8_t* s) const {
        return {format,n,k,p,s,format==PackedFormat::NVFP4?amax/denom:1.f};
    }
    PackedRows scalar() const { return {format,1,n,k,data.data(),scale.data(),&amax,denom}; }
};
int input_index(int m) { return (m*131+15)%H; }
float round_bf(float x) { return b70::bf16_to_f32(b70::f32_to_bf16(x)); }
Bf16 as_bf(float x) { return std::bit_cast<Bf16>(b70::f32_to_bf16(x).bits); }

struct Allocations {
    sycl::queue& q;
    std::vector<void*> pointers;
    explicit Allocations(sycl::queue& q) : q(q) { pointers.reserve(10); }
    template<typename T> T* alloc(size_t n) {
        auto* p=sycl::aligned_alloc_device<T>(64,n,q);
        if(!p) throw std::bad_alloc();
        pointers.push_back(p);
        return p;
    }
    ~Allocations() {
        // Fixture cleanup only. The launcher itself never waits.
        q.wait();
        for(auto* p:pointers) sycl::free(p,q);
    }
};

void test(sycl::queue& q,PackedFormat f,int m) {
    const Fixture gu(f,2*I,H,2.7f,false), dn(f,H,I,1.4f,true);
    std::vector<Bf16> input(size_t(m)*H,as_bf(0.f)), result(input.size());
    for(int t=0;t<m;++t) input[size_t(t)*H+input_index(t)]=as_bf(t%2?3.f:-2.f);
    Allocations mem(q);
    auto* x=mem.alloc<Bf16>(input.size()); auto* y=mem.alloc<Bf16>(result.size());
    auto* g=mem.alloc<uint8_t>(gu.data.size()); auto* gs=mem.alloc<uint8_t>(gu.scale.size());
    auto* d=mem.alloc<uint8_t>(dn.data.size()); auto* ds=mem.alloc<uint8_t>(dn.scale.size());
    const auto size=expert_scratch_sizes(m,H,I);
    ExpertScratch scratch{mem.alloc<uint32_t>(size.weight_pairs),mem.alloc<float>(size.projection),
                          mem.alloc<Bf16>(size.activation)};
    const std::vector<sycl::event> ready{
        q.memcpy(x,input.data(),input.size()*sizeof(Bf16)),
        q.memcpy(g,gu.data.data(),gu.data.size()), q.memcpy(gs,gu.scale.data(),gu.scale.size()),
        q.memcpy(d,dn.data.data(),dn.data.size()), q.memcpy(ds,dn.scale.data(),dn.scale.size())};
    auto done=launch_expert_ffn(q,gu.view(g,gs),dn.view(d,ds),m,x,y,gate_cap,up_cap,scratch,ready);
    q.submit([&](sycl::handler& h) {
        h.depends_on(done); h.memcpy(result.data(),y,result.size()*sizeof(Bf16));
    }).wait_and_throw();

    const auto g_ref=gu.scalar(), d_ref=dn.scalar();
    std::vector<float> gbuf(H), ubuf(H), dbuf(I), act(size_t(m)*I);
    for(int j=0;j<I;++j) {
        g_ref.row(0,2*j,gbuf); g_ref.row(0,2*j+1,ubuf);
        for(int t=0;t<m;++t) {
            const int k=input_index(t);
            const float v=float(input[size_t(t)*H+k]);
            act[size_t(t)*I+j]=round_bf(swiglu(round_bf(v*round_bf(gbuf[k])),
                round_bf(v*round_bf(ubuf[k])),gate_cap,up_cap));
        }
    }
    double sq=0, norm=0;
    float max_abs=0;
    for(int col=0;col<H;++col) {
        d_ref.row(0,col,dbuf);
        const int k=Fixture::down_index(col,I);
        for(int t=0;t<m;++t) {
            const float expected=round_bf(act[size_t(t)*I+k]*round_bf(dbuf[k]));
            const float saw=float(result[size_t(t)*H+col]);
            if(!std::isfinite(saw)) throw std::runtime_error("non-finite GPU result");
            const double diff=double(saw)-expected;
            sq+=diff*diff; norm+=double(expected)*expected;
            max_abs=std::max(max_abs,float(std::abs(diff)));
        }
    }
    const double rel=std::sqrt(sq/norm);
    std::cout<<(f==PackedFormat::NVFP4?"NVFP4":"MXFP8")<<" rows="<<m<<" rel_l2="<<rel<<" max_abs="<<max_abs<<'\n';
    if(!std::isfinite(rel) || rel>0.005) throw std::runtime_error("Inkling2 GPU/CPU expert mismatch");
}
} // namespace

int main(int argc,char** argv) try {
    const auto sizes=expert_scratch_sizes(M,H,I);
    if(sizes.weight_pairs!=size_t(H)*I || sizes.projection!=size_t(M)*H || sizes.activation!=size_t(M)*I)
        throw std::runtime_error("Mini scratch size mismatch");
    if(argc==2 && std::string_view(argv[1])=="--contract") {
        try { (void)expert_scratch_sizes(1,2303,1152); }
        catch(const std::invalid_argument&) {
            std::cout<<"PASS Mini scratch and shape contract (no queue or device enumeration)\n";
            return 0;
        }
        throw std::runtime_error("wrong shape accepted");
    }
    if(argc!=2 || std::string_view(argv[1])!="--reserved-b60" ||
       !std::getenv("ONEAPI_DEVICE_SELECTOR") || !std::string_view(std::getenv("ONEAPI_DEVICE_SELECTOR")).starts_with("level_zero:"))
        throw std::invalid_argument("use --contract, or --reserved-b60 with an explicitly reserved, isolated Level Zero device");
    const auto devices=sycl::device::get_devices(sycl::info::device_type::gpu);
    if(devices.size()!=1 || devices.front().get_backend()!=sycl::backend::ext_oneapi_level_zero ||
        devices.front().get_info<sycl::info::device::vendor_id>()!=0x8086)
        throw std::runtime_error("expected one visible Intel Level Zero GPU");
    std::cout<<devices.front().get_info<sycl::info::device::name>()<<'\n';
    auto async=[](sycl::exception_list xs){ for(auto& x:xs) std::rethrow_exception(x); };
    sycl::queue q(devices.front(),async); // intentionally not in-order
    for(auto f:{PackedFormat::NVFP4,PackedFormat::MXFP8})
        for(int m:{1,M}) test(q,f,m);
    q.wait_and_throw();
    return 0;
} catch(const std::exception& e) {
    std::cerr<<"inkling2 expert fixture: "<<e.what()<<'\n';
    return 1;
}
