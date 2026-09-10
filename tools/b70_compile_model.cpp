// Native C++ offline quantizer. No GPU or inference framework is required.
#include "b70/native_model.hpp"
#include "b70/qwen35.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
using namespace b70;
void require(bool yes, const std::string& why) { if (!yes) throw std::runtime_error(why); }
uint64_t aligned(uint64_t n) {
    require(n<=uint64_t(INT64_MAX)-kNativeAlignment,"output too large");
    return (n+kNativeAlignment-1)/kNativeAlignment*kNativeAlignment;
}
bool ends(const std::string& s,const char* tail) {
    const size_t n=std::strlen(tail); return s.size()>=n && s.compare(s.size()-n,n,tail)==0;
}
Fmt parse_format(const std::string& s) {
    if(s=="int4") return Fmt::INT4;
    if(s=="fp8" || s=="fp8-e4m3") return Fmt::FP8_E4M3;
    if(s=="fp8-e5m2") return Fmt::FP8_E5M2;
    if(s=="int8") return Fmt::INT8;
    if(s=="mxfp4") return Fmt::MXFP4;
    if(s=="mxfp8") return Fmt::MXFP8;
    if(s=="bf16") return Fmt::BF16;
    throw std::runtime_error("unknown format: "+s);
}
struct Options {
    std::string input,output,policy="experts";
    Fmt format=Fmt::INT4;
    bool bundle=false;
};
void usage(const char* app) {
    std::printf("usage: %s BF16_MODEL_DIR OUTPUT [--format FORMAT] [--policy experts|linear] [--bundle]\n"
        "FORMAT: int4 (default), fp8, fp8-e4m3, fp8-e5m2, int8, mxfp4, mxfp8, bf16\n"
        "Default policy quantizes routed experts; linear also quantizes non-sensitive projections.\n"
        "--bundle creates OUTPUT/model-v3.b70 and copies config/tokenizer files.\n"
        "Existing outputs are never overwritten. Conversion uses round-to-nearest weights, no calibration.\n",app);
}
struct AtomicFile {
    std::string temp;
    FILE* file=nullptr;
    explicit AtomicFile(const fs::path& target) {
        std::string pattern=target.string()+".tmp-XXXXXX";
        std::vector<char> p(pattern.begin(),pattern.end());p.push_back(0);
        int fd=::mkstemp(p.data());require(fd>=0,"cannot create temporary output: "+std::string(std::strerror(errno)));
        temp=p.data();file=::fdopen(fd,"wb+");
        if(!file){::close(fd);::unlink(temp.c_str());throw std::runtime_error("fdopen failed");}
    }
    ~AtomicFile(){if(file)std::fclose(file);if(!temp.empty())::unlink(temp.c_str());}
    void write(uint64_t off,const void* data,size_t n) {
        require(off<=uint64_t(INT64_MAX) && n<=uint64_t(INT64_MAX)-off,"file offset overflow");
        require(::fseeko(file,off_t(off),SEEK_SET)==0 && (!n || std::fwrite(data,1,n,file)==n),
                "write failed (check free disk space)");
    }
    void finish(const fs::path& target) {
        require(std::fflush(file)==0 && ::fsync(::fileno(file))==0,"cannot flush output");
        FILE* f=file;file=nullptr;require(std::fclose(f)==0,"cannot close output");
        NativeModel check;std::string err;
        if(!check.open(temp,err)) throw std::runtime_error("export validation failed: "+err);
        check.close();
        // link() is atomic and refuses existing files, including symlinks.
        require(::link(temp.c_str(),target.c_str())==0,"cannot publish output (destination exists or filesystem error)");
        ::unlink(temp.c_str());temp.clear();
    }
};
struct TempDir {
    fs::path path;
    ~TempDir(){if(!path.empty()){std::error_code ec;fs::remove_all(path,ec);}}
};
struct Stats {uint64_t raw=0,quantized=0,fallback=0;double sqerr=0,energy=0,maxerr=0;};
Stats convert(Qwen35Model& model,const fs::path& target,const Options& opt) {
    AtomicFile out(target);Stats stats;
    NativeFileHeader hdr{kNativeMagic,kNativeVersion,kNativeAlignment,0,0,0,0x031,0};
    out.write(0,&hdr,sizeof hdr);
    uint64_t pos=kNativeAlignment;
    std::vector<NativeTensorRecord> toc;
    for(const auto& kv:model.index) {
        const auto& name=kv.first;const auto& ref=kv.second;
        require(!ref.native && ref.shard>=0,"input must be the original BF16 safetensors checkpoint");
        require(!name.empty() && name.size()<sizeof(NativeTensorRecord::name),"tensor name too long: "+name);
        require(ref.t.shape.size()<=4,"tensor rank exceeds native format: "+name);
        uint64_t count=1;
        for(auto d:ref.t.shape){require(d>0 && count<=uint64_t(INT64_MAX)/uint64_t(d),"invalid shape: "+name);count*=uint64_t(d);}
        const int unit=st_dtype_size(ref.t.dtype);
        require(unit>0 && count<=SIZE_MAX/size_t(unit) && ref.t.end>=ref.t.begin &&
            ref.t.end-ref.t.begin==count*size_t(unit),"invalid tensor byte size: "+name);
        require(name.find(".qweight")==std::string::npos && name.find(".weight_packed")==std::string::npos &&
            name.find(".weight_scale")==std::string::npos,"prequantized input is not supported by this BF16 converter: "+name);
        const bool expert=name.find(".mlp.experts.")!=std::string::npos;
        const bool matrix=(ref.t.shape.size()==2 && ends(name,".weight")) ||
                           (ref.t.shape.size()==3 && expert);
        if(matrix)require(ref.t.dtype==STDtype::BF16,"expected BF16 source matrix: "+name);
        bool pack=matrix && !keep_qwen_bf16(name) && (opt.policy=="linear" || expert) && opt.format!=Fmt::BF16;
        NativeTensorRecord r{};
        std::memcpy(r.name,name.c_str(),name.size()+1);
        r.source_dtype=uint32_t(ref.t.dtype);r.rank=uint32_t(ref.t.shape.size());
        std::copy(ref.t.shape.begin(),ref.t.shape.end(),r.shape);
        int N=0,K=0;
        if(pack){
            require(ref.t.shape.back()<=INT32_MAX/2,"row width exceeds kernel limit: "+name);
            K=int(ref.t.shape.back());require(count/uint64_t(K)<=INT32_MAX,"row count exceeds kernel limit: "+name);
            N=int(count/uint64_t(K));
            const int group=opt.format==Fmt::INT4?kInt4Group:
                (opt.format==Fmt::MXFP4 || opt.format==Fmt::MXFP8)?kMXBlock:1;
            if(K%group){
                require(!expert,"expert width is incompatible with selected groups; use FP8 or BF16: "+name);
                pack=false;++stats.fallback;
            }
        }
        r.payload_offset=aligned(pos);
        if(pack){
            r.encoding=uint32_t(native_encoding(opt.format));
            r.padded_n=N;r.padded_k=K;r.tile_n=64;r.tile_k=256;
            r.payload_bytes=uint64_t(N)*uint64_t(bytes_per_row(opt.format,K));
            const uint64_t scale_row=uint64_t(scales_per_row(opt.format,K))*scale_element_bytes(opt.format);
            const uint64_t zero_row=opt.format==Fmt::INT4?uint64_t(K/kInt4Group):0;
            r.scales_offset=aligned(r.payload_offset+r.payload_bytes);
            r.scales_bytes=uint64_t(N)*(scale_row+zero_row);
            // Bound conversion memory even for a fused [experts,2I,H] tensor.
            const int chunk=std::max(1,int((32u*1024u*1024u)/(uint64_t(K)*sizeof(float))));
            for(int row=0;row<N;){
                const int nr=std::min(chunk,N-row);
                STTensor sub=ref.t;sub.shape={nr,K};sub.begin=ref.t.begin+uint64_t(row)*K*2;
                sub.end=sub.begin+uint64_t(nr)*K*2;
                std::vector<float> fp(size_t(nr)*K);std::string err;
                if(!model.shards[ref.shard]->read_f32(sub,fp.data(),err))
                    throw std::runtime_error("read "+name+": "+err);
                const auto q=quantize(fp.data(),nr,K,opt.format);const auto w=q.view();
                for(int n=0;n<nr;++n)for(int k=0;k<K;++k){
                    const double a=fp[size_t(n)*K+k],b=w.at(n,k);
                    if(!std::isfinite(b)) throw std::runtime_error("non-finite reconstructed weight: "+name);
                    stats.sqerr+=(a-b)*(a-b);stats.energy+=a*a;
                    stats.maxerr=std::max(stats.maxerr,std::fabs(a-b));
                }
                out.write(r.payload_offset+uint64_t(row)*q.row_bytes,q.payload.data(),q.payload.size());
                out.write(r.scales_offset+uint64_t(row)*scale_row,q.scales_raw.data(),q.scales_raw.size());
                if(zero_row)out.write(r.scales_offset+uint64_t(N)*scale_row+uint64_t(row)*zero_row,q.zeros.data(),q.zeros.size());
                row+=nr;
            }
            pos=r.scales_offset+r.scales_bytes;++stats.quantized;
        }else{
            r.encoding=uint32_t(NativeEncoding::RAW);r.payload_bytes=count*size_t(unit);
            std::vector<uint8_t> raw(size_t(std::min<uint64_t>(r.payload_bytes,4u*1024u*1024u)));
            for(uint64_t off=0;off<r.payload_bytes;){
                const size_t n=size_t(std::min<uint64_t>(raw.size(),r.payload_bytes-off));
                TensorRef sub=ref;sub.t.begin=ref.t.begin+off;sub.t.end=sub.t.begin+n;std::string err;
                if(!model.read_raw(sub,raw.data(),err)) throw std::runtime_error("read "+name+": "+err);
                if(ref.t.dtype==STDtype::BF16 || ref.t.dtype==STDtype::F32){
                    std::vector<float> f(n/size_t(unit));st_to_f32(raw.data(),ref.t.dtype,int64_t(f.size()),f.data());
                    for(float v:f)if(!std::isfinite(v)) throw std::runtime_error("non-finite source weight: "+name);
                }
                out.write(r.payload_offset+off,raw.data(),n);off+=n;
            }
            pos=r.payload_offset+r.payload_bytes;++stats.raw;
        }
        toc.push_back(r);
        std::printf("  [%zu/%zu] %-12s %s\n",toc.size(),model.index.size(),native_encoding_name(r.encoding),name.c_str());
    }
    require(opt.format==Fmt::BF16 || stats.quantized>0,"no eligible matrices; choose --policy linear for a dense model");
    hdr.toc_offset=aligned(pos);hdr.tensor_count=toc.size();
    hdr.file_size=hdr.toc_offset+toc.size()*sizeof(NativeTensorRecord);
    out.write(hdr.toc_offset,toc.data(),toc.size()*sizeof(NativeTensorRecord));out.write(0,&hdr,sizeof hdr);
    out.finish(target);
    std::printf("Saved %llu quantized, %llu raw tensors; %llu unaligned matrices kept BF16; %.3f GiB.\n",
        (unsigned long long)stats.quantized,(unsigned long long)stats.raw,(unsigned long long)stats.fallback,
        double(hdr.file_size)/1073741824.0);
    std::printf("Quantized-weight relative RMS %.8g; maximum absolute error %.8g. These are weight errors, not language-quality scores.\n",
        stats.energy?std::sqrt(stats.sqerr/stats.energy):0,stats.maxerr);
    return stats;
}
}
int main(int argc,char** argv){
    if(argc==2 && std::string(argv[1])=="--help"){usage(argv[0]);return 0;}
    if(argc<3){usage(argv[0]);return 2;}
    try{
        Options o;o.input=argv[1];o.output=argv[2];
        for(int i=3;i<argc;++i){
            const std::string arg=argv[i];
            if(arg=="--bundle")o.bundle=true;
            else if(arg=="--format" && i+1<argc)o.format=parse_format(argv[++i]);
            else if(arg=="--policy" && i+1<argc)o.policy=argv[++i];
            else throw std::runtime_error("unknown or incomplete option: "+arg);
        }
        require(o.policy=="experts" || o.policy=="linear","policy must be experts or linear");
        const fs::path output=fs::absolute(o.output).lexically_normal();
        require(!fs::exists(fs::symlink_status(output)),"output already exists: "+output.string());
        require(fs::is_directory(output.parent_path()),"output parent directory does not exist");
        require(!fs::exists(fs::path(o.input)/"model-v3.b70") && !fs::exists(fs::path(o.input)/"model-v2.b70"),
                "input contains a native artifact; select the original BF16 safetensors directory");
        Qwen35Model model;std::string err;
        if(!model.load(o.input,err,true,true)) throw std::runtime_error("load: "+err);
        model.unmap_all();
        TempDir tmp;
        fs::path destination=output;
        if(o.bundle){
            require(fs::is_regular_file(fs::path(o.input)/"tokenizer.json"),"bundle requires tokenizer.json");
            std::string pattern=output.string()+".tmp-XXXXXX";std::vector<char> p(pattern.begin(),pattern.end());p.push_back(0);
            require(::mkdtemp(p.data())!=nullptr,"cannot create temporary bundle");tmp.path=p.data();
            for(const char* name:{"config.json","tokenizer.json","tokenizer_config.json","special_tokens_map.json",
                "generation_config.json","chat_template.jinja","vocab.json","merges.txt","added_tokens.json","README.md","LICENSE","LICENSE.txt"}){
                const auto source=fs::path(o.input)/name;
                if(fs::is_regular_file(source))fs::copy_file(source,tmp.path/name);
            }
            destination=tmp.path/"model-v3.b70";
        }
        const Stats s=convert(model,destination,o);
        if(o.bundle){
            std::ofstream manifest(tmp.path/"grimoire-quantization.json");
            manifest<<"{\n  \"container_version\": 3,\n  \"format\": \""<<native_encoding_name(uint32_t(native_encoding(o.format)))
                <<"\",\n  \"policy\": \""<<o.policy<<"\",\n  \"method\": \"round-to-nearest, no activation calibration\",\n"
                <<"  \"quantized_tensors\": "<<s.quantized<<",\n  \"raw_tensors\": "<<s.raw<<"\n}\n";
            manifest.close();require(bool(manifest),"cannot write quantization manifest");
            // Linux no-replace rename protects even an existing empty directory.
            require(::syscall(SYS_renameat2,AT_FDCWD,tmp.path.c_str(),AT_FDCWD,output.c_str(),1)==0,
                    "cannot publish bundle (destination exists or rename unsupported)");
            tmp.path.clear();
        }
        std::printf("Output: %s\n",output.c_str());return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"quantizer: %s\n",e.what());return 1;}
}
