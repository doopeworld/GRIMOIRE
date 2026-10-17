// GRIMOIRE -- host GGUF parser and independently generated decoder goldens.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf.hpp"
#include "b70/gguf_decode.hpp"
#include "gguf/test_writer.hpp"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>

using namespace b70;
static int failures=0, cases=0;
#define CHECK(x) do {++cases;if(!(x)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);++failures;}}while(0)
using namespace gguf_test_writer;
static std::string temp_path() {return "/tmp/grimoire-test-gguf-"+std::to_string(getpid())+".gguf";}
static void write_file(const Bytes& b) {std::ofstream o(temp_path(),std::ios::binary);o.write(reinterpret_cast<const char*>(b.data()),b.size());}
static bool rejected(const Bytes& b,const std::string& reason) {
    write_file(b);GGUFFile f;std::string err;bool ok=f.open(temp_path(),err);
    CHECK(!ok);CHECK(!err.empty());
    if(!reason.empty()) {CHECK(err.find(reason)!=std::string::npos);if(err.find(reason)==std::string::npos)std::fprintf(stderr,"  got: %s\n",err.c_str());}
    return !ok;
}
static void parse_tests() {
    std::vector<KV> kvs={{"general.alignment",4,num(64,4)},
        kv_string("general.architecture","qwen35moe"),kv_array_strings("tokenizer.ggml.tokens",{"a","bb","ccc"}),
        {"u8",0,num(255,1)},{"i8",1,num(0x80,1)},{"u16",2,num(65535,2)},
        {"i16",3,num(0x8000,2)},{"i32",5,num(0x80000000,4)},
        {"f32",6,num(0x3f800000,4)},{"bool",7,num(1,1)},
        {"u64",10,num(0x12345678abcdef00ull,8)},{"i64",11,num(0xffffffffffffffffull,8)},
        {"f64",12,num(0x3ff0000000000000ull,8)}};
    std::vector<TS> ts={{"f32",{3,2},0,0,Bytes(24,0x12)},
        {"f16",{5,2},1,64,Bytes(20,0x34)},
        {"bf16",{2,2},30,128,Bytes(8,0x56)},
        {"q8",{32},8,192,Bytes(34,0x78)},
        {"q4k",{256},12,256,Bytes(144,0x9a)},
        {"q6k",{256,2},14,448,Bytes(420,0xbc)}};
    Bytes valid=build_file(kvs,ts);
    write_file(valid);GGUFFile f;std::string err;
    CHECK(f.open(temp_path(),err));if(!err.empty())std::fprintf(stderr,"open: %s\n",err.c_str());
    CHECK(f.version()==3 && f.alignment()==64 && f.tensors().size()==6);
    uint32_t align=0;CHECK(f.get_u32("general.alignment",align,err) && align==64);
    uint64_t u64=0;CHECK(f.get_u64("u64",u64,err) && u64==0x12345678abcdef00ull);
    float x=0;CHECK(f.get_f32("f32",x,err) && x==1.0f);
    std::string arch;CHECK(f.get_string("general.architecture",arch,err) && arch=="qwen35moe");
    const GGUFValue* arr=nullptr;CHECK(f.get_array("tokenizer.ggml.tokens",arr,err));
    CHECK(arr && arr->element_type==8 && std::get<std::vector<std::string>>(arr->data)[1]=="bb");
    CHECK(f.get("u8") && std::get<uint8_t>(f.get("u8")->data)==255);
    CHECK(f.get("i8") && std::get<int8_t>(f.get("i8")->data)==-128);
    CHECK(f.get("u16") && std::get<uint16_t>(f.get("u16")->data)==65535);
    CHECK(f.get("i16") && std::get<int16_t>(f.get("i16")->data)==-32768);
    CHECK(f.get("i32") && std::get<int32_t>(f.get("i32")->data)==INT32_MIN);
    CHECK(f.get("bool") && std::get<bool>(f.get("bool")->data));
    CHECK(f.get("i64") && std::get<int64_t>(f.get("i64")->data)==-1);
    CHECK(f.get("f64") && std::get<double>(f.get("f64")->data)==1.0);
    CHECK(!f.get_u64("general.alignment",u64,err) && err.find("wrong")!=std::string::npos);
    CHECK(!f.get_u32("absent",align,err) && err.find("missing")!=std::string::npos);
    CHECK(f.find("q6k") && f.find("q6k")->ne[0]==256 && f.find("q6k")->ne[1]==2);
    CHECK(f.find("q6k") && f.find("q6k")->n_rows==2 && f.find("q6k")->row_bytes==210);
    for(const auto& t:ts) {
        auto* found=f.find(t.name);CHECK(found!=nullptr);
        if(found) {CHECK(found->offset==f.data_start()+t.offset);Bytes raw(found->nbytes);CHECK(f.read_raw(*found,raw.data(),err));CHECK(raw==t.raw);}
    }
    if(auto* q6=f.find("q6k")) {
        Bytes row(q6->row_bytes);
        CHECK(f.read_raw_row(*q6,1,row.data(),err));
        CHECK(row==Bytes(210,0xbc));
        CHECK(!f.read_raw_row(*q6,2,row.data(),err) && err.find("out of range")!=std::string::npos);
    }
    f.close();
    std::vector<KV> smallkv={{"general.alignment",4,num(64,4)}};
    std::vector<TS> smallts={{"x",{1},0,0,Bytes(4,0)}};
    Bytes small=build_file(smallkv,smallts);
    for(size_t i=0;i<small.size();++i) rejected(Bytes(small.begin(),small.begin()+i),"");
    std::printf("truncation prefixes checked: %zu\n",small.size());
    {Bytes b=small;b[0]='X';rejected(b,"magic");}
    {Bytes b=small;b[0]='F';b[1]='U';b[2]='G';b[3]='G';rejected(b,"byte-swapped GGUF magic");}
    {Bytes b=small;b[4]=2;rejected(b,"version");}
    {Bytes b=small;b[4]=0;b[7]=3;rejected(b,"byte-swapped");}
    {auto v=smallkv;v.push_back({"bad",8,num(1000,8)});rejected(build_file(v,smallts),"string length past EOF");}
    {auto v=smallkv;Bytes p;put(p,0,4);put(p,UINT64_MAX,8);v.push_back({"bad",9,p});rejected(build_file(v,smallts),"array length");}
    {auto v=smallkv;Bytes p;put(p,9,4);put(p,0,8);v.push_back({"bad",9,p});rejected(build_file(v,smallts),"nested");}
    {auto v=smallkv;v.push_back(v[0]);rejected(build_file(v,smallts),"duplicate GGUF metadata");}
    {auto v=smallts;v.push_back({"x",{1},0,64,Bytes(4)});rejected(build_file(smallkv,v),"duplicate GGUF tensor");}
    {auto v=smallts;v[0].ne.clear();rejected(build_file(smallkv,v),"n_dims");}
    {auto v=smallts;v[0].ne={1,1,1,1,1};rejected(build_file(smallkv,v),"n_dims");}
    {auto v=smallts;v[0].ne={0};rejected(build_file(smallkv,v),"dimension is zero");}
    {auto v=smallts;v[0].type=99;rejected(build_file(smallkv,v),"type id 99");}
    {auto v=smallts;v[0].type=12;v[0].ne={31};rejected(build_file(smallkv,v),"not divisible");}
    {auto v=smallts;v[0].offset=1;rejected(build_file(smallkv,v),"misaligned");}
    {auto v=smallts;v.push_back({"y",{1},0,0,Bytes(4)});Bytes b=build_file(smallkv,v);rejected(b,"overlapping");
        write_file(b);GGUFFile h;CHECK(!h.open(temp_path(),err,true) && err.find("overlapping")!=std::string::npos);}
    {Bytes b=small;b.pop_back();rejected(b,"data past EOF");write_file(b);GGUFFile h;
        CHECK(h.open(temp_path(),err,true));
        if(auto* t=h.find("x")) {Bytes row(t->row_bytes);CHECK(!h.read_raw_row(*t,0,row.data(),err));}
    }
    for(uint32_t a:{0u,3u}) {auto v=smallkv;v[0].payload=num(a,4);rejected(build_file(v,smallts),"power of two");}
    {auto v=smallkv;v.push_back({"split.count",4,num(2,4)});rejected(build_file(v,smallts),"split filename");}
    ::unlink(temp_path().c_str());
}
static void split_tests() {
    const std::string prefix="/tmp/grimoire-test-split-"+std::to_string(getpid());
    const std::string one=prefix+"-00001-of-00002.gguf",two=prefix+"-00002-of-00002.gguf";
    auto write=[&](const std::string& p,const Bytes& b){
        std::ofstream o(p,std::ios::binary);o.write(reinterpret_cast<const char*>(b.data()),b.size());
    };
    auto shard=[&](uint32_t no,const std::string& name,uint32_t total=2,uint32_t align=64){
        std::vector<KV> kv={{"general.alignment",4,num(align,4)},
            // the exact types llama.cpp's gguf-split writes: u16, u16, i32
            {"split.count",2,num(2,2)},{"split.no",2,num(no,2)},
            {"split.tensors.count",5,num(total,4)}};
        return build_file(kv,{{name,{2,2},0,0,Bytes(16,uint8_t(no+1))}},align);
    };
    std::string err; GGUFFile f;
    write(one,shard(0,"first"));write(two,shard(1,"second"));
    CHECK(f.open(two,err));
    CHECK(f.tensors().size()==2 && f.find("first") && f.find("second"));
    if (const auto* t=f.find("second")) {
        Bytes bytes(t->nbytes);CHECK(f.read_raw(*t,bytes.data(),err));CHECK(bytes==Bytes(16,2));
        Bytes row(t->row_bytes);CHECK(f.read_raw_row(*t,1,row.data(),err));CHECK(row==Bytes(8,2));
    }
    f.close();::unlink(two.c_str());
    CHECK(!f.open(one,err) && err.find("missing or invalid shard")!=std::string::npos);
    write(two,shard(0,"second"));
    CHECK(!f.open(one,err) && err.find("split.no")!=std::string::npos);
    write(two,shard(1,"first"));
    CHECK(!f.open(one,err) && err.find("duplicate GGUF tensor")!=std::string::npos);
    write(two,shard(1,"second",3));
    CHECK(!f.open(one,err) && err.find("split.tensors.count")!=std::string::npos);
    write(one,shard(0,"first",3));
    CHECK(!f.open(one,err) && err.find("split.tensors.count mismatch")!=std::string::npos);
    write(one,shard(0,"first"));
    write(two,shard(1,"second",2,32));
    CHECK(!f.open(one,err) && err.find("alignment")!=std::string::npos);
    const std::string dir=prefix+"-dir";::mkdir(dir.c_str(),0700);
    const std::string a=dir+"/model-00001-of-00002.gguf";
    const std::string b=dir+"/model-00002-of-00002.gguf";
    write(a,shard(0,"first"));write(b,shard(1,"second"));
    write(dir+"/mmproj-vision.gguf",shard(0,"projector"));
    write(dir+"/imatrix.gguf",shard(0,"imatrix"));
    std::string selected;
    CHECK(gguf_model_path(dir,selected,err) && selected==a);
    CHECK(gguf_model_path(b,selected,err) && selected==b);
    write(dir+"/other.gguf",shard(0,"other"));
    CHECK(!gguf_model_path(dir,selected,err) && err.find("multiple GGUF models")!=std::string::npos);
    for (const auto& p:{one,two,a,b,dir+"/mmproj-vision.gguf",dir+"/imatrix.gguf",dir+"/other.gguf"})::unlink(p.c_str());
    ::rmdir(dir.c_str());
}
static uint32_t get32(const Bytes& b,size_t i) {return uint32_t(b[i])|(uint32_t(b[i+1])<<8)|(uint32_t(b[i+2])<<16)|(uint32_t(b[i+3])<<24);}
static void golden_tests() {
    {Bytes raw;put(raw,0x3f800000,4);put(raw,0xbf000000,4);
        float out[2]{};std::string err;
        CHECK(gguf_dequantize(0,raw.data(),2,out,err) && out[0]==1.0f && out[1]==-0.5f);
    }
    for(const char* name:{"F32","F16","BF16","F64","I8","I16","I32","I64",
            "Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K","Q4_K",
            "Q5_K","Q6_K","Q8_K","IQ1_S","IQ1_M","IQ2_XXS","IQ2_XS","IQ2_S",
            "IQ3_XXS","IQ3_S","IQ4_NL","IQ4_XS","TQ1_0","TQ2_0","MXFP4",
            "NVFP4","Q1_0","Q2_0"}) {
        std::string path=std::string("tests/gguf/data/")+name+".bin";
        std::ifstream in(path,std::ios::binary);
        if(!in) {std::fprintf(stderr,"MISSING GGUF golden file: %s; run /tmp/gguf-golden-venv/bin/python tests/gguf/gen_golden.py\n",path.c_str());++failures;continue;}
        Bytes b((std::istreambuf_iterator<char>(in)),{});
        CHECK(b.size()>=16 && std::memcmp(b.data(),"GGLD",4)==0);if(b.size()<16)continue;
        uint32_t type=get32(b,4),n=get32(b,8),raw_size=get32(b,12);
        CHECK(b.size()==16ull+raw_size+4ull*n);if(b.size()!=16ull+raw_size+4ull*n)continue;
        std::vector<float> out(n);
        std::string err;CHECK(gguf_dequantize(type,b.data()+16,n,out.data(),err,"test.weight"));
        for(uint32_t i=0;i<n;++i) if(std::memcmp(&out[i],b.data()+16+raw_size+4ull*i,4)!=0) {
            float expected;std::memcpy(&expected,b.data()+16+raw_size+4ull*i,4);
            std::fprintf(stderr,"golden mismatch %s[%u]: got %.9g expected %.9g\n",name,i,out[i],expected);
            ++failures;break;
        }
    }
    float out[32]{};uint8_t bytes[34]{};std::string err;
    CHECK(!gguf_dequantize(99,bytes,32,out,err) && err.find("unknown")!=std::string::npos);
    CHECK(!gguf_dequantize(24,bytes,1,out,err,"tokenizer.ids") && err.find("tokenizer.ids")!=std::string::npos);
}
int main() {parse_tests();split_tests();golden_tests();std::printf("GGUF: %d assertions, %d failures\n",cases,failures);return failures?1:0;}
