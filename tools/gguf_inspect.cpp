// GRIMOIRE -- inspect a GGUF v3 header. SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf.hpp"
#include "b70/gguf_decode.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <type_traits>

using namespace b70;
namespace {
std::string json_quote(const std::string& s) {
    std::ostringstream o;o<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') o<<'\\'<<char(c);
        else if(c=='\n') o<<"\\n";
        else if(c=='\r') o<<"\\r";
        else if(c=='\t') o<<"\\t";
        else if(c<32) {o<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<int(c)<<std::dec;}
        else o<<char(c);
    }
    o<<'"';return o.str();
}
std::string value_text(const std::string& key,const GGUFValue& v) {
    if(v.type==9) {
        const auto count=v.array_size();
        if(count>16) return "array<"+std::to_string(v.element_type)+">["+std::to_string(count)+"]";
        return std::visit([](const auto& x)->std::string {
            using X=std::decay_t<decltype(x)>;
            if constexpr(std::is_same_v<X,std::vector<uint8_t>> || std::is_same_v<X,std::vector<int8_t>> ||
                std::is_same_v<X,std::vector<uint16_t>> || std::is_same_v<X,std::vector<int16_t>> ||
                std::is_same_v<X,std::vector<uint32_t>> || std::is_same_v<X,std::vector<int32_t>> ||
                std::is_same_v<X,std::vector<float>> || std::is_same_v<X,std::vector<std::string>> ||
                std::is_same_v<X,std::vector<uint64_t>> || std::is_same_v<X,std::vector<int64_t>> ||
                std::is_same_v<X,std::vector<double>>) {
                std::ostringstream o;o<<'[';bool first=true;
                for(const auto& item:x) {if(!first)o<<", ";first=false;
                    if constexpr(std::is_same_v<typename X::value_type,std::string>)o<<json_quote(item);
                    else o<<+item;}
                o<<']';return o.str();
            } else return std::string{};
        },v.data);
    }
    if(v.type==8) {
        const auto& s=std::get<std::string>(v.data);
        if(key.find("chat_template")!=std::string::npos) return "string["+std::to_string(s.size())+"]";
        return s.size()>160?"string["+std::to_string(s.size())+"]":json_quote(s);
    }
    return std::visit([](const auto& x)->std::string {
        using X=std::decay_t<decltype(x)>;
        if constexpr(std::is_arithmetic_v<X>) {std::ostringstream o;if constexpr(std::is_same_v<X,bool>) o<<(x?"true":"false");else o<<+x;return o.str();}
        else return "";
    },v.data);
}
std::string json_value(const std::string& key,const GGUFValue& v) {
    if(v.type==8) {
        const auto& s=std::get<std::string>(v.data);
        if(key.find("chat_template")!=std::string::npos)
            return "{\"length\": "+std::to_string(s.size())+"}";
        return json_quote(s);
    }
    if(v.type==9 && v.array_size()>16)
        return "{\"element_type\": "+std::to_string(v.element_type)+
               ", \"count\": "+std::to_string(v.array_size())+"}";
    return std::visit([&v](const auto& x)->std::string {
        using X=std::decay_t<decltype(x)>;
        std::ostringstream o;
        if constexpr(std::is_same_v<X,std::string>) return json_quote(x);
        else if constexpr(std::is_arithmetic_v<X>) {
            if constexpr(std::is_same_v<X,bool>) o<<(x?"true":"false");
            else if constexpr(std::is_floating_point_v<X>) {if(!std::isfinite(x))return "null";o<<std::setprecision(17)<<x;}
            else o<<+x;
        } else {
            o<<'[';bool first=true;
            for(const auto& item:x) {
                if(!first)o<<", ";first=false;
                if constexpr(std::is_same_v<typename X::value_type,std::string>)o<<json_quote(item);
                else if constexpr(std::is_floating_point_v<typename X::value_type>) {if(std::isfinite(item))o<<std::setprecision(17)<<item;else o<<"null";}
                else if constexpr(std::is_same_v<typename X::value_type,uint8_t>) {
                    if(v.element_type==7)o<<(item?"true":"false");else o<<unsigned(item);
                }
                else o<<+item;
            }
            o<<']';
        }
        return o.str();
    },v.data);
}
std::string role_of(const std::string& name) {
    if(name.rfind("blk.",0)!=0) return name;
    size_t i=4;while(i<name.size() && name[i]>='0' && name[i]<='9') ++i;
    if(i==4 || i==name.size() || name[i]!='.') return name;
    return name.substr(i+1);
}
std::string shape(const GGUFTensor& t) {
    std::ostringstream o;o<<'[';
    for(uint32_t i=0;i<t.n_dims;++i) {if(i)o<<',';o<<t.ne[i];}
    o<<']';return o.str();
}
}
int main(int argc,char** argv) {
    if(argc<2) {std::cerr<<"usage: gguf_inspect <file.gguf> [--header-only] [--json out.json] [--decode-check name]\n";return 2;}
    std::string path=argv[1],json_path,decode_name;bool header_only=false;
    for(int i=2;i<argc;++i) {
        std::string a=argv[i];
        if(a=="--header-only") header_only=true;
        else if(a=="--json" && i+1<argc) json_path=argv[++i];
        else if(a=="--decode-check" && i+1<argc) decode_name=argv[++i];
        else {std::cerr<<"unknown or incomplete option: "<<a<<'\n';return 2;}
    }
    GGUFFile file;std::string err;
    if(!file.open(path,err,header_only)) {std::cerr<<err<<'\n';return 1;}
    std::cout<<"GGUF version: "<<file.version()<<'\n';
    if(header_only) std::cout<<"HEADER ONLY: tensor data range vs file size validation skipped\n";
    std::cout<<"Alignment: "<<file.alignment()<<"  data_start: "<<file.data_start()<<'\n';
    std::cout<<"Metadata ("<<file.metadata().size()<<"):\n";
    for(const auto& kv:file.metadata()) std::cout<<"  "<<kv.first<<" = "<<value_text(kv.first,kv.second)<<'\n';
    std::map<std::string,std::pair<uint64_t,uint64_t>> types;
    struct Role {std::map<std::string,uint64_t> types;std::string example;};
    std::map<std::string,Role> roles;
    for(const auto& t:file.tensors()) {
        std::string ty=gguf_type_info(t.type)->name;
        auto& st=types[ty];++st.first;st.second+=t.nbytes;
        auto& role=roles[role_of(t.name)];++role.types[ty];if(role.example.empty()) role.example=shape(t);
    }
    std::cout<<"Tensors: "<<file.tensors().size()<<'\n';
    for(const auto& kv:types) std::cout<<"  "<<kv.first<<": "<<kv.second.first<<" tensors, "
        <<std::fixed<<std::setprecision(3)<<double(kv.second.second)/1073741824.0<<" GiB\n";
    std::cout<<"Roles ("<<roles.size()<<"):\n";
    for(const auto& kv:roles) {
        std::cout<<"  "<<kv.first<<" "<<kv.second.example<<" {";
        bool first=true;for(const auto& tc:kv.second.types) {if(!first)std::cout<<", ";first=false;std::cout<<tc.first<<":"<<tc.second;}
        std::cout<<"}\n";
    }
    if(!decode_name.empty()) {
        auto* t=file.find(decode_name);
        if(!t) {std::cerr<<"tensor not found: "<<decode_name<<'\n';return 1;}
        if(t->row_bytes>SIZE_MAX || t->ne[0]>SIZE_MAX/sizeof(float)) {std::cerr<<"row too large\n";return 1;}
        std::vector<uint8_t> raw(static_cast<size_t>(t->row_bytes));
        if(!file.read_raw_row(*t,0,raw.data(),err)) {std::cerr<<err<<'\n';return 1;}
        std::vector<float> out(static_cast<size_t>(t->ne[0]));
        if(!gguf_dequantize(t->type,raw.data(),t->ne[0],out.data(),err)) {std::cerr<<err<<'\n';return 1;}
        float lo=std::numeric_limits<float>::infinity(),hi=-lo;double sum=0;uint64_t finite=0;
        for(float v:out) if(std::isfinite(v)) {lo=std::min(lo,v);hi=std::max(hi,v);sum+=v;++finite;}
        std::cout<<"Decode "<<decode_name<<" first row: min="<<lo<<" max="<<hi
                 <<" mean="<<(finite?sum/double(finite):0)<<" finite="<<finite<<"/"<<out.size()<<'\n';
    }
    if(!json_path.empty()) {
        std::ofstream o(json_path);
        if(!o) {std::cerr<<"cannot write JSON: "<<json_path<<'\n';return 1;}
        o<<"{\n  \"version\": "<<file.version()<<", \"header_only\": "<<(header_only?"true":"false")
         <<", \"alignment\": "<<file.alignment()<<", \"data_start\": "<<file.data_start()
         <<", \"tensor_count\": "<<file.tensors().size()<<",\n  \"metadata\": {\n";
        bool first=true;for(const auto& kv:file.metadata()) {if(!first)o<<",\n";first=false;o<<"    "<<json_quote(kv.first)<<": "<<json_value(kv.first,kv.second);}
        o<<"\n  },\n  \"types\": {\n";first=true;
        for(const auto& kv:types) {if(!first)o<<",\n";first=false;o<<"    "<<json_quote(kv.first)<<": {\"count\": "<<kv.second.first<<", \"bytes\": "<<kv.second.second<<"}";}
        o<<"\n  },\n  \"roles\": {\n";first=true;
        for(const auto& kv:roles) {if(!first)o<<",\n";first=false;o<<"    "<<json_quote(kv.first)<<": {\"shape\": "<<json_quote(kv.second.example)<<", \"types\": {";
            bool inner=true;for(const auto& tc:kv.second.types) {if(!inner)o<<", ";inner=false;o<<json_quote(tc.first)<<": "<<tc.second;}o<<"}}";}
        o<<"\n  }\n}\n";
        if(!o) {std::cerr<<"JSON write failed: "<<json_path<<'\n';return 1;}
    }
    return 0;
}
