// Small hand-checked SentencePiece BPE fixture, independent of the HF golden.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/gguf.hpp"
#include "b70/tokenizer.hpp"
#include "gguf/test_writer.hpp"
#include <cstdio>
#include <fstream>
#include <unistd.h>
using namespace b70;
using namespace gguf_test_writer;
static int failed=0;
#define CHECK(x) do {if(!(x)){std::fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);++failed;}}while(0)
int main() {
    const std::string path="/tmp/grimoire-spm-test-"+std::to_string(getpid())+".gguf";
    const std::vector<std::string> tokens={"<unk>","h","\xE2\x96\x81","i",
        "\xE2\x96\x81h","\xE2\x96\x81hi","<0x7A>","<s>","</s>","<tag>"};
    const std::vector<KV> kv={kv_string("tokenizer.ggml.model","llama"),
        kv_string("tokenizer.ggml.pre","default"),
        kv_array_strings("tokenizer.ggml.tokens",tokens),
        kv_array_i32("tokenizer.ggml.token_type",{2,1,1,1,1,1,6,3,3,4}),
        kv_array_f32("tokenizer.ggml.scores",{-1,0,0,0,1,2,0,0,0,0}),
        {"tokenizer.ggml.unknown_token_id",4,num(0,4)},
        {"tokenizer.ggml.bos_token_id",4,num(7,4)},
        {"tokenizer.ggml.eos_token_id",4,num(8,4)},
        {"tokenizer.ggml.add_bos_token",7,num(1,1)},
        {"tokenizer.ggml.add_eos_token",7,num(1,1)}};
    auto write=[&](const std::vector<KV>& values){
        auto raw=build_file(values,{});std::ofstream o(path,std::ios::binary);
        o.write(reinterpret_cast<const char*>(raw.data()),raw.size());
    };
    write(kv);
    GGUFFile file;std::string err;Tokenizer tk;
    CHECK(file.open(path,err));CHECK(tk.load_gguf(file,err));
    CHECK(tk.encode("hi hi")==std::vector<int32_t>({5,5}));
    CHECK(tk.encode(" hi")==std::vector<int32_t>({2,5}));
    CHECK(tk.encode("hi hi",true)==std::vector<int32_t>({7,5,5,8}));
    CHECK(tk.encode("<tag>hi")==std::vector<int32_t>({9,5}));
    CHECK(tk.encode("z")==std::vector<int32_t>({2,6}));
    CHECK(tk.decode({5,5})=="hi hi");CHECK(tk.decode_one(6)=="z");
    CHECK(tk.decode({2,5})==" hi");
    CHECK(tk.bos()==7 && tk.eos()==8 && tk.add_bos_token() && tk.add_eos_token());
    file.close();
    auto bad=kv;bad[0]=kv_string("tokenizer.ggml.model","bert");write(bad);
    CHECK(file.open(path,err));CHECK(!tk.load_gguf(file,err) && err.find("model: bert")!=std::string::npos);
    file.close();bad[0]=kv_string("tokenizer.ggml.model","gpt2");
    bad[1]=kv_string("tokenizer.ggml.pre","gpt-4o");write(bad);
    CHECK(file.open(path,err));CHECK(!tk.load_gguf(file,err) && err.find("pre: gpt-4o")!=std::string::npos);
    file.close();::unlink(path.c_str());
    std::printf("GGUF synthetic SPM: %d failures\n",failed);return failed?1:0;
}
