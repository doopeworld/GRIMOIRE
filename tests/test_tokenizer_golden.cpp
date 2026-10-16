// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/json.hpp"
#include "b70/tokenizer.hpp"
#include "b70/gguf.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

int main() {
    // Defaults point at the reference material under .ref/ (tokenizer.json is
    // the HF reference for this vocabulary); each is overridable by env var.
    const char* env = std::getenv("GRIMOIRE_TOKENIZER_DIR");
    std::string dir = env && *env ? env : ".ref";
    if (!(env && *env) && !std::ifstream(dir + "/tokenizer.json", std::ios::binary).good() &&
        std::ifstream("/tmp/toktest/tokenizer.json", std::ios::binary).good())
        dir = "/tmp/toktest";
    std::ifstream model(dir + "/tokenizer.json", std::ios::binary);
    if (!model) {
        std::printf("SKIP tokenizer HF goldens: %s/tokenizer.json absent\n", dir.c_str());
        return 0;
    }
    model.close();
    b70::Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(dir, error)) {
        std::fprintf(stderr, "tokenizer load failed: %s\n", error.c_str());
        return 1;
    }
    b70::Tokenizer gguf_tokenizer;
    b70::GGUFFile gguf;
    const char* header_env=std::getenv("GRIMOIRE_GGUF_HEADER");
    const std::string header=header_env&&*header_env ? header_env : ".ref/q38-UD-Q4_K_XL.bin";
    const bool have_gguf=std::ifstream(header,std::ios::binary).good();
    if (!have_gguf)
        std::printf("SKIP GGUF-built tokenizer checks: %s absent (GRIMOIRE_GGUF_HEADER)\n",header.c_str());
    if (have_gguf) {
        if (!gguf.open(header,error,true) || !gguf_tokenizer.load_gguf(gguf,error)) {
            std::fprintf(stderr,"GGUF tokenizer load failed: %s\n",error.c_str());return 1;
        }
        const b70::GGUFValue* v=nullptr;
        if (!gguf.get_array("tokenizer.ggml.tokens",v,error)) return 1;
        const auto& tokens=std::get<std::vector<std::string>>(v->data);
        bool same=tokens.size()>=tokenizer.vocab_size();
        for (size_t i=0;same && i<tokenizer.vocab_size();++i)
            same=tokens[i]==tokenizer.raw_token(i);
        if (!same) {
            std::fprintf(stderr,"GGUF vocabulary differs from Ornith tokenizer.json\n");return 1;
        }
        if (!gguf.get_array("tokenizer.ggml.token_type",v,error)) return 1;
        const auto& types=std::get<std::vector<int32_t>>(v->data);
        for (size_t i=tokenizer.vocab_size();i<tokens.size();++i)
            if (types[i]!=3 && types[i]!=5) {
                std::fprintf(stderr,"GGUF extra token %zu is not CONTROL/UNUSED\n",i);return 1;
            }
        if (gguf_tokenizer.merge_count()!=tokenizer.merge_count()) {
            std::fprintf(stderr,"GGUF merge count differs from Ornith tokenizer.json\n");return 1;
        }
        if (!gguf.get_array("tokenizer.ggml.merges",v,error)) return 1;
        const auto& merges=std::get<std::vector<std::string>>(v->data);
        for (size_t i=0;i<merges.size();++i)
            if (tokenizer.merge_rank(merges[i])!=int32_t(i)) {
                std::fprintf(stderr,"GGUF merge differs from Ornith tokenizer.json at %zu\n",i);return 1;
            }
        size_t bad_decode=0;
        for (size_t i=0;i<tokenizer.vocab_size();++i)
            if (gguf_tokenizer.decode_one(int32_t(i))!=tokenizer.decode_one(int32_t(i))) {
                if (bad_decode<30)std::fprintf(stderr,"GGUF decode differs at id %zu (%s), type %d: GGUF %s, HF %s\n",
                    i,tokens[i].c_str(),types[i],
                    b70::json_dump_string(gguf_tokenizer.decode_one(int32_t(i))).c_str(),
                    b70::json_dump_string(tokenizer.decode_one(int32_t(i))).c_str());
                ++bad_decode;
            }
        if (bad_decode) {std::fprintf(stderr,"GGUF decode mismatches: %zu\n",bad_decode);return 1;}
        if (gguf_tokenizer.supports_tools()!=tokenizer.supports_tools() ||
            gguf_tokenizer.apply_chat_template("hello")!=tokenizer.apply_chat_template("hello")) {
            std::fprintf(stderr,"GGUF chat template family differs from Ornith sidecar\n");return 1;
        }
        std::printf("PASS GGUF/Ornith vocabulary: %zu common ids, %zu merges\n",
                    tokenizer.vocab_size(),tokenizer.merge_count());
    }
    std::ifstream f("tests/tokenizer_golden/ornith_hf_golden.json", std::ios::binary);
    if (!f) { std::fprintf(stderr, "HF golden fixture absent\n"); return 1; }
    const std::string raw((std::istreambuf_iterator<char>(f)), {});
    const b70::Json cases = b70::Json::parse(raw).at("cases");
    for (const auto& name : cases.keys) {
        const b70::Json& test = cases.at(name);
        const std::string input = test.at("text").text();
        std::vector<int32_t> expected;
        for (const b70::Json& id : test.at("ids").array) expected.push_back(id.integer());
        const std::vector<int32_t> actual = tokenizer.encode(input);
        if (have_gguf && gguf_tokenizer.encode(input)!=expected) {
            std::fprintf(stderr,"FAIL GGUF HF golden %s\n",name.c_str());return 1;
        }
        size_t i = 0;
        while (i < expected.size() && i < actual.size() && expected[i] == actual[i]) ++i;
        if (i != expected.size() || i != actual.size()) {
            const std::vector<int32_t> prefix(expected.begin(), expected.begin() + std::min(i, expected.size()));
            const size_t byte = tokenizer.decode(prefix).size();
            const size_t left = byte > 40 ? byte - 40 : 0;
            const std::string context = input.substr(left, 80);
            std::fprintf(stderr,
                "FAIL %s: first divergence at index %zu, HF %d, GRIMOIRE %d "
                "(HF %zu tokens, GRIMOIRE %zu); text near byte %zu: %s\n",
                name.c_str(), i, i < expected.size() ? expected[i] : -1,
                i < actual.size() ? actual[i] : -1, expected.size(), actual.size(),
                byte, b70::json_dump_string(context).c_str());
            return 1;
        }
        std::printf("PASS %s: %zu HF ids identical%s\n", name.c_str(), actual.size(),
                    have_gguf ? " (GGUF-built tokenizer too)" : "");
    }
    // The tokenizer.json path is verified equal to HF on wiki.test.raw
    // (297,193 tokens); the GGUF-built tokenizer must give exactly the same ids.
    if (have_gguf) {
        const char* wiki_env=std::getenv("GRIMOIRE_WIKI_RAW");
        const std::string wiki_path=wiki_env&&*wiki_env ? wiki_env : ".ref/wiki.test.raw";
        std::ifstream wf(wiki_path, std::ios::binary);
        if (!wf) {
            std::printf("SKIP GGUF vs tokenizer.json on wiki: %s absent (GRIMOIRE_WIKI_RAW)\n",wiki_path.c_str());
        } else {
            const std::string wiki((std::istreambuf_iterator<char>(wf)), {});
            const std::vector<int32_t> a = tokenizer.encode(wiki), g = gguf_tokenizer.encode(wiki);
            size_t i = 0;
            while (i < a.size() && i < g.size() && a[i] == g[i]) ++i;
            if (i != a.size() || i != g.size()) {
                std::fprintf(stderr,"FAIL GGUF vs tokenizer.json on %s: first divergence at %zu "
                             "(json %zu ids, GGUF %zu ids)\n",wiki_path.c_str(),i,a.size(),g.size());
                return 1;
            }
            std::printf("PASS GGUF-built tokenizer == tokenizer.json on %s: %zu ids\n",
                        wiki_path.c_str(), a.size());
            if (!(wiki_env&&*wiki_env) && a.size()!=297193) {
                std::fprintf(stderr,"FAIL expected 297193 ids on default wiki.test.raw, got %zu\n",a.size());
                return 1;
            }
        }
    }
    return 0;
}
