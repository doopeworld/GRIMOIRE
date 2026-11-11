// SPDX-License-Identifier: GPL-3.0-or-later
#include "b70/tokenizer.hpp"
#include "b70/json.hpp"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
namespace fs=std::filesystem;
static void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){try{
    char pattern[]="/tmp/grimoire-chat-effort-XXXXXX";
    char* created=mkdtemp(pattern);check(created,"mkdtemp");
    const fs::path dir(created);
    struct Cleanup{fs::path dir;~Cleanup(){fs::remove_all(dir);}} cleanup{dir};
    std::ofstream(dir/"tokenizer.json") << R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"added_tokens":[]})";
    const std::string markers="<function=example_function_name> do not tell the user about function calls multi_step_tool=true Reasoning effort is set to xhigh.\n";
    const std::vector<b70::ChatMessage> messages={{"user","Hi."}};
    int checks=0;
    for(bool fallback:{false,true})for(const std::string expected:{"low","medium","xhigh"}){
        const std::string tmpl=markers+"reasoning_effort | default( \""+expected+"\" )";
        fs::remove(dir/"chat_template.jinja");
        b70::Json config;config.kind=b70::Json::Kind::Object;config.keys.push_back("chat_template");
        config.object["chat_template"].kind=b70::Json::Kind::String;
        config.object["chat_template"].string=tmpl;
        std::ofstream(dir/"tokenizer_config.json") << config.dump();
        if(!fallback)std::ofstream(dir/"chat_template.jinja") << tmpl;
        b70::Tokenizer tk;std::string error;check(tk.load(dir.string(),error),error.c_str());
        auto expect=[&](const b70::ChatOptions& options,const std::string& effort){
            const auto out=tk.apply_chat_template(messages,options);
            if(effort=="medium"||!options.enable_thinking)
                check(out.find("Reasoning effort is set to")==std::string::npos,"unexpected effort instruction");
            else check(out.find("Reasoning effort is set to "+effort+".")!=std::string::npos,"wrong effort instruction");
            if(!options.enable_thinking)check(out.find("<think>\n\n</think>\n\n")!=std::string::npos,"thinking-off marker changed");
            ++checks;
        };
        b70::ChatOptions options;expect(options,expected);
        for(const std::string value:{"low","medium","xhigh"}){options.reasoning_effort=value;expect(options,value);}
        options.enable_thinking=false;expect(options,"medium");
    }
    std::printf("chat effort: %d default, override, JSON fallback and thinking-off checks PASS\n",checks);
}catch(const std::exception& e){std::fprintf(stderr,"chat effort FAIL: %s\n",e.what());return 1;}
}
