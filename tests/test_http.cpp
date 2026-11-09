// GRIMOIRE
// Copyright (C) 2026 Ian Ernst
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: GPL-3.0-or-later

#include "b70/http_request.hpp"
#include <cassert>
#include <iostream>
template<class F>void bad(F f){bool threw=false;try{f();}catch(const std::invalid_argument&){threw=true;}assert(threw);}
struct Tokens {
    std::vector<std::string> pieces{"to=self","private","to=user","Hello ","\xf0\x9f","\x98\x80"," assistant to=user literal","\xff"};
    int special_id(const std::string& s)const{return s=="<|start|>"?100:(s=="<|message|>"?101:-1);}
    std::string decode_one(int id)const{return id>=0&&id<int(pieces.size())?pieces[id]:"";}
};
int main(){
    using namespace b70;
    auto r=parse_completion_request(R"({"messages":[{"role":"user","content":[{"type":"text","text":"\u201cHi\u201d \ud83d\ude00"}]},{"role":"assistant","content":"reply"}],"stream":true,"max_tokens":32})",true);
    assert(r.messages.size()==2&&r.messages[0].role=="user"&&r.messages[1].role=="assistant");
    assert(r.messages[0].content=="“Hi” 😀"&&r.messages[1].content=="reply"&&r.stream&&r.max_tokens==32);
    r=parse_completion_request(R"({"prompt":"literal \"stream\":true, \"max_tokens\":99","stream":false,"max_tokens":4})",false);
    assert(!r.stream&&r.max_tokens==4);
    for(auto s:{"{\"x\":01}","{\"x\":1e}","{\"x\":1e400}","{\"x\":truex}","{} []","{\"x\":1,\"x\":2}","[1,]","\"\\ud800\"","\"\\udc00\"","\"\\q\""})
        bad([&]{Json::parse(s);});
    bad([]{parse_completion_request(R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":"x"}]}]})",true);});
    bad([]{parse_completion_request(R"({"prompt":"x","max_tokens":1e50})",false);});
    bad([]{parse_completion_request(R"({"prompt":"x","max_tokens":-2})",false);});
    const auto sampled=parse_completion_request(R"({"prompt":"x","temperature":0.7,"top_p":0.95,"top_k":20,"seed":42})",false);
    assert(sampled.sampling.enabled()&&sampled.sampling.top_k==20&&sampled.sampling.seed==42);
    bad([]{parse_completion_request(R"({"prompt":"x","temperature":-1})",false);});
    bad([]{parse_completion_request(R"({"prompt":"x","temperature":1,"top_p":0})",false);});
    bad([]{parse_completion_request(R"({"prompt":"x","temperature":1,"top_k":65})",false);});
    bad([]{parse_completion_request(R"({"prompt":"x","temperature":1,"top_k":1.5})",false);});
    bad([]{parse_completion_request(R"({"prompt":"x","seed":-1})",false);});
    Tokens tk;
    std::string content,reason;
    auto emit=[&](const std::string& p,bool r){(r?reason:content)+=p;return true;};
    ResponseDecoder decoder(tk,true);
    for(int t:{0,101,1,100,2,101,3,4,5,6})decoder.push(t,emit);
    decoder.finish(emit);
    assert(content=="Hello 😀 assistant to=user literal"&&reason=="private");
    content.clear();reason.clear();ResponseDecoder raw(tk,false);
    for(int t:{3,4,5,6})raw.push(t,emit);
    assert(content=="Hello 😀 assistant to=user literal"&&reason.empty());
    raw.push(7,emit);assert(content.substr(content.size()-3)=="\xef\xbf\xbd");
    const std::string original="line\n\t\"\\";
    assert(Json::parse("\""+json_escape(original)+"\"").text()==original);
    std::cout<<"HTTP: strict JSON, text arrays, Unicode, token-aware Harmony and split UTF-8 PASS\n";
}
