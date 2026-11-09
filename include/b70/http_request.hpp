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

#pragma once
#include "b70/json.hpp"
#include "b70/sampling.hpp"
#include <algorithm>
#include <cctype>
#include "b70/tokenizer.hpp"

namespace b70 {
struct CompletionRequest {
    std::vector<ChatMessage> messages;
    std::string prompt;
    int max_tokens=4096;
    bool stream=false;
    std::vector<Json> tools;   // OpenAI function tools, kept for argument typing
    SamplingOptions sampling;
    ChatOptions chat;          // template inputs: tools (tojson), thinking, effort
};
// A message's text: a string, an array of text parts, or null.
inline std::string message_text(const Json& c) {
    if(c.kind==Json::Kind::Null)return {};
    if(c.kind==Json::Kind::String)return c.string;
    if(c.kind==Json::Kind::Array) {
        std::string out;
        for(const auto& part:c.array) {
            if(part.at("type").text()!="text")
                throw std::invalid_argument("only text message parts are supported");
            out+=part.at("text").text();
        }
        return out;
    }
    throw std::invalid_argument("message content must be a string or text array");
}
// An assistant tool call from the history, in the order and form the
// template prints it: arguments (a JSON-encoded string or an object) as
// name -> value, strings as is and anything else as JSON.
inline ChatToolCall history_tool_call(const Json& t) {
    const Json& f=t.find("function")?t.at("function"):t;
    ChatToolCall tc;tc.name=f.at("name").text();
    if(const auto* a=f.find("arguments")) {
        Json args=a->kind==Json::Kind::String?(a->string.empty()?Json{}:Json::parse(a->string)):*a;
        if(args.kind==Json::Kind::Object)
            for(const auto& k:args.keys) {
                const Json& v=args.object.at(k);
                tc.args.push_back({k,v.kind==Json::Kind::String?v.string:v.dump()});
            }
        else if(args.kind!=Json::Kind::Null)
            throw std::invalid_argument("tool call arguments must be a JSON object");
    }
    return tc;
}
inline CompletionRequest parse_completion_request(const std::string& body, bool chat) {
    auto j=Json::parse(body);
    if(j.kind!=Json::Kind::Object)throw std::invalid_argument("request must be a JSON object");
    CompletionRequest r;
    r.stream=j.bool_or("stream",false);
    if(const auto* n=j.find("max_completion_tokens"))r.max_tokens=n->integer();
    else r.max_tokens=j.int_or("max_tokens",4096);
    if(r.max_tokens<=0)throw std::invalid_argument("max_tokens must be positive");
    if(j.int_or("n",1)!=1)throw std::invalid_argument("only n=1 is supported");
    auto number=[&](const char* name,double fallback) {
        const auto* v=j.find(name);if(!v)return fallback;
        if(v->kind!=Json::Kind::Number||!std::isfinite(v->number))
            throw std::invalid_argument(std::string(name)+" must be a finite number");
        return v->number;
    };
    r.sampling.temperature=float(number("temperature",0));
    r.sampling.top_p=float(number("top_p",1));
    const double k=number("top_k",20);
    if(k!=std::floor(k)||k<-1||k>INT32_MAX)throw std::invalid_argument("top_k must be an integer");
    r.sampling.top_k=int(k);
    if(const auto* seed=j.find("seed");seed&&seed->kind!=Json::Kind::Null) {
        const double v=number("seed",0);
        if(v!=std::floor(v)||v<0||v>9007199254740991.0)
            throw std::invalid_argument("seed must be an integer in [0,2^53-1]");
        r.sampling.seed=uint64_t(v);r.sampling.seeded=true;
    }
    r.sampling.validate();
    for(auto field:{"presence_penalty","frequency_penalty"})
        if(number(field,0)!=0)throw std::invalid_argument(std::string("unsupported nonzero ")+field);
    for(auto field:{"stop","logprobs","response_format"}) {
        const auto* v=j.find(field);
        if(v && v->kind!=Json::Kind::Null && !(v->kind==Json::Kind::Boolean&&!v->boolean))
            throw std::invalid_argument(std::string("unsupported request field: ")+field);
    }
    if(!chat){
        for(auto field:{"tools","tool_choice"}) {
            const auto* v=j.find(field);
            if(v && v->kind!=Json::Kind::Null)
                throw std::invalid_argument(std::string("unsupported request field: ")+field);
        }
        r.prompt=j.at("prompt").text();return r;
    }
    // Tools: OpenAI function definitions, printed into the prompt by the
    // model's own template.  tool_choice "auto" (default) or "none"; forcing
    // a call would need constrained decoding, which GRIMOIRE does not do.
    if(const auto* t=j.find("tools"); t && t->kind!=Json::Kind::Null) {
        if(t->kind!=Json::Kind::Array)throw std::invalid_argument("tools must be an array");
        for(const auto& tool:t->array) {
            if(tool.kind!=Json::Kind::Object || tool.find("type")==nullptr ||
               tool.at("type").text()!="function" || !tool.find("function") ||
               tool.at("function").at("name").kind!=Json::Kind::String)
                throw std::invalid_argument("each tool must be {\"type\": \"function\", \"function\": {\"name\": ...}}");
            r.tools.push_back(tool);
        }
    }
    if(const auto* c=j.find("tool_choice"); c && c->kind!=Json::Kind::Null) {
        const std::string v=c->kind==Json::Kind::String?c->string:"";
        if(v=="none")r.tools.clear();
        else if(v!="auto")throw std::invalid_argument("tool_choice must be \"auto\" or \"none\"");
    }
    for(const auto& tool:r.tools)r.chat.tools.push_back(tool.dump());
    if(const auto* e=j.find("reasoning_effort"); e && e->kind!=Json::Kind::Null)r.chat.reasoning_effort=e->text();
    if(const auto* k=j.find("chat_template_kwargs"); k && k->kind==Json::Kind::Object) {
        if(const auto* t=k->find("enable_thinking"); t && t->kind==Json::Kind::Boolean)r.chat.enable_thinking=t->boolean;
        if(const auto* e=k->find("reasoning_effort"); e && e->kind==Json::Kind::String)r.chat.reasoning_effort=e->string;
    }
    const auto& messages=j.at("messages");
    if(messages.kind!=Json::Kind::Array || messages.array.empty())
        throw std::invalid_argument("messages must be a nonempty array");
    for(const auto& m:messages.array) {
        ChatMessage msg;
        msg.role=m.at("role").text();
        if(msg.role!="system"&&msg.role!="user"&&msg.role!="assistant"&&msg.role!="tool")
            throw std::invalid_argument("unsupported message role: "+msg.role);
        const auto* c=m.find("content");
        msg.content=c?message_text(*c):std::string();
        if(msg.role=="assistant") {
            if(const auto* rc=m.find("reasoning_content"); rc && rc->kind==Json::Kind::String) {
                msg.reasoning=rc->string;msg.has_reasoning=true;
            }
            if(const auto* tc=m.find("tool_calls"); tc && tc->kind==Json::Kind::Array)
                for(const auto& t:tc->array)msg.tool_calls.push_back(history_tool_call(t));
        } else if(m.find("tool_calls") && m.at("tool_calls").kind!=Json::Kind::Null)
            throw std::invalid_argument("only assistant messages carry tool_calls");
        r.messages.push_back(std::move(msg));
    }
    return r;
}

// Qwen3.5-family XML tool calls in generated text -> OpenAI tool_calls:
//   <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
// Each value is typed by the tool's JSON schema (string as a string; integer,
// number, boolean; object / array parsed as JSON); a parameter the schema does
// not list stays a string.  arguments is the JSON object text.
struct ParsedToolCall { std::string name, arguments; };
inline std::string tool_param_json(const std::string& value, const Json* schema) {
    std::string type;
    if(schema && schema->kind==Json::Kind::Object) {
        if(const auto* t=schema->find("type")) {
            if(t->kind==Json::Kind::String)type=t->string;
            else if(t->kind==Json::Kind::Array)
                for(const auto& x:t->array)
                    if(x.kind==Json::Kind::String && x.string!="null"){type=x.string;break;}
        }
    }
    const std::string v=value;
    auto as_json=[&](Json::Kind want)->std::string {
        try {Json p=Json::parse(v);if(p.kind==want)return p.dump();} catch(const std::invalid_argument&) {}
        return {};
    };
    if(type=="integer"||type=="number") {
        std::string out=as_json(Json::Kind::Number);
        if(!out.empty() && (type=="number"||out.find_first_of(".eE")==std::string::npos))return out;
    } else if(type=="boolean") {
        std::string l=v;for(auto& ch:l)ch=char(std::tolower(static_cast<unsigned char>(ch)));
        if(l=="true"||l=="false")return l;
    } else if(type=="object") {
        std::string out=as_json(Json::Kind::Object);if(!out.empty())return out;
    } else if(type=="array") {
        std::string out=as_json(Json::Kind::Array);if(!out.empty())return out;
    } else if(type=="null" && v=="null") return "null";
    return json_dump_string(v);
}
inline std::vector<ParsedToolCall> parse_xml_tool_calls(const std::string& text,
                                                        const std::vector<Json>& tools) {
    std::vector<ParsedToolCall> calls;
    const std::string open="<tool_call>", close="</tool_call>";
    auto trim=[](std::string x){const char* ws=" \t\r\n";const size_t b=x.find_first_not_of(ws);
        if(b==std::string::npos)return std::string();return x.substr(b,x.find_last_not_of(ws)-b+1);};
    for(size_t p=text.find(open);p!=std::string::npos;p=text.find(open,p)) {
        const size_t e=text.find(close,p+open.size());
        const std::string block=text.substr(p+open.size(),(e==std::string::npos?text.size():e)-p-open.size());
        p=e==std::string::npos?text.size():e+close.size();
        const size_t f=block.find("<function=");
        if(f==std::string::npos)return {};
        const size_t fe=block.find('>',f);
        if(fe==std::string::npos)return {};
        ParsedToolCall call;call.name=trim(block.substr(f+10,fe-f-10));
        const Json* props=nullptr;
        for(const auto& t:tools) {
            const Json& fn=t.at("function");
            if(fn.at("name").string!=call.name)continue;
            if(const auto* ps=fn.find("parameters"))props=ps->find("properties");
            break;
        }
        call.arguments="{";
        bool first=true;
        for(size_t q=block.find("<parameter=",fe);q!=std::string::npos;q=block.find("<parameter=",q)) {
            const size_t ne=block.find('>',q);
            if(ne==std::string::npos)return {};
            const std::string pname=trim(block.substr(q+11,ne-q-11));
            size_t ve=block.find("</parameter>",ne+1);
            const size_t next=std::min(block.find("<parameter=",ne+1),block.find("</function>",ne+1));
            const bool closed=ve!=std::string::npos && ve<=next;
            const size_t vend=closed?ve:(next==std::string::npos?block.size():next);
            std::string value=block.substr(ne+1,vend-ne-1);
            if(!value.empty()&&value.front()=='\n')value.erase(0,1);
            if(!value.empty()&&value.back()=='\n')value.pop_back();
            const Json* schema=props?props->find(pname):nullptr;
            call.arguments+=(first?"":", ")+json_dump_string(pname)+": "+tool_param_json(value,schema);
            first=false;
            q=closed?ve+12:vend;
        }
        call.arguments+="}";
        calls.push_back(std::move(call));
    }
    return calls;
}

// Token-aware Harmony decoding. Literal 'assistant to=user' in answer text
// is not a protocol transition. Both HTTP modes use exactly this decoder.
// Thinking models (ChatML template ending in "<think>\n"): the answer starts
// in reasoning and </think> switches to content, the split OpenAI clients
// expect (reasoning_content / content).  The tags themselves are not emitted,
// and the blank lines the template puts after </think> are dropped.
template<class T=Tokenizer> class ResponseDecoder {
    const T& tk;
    bool harmony, header, reasoning=false, strip_lead=false, tool_mode=false;
    int32_t think_open=-1, think_close=-1, tool_open=-1;
    std::string header_text, pending, tool_text;
public:
    explicit ResponseDecoder(const T& t, bool h, bool think=false):tk(t),harmony(h),header(h){
        if(think && !h) {
            think_open=tk.special_id("<think>"); think_close=tk.special_id("</think>");
            if(think_close<0) {   // K2-Horizon: <ifm|think> ... </ifm|think>
                think_open=tk.special_id("<ifm|think>"); think_close=tk.special_id("</ifm|think>");
            }
            reasoning=think_close>=0;
        }
    }
    // With tools in the request, everything from the first <tool_call> of the
    // answer (not of the reasoning) on is held back for parse_xml_tool_calls.
    void enable_tools() { tool_open=tk.special_id("<tool_call>"); }
    const std::string& tool_call_text() const { return tool_text; }
    template<class Emit> bool push(int32_t id, Emit emit) {
        if(tool_mode){tool_text+=tk.decode_one(id);return true;}
        if(tool_open>=0 && id==tool_open && !reasoning) {
            if(!finish(emit))return false;
            tool_mode=true;tool_text="<tool_call>";return true;
        }
        if(think_close>=0 && (id==think_close || id==think_open)) {
            if(!finish(emit))return false;
            if(id==think_close){reasoning=false;strip_lead=true;}
            return true;
        }
        if(strip_lead) {
            auto out=[&](const std::string& piece,bool r){
                if(!strip_lead)return emit(piece,r);
                const size_t k=piece.find_first_not_of("\r\n");
                if(k==std::string::npos)return true;
                strip_lead=false;return emit(piece.substr(k),r);
            };
            return push_text(id,out);
        }
        return push_text(id,emit);
    }
    template<class Emit> bool push_text(int32_t id, Emit emit) {
        if(harmony && id==tk.special_id("<|start|>")) {
            if(!finish(emit))return false;
            header=true;header_text.clear();return true;
        }
        if(harmony && header) {
            if(id==tk.special_id("<|message|>")) {
                reasoning=header_text.find("to=self")!=std::string::npos;
                header=false;header_text.clear();
            } else header_text+=tk.decode_one(id);
            return true;
        }
        pending+=tk.decode_one(id);
        size_t n=0;
        try {n=utf8_prefix(pending);}
        catch(const std::invalid_argument&) {
            // Byte-level tokenizers can generate malformed bytes. Match text
            // decoding with replacement rather than emitting invalid JSON.
            std::string clean;
            for(size_t p=0;p<pending.size();) {
                size_t len=1;unsigned c=static_cast<unsigned char>(pending[p]);
                if(c>=0xc2&&c<=0xdf)len=2;else if(c>=0xe0&&c<=0xef)len=3;else if(c>=0xf0&&c<=0xf4)len=4;
                const auto part=std::string_view(pending).substr(p,std::min(len,pending.size()-p));
                try {if(utf8_prefix(part)!=part.size())break;clean+=part;p+=part.size();}
                catch(const std::invalid_argument&){clean+="\xef\xbf\xbd";++p;}
                n=p;
            }
            pending.erase(0,n);return clean.empty()||emit(clean,reasoning);
        }
        if(!n)return true;
        const auto piece=pending.substr(0,n);pending.erase(0,n);
        return emit(piece,reasoning);
    }
    template<class Emit> bool finish(Emit emit) {
        if(pending.empty())return true;
        pending.clear();return emit("\xef\xbf\xbd",reasoning);
    }
};
} // namespace b70
