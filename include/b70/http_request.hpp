#pragma once
#include "b70/json.hpp"
#include "b70/tokenizer.hpp"

namespace b70 {
struct CompletionRequest {
    std::vector<ChatMessage> messages;
    std::string prompt;
    int max_tokens=4096;
    bool stream=false;
};
inline CompletionRequest parse_completion_request(const std::string& body, bool chat) {
    auto j=Json::parse(body);
    if(j.kind!=Json::Kind::Object)throw std::invalid_argument("request must be a JSON object");
    CompletionRequest r;
    r.stream=j.bool_or("stream",false);
    if(const auto* n=j.find("max_completion_tokens"))r.max_tokens=n->integer();
    else r.max_tokens=j.int_or("max_tokens",4096);
    if(r.max_tokens<=0)throw std::invalid_argument("max_tokens must be positive");
    if(j.int_or("n",1)!=1)throw std::invalid_argument("only n=1 is supported");
    // The engine implements greedy target verification. Do not silently
    // accept sampling parameters which would change that distribution.
    for(auto field:{"temperature","top_p","presence_penalty","frequency_penalty"}) {
        const auto* v=j.find(field); if(!v)continue;
        const double allowed=std::string(field)=="top_p"?1:0;
        if(v->kind!=Json::Kind::Number || v->number!=allowed)
            throw std::invalid_argument(std::string("greedy decoding requires ")+field+"="+std::to_string(allowed));
    }
    for(auto field:{"tools","tool_choice","stop","logprobs","response_format"}) {
        const auto* v=j.find(field);
        if(v && v->kind!=Json::Kind::Null && !(v->kind==Json::Kind::Boolean&&!v->boolean))
            throw std::invalid_argument(std::string("unsupported request field: ")+field);
    }
    if(!chat){r.prompt=j.at("prompt").text();return r;}
    const auto& messages=j.at("messages");
    if(messages.kind!=Json::Kind::Array || messages.array.empty())
        throw std::invalid_argument("messages must be a nonempty array");
    for(const auto& m:messages.array) {
        const std::string role=m.at("role").text();
        if(role!="system"&&role!="user"&&role!="assistant")
            throw std::invalid_argument("unsupported message role: "+role);
        const auto& c=m.at("content");
        std::string content;
        if(c.kind==Json::Kind::String)content=c.string;
        else if(c.kind==Json::Kind::Array) {
            for(const auto& part:c.array) {
                if(part.at("type").text()!="text")
                    throw std::invalid_argument("only text message parts are supported");
                content+=part.at("text").text();
            }
        } else throw std::invalid_argument("message content must be a string or text array");
        if(m.find("tool_calls"))throw std::invalid_argument("tool calls are not supported");
        r.messages.push_back({role,content});
    }
    return r;
}

// Token-aware Harmony decoding. Literal 'assistant to=user' in answer text
// is not a protocol transition. Both HTTP modes use exactly this decoder.
template<class T=Tokenizer> class ResponseDecoder {
    const T& tk;
    bool harmony, header, reasoning=false;
    std::string header_text, pending;
public:
    explicit ResponseDecoder(const T& t, bool h):tk(t),harmony(h),header(h){}
    template<class Emit> bool push(int32_t id, Emit emit) {
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
