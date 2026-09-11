#pragma once
// Strict JSON for configuration and HTTP requests. No framework dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace b70 {
struct Json {
    enum class Kind { Null, Boolean, Number, String, Array, Object } kind = Kind::Null;
    bool boolean = false;
    double number = 0;
    std::string string;
    std::vector<Json> array;
    std::map<std::string, Json> object;

    const Json* find(const std::string& key) const {
        if (kind != Kind::Object) return nullptr;
        auto it = object.find(key);
        return it == object.end() ? nullptr : &it->second;
    }
    const Json& at(const std::string& key) const {
        const auto* v = find(key);
        if (!v) throw std::invalid_argument("missing JSON field: " + key);
        return *v;
    }
    int integer() const {
        if (kind != Kind::Number || !std::isfinite(number) ||
            std::trunc(number) != number || number < std::numeric_limits<int>::min() ||
            number > std::numeric_limits<int>::max())
            throw std::invalid_argument("expected an integer in int32 range");
        return static_cast<int>(number);
    }
    std::string text() const {
        if (kind != Kind::String) throw std::invalid_argument("expected a JSON string");
        return string;
    }
    bool flag() const {
        if (kind != Kind::Boolean) throw std::invalid_argument("expected a JSON boolean");
        return boolean;
    }
    int int_or(const std::string& k, int d) const { auto* v=find(k); return v?v->integer():d; }
    bool bool_or(const std::string& k, bool d) const { auto* v=find(k); return v?v->flag():d; }
    std::string text_or(const std::string& k, const std::string& d) const {
        auto* v=find(k); return v?v->text():d;
    }
    static Json parse(std::string_view input);
};

inline void append_utf8(std::string& out, uint32_t c) {
    if (c < 0x80) out += char(c);
    else if (c < 0x800) { out += char(0xc0|(c>>6)); out += char(0x80|(c&63)); }
    else if (c < 0x10000) {
        out += char(0xe0|(c>>12)); out += char(0x80|((c>>6)&63)); out += char(0x80|(c&63));
    } else {
        out += char(0xf0|(c>>18)); out += char(0x80|((c>>12)&63));
        out += char(0x80|((c>>6)&63)); out += char(0x80|(c&63));
    }
}

// Returns the complete UTF-8 prefix; throws on invalid bytes, retaining a
// trailing incomplete codepoint for streaming token pieces.
inline size_t utf8_prefix(std::string_view s) {
    size_t p=0;
    while (p<s.size()) {
        unsigned c=static_cast<unsigned char>(s[p]);
        unsigned n=c<0x80?1:(c>=0xc2&&c<=0xdf?2:(c>=0xe0&&c<=0xef?3:(c>=0xf0&&c<=0xf4?4:0)));
        if (!n) throw std::invalid_argument("invalid UTF-8 lead byte");
        uint32_t cp=c & (n==1?127:(n==2?31:(n==3?15:7)));
        const size_t available=std::min(size_t(n),s.size()-p);
        for (size_t j=1;j<available;++j) {
            unsigned b=static_cast<unsigned char>(s[p+j]);
            if ((b&0xc0)!=0x80 || (j==1 && ((c==0xe0&&b<0xa0) ||
                (c==0xed&&b>=0xa0) || (c==0xf0&&b<0x90) || (c==0xf4&&b>=0x90))))
                throw std::invalid_argument("invalid UTF-8 continuation");
            cp=(cp<<6)|(b&63);
        }
        if (available<n) break;
        if (cp>0x10ffff || (cp>=0xd800&&cp<=0xdfff))
            throw std::invalid_argument("invalid Unicode codepoint");
        p+=n;
    }
    return p;
}

namespace json_detail {
struct Parser {
    std::string_view s;
    size_t p=0;
    [[noreturn]] void fail() const { throw std::invalid_argument("invalid JSON at byte " + std::to_string(p)); }
    void ws() { while(p<s.size()&&(s[p]==' '||s[p]=='\t'||s[p]=='\r'||s[p]=='\n'))++p; }
    bool eat(char c) { ws(); if(p<s.size()&&s[p]==c){++p;return true;}return false; }
    unsigned hex4() {
        unsigned v=0;
        for(int i=0;i<4;++i) {
            if(p==s.size())fail();
            const char c=s[p++];
            int d=c>='0'&&c<='9'?c-'0':(c>='a'&&c<='f'?c-'a'+10:(c>='A'&&c<='F'?c-'A'+10:-1));
            if(d<0)fail();
            v=(v<<4)|unsigned(d);
        }
        return v;
    }
    std::string str() {
        if(!eat('"'))fail();
        std::string out;
        while(p<s.size()) {
            unsigned char c=s[p++];
            if(c=='"') {
                if(utf8_prefix(out)!=out.size())fail();
                return out;
            }
            if(c<32)fail();
            if(c!='\\'){out+=char(c);continue;}
            if(p==s.size())fail();
            switch(s[p++]) {
                case '"':out+='"';break; case '\\':out+='\\';break; case '/':out+='/';break;
                case 'b':out+='\b';break; case 'f':out+='\f';break; case 'n':out+='\n';break;
                case 'r':out+='\r';break; case 't':out+='\t';break;
                case 'u': {
                    unsigned cp=hex4();
                    if(cp>=0xd800&&cp<=0xdbff) {
                        if(p+2>s.size()||s[p]!='\\'||s[p+1]!='u')fail();
                        p+=2; unsigned lo=hex4();
                        if(lo<0xdc00||lo>0xdfff)fail();
                        cp=0x10000+((cp-0xd800)<<10)+(lo-0xdc00);
                    } else if(cp>=0xdc00&&cp<=0xdfff)fail();
                    append_utf8(out,cp);break;
                }
                default:fail();
            }
        }
        fail();
    }
    Json value(unsigned depth=0) {
        if(depth>64)fail();
        ws(); if(p==s.size())fail(); Json v;
        if(s[p]=='"'){v.kind=Json::Kind::String;v.string=str();return v;}
        if(eat('{')) {
            v.kind=Json::Kind::Object; if(eat('}'))return v;
            do {auto k=str();if(!eat(':'))fail();
                if(!v.object.emplace(k,value(depth+1)).second)fail();
            }while(eat(','));
            if(!eat('}'))fail();
            return v;
        }
        if(eat('[')) {
            v.kind=Json::Kind::Array;if(eat(']'))return v;
            do {v.array.push_back(value(depth+1));}while(eat(','));
            if(!eat(']'))fail();
            return v;
        }
        for(const char* word:{"true","false","null"}) {
            const std::string_view w=word;
            if(s.substr(p,w.size())==w) {
                p+=w.size();v.kind=w=="null"?Json::Kind::Null:Json::Kind::Boolean;
                v.boolean=w=="true";return v;
            }
        }
        size_t begin=p;
        if(s[p]=='-')++p;
        if(p==s.size())fail();
        if(s[p]=='0')++p;
        else {if(s[p]<'1'||s[p]>'9')fail();while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;}
        auto digits=[&]{size_t b=p;while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;if(b==p)fail();};
        if(p<s.size()&&s[p]=='.'){++p;digits();}
        if(p<s.size()&&(s[p]=='e'||s[p]=='E')){++p;if(p<s.size()&&(s[p]=='+'||s[p]=='-'))++p;digits();}
        const std::string n(s.substr(begin,p-begin));char* end=nullptr;
        v.number=std::strtod(n.c_str(),&end);
        if(end!=n.c_str()+n.size()||!std::isfinite(v.number))fail();
        v.kind=Json::Kind::Number;return v;
    }
};
}
inline Json Json::parse(std::string_view input) {
    json_detail::Parser p{input}; auto v=p.value();p.ws();if(p.p!=input.size())p.fail();return v;
}
inline std::string json_escape(std::string_view s) {
    std::string out;
    const char* hex="0123456789abcdef";
    for(unsigned char c:s) {
        if(c=='"'||c=='\\'){out+='\\';out+=char(c);}
        else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
        else out+=char(c);
    }
    return out;
}
} // namespace b70
