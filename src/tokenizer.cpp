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

// =====================================================================
//  tokenizer.cpp
// =====================================================================
#include "b70/tokenizer.hpp"
#include "b70/gguf.hpp"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <stdexcept>
#include <limits>

namespace b70 {
namespace {

std::string read_file(const std::string& p) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return {};
    std::string out;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

// Decode a JSON string literal starting at s[i] == '"'. Advances i past
// the closing quote. \uXXXX is decoded to UTF-8, which matters: the
// byte-level placeholders live in U+0100..U+017F and appear escaped in
// some tokenizer.json files.
std::string json_string(const std::string& s, size_t& i) {
    std::string out;
    // Skip leading whitespace. tokenizer.json is pretty-printed, so a
    // string is preceded by a newline and indentation. Without this the
    // merges parser returns EMPTY for the first element of every pair
    // and stores keys like " Ġ" instead of "Ġ Ġ" -- merges appear to
    // load, none of them ever match, and every word degrades to single
    // characters.
    while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r')) ++i;
    if (i >= s.size() || s[i] != '"') return out;
    ++i;
    while (i < s.size() && s[i] != '"') {
        if (s[i] == '\\' && i + 1 < s.size()) {
            ++i;
            switch (s[i]) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case '/': out += '/';  break;
                case '"': out += '"';  break;
                case '\\': out += '\\'; break;
                case 'u': {
                    if (i + 4 < s.size()) {
                        const uint32_t cp = uint32_t(std::strtoul(
                            s.substr(i + 1, 4).c_str(), nullptr, 16));
                        i += 4;
                        if (cp < 0x80) out += char(cp);
                        else if (cp < 0x800) {
                            out += char(0xC0 | (cp >> 6));
                            out += char(0x80 | (cp & 0x3F));
                        } else {
                            out += char(0xE0 | (cp >> 12));
                            out += char(0x80 | ((cp >> 6) & 0x3F));
                            out += char(0x80 | (cp & 0x3F));
                        }
                    }
                    break;
                }
                default: out += s[i];
            }
            ++i;
        } else {
            out += s[i++];
        }
    }
    ++i;   // closing quote
    return out;
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i]==' '||s[i]=='\n'||s[i]=='\t'||s[i]=='\r')) ++i;
}

} // namespace

void Tokenizer::set_chat_family(const std::string& tmpl) {
    tmpl_xml_tools_ = tmpl.find("<function=example_function_name>") != std::string::npos &&
                      tmpl.find("do not tell the user about function calls") != std::string::npos &&
                      tmpl.find("multi_step_tool=true") != std::string::npos;
    tmpl_effort_ = tmpl_xml_tools_ && tmpl.find("Reasoning effort is set to xhigh.") != std::string::npos;
    tmpl_think_split_ = tmpl_xml_tools_ && tmpl.find("content.split('</think>')") != std::string::npos;
}

// ---------------------------------------------------------------------
// GPT-2 bytes_to_unicode: every one of the 256 byte values gets a
// printable codepoint so BPE can operate on text. Bytes that are already
// printable map to themselves; the rest are shifted into U+0100.
// ---------------------------------------------------------------------
void Tokenizer::build_byte_tables() {
    std::vector<int> bs;
    for (int b = '!'; b <= '~'; ++b) bs.push_back(b);
    for (int b = 0xA1; b <= 0xAC; ++b) bs.push_back(b);
    for (int b = 0xAE; b <= 0xFF; ++b) bs.push_back(b);

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n);
            ++n;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        const uint32_t cp = uint32_t(cs[i]);
        std::string u;
        if (cp < 0x80) u += char(cp);
        else if (cp < 0x800) {
            u += char(0xC0 | (cp >> 6));
            u += char(0x80 | (cp & 0x3F));
        } else {
            u += char(0xE0 | (cp >> 12));
            u += char(0x80 | ((cp >> 6) & 0x3F));
            u += char(0x80 | (cp & 0x3F));
        }
        byte_to_uni_[bs[i]] = u;
        uni_to_byte_[u] = uint8_t(bs[i]);
    }
}

bool Tokenizer::load(const std::string& dir, std::string& err) {
    const bool direct=dir.size()>=5 && dir.compare(dir.size()-5,5,".gguf")==0;
    const std::string root=direct ? (dir.find_last_of('/')==std::string::npos ? "." :
                           dir.substr(0,dir.find_last_of('/'))) : dir;
    const std::string sidecar = root + "/tokenizer.json";
    if (std::FILE* probe=std::fopen(sidecar.c_str(),"rb")) std::fclose(probe);
    else {
        std::string selected;
        if (!gguf_model_path(dir,selected,err)) return false;
        if (selected.empty()) { err="cannot read "+sidecar+" and no GGUF model found";return false; }
        GGUFFile file;
        if (!file.open(selected,err)) return false;
        return load_gguf(file,err);
    }
    build_byte_tables();
    add_bos_token_ = false;
    // Raw-text scoring follows the checkpoint's tokenizer setting, not the
    // presence of a BOS id. Qwen-family tokenizers normally set this false.
    const std::string tokenizer_cfg = read_file(root + "/tokenizer_config.json");
    const size_t bos_key = tokenizer_cfg.find("\"add_bos_token\"");
    if (bos_key != std::string::npos) {
        size_t value = tokenizer_cfg.find(':', bos_key);
        if (value == std::string::npos) { err = "malformed add_bos_token"; return false; }
        ++value;
        skip_ws(tokenizer_cfg, value);
        if (tokenizer_cfg.compare(value, 4, "true") == 0) add_bos_token_ = true;
        else if (tokenizer_cfg.compare(value, 5, "false") == 0) add_bos_token_ = false;
        else { err = "invalid add_bos_token"; return false; }
    }

    const std::string js = read_file(sidecar);
    if (js.empty()) { err = "cannot read " + sidecar; return false; }

    // ---- vocab -------------------------------------------------------
    // Scanning for the key rather than parsing the whole document: the
    // file is ~10 MB and a general parser would cost more than the model
    // load itself.
    size_t p = js.find("\"vocab\"");
    if (p == std::string::npos) { err = "no \"vocab\" in tokenizer.json"; return false; }
    p = js.find('{', p);
    if (p == std::string::npos) { err = "malformed vocab"; return false; }
    ++p;

    int32_t max_id = -1;
    std::vector<std::pair<std::string,int32_t>> entries;
    entries.reserve(300000);
    while (true) {
        skip_ws(js, p);
        if (p >= js.size() || js[p] == '}') { ++p; break; }
        const std::string tok = json_string(js, p);
        skip_ws(js, p);
        if (p < js.size() && js[p] == ':') ++p;
        skip_ws(js, p);
        char* endp = nullptr;
        const long id = std::strtol(js.c_str() + p, &endp, 10);
        p = size_t(endp - js.c_str());
        entries.emplace_back(tok, int32_t(id));
        max_id = std::max(max_id, int32_t(id));
        skip_ws(js, p);
        if (p < js.size() && js[p] == ',') ++p;
        else if (p < js.size() && js[p] == '}') { ++p; break; }
    }
    if (entries.empty()) { err = "vocab parsed empty"; return false; }

    id_to_tok_.assign(size_t(max_id) + 1, std::string());
    for (const auto& e : entries) {
        id_to_tok_[size_t(e.second)] = e.first;
        tok_to_id_[e.first] = e.second;
    }

    // ---- merges ------------------------------------------------------
    // Rank IS the priority. Storing them without order would produce a
    // tokenization that decodes correctly but uses entirely different
    // ids -- invisible in the output, fatal to the model.
    p = js.find("\"merges\"");
    if (p != std::string::npos) {
        p = js.find('[', p);
        ++p;
        int rank = 0;
        while (p < js.size()) {
            skip_ws(js, p);
            if (js[p] == ']') break;
            if (js[p] == '"') {
                const std::string m = json_string(js, p);
                merge_rank_[m] = rank++;
            } else if (js[p] == '[') {
                // pair format: ["a","b"]
                ++p;
                const std::string a = json_string(js, p);
                skip_ws(js, p); if (p < js.size() && js[p] == ',') ++p;
                const std::string b = json_string(js, p);
                skip_ws(js, p); if (p < js.size() && js[p] == ']') ++p;
                // A merge with an empty side means the parse desynced.
                // Storing it silently poisons the rank table.
                if (a.empty() || b.empty()) { bad_merges_++; }
                else merge_rank_[a + " " + b] = rank++;
            } else {
                ++p;
                continue;
            }
            skip_ws(js, p);
            if (p < js.size() && js[p] == ',') ++p;
        }
    }

    // ---- chat template family ----------------------------------------
    // The checkpoint's own template decides how tools and past turns are
    // rendered.  Markers that only the Qwen3.5-family XML-tool template
    // (Qwen3.8 / Agnes / Ornith) has -- the Qwen3.6 template words its tool
    // block differently and keeps its own (legacy) rendering here.
    {
        std::string tmpl = read_file(root + "/chat_template.jinja");
        if (tmpl.empty()) tmpl = read_file(root + "/tokenizer_config.json");
        set_chat_family(tmpl);
    }

    // ---- special tokens ----------------------------------------------
    p = js.find("\"added_tokens\"");
    if (p != std::string::npos) {
        size_t q = js.find('[', p);
        const size_t end = js.find(']', q);
        while (q != std::string::npos && q < end) {
            const size_t idp = js.find("\"id\"", q);
            if (idp == std::string::npos || idp > end) break;
            size_t r = js.find(':', idp) + 1;
            const int32_t id = int32_t(std::strtol(js.c_str() + r, nullptr, 10));
            const size_t cp = js.find("\"content\"", idp);
            if (cp == std::string::npos || cp > end) break;
            r = js.find('"', js.find(':', cp)) ;
            const std::string content = json_string(js, r);
            // Only tokens flagged "special": true are control tokens that
            // decoding hides (HF skip_special_tokens).  Qwen-family
            // <think>, </think>, <tool_call>, </tool_call> are added but NOT
            // special, and every client parses them from the text: hiding
            // them turned a tool call into a bare <function=...> block and
            // erased the reasoning / answer boundary (2026-10-05).  A token
            // without the flag stays hidden, as before.
            {
                const size_t nx = js.find("\"id\"", idp + 1);
                const size_t sp = js.find("\"special\"", cp);
                bool special = true;
                if (sp != std::string::npos && sp < std::min(nx, end)) {
                    size_t v = js.find(':', sp) + 1;
                    skip_ws(js, v);
                    special = js.compare(v, 4, "true") == 0;
                }
                if (special) special_ids_[id] = 1;
            }
            special_by_text_[content] = id;
            // Added/special tokens are almost always appended AFTER the base
            // BPE vocab, at ids beyond id_to_tok_'s initial size (set from
            // the base vocab alone, before this loop runs).  The old guard
            // (`if id < size()`) silently dropped every such mapping instead
            // of growing the vector -- vocab_size() (== id_to_tok_.size())
            // then under-reported the model's real vocabulary by exactly
            // the special-token count.  MEASURED 2026-10-02: Ornith's base
            // vocab is 248,044; <|im_start|>/<|im_end|> sit at 248,045 and
            // 248,046.  Every chat-templated prompt starts with
            // <|im_start|>, so grimoire-server's bounds check
            // (generation_budget, using this vocab_size()) rejected EVERY
            // /v1/chat/completions request with "prompt token outside model
            // vocabulary" -- while /v1/completions, which never emits a
            // special token, was unaffected.  Growing the vector here fixes
            // vocab_size() for every caller, not just the server.
            if (size_t(id) >= id_to_tok_.size()) id_to_tok_.resize(size_t(id) + 1);
            id_to_tok_[size_t(id)] = content;
            tok_to_id_[content] = id;
            q = js.find("\"id\"", idp + 1);
            if (q == std::string::npos || q > end) break;
        }
    }

    auto find_special = [&](const char* t) -> int32_t {
        auto it = special_by_text_.find(t);
        return it == special_by_text_.end() ? -1 : it->second;
    };
    // End of turn: ChatML's <|im_end|>, K2-Horizon's <|ifm|im_end|> (IFM
    // prefixes every control token; with neither found K2 had NO stop token
    // and every answer ran on to max_tokens), else <|endoftext|>.
    eos_ = find_special("<|im_end|>");
    if (eos_ < 0) eos_ = find_special("<|ifm|im_end|>");
    if (eos_ < 0) eos_ = find_special("<|endoftext|>");
    for (const char* t : {"<|eot|>", "<|ifm|endoftext|>", "<|endoftext|>"}) {
        const int32_t id = find_special(t);
        if (id >= 0 && id != eos_) { eos2_ = id; break; }
    }
    bos_ = find_special("<|im_start|>");
    if (add_bos_token_) {
        // A tokenizer that requests BOS may use a different literal from
        // ChatML's <|im_start|>. Resolve its configured token explicitly.
        bos_ = -1;
        const size_t key = tokenizer_cfg.find("\"bos_token\"");
        if (key != std::string::npos) {
            size_t value = tokenizer_cfg.find(':', key);
            if (value != std::string::npos) {
                ++value; skip_ws(tokenizer_cfg, value);
                if (value < tokenizer_cfg.size() && tokenizer_cfg[value] == '{') {
                    const size_t content = tokenizer_cfg.find("\"content\"", value);
                    const size_t close = tokenizer_cfg.find('}', value);
                    if (content != std::string::npos && content < close) {
                        value = tokenizer_cfg.find(':', content) + 1;
                        skip_ws(tokenizer_cfg, value);
                    }
                }
                if (value < tokenizer_cfg.size() && tokenizer_cfg[value] == '"') {
                    const std::string literal = json_string(tokenizer_cfg, value);
                    const auto it = tok_to_id_.find(literal);
                    if (it != tok_to_id_.end()) bos_ = it->second;
                }
            }
        }
    }

    return true;
}

bool Tokenizer::load_gguf(const GGUFFile& file, std::string& err) {
    *this=Tokenizer{};
    build_byte_tables();
    std::string model, pre;
    if (!file.get_string("tokenizer.ggml.model",model,err)) return false;
    if (model!="gpt2" && model!="llama") {
        err="unsupported tokenizer.ggml.model: "+model; return false;
    }
    sentencepiece_=model=="llama";
    if (sentencepiece_) add_bos_token_=true; // llama.cpp SPM default
    if (const GGUFValue* p=file.get("tokenizer.ggml.pre")) {
        if (p->type!=8) {err="wrong tokenizer.ggml.pre type";return false;}
        pre=std::get<std::string>(p->data);
    }
    // llama.cpp 711780ffc75b, llama-vocab.cpp QWEN35: the existing Unicode
    // splitter implements its Split expression exactly (including marks and
    // single-codepoint numbers). Other pre names select different regexes.
    if ((!sentencepiece_ && pre!="qwen35") ||
        (sentencepiece_ && !pre.empty() && pre!="default")) {
        err="unsupported tokenizer.ggml.pre: "+(pre.empty()?"<missing>":pre); return false;
    }
    const GGUFValue *v=nullptr,*types=nullptr;
    if (!file.get_array("tokenizer.ggml.tokens",v,err)) return false;
    const auto* strings=std::get_if<std::vector<std::string>>(&v->data);
    if (!strings || strings->empty()) {err="tokenizer.ggml.tokens must be a nonempty string array";return false;}
    if (!file.get_array("tokenizer.ggml.token_type",types,err)) return false;
    const auto* kind=std::get_if<std::vector<int32_t>>(&types->data);
    if (!kind || kind->size()!=strings->size()) {
        err="tokenizer.ggml.token_type must be an i32 array matching tokens";return false;
    }
    id_to_tok_=*strings; token_type_=*kind;
    for (size_t i=0;i<strings->size();++i) {
        const int32_t t=(*kind)[i];
        if (t<1 || t>6) {err="invalid tokenizer.ggml.token_type at "+std::to_string(i);return false;}
        const std::string& s=(*strings)[i];
        if (t!=5 && !s.empty()) tok_to_id_.emplace(s,int32_t(i));
        if ((t==2 || t==3 || t==4) && !s.empty()) {
            special_by_text_.emplace(s,int32_t(i));
            // Qwen3.5's GGUF marks the six FIM/repository delimiters as
            // CONTROL, although the HF added_tokens entries set special=false.
            // They must remain atomic on encode and visible on decode.
            const bool visible_qwen_control=!sentencepiece_ &&
                (s=="<|fim_prefix|>" || s=="<|fim_middle|>" ||
                 s=="<|fim_suffix|>" || s=="<|fim_pad|>" ||
                 s=="<|repo_name|>" || s=="<|file_sep|>");
            if (t==2 || (t==3 && !visible_qwen_control)) special_ids_[int32_t(i)]=1;
        }
    }
    if (sentencepiece_) {
        const GGUFValue* sv=nullptr;
        if (!file.get_array("tokenizer.ggml.scores",sv,err)) return false;
        const auto* scores=std::get_if<std::vector<float>>(&sv->data);
        if (!scores || scores->size()!=strings->size()) {
            err="tokenizer.ggml.scores must match tokens";return false;
        }
        scores_=*scores;
    } else {
        const GGUFValue* mv=nullptr;
        if (!file.get_array("tokenizer.ggml.merges",mv,err)) return false;
        const auto* merges=std::get_if<std::vector<std::string>>(&mv->data);
        if (!merges) {err="tokenizer.ggml.merges must be a string array";return false;}
        for (size_t i=0;i<merges->size();++i) {
            if ((*merges)[i].find(' ')==std::string::npos) {
                err="invalid tokenizer.ggml.merges entry "+std::to_string(i);return false;
            }
            merge_rank_.emplace((*merges)[i],int32_t(i));
        }
    }
    auto id=[&](const char* key, int32_t& target)->bool {
        const GGUFValue* p=file.get(key);
        if (!p) return true;
        uint64_t n=0;
        if (p->type==4) n=std::get<uint32_t>(p->data);
        else if (p->type==2) n=std::get<uint16_t>(p->data);
        else {err=std::string("wrong GGUF token id type: ")+key;return false;}
        if (n>=id_to_tok_.size()) {err=std::string("GGUF token id out of range: ")+key;return false;}
        target=int32_t(n);return true;
    };
    if (!id("tokenizer.ggml.bos_token_id",bos_) || !id("tokenizer.ggml.eos_token_id",eos_) ||
        !id("tokenizer.ggml.eot_token_id",eot_) || !id("tokenizer.ggml.padding_token_id",pad_) ||
        !id("tokenizer.ggml.unknown_token_id",unk_)) return false;
    if (sentencepiece_ && unk_<0) {
        for (size_t i=0;i<token_type_.size();++i)
            if (token_type_[i]==2) {unk_=int32_t(i);break;}
        if (unk_<0) {
            for (unsigned c=0;c<256;++c) {
                char byte[7];std::snprintf(byte,sizeof byte,"<0x%02X>",c);
                auto it=tok_to_id_.find(byte);
                if (it==tok_to_id_.end() || token_type_[it->second]!=6) {
                    err="llama tokenizer lacks unknown id and complete byte fallback";return false;
                }
            }
        }
    }
    auto flag=[&](const char* key, bool& dst)->bool {
        const GGUFValue* p=file.get(key);
        if (!p) return true;
        if (p->type!=7) {err=std::string("wrong GGUF bool type: ")+key;return false;}
        dst=std::get<bool>(p->data);return true;
    };
    if (!flag("tokenizer.ggml.add_bos_token",add_bos_token_) ||
        !flag("tokenizer.ggml.add_eos_token",add_eos_token_) ||
        !flag("tokenizer.ggml.add_space_prefix",spm_add_prefix_)) return false;
    if (eot_>=0 && eot_!=eos_) eos2_=eot_;
    else for (const char* s:{"<|eot|>","<|ifm|endoftext|>","<|endoftext|>"}) {
        auto it=special_by_text_.find(s);
        if (it!=special_by_text_.end() && it->second!=eos_) {eos2_=it->second;break;}
    }
    if (const GGUFValue* p=file.get("tokenizer.chat_template")) {
        if (p->type!=8) {err="wrong tokenizer.chat_template type";return false;}
        set_chat_family(std::get<std::string>(p->data));
    }
    return true;
}

// ---------------------------------------------------------------------
// BPE over one pre-tokenized word, expressed in placeholder characters.
// Repeatedly merge the adjacent pair with the LOWEST rank.
// ---------------------------------------------------------------------
std::vector<std::string> Tokenizer::bpe_word(const std::string& word) const {
    // split into UTF-8 characters
    std::vector<std::string> sym;
    for (size_t i = 0; i < word.size();) {
        const unsigned char c = word[i];
        size_t len = 1;
        if ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        sym.push_back(word.substr(i, len));
        i += len;
    }
    if (sym.size() < 2) return sym;

    for (;;) {
        int best_rank = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank == INT32_MAX) break;
        sym[best_i] += sym[best_i + 1];
        sym.erase(sym.begin() + long(best_i) + 1);
        if (sym.size() == 1) break;
    }
    return sym;
}

std::vector<int32_t> Tokenizer::spm_word(const std::string& text) const {
    std::string normalized;
    if (!text.empty() && spm_add_prefix_) normalized="\xE2\x96\x81";
    for (char c:text) normalized += c==' ' ? "\xE2\x96\x81" : std::string(1,c);
    std::vector<std::string> parts;
    for (size_t i=0;i<normalized.size();) {
        const unsigned char c=normalized[i];
        const size_t width=(c&0xF8)==0xF0?4:(c&0xF0)==0xE0?3:(c&0xE0)==0xC0?2:1;
        const size_t n=std::min(width,normalized.size()-i);
        parts.push_back(normalized.substr(i,n)); i+=n;
    }
    for (;;) {
        float best=-std::numeric_limits<float>::infinity(); size_t at=parts.size();
        for (size_t i=0;i+1<parts.size();++i) {
            auto it=tok_to_id_.find(parts[i]+parts[i+1]);
            if (it==tok_to_id_.end() || token_type_[it->second]!=1) continue;
            if (scores_[it->second]>best) {best=scores_[it->second];at=i;}
        }
        if (at==parts.size()) break;
        parts[at]+=parts[at+1];parts.erase(parts.begin()+long(at)+1);
    }
    std::vector<int32_t> result;
    for (const auto& p:parts) {
        auto it=tok_to_id_.find(p);
        if (it!=tok_to_id_.end() && token_type_[it->second]!=5) {result.push_back(it->second);continue;}
        for (unsigned char c:p) {
            char byte[7];std::snprintf(byte,sizeof byte,"<0x%02X>",unsigned(c));
            auto b=tok_to_id_.find(byte);
            if (b!=tok_to_id_.end() && token_type_[b->second]==6) result.push_back(b->second);
            else if (unk_>=0) result.push_back(unk_);
        }
    }
    return result;
}

// ---------------------------------------------------------------------
// Pre-tokenizer. Alternatives are attempted in order at each codepoint,
// exactly as the checkpoint's Split(Isolated) regex specifies:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
//   |\p{N}
//   | ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// std::regex has no Unicode properties; use a pinned category table instead.
namespace {

inline bool is_nl(uint32_t cp) { return cp == '\n' || cp == '\r'; }
inline int u8len(unsigned char c) {
    if ((c & 0x80) == 0)    return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}
inline uint32_t u8cp(const std::string& t, size_t i, int len) {
    const unsigned char c0 = t[i];
    if (len == 1) return c0;
    if (i + size_t(len) > t.size()) return 0;
    uint32_t cp = c0 & (0xFF >> (len + 1));
    for (int k = 1; k < len; ++k) {
        const unsigned char ck = t[i + size_t(k)];
        if ((ck & 0xC0) != 0x80) return 0;
        cp = (cp << 6) | (ck & 0x3F);
    }
    return cp;
}

// Regex character classes derived from HF tokenizers 0.22.2's Split output
// over Unicode scalar values, using UnicodeData 13.0 as the baseline.
// A means \p{L} or \p{M}; N means \p{N}. These are the classes that
// affect output boundaries. Treating L and M together is equivalent here:
// consuming an initial mark as the optional prefix still produces the same
// span as consuming it in the following [\p{L}\p{M}]+ run.
struct CategoryRange { uint32_t first, last; char category; };
static constexpr CategoryRange kCategories[] = {
    {0x30, 0x39, 'N'},
    {0x41, 0x5A, 'A'},
    {0x61, 0x7A, 'A'},
    {0xAA, 0xAA, 'A'},
    {0xB2, 0xB3, 'N'},
    {0xB5, 0xB5, 'A'},
    {0xB9, 0xB9, 'N'},
    {0xBA, 0xBA, 'A'},
    {0xBC, 0xBE, 'N'},
    {0xC0, 0xD6, 'A'},
    {0xD8, 0xF6, 'A'},
    {0xF8, 0x2C1, 'A'},
    {0x2C6, 0x2D1, 'A'},
    {0x2E0, 0x2E4, 'A'},
    {0x2EC, 0x2EC, 'A'},
    {0x2EE, 0x2EE, 'A'},
    {0x300, 0x374, 'A'},
    {0x376, 0x377, 'A'},
    {0x37A, 0x37D, 'A'},
    {0x37F, 0x37F, 'A'},
    {0x386, 0x386, 'A'},
    {0x388, 0x38A, 'A'},
    {0x38C, 0x38C, 'A'},
    {0x38E, 0x3A1, 'A'},
    {0x3A3, 0x3F5, 'A'},
    {0x3F7, 0x481, 'A'},
    {0x483, 0x52F, 'A'},
    {0x531, 0x556, 'A'},
    {0x559, 0x559, 'A'},
    {0x560, 0x588, 'A'},
    {0x591, 0x5BD, 'A'},
    {0x5BF, 0x5BF, 'A'},
    {0x5C1, 0x5C2, 'A'},
    {0x5C4, 0x5C5, 'A'},
    {0x5C7, 0x5C7, 'A'},
    {0x5D0, 0x5EA, 'A'},
    {0x5EF, 0x5F2, 'A'},
    {0x610, 0x61A, 'A'},
    {0x620, 0x65F, 'A'},
    {0x660, 0x669, 'N'},
    {0x66E, 0x6D3, 'A'},
    {0x6D5, 0x6DC, 'A'},
    {0x6DF, 0x6E8, 'A'},
    {0x6EA, 0x6EF, 'A'},
    {0x6F0, 0x6F9, 'N'},
    {0x6FA, 0x6FC, 'A'},
    {0x6FF, 0x6FF, 'A'},
    {0x710, 0x74A, 'A'},
    {0x74D, 0x7B1, 'A'},
    {0x7C0, 0x7C9, 'N'},
    {0x7CA, 0x7F5, 'A'},
    {0x7FA, 0x7FA, 'A'},
    {0x7FD, 0x7FD, 'A'},
    {0x800, 0x82D, 'A'},
    {0x840, 0x85B, 'A'},
    {0x860, 0x86A, 'A'},
    {0x870, 0x887, 'A'},
    {0x889, 0x88E, 'A'},
    {0x897, 0x8E1, 'A'},
    {0x8E3, 0x963, 'A'},
    {0x966, 0x96F, 'N'},
    {0x971, 0x983, 'A'},
    {0x985, 0x98C, 'A'},
    {0x98F, 0x990, 'A'},
    {0x993, 0x9A8, 'A'},
    {0x9AA, 0x9B0, 'A'},
    {0x9B2, 0x9B2, 'A'},
    {0x9B6, 0x9B9, 'A'},
    {0x9BC, 0x9C4, 'A'},
    {0x9C7, 0x9C8, 'A'},
    {0x9CB, 0x9CE, 'A'},
    {0x9D7, 0x9D7, 'A'},
    {0x9DC, 0x9DD, 'A'},
    {0x9DF, 0x9E3, 'A'},
    {0x9E6, 0x9EF, 'N'},
    {0x9F0, 0x9F1, 'A'},
    {0x9F4, 0x9F9, 'N'},
    {0x9FC, 0x9FC, 'A'},
    {0x9FE, 0x9FE, 'A'},
    {0xA01, 0xA03, 'A'},
    {0xA05, 0xA0A, 'A'},
    {0xA0F, 0xA10, 'A'},
    {0xA13, 0xA28, 'A'},
    {0xA2A, 0xA30, 'A'},
    {0xA32, 0xA33, 'A'},
    {0xA35, 0xA36, 'A'},
    {0xA38, 0xA39, 'A'},
    {0xA3C, 0xA3C, 'A'},
    {0xA3E, 0xA42, 'A'},
    {0xA47, 0xA48, 'A'},
    {0xA4B, 0xA4D, 'A'},
    {0xA51, 0xA51, 'A'},
    {0xA59, 0xA5C, 'A'},
    {0xA5E, 0xA5E, 'A'},
    {0xA66, 0xA6F, 'N'},
    {0xA70, 0xA75, 'A'},
    {0xA81, 0xA83, 'A'},
    {0xA85, 0xA8D, 'A'},
    {0xA8F, 0xA91, 'A'},
    {0xA93, 0xAA8, 'A'},
    {0xAAA, 0xAB0, 'A'},
    {0xAB2, 0xAB3, 'A'},
    {0xAB5, 0xAB9, 'A'},
    {0xABC, 0xAC5, 'A'},
    {0xAC7, 0xAC9, 'A'},
    {0xACB, 0xACD, 'A'},
    {0xAD0, 0xAD0, 'A'},
    {0xAE0, 0xAE3, 'A'},
    {0xAE6, 0xAEF, 'N'},
    {0xAF9, 0xAFF, 'A'},
    {0xB01, 0xB03, 'A'},
    {0xB05, 0xB0C, 'A'},
    {0xB0F, 0xB10, 'A'},
    {0xB13, 0xB28, 'A'},
    {0xB2A, 0xB30, 'A'},
    {0xB32, 0xB33, 'A'},
    {0xB35, 0xB39, 'A'},
    {0xB3C, 0xB44, 'A'},
    {0xB47, 0xB48, 'A'},
    {0xB4B, 0xB4D, 'A'},
    {0xB55, 0xB57, 'A'},
    {0xB5C, 0xB5D, 'A'},
    {0xB5F, 0xB63, 'A'},
    {0xB66, 0xB6F, 'N'},
    {0xB71, 0xB71, 'A'},
    {0xB72, 0xB77, 'N'},
    {0xB82, 0xB83, 'A'},
    {0xB85, 0xB8A, 'A'},
    {0xB8E, 0xB90, 'A'},
    {0xB92, 0xB95, 'A'},
    {0xB99, 0xB9A, 'A'},
    {0xB9C, 0xB9C, 'A'},
    {0xB9E, 0xB9F, 'A'},
    {0xBA3, 0xBA4, 'A'},
    {0xBA8, 0xBAA, 'A'},
    {0xBAE, 0xBB9, 'A'},
    {0xBBE, 0xBC2, 'A'},
    {0xBC6, 0xBC8, 'A'},
    {0xBCA, 0xBCD, 'A'},
    {0xBD0, 0xBD0, 'A'},
    {0xBD7, 0xBD7, 'A'},
    {0xBE6, 0xBF2, 'N'},
    {0xC00, 0xC0C, 'A'},
    {0xC0E, 0xC10, 'A'},
    {0xC12, 0xC28, 'A'},
    {0xC2A, 0xC39, 'A'},
    {0xC3C, 0xC44, 'A'},
    {0xC46, 0xC48, 'A'},
    {0xC4A, 0xC4D, 'A'},
    {0xC55, 0xC56, 'A'},
    {0xC58, 0xC5A, 'A'},
    {0xC5D, 0xC5D, 'A'},
    {0xC60, 0xC63, 'A'},
    {0xC66, 0xC6F, 'N'},
    {0xC78, 0xC7E, 'N'},
    {0xC80, 0xC83, 'A'},
    {0xC85, 0xC8C, 'A'},
    {0xC8E, 0xC90, 'A'},
    {0xC92, 0xCA8, 'A'},
    {0xCAA, 0xCB3, 'A'},
    {0xCB5, 0xCB9, 'A'},
    {0xCBC, 0xCC4, 'A'},
    {0xCC6, 0xCC8, 'A'},
    {0xCCA, 0xCCD, 'A'},
    {0xCD5, 0xCD6, 'A'},
    {0xCDD, 0xCDE, 'A'},
    {0xCE0, 0xCE3, 'A'},
    {0xCE6, 0xCEF, 'N'},
    {0xCF1, 0xCF3, 'A'},
    {0xD00, 0xD0C, 'A'},
    {0xD0E, 0xD10, 'A'},
    {0xD12, 0xD44, 'A'},
    {0xD46, 0xD48, 'A'},
    {0xD4A, 0xD4E, 'A'},
    {0xD54, 0xD57, 'A'},
    {0xD58, 0xD5E, 'N'},
    {0xD5F, 0xD63, 'A'},
    {0xD66, 0xD78, 'N'},
    {0xD7A, 0xD7F, 'A'},
    {0xD81, 0xD83, 'A'},
    {0xD85, 0xD96, 'A'},
    {0xD9A, 0xDB1, 'A'},
    {0xDB3, 0xDBB, 'A'},
    {0xDBD, 0xDBD, 'A'},
    {0xDC0, 0xDC6, 'A'},
    {0xDCA, 0xDCA, 'A'},
    {0xDCF, 0xDD4, 'A'},
    {0xDD6, 0xDD6, 'A'},
    {0xDD8, 0xDDF, 'A'},
    {0xDE6, 0xDEF, 'N'},
    {0xDF2, 0xDF3, 'A'},
    {0xE01, 0xE3A, 'A'},
    {0xE40, 0xE4E, 'A'},
    {0xE50, 0xE59, 'N'},
    {0xE81, 0xE82, 'A'},
    {0xE84, 0xE84, 'A'},
    {0xE86, 0xE8A, 'A'},
    {0xE8C, 0xEA3, 'A'},
    {0xEA5, 0xEA5, 'A'},
    {0xEA7, 0xEBD, 'A'},
    {0xEC0, 0xEC4, 'A'},
    {0xEC6, 0xEC6, 'A'},
    {0xEC8, 0xECE, 'A'},
    {0xED0, 0xED9, 'N'},
    {0xEDC, 0xEDF, 'A'},
    {0xF00, 0xF00, 'A'},
    {0xF18, 0xF19, 'A'},
    {0xF20, 0xF33, 'N'},
    {0xF35, 0xF35, 'A'},
    {0xF37, 0xF37, 'A'},
    {0xF39, 0xF39, 'A'},
    {0xF3E, 0xF47, 'A'},
    {0xF49, 0xF6C, 'A'},
    {0xF71, 0xF84, 'A'},
    {0xF86, 0xF97, 'A'},
    {0xF99, 0xFBC, 'A'},
    {0xFC6, 0xFC6, 'A'},
    {0x1000, 0x103F, 'A'},
    {0x1040, 0x1049, 'N'},
    {0x1050, 0x108F, 'A'},
    {0x1090, 0x1099, 'N'},
    {0x109A, 0x109D, 'A'},
    {0x10A0, 0x10C5, 'A'},
    {0x10C7, 0x10C7, 'A'},
    {0x10CD, 0x10CD, 'A'},
    {0x10D0, 0x10FA, 'A'},
    {0x10FC, 0x1248, 'A'},
    {0x124A, 0x124D, 'A'},
    {0x1250, 0x1256, 'A'},
    {0x1258, 0x1258, 'A'},
    {0x125A, 0x125D, 'A'},
    {0x1260, 0x1288, 'A'},
    {0x128A, 0x128D, 'A'},
    {0x1290, 0x12B0, 'A'},
    {0x12B2, 0x12B5, 'A'},
    {0x12B8, 0x12BE, 'A'},
    {0x12C0, 0x12C0, 'A'},
    {0x12C2, 0x12C5, 'A'},
    {0x12C8, 0x12D6, 'A'},
    {0x12D8, 0x1310, 'A'},
    {0x1312, 0x1315, 'A'},
    {0x1318, 0x135A, 'A'},
    {0x135D, 0x135F, 'A'},
    {0x1369, 0x137C, 'N'},
    {0x1380, 0x138F, 'A'},
    {0x13A0, 0x13F5, 'A'},
    {0x13F8, 0x13FD, 'A'},
    {0x1401, 0x166C, 'A'},
    {0x166F, 0x167F, 'A'},
    {0x1681, 0x169A, 'A'},
    {0x16A0, 0x16EA, 'A'},
    {0x16EE, 0x16F0, 'N'},
    {0x16F1, 0x16F8, 'A'},
    {0x1700, 0x1715, 'A'},
    {0x171F, 0x1734, 'A'},
    {0x1740, 0x1753, 'A'},
    {0x1760, 0x176C, 'A'},
    {0x176E, 0x1770, 'A'},
    {0x1772, 0x1773, 'A'},
    {0x1780, 0x17D3, 'A'},
    {0x17D7, 0x17D7, 'A'},
    {0x17DC, 0x17DD, 'A'},
    {0x17E0, 0x17E9, 'N'},
    {0x17F0, 0x17F9, 'N'},
    {0x180B, 0x180D, 'A'},
    {0x180F, 0x180F, 'A'},
    {0x1810, 0x1819, 'N'},
    {0x1820, 0x1878, 'A'},
    {0x1880, 0x18AA, 'A'},
    {0x18B0, 0x18F5, 'A'},
    {0x1900, 0x191E, 'A'},
    {0x1920, 0x192B, 'A'},
    {0x1930, 0x193B, 'A'},
    {0x1946, 0x194F, 'N'},
    {0x1950, 0x196D, 'A'},
    {0x1970, 0x1974, 'A'},
    {0x1980, 0x19AB, 'A'},
    {0x19B0, 0x19C9, 'A'},
    {0x19D0, 0x19DA, 'N'},
    {0x1A00, 0x1A1B, 'A'},
    {0x1A20, 0x1A5E, 'A'},
    {0x1A60, 0x1A7C, 'A'},
    {0x1A7F, 0x1A7F, 'A'},
    {0x1A80, 0x1A89, 'N'},
    {0x1A90, 0x1A99, 'N'},
    {0x1AA7, 0x1AA7, 'A'},
    {0x1AB0, 0x1ACE, 'A'},
    {0x1B00, 0x1B4C, 'A'},
    {0x1B50, 0x1B59, 'N'},
    {0x1B6B, 0x1B73, 'A'},
    {0x1B80, 0x1BAF, 'A'},
    {0x1BB0, 0x1BB9, 'N'},
    {0x1BBA, 0x1BF3, 'A'},
    {0x1C00, 0x1C37, 'A'},
    {0x1C40, 0x1C49, 'N'},
    {0x1C4D, 0x1C4F, 'A'},
    {0x1C50, 0x1C59, 'N'},
    {0x1C5A, 0x1C7D, 'A'},
    {0x1C80, 0x1C8A, 'A'},
    {0x1C90, 0x1CBA, 'A'},
    {0x1CBD, 0x1CBF, 'A'},
    {0x1CD0, 0x1CD2, 'A'},
    {0x1CD4, 0x1CFA, 'A'},
    {0x1D00, 0x1F15, 'A'},
    {0x1F18, 0x1F1D, 'A'},
    {0x1F20, 0x1F45, 'A'},
    {0x1F48, 0x1F4D, 'A'},
    {0x1F50, 0x1F57, 'A'},
    {0x1F59, 0x1F59, 'A'},
    {0x1F5B, 0x1F5B, 'A'},
    {0x1F5D, 0x1F5D, 'A'},
    {0x1F5F, 0x1F7D, 'A'},
    {0x1F80, 0x1FB4, 'A'},
    {0x1FB6, 0x1FBC, 'A'},
    {0x1FBE, 0x1FBE, 'A'},
    {0x1FC2, 0x1FC4, 'A'},
    {0x1FC6, 0x1FCC, 'A'},
    {0x1FD0, 0x1FD3, 'A'},
    {0x1FD6, 0x1FDB, 'A'},
    {0x1FE0, 0x1FEC, 'A'},
    {0x1FF2, 0x1FF4, 'A'},
    {0x1FF6, 0x1FFC, 'A'},
    {0x2070, 0x2070, 'N'},
    {0x2071, 0x2071, 'A'},
    {0x2074, 0x2079, 'N'},
    {0x207F, 0x207F, 'A'},
    {0x2080, 0x2089, 'N'},
    {0x2090, 0x209C, 'A'},
    {0x20D0, 0x20F0, 'A'},
    {0x2102, 0x2102, 'A'},
    {0x2107, 0x2107, 'A'},
    {0x210A, 0x2113, 'A'},
    {0x2115, 0x2115, 'A'},
    {0x2119, 0x211D, 'A'},
    {0x2124, 0x2124, 'A'},
    {0x2126, 0x2126, 'A'},
    {0x2128, 0x2128, 'A'},
    {0x212A, 0x212D, 'A'},
    {0x212F, 0x2139, 'A'},
    {0x213C, 0x213F, 'A'},
    {0x2145, 0x2149, 'A'},
    {0x214E, 0x214E, 'A'},
    {0x2150, 0x2182, 'N'},
    {0x2183, 0x2184, 'A'},
    {0x2185, 0x2189, 'N'},
    {0x2460, 0x249B, 'N'},
    {0x24EA, 0x24FF, 'N'},
    {0x2776, 0x2793, 'N'},
    {0x2C00, 0x2CE4, 'A'},
    {0x2CEB, 0x2CF3, 'A'},
    {0x2CFD, 0x2CFD, 'N'},
    {0x2D00, 0x2D25, 'A'},
    {0x2D27, 0x2D27, 'A'},
    {0x2D2D, 0x2D2D, 'A'},
    {0x2D30, 0x2D67, 'A'},
    {0x2D6F, 0x2D6F, 'A'},
    {0x2D7F, 0x2D96, 'A'},
    {0x2DA0, 0x2DA6, 'A'},
    {0x2DA8, 0x2DAE, 'A'},
    {0x2DB0, 0x2DB6, 'A'},
    {0x2DB8, 0x2DBE, 'A'},
    {0x2DC0, 0x2DC6, 'A'},
    {0x2DC8, 0x2DCE, 'A'},
    {0x2DD0, 0x2DD6, 'A'},
    {0x2DD8, 0x2DDE, 'A'},
    {0x2DE0, 0x2DFF, 'A'},
    {0x2E2F, 0x2E2F, 'A'},
    {0x3005, 0x3006, 'A'},
    {0x3007, 0x3007, 'N'},
    {0x3021, 0x3029, 'N'},
    {0x302A, 0x302F, 'A'},
    {0x3031, 0x3035, 'A'},
    {0x3038, 0x303A, 'N'},
    {0x303B, 0x303C, 'A'},
    {0x3041, 0x3096, 'A'},
    {0x3099, 0x309A, 'A'},
    {0x309D, 0x309F, 'A'},
    {0x30A1, 0x30FA, 'A'},
    {0x30FC, 0x30FF, 'A'},
    {0x3105, 0x312F, 'A'},
    {0x3131, 0x318E, 'A'},
    {0x3192, 0x3195, 'N'},
    {0x31A0, 0x31BF, 'A'},
    {0x31F0, 0x31FF, 'A'},
    {0x3220, 0x3229, 'N'},
    {0x3248, 0x324F, 'N'},
    {0x3251, 0x325F, 'N'},
    {0x3280, 0x3289, 'N'},
    {0x32B1, 0x32BF, 'N'},
    {0x3400, 0x4DBF, 'A'},
    {0x4E00, 0xA48C, 'A'},
    {0xA4D0, 0xA4FD, 'A'},
    {0xA500, 0xA60C, 'A'},
    {0xA610, 0xA61F, 'A'},
    {0xA620, 0xA629, 'N'},
    {0xA62A, 0xA62B, 'A'},
    {0xA640, 0xA672, 'A'},
    {0xA674, 0xA67D, 'A'},
    {0xA67F, 0xA6E5, 'A'},
    {0xA6E6, 0xA6EF, 'N'},
    {0xA6F0, 0xA6F1, 'A'},
    {0xA717, 0xA71F, 'A'},
    {0xA722, 0xA788, 'A'},
    {0xA78B, 0xA7CD, 'A'},
    {0xA7D0, 0xA7D1, 'A'},
    {0xA7D3, 0xA7D3, 'A'},
    {0xA7D5, 0xA7DC, 'A'},
    {0xA7F2, 0xA827, 'A'},
    {0xA82C, 0xA82C, 'A'},
    {0xA830, 0xA835, 'N'},
    {0xA840, 0xA873, 'A'},
    {0xA880, 0xA8C5, 'A'},
    {0xA8D0, 0xA8D9, 'N'},
    {0xA8E0, 0xA8F7, 'A'},
    {0xA8FB, 0xA8FB, 'A'},
    {0xA8FD, 0xA8FF, 'A'},
    {0xA900, 0xA909, 'N'},
    {0xA90A, 0xA92D, 'A'},
    {0xA930, 0xA953, 'A'},
    {0xA960, 0xA97C, 'A'},
    {0xA980, 0xA9C0, 'A'},
    {0xA9CF, 0xA9CF, 'A'},
    {0xA9D0, 0xA9D9, 'N'},
    {0xA9E0, 0xA9EF, 'A'},
    {0xA9F0, 0xA9F9, 'N'},
    {0xA9FA, 0xA9FE, 'A'},
    {0xAA00, 0xAA36, 'A'},
    {0xAA40, 0xAA4D, 'A'},
    {0xAA50, 0xAA59, 'N'},
    {0xAA60, 0xAA76, 'A'},
    {0xAA7A, 0xAAC2, 'A'},
    {0xAADB, 0xAADD, 'A'},
    {0xAAE0, 0xAAEF, 'A'},
    {0xAAF2, 0xAAF6, 'A'},
    {0xAB01, 0xAB06, 'A'},
    {0xAB09, 0xAB0E, 'A'},
    {0xAB11, 0xAB16, 'A'},
    {0xAB20, 0xAB26, 'A'},
    {0xAB28, 0xAB2E, 'A'},
    {0xAB30, 0xAB5A, 'A'},
    {0xAB5C, 0xAB69, 'A'},
    {0xAB70, 0xABEA, 'A'},
    {0xABEC, 0xABED, 'A'},
    {0xABF0, 0xABF9, 'N'},
    {0xAC00, 0xD7A3, 'A'},
    {0xD7B0, 0xD7C6, 'A'},
    {0xD7CB, 0xD7FB, 'A'},
    {0xF900, 0xFA6D, 'A'},
    {0xFA70, 0xFAD9, 'A'},
    {0xFB00, 0xFB06, 'A'},
    {0xFB13, 0xFB17, 'A'},
    {0xFB1D, 0xFB28, 'A'},
    {0xFB2A, 0xFB36, 'A'},
    {0xFB38, 0xFB3C, 'A'},
    {0xFB3E, 0xFB3E, 'A'},
    {0xFB40, 0xFB41, 'A'},
    {0xFB43, 0xFB44, 'A'},
    {0xFB46, 0xFBB1, 'A'},
    {0xFBD3, 0xFD3D, 'A'},
    {0xFD50, 0xFD8F, 'A'},
    {0xFD92, 0xFDC7, 'A'},
    {0xFDF0, 0xFDFB, 'A'},
    {0xFE00, 0xFE0F, 'A'},
    {0xFE20, 0xFE2F, 'A'},
    {0xFE70, 0xFE74, 'A'},
    {0xFE76, 0xFEFC, 'A'},
    {0xFF10, 0xFF19, 'N'},
    {0xFF21, 0xFF3A, 'A'},
    {0xFF41, 0xFF5A, 'A'},
    {0xFF66, 0xFFBE, 'A'},
    {0xFFC2, 0xFFC7, 'A'},
    {0xFFCA, 0xFFCF, 'A'},
    {0xFFD2, 0xFFD7, 'A'},
    {0xFFDA, 0xFFDC, 'A'},
    {0x10000, 0x1000B, 'A'},
    {0x1000D, 0x10026, 'A'},
    {0x10028, 0x1003A, 'A'},
    {0x1003C, 0x1003D, 'A'},
    {0x1003F, 0x1004D, 'A'},
    {0x10050, 0x1005D, 'A'},
    {0x10080, 0x100FA, 'A'},
    {0x10107, 0x10133, 'N'},
    {0x10140, 0x10178, 'N'},
    {0x1018A, 0x1018B, 'N'},
    {0x101FD, 0x101FD, 'A'},
    {0x10280, 0x1029C, 'A'},
    {0x102A0, 0x102D0, 'A'},
    {0x102E0, 0x102E0, 'A'},
    {0x102E1, 0x102FB, 'N'},
    {0x10300, 0x1031F, 'A'},
    {0x10320, 0x10323, 'N'},
    {0x1032D, 0x10340, 'A'},
    {0x10341, 0x10341, 'N'},
    {0x10342, 0x10349, 'A'},
    {0x1034A, 0x1034A, 'N'},
    {0x10350, 0x1037A, 'A'},
    {0x10380, 0x1039D, 'A'},
    {0x103A0, 0x103C3, 'A'},
    {0x103C8, 0x103CF, 'A'},
    {0x103D1, 0x103D5, 'N'},
    {0x10400, 0x1049D, 'A'},
    {0x104A0, 0x104A9, 'N'},
    {0x104B0, 0x104D3, 'A'},
    {0x104D8, 0x104FB, 'A'},
    {0x10500, 0x10527, 'A'},
    {0x10530, 0x10563, 'A'},
    {0x10570, 0x1057A, 'A'},
    {0x1057C, 0x1058A, 'A'},
    {0x1058C, 0x10592, 'A'},
    {0x10594, 0x10595, 'A'},
    {0x10597, 0x105A1, 'A'},
    {0x105A3, 0x105B1, 'A'},
    {0x105B3, 0x105B9, 'A'},
    {0x105BB, 0x105BC, 'A'},
    {0x105C0, 0x105F3, 'A'},
    {0x10600, 0x10736, 'A'},
    {0x10740, 0x10755, 'A'},
    {0x10760, 0x10767, 'A'},
    {0x10780, 0x10785, 'A'},
    {0x10787, 0x107B0, 'A'},
    {0x107B2, 0x107BA, 'A'},
    {0x10800, 0x10805, 'A'},
    {0x10808, 0x10808, 'A'},
    {0x1080A, 0x10835, 'A'},
    {0x10837, 0x10838, 'A'},
    {0x1083C, 0x1083C, 'A'},
    {0x1083F, 0x10855, 'A'},
    {0x10858, 0x1085F, 'N'},
    {0x10860, 0x10876, 'A'},
    {0x10879, 0x1087F, 'N'},
    {0x10880, 0x1089E, 'A'},
    {0x108A7, 0x108AF, 'N'},
    {0x108E0, 0x108F2, 'A'},
    {0x108F4, 0x108F5, 'A'},
    {0x108FB, 0x108FF, 'N'},
    {0x10900, 0x10915, 'A'},
    {0x10916, 0x1091B, 'N'},
    {0x10920, 0x10939, 'A'},
    {0x10980, 0x109B7, 'A'},
    {0x109BC, 0x109BD, 'N'},
    {0x109BE, 0x109BF, 'A'},
    {0x109C0, 0x109CF, 'N'},
    {0x109D2, 0x109FF, 'N'},
    {0x10A00, 0x10A03, 'A'},
    {0x10A05, 0x10A06, 'A'},
    {0x10A0C, 0x10A13, 'A'},
    {0x10A15, 0x10A17, 'A'},
    {0x10A19, 0x10A35, 'A'},
    {0x10A38, 0x10A3A, 'A'},
    {0x10A3F, 0x10A3F, 'A'},
    {0x10A40, 0x10A48, 'N'},
    {0x10A60, 0x10A7C, 'A'},
    {0x10A7D, 0x10A7E, 'N'},
    {0x10A80, 0x10A9C, 'A'},
    {0x10A9D, 0x10A9F, 'N'},
    {0x10AC0, 0x10AC7, 'A'},
    {0x10AC9, 0x10AE6, 'A'},
    {0x10AEB, 0x10AEF, 'N'},
    {0x10B00, 0x10B35, 'A'},
    {0x10B40, 0x10B55, 'A'},
    {0x10B58, 0x10B5F, 'N'},
    {0x10B60, 0x10B72, 'A'},
    {0x10B78, 0x10B7F, 'N'},
    {0x10B80, 0x10B91, 'A'},
    {0x10BA9, 0x10BAF, 'N'},
    {0x10C00, 0x10C48, 'A'},
    {0x10C80, 0x10CB2, 'A'},
    {0x10CC0, 0x10CF2, 'A'},
    {0x10CFA, 0x10CFF, 'N'},
    {0x10D00, 0x10D27, 'A'},
    {0x10D30, 0x10D39, 'N'},
    {0x10D40, 0x10D49, 'N'},
    {0x10D4A, 0x10D65, 'A'},
    {0x10D69, 0x10D6D, 'A'},
    {0x10D6F, 0x10D85, 'A'},
    {0x10E60, 0x10E7E, 'N'},
    {0x10E80, 0x10EA9, 'A'},
    {0x10EAB, 0x10EAC, 'A'},
    {0x10EB0, 0x10EB1, 'A'},
    {0x10EC2, 0x10EC4, 'A'},
    {0x10EFC, 0x10F1C, 'A'},
    {0x10F1D, 0x10F26, 'N'},
    {0x10F27, 0x10F27, 'A'},
    {0x10F30, 0x10F50, 'A'},
    {0x10F51, 0x10F54, 'N'},
    {0x10F70, 0x10F85, 'A'},
    {0x10FB0, 0x10FC4, 'A'},
    {0x10FC5, 0x10FCB, 'N'},
    {0x10FE0, 0x10FF6, 'A'},
    {0x11000, 0x11046, 'A'},
    {0x11052, 0x1106F, 'N'},
    {0x11070, 0x11075, 'A'},
    {0x1107F, 0x110BA, 'A'},
    {0x110C2, 0x110C2, 'A'},
    {0x110D0, 0x110E8, 'A'},
    {0x110F0, 0x110F9, 'N'},
    {0x11100, 0x11134, 'A'},
    {0x11136, 0x1113F, 'N'},
    {0x11144, 0x11147, 'A'},
    {0x11150, 0x11173, 'A'},
    {0x11176, 0x11176, 'A'},
    {0x11180, 0x111C4, 'A'},
    {0x111C9, 0x111CC, 'A'},
    {0x111CE, 0x111CF, 'A'},
    {0x111D0, 0x111D9, 'N'},
    {0x111DA, 0x111DA, 'A'},
    {0x111DC, 0x111DC, 'A'},
    {0x111E1, 0x111F4, 'N'},
    {0x11200, 0x11211, 'A'},
    {0x11213, 0x11237, 'A'},
    {0x1123E, 0x11241, 'A'},
    {0x11280, 0x11286, 'A'},
    {0x11288, 0x11288, 'A'},
    {0x1128A, 0x1128D, 'A'},
    {0x1128F, 0x1129D, 'A'},
    {0x1129F, 0x112A8, 'A'},
    {0x112B0, 0x112EA, 'A'},
    {0x112F0, 0x112F9, 'N'},
    {0x11300, 0x11303, 'A'},
    {0x11305, 0x1130C, 'A'},
    {0x1130F, 0x11310, 'A'},
    {0x11313, 0x11328, 'A'},
    {0x1132A, 0x11330, 'A'},
    {0x11332, 0x11333, 'A'},
    {0x11335, 0x11339, 'A'},
    {0x1133B, 0x11344, 'A'},
    {0x11347, 0x11348, 'A'},
    {0x1134B, 0x1134D, 'A'},
    {0x11350, 0x11350, 'A'},
    {0x11357, 0x11357, 'A'},
    {0x1135D, 0x11363, 'A'},
    {0x11366, 0x1136C, 'A'},
    {0x11370, 0x11374, 'A'},
    {0x11380, 0x11389, 'A'},
    {0x1138B, 0x1138B, 'A'},
    {0x1138E, 0x1138E, 'A'},
    {0x11390, 0x113B5, 'A'},
    {0x113B7, 0x113C0, 'A'},
    {0x113C2, 0x113C2, 'A'},
    {0x113C5, 0x113C5, 'A'},
    {0x113C7, 0x113CA, 'A'},
    {0x113CC, 0x113D3, 'A'},
    {0x113E1, 0x113E2, 'A'},
    {0x11400, 0x1144A, 'A'},
    {0x11450, 0x11459, 'N'},
    {0x1145E, 0x11461, 'A'},
    {0x11480, 0x114C5, 'A'},
    {0x114C7, 0x114C7, 'A'},
    {0x114D0, 0x114D9, 'N'},
    {0x11580, 0x115B5, 'A'},
    {0x115B8, 0x115C0, 'A'},
    {0x115D8, 0x115DD, 'A'},
    {0x11600, 0x11640, 'A'},
    {0x11644, 0x11644, 'A'},
    {0x11650, 0x11659, 'N'},
    {0x11680, 0x116B8, 'A'},
    {0x116C0, 0x116C9, 'N'},
    {0x116D0, 0x116E3, 'N'},
    {0x11700, 0x1171A, 'A'},
    {0x1171D, 0x1172B, 'A'},
    {0x11730, 0x1173B, 'N'},
    {0x11740, 0x11746, 'A'},
    {0x11800, 0x1183A, 'A'},
    {0x118A0, 0x118DF, 'A'},
    {0x118E0, 0x118F2, 'N'},
    {0x118FF, 0x11906, 'A'},
    {0x11909, 0x11909, 'A'},
    {0x1190C, 0x11913, 'A'},
    {0x11915, 0x11916, 'A'},
    {0x11918, 0x11935, 'A'},
    {0x11937, 0x11938, 'A'},
    {0x1193B, 0x11943, 'A'},
    {0x11950, 0x11959, 'N'},
    {0x119A0, 0x119A7, 'A'},
    {0x119AA, 0x119D7, 'A'},
    {0x119DA, 0x119E1, 'A'},
    {0x119E3, 0x119E4, 'A'},
    {0x11A00, 0x11A3E, 'A'},
    {0x11A47, 0x11A47, 'A'},
    {0x11A50, 0x11A99, 'A'},
    {0x11A9D, 0x11A9D, 'A'},
    {0x11AB0, 0x11AF8, 'A'},
    {0x11BC0, 0x11BE0, 'A'},
    {0x11BF0, 0x11BF9, 'N'},
    {0x11C00, 0x11C08, 'A'},
    {0x11C0A, 0x11C36, 'A'},
    {0x11C38, 0x11C40, 'A'},
    {0x11C50, 0x11C6C, 'N'},
    {0x11C72, 0x11C8F, 'A'},
    {0x11C92, 0x11CA7, 'A'},
    {0x11CA9, 0x11CB6, 'A'},
    {0x11D00, 0x11D06, 'A'},
    {0x11D08, 0x11D09, 'A'},
    {0x11D0B, 0x11D36, 'A'},
    {0x11D3A, 0x11D3A, 'A'},
    {0x11D3C, 0x11D3D, 'A'},
    {0x11D3F, 0x11D47, 'A'},
    {0x11D50, 0x11D59, 'N'},
    {0x11D60, 0x11D65, 'A'},
    {0x11D67, 0x11D68, 'A'},
    {0x11D6A, 0x11D8E, 'A'},
    {0x11D90, 0x11D91, 'A'},
    {0x11D93, 0x11D98, 'A'},
    {0x11DA0, 0x11DA9, 'N'},
    {0x11EE0, 0x11EF6, 'A'},
    {0x11F00, 0x11F10, 'A'},
    {0x11F12, 0x11F3A, 'A'},
    {0x11F3E, 0x11F42, 'A'},
    {0x11F50, 0x11F59, 'N'},
    {0x11F5A, 0x11F5A, 'A'},
    {0x11FB0, 0x11FB0, 'A'},
    {0x11FC0, 0x11FD4, 'N'},
    {0x12000, 0x12399, 'A'},
    {0x12400, 0x1246E, 'N'},
    {0x12480, 0x12543, 'A'},
    {0x12F90, 0x12FF0, 'A'},
    {0x13000, 0x1342F, 'A'},
    {0x13440, 0x13455, 'A'},
    {0x13460, 0x143FA, 'A'},
    {0x14400, 0x14646, 'A'},
    {0x16100, 0x1612F, 'A'},
    {0x16130, 0x16139, 'N'},
    {0x16800, 0x16A38, 'A'},
    {0x16A40, 0x16A5E, 'A'},
    {0x16A60, 0x16A69, 'N'},
    {0x16A70, 0x16ABE, 'A'},
    {0x16AC0, 0x16AC9, 'N'},
    {0x16AD0, 0x16AED, 'A'},
    {0x16AF0, 0x16AF4, 'A'},
    {0x16B00, 0x16B36, 'A'},
    {0x16B40, 0x16B43, 'A'},
    {0x16B50, 0x16B59, 'N'},
    {0x16B5B, 0x16B61, 'N'},
    {0x16B63, 0x16B77, 'A'},
    {0x16B7D, 0x16B8F, 'A'},
    {0x16D40, 0x16D6C, 'A'},
    {0x16D70, 0x16D79, 'N'},
    {0x16E40, 0x16E7F, 'A'},
    {0x16E80, 0x16E96, 'N'},
    {0x16F00, 0x16F4A, 'A'},
    {0x16F4F, 0x16F87, 'A'},
    {0x16F8F, 0x16F9F, 'A'},
    {0x16FE0, 0x16FE1, 'A'},
    {0x16FE3, 0x16FE4, 'A'},
    {0x16FF0, 0x16FF1, 'A'},
    {0x17000, 0x187F7, 'A'},
    {0x18800, 0x18CD5, 'A'},
    {0x18CFF, 0x18D08, 'A'},
    {0x1AFF0, 0x1AFF3, 'A'},
    {0x1AFF5, 0x1AFFB, 'A'},
    {0x1AFFD, 0x1AFFE, 'A'},
    {0x1B000, 0x1B122, 'A'},
    {0x1B132, 0x1B132, 'A'},
    {0x1B150, 0x1B152, 'A'},
    {0x1B155, 0x1B155, 'A'},
    {0x1B164, 0x1B167, 'A'},
    {0x1B170, 0x1B2FB, 'A'},
    {0x1BC00, 0x1BC6A, 'A'},
    {0x1BC70, 0x1BC7C, 'A'},
    {0x1BC80, 0x1BC88, 'A'},
    {0x1BC90, 0x1BC99, 'A'},
    {0x1BC9D, 0x1BC9E, 'A'},
    {0x1CCF0, 0x1CCF9, 'N'},
    {0x1CF00, 0x1CF2D, 'A'},
    {0x1CF30, 0x1CF46, 'A'},
    {0x1D165, 0x1D169, 'A'},
    {0x1D16D, 0x1D172, 'A'},
    {0x1D17B, 0x1D182, 'A'},
    {0x1D185, 0x1D18B, 'A'},
    {0x1D1AA, 0x1D1AD, 'A'},
    {0x1D242, 0x1D244, 'A'},
    {0x1D2C0, 0x1D2D3, 'N'},
    {0x1D2E0, 0x1D2F3, 'N'},
    {0x1D360, 0x1D378, 'N'},
    {0x1D400, 0x1D454, 'A'},
    {0x1D456, 0x1D49C, 'A'},
    {0x1D49E, 0x1D49F, 'A'},
    {0x1D4A2, 0x1D4A2, 'A'},
    {0x1D4A5, 0x1D4A6, 'A'},
    {0x1D4A9, 0x1D4AC, 'A'},
    {0x1D4AE, 0x1D4B9, 'A'},
    {0x1D4BB, 0x1D4BB, 'A'},
    {0x1D4BD, 0x1D4C3, 'A'},
    {0x1D4C5, 0x1D505, 'A'},
    {0x1D507, 0x1D50A, 'A'},
    {0x1D50D, 0x1D514, 'A'},
    {0x1D516, 0x1D51C, 'A'},
    {0x1D51E, 0x1D539, 'A'},
    {0x1D53B, 0x1D53E, 'A'},
    {0x1D540, 0x1D544, 'A'},
    {0x1D546, 0x1D546, 'A'},
    {0x1D54A, 0x1D550, 'A'},
    {0x1D552, 0x1D6A5, 'A'},
    {0x1D6A8, 0x1D6C0, 'A'},
    {0x1D6C2, 0x1D6DA, 'A'},
    {0x1D6DC, 0x1D6FA, 'A'},
    {0x1D6FC, 0x1D714, 'A'},
    {0x1D716, 0x1D734, 'A'},
    {0x1D736, 0x1D74E, 'A'},
    {0x1D750, 0x1D76E, 'A'},
    {0x1D770, 0x1D788, 'A'},
    {0x1D78A, 0x1D7A8, 'A'},
    {0x1D7AA, 0x1D7C2, 'A'},
    {0x1D7C4, 0x1D7CB, 'A'},
    {0x1D7CE, 0x1D7FF, 'N'},
    {0x1DA00, 0x1DA36, 'A'},
    {0x1DA3B, 0x1DA6C, 'A'},
    {0x1DA75, 0x1DA75, 'A'},
    {0x1DA84, 0x1DA84, 'A'},
    {0x1DA9B, 0x1DA9F, 'A'},
    {0x1DAA1, 0x1DAAF, 'A'},
    {0x1DF00, 0x1DF1E, 'A'},
    {0x1DF25, 0x1DF2A, 'A'},
    {0x1E000, 0x1E006, 'A'},
    {0x1E008, 0x1E018, 'A'},
    {0x1E01B, 0x1E021, 'A'},
    {0x1E023, 0x1E024, 'A'},
    {0x1E026, 0x1E02A, 'A'},
    {0x1E030, 0x1E06D, 'A'},
    {0x1E08F, 0x1E08F, 'A'},
    {0x1E100, 0x1E12C, 'A'},
    {0x1E130, 0x1E13D, 'A'},
    {0x1E140, 0x1E149, 'N'},
    {0x1E14E, 0x1E14E, 'A'},
    {0x1E290, 0x1E2AE, 'A'},
    {0x1E2C0, 0x1E2EF, 'A'},
    {0x1E2F0, 0x1E2F9, 'N'},
    {0x1E4D0, 0x1E4EF, 'A'},
    {0x1E4F0, 0x1E4F9, 'N'},
    {0x1E5D0, 0x1E5F0, 'A'},
    {0x1E5F1, 0x1E5FA, 'N'},
    {0x1E7E0, 0x1E7E6, 'A'},
    {0x1E7E8, 0x1E7EB, 'A'},
    {0x1E7ED, 0x1E7EE, 'A'},
    {0x1E7F0, 0x1E7FE, 'A'},
    {0x1E800, 0x1E8C4, 'A'},
    {0x1E8C7, 0x1E8CF, 'N'},
    {0x1E8D0, 0x1E8D6, 'A'},
    {0x1E900, 0x1E94B, 'A'},
    {0x1E950, 0x1E959, 'N'},
    {0x1EC71, 0x1ECAB, 'N'},
    {0x1ECAD, 0x1ECAF, 'N'},
    {0x1ECB1, 0x1ECB4, 'N'},
    {0x1ED01, 0x1ED2D, 'N'},
    {0x1ED2F, 0x1ED3D, 'N'},
    {0x1EE00, 0x1EE03, 'A'},
    {0x1EE05, 0x1EE1F, 'A'},
    {0x1EE21, 0x1EE22, 'A'},
    {0x1EE24, 0x1EE24, 'A'},
    {0x1EE27, 0x1EE27, 'A'},
    {0x1EE29, 0x1EE32, 'A'},
    {0x1EE34, 0x1EE37, 'A'},
    {0x1EE39, 0x1EE39, 'A'},
    {0x1EE3B, 0x1EE3B, 'A'},
    {0x1EE42, 0x1EE42, 'A'},
    {0x1EE47, 0x1EE47, 'A'},
    {0x1EE49, 0x1EE49, 'A'},
    {0x1EE4B, 0x1EE4B, 'A'},
    {0x1EE4D, 0x1EE4F, 'A'},
    {0x1EE51, 0x1EE52, 'A'},
    {0x1EE54, 0x1EE54, 'A'},
    {0x1EE57, 0x1EE57, 'A'},
    {0x1EE59, 0x1EE59, 'A'},
    {0x1EE5B, 0x1EE5B, 'A'},
    {0x1EE5D, 0x1EE5D, 'A'},
    {0x1EE5F, 0x1EE5F, 'A'},
    {0x1EE61, 0x1EE62, 'A'},
    {0x1EE64, 0x1EE64, 'A'},
    {0x1EE67, 0x1EE6A, 'A'},
    {0x1EE6C, 0x1EE72, 'A'},
    {0x1EE74, 0x1EE77, 'A'},
    {0x1EE79, 0x1EE7C, 'A'},
    {0x1EE7E, 0x1EE7E, 'A'},
    {0x1EE80, 0x1EE89, 'A'},
    {0x1EE8B, 0x1EE9B, 'A'},
    {0x1EEA1, 0x1EEA3, 'A'},
    {0x1EEA5, 0x1EEA9, 'A'},
    {0x1EEAB, 0x1EEBB, 'A'},
    {0x1F100, 0x1F10C, 'N'},
    {0x1FBF0, 0x1FBF9, 'N'},
    {0x20000, 0x2A6DF, 'A'},
    {0x2A700, 0x2B739, 'A'},
    {0x2B740, 0x2B81D, 'A'},
    {0x2B820, 0x2CEA1, 'A'},
    {0x2CEB0, 0x2EBE0, 'A'},
    {0x2EBF0, 0x2EE5D, 'A'},
    {0x2F800, 0x2FA1D, 'A'},
    {0x30000, 0x3134A, 'A'},
    {0x31350, 0x323AF, 'A'},
    {0xE0100, 0xE01EF, 'A'},
};
inline char category(uint32_t cp) {
    size_t lo = 0, hi = sizeof(kCategories) / sizeof(kCategories[0]);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kCategories[mid].last < cp) lo = mid + 1;
        else hi = mid;
    }
    return lo < sizeof(kCategories) / sizeof(kCategories[0]) &&
           kCategories[lo].first <= cp ? kCategories[lo].category : 0;
}
// Unicode White_Space (the meaning of regex \s), including NBSP and
// paragraph separators. A codepoint's UTF-8 byte length is kept intact.
inline bool is_space(uint32_t cp) {
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 ||
           cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}
struct Char { uint32_t cp; size_t len; char cat; };
inline Char char_at(const std::string& text, size_t i) {
    const size_t len = size_t(u8len((unsigned char)text[i]));
    const uint32_t cp = u8cp(text, i, int(len));
    return {cp, std::min(len, text.size() - i), category(cp)};
}

} // namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = text.size();
    while (i < n) {
        const size_t start = i;
        const Char c = char_at(text, i);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c.cp == '\'') {
            // Unicode case folding also equates long s (U+017F) with s.
            if (text.compare(i, 3, "'\xC5\xBF") == 0) {
                i += 3;
                out.push_back(text.substr(start, i - start));
                continue;
            }
            static const char* conts[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char* suffix : conts) {
                const size_t len = std::strlen(suffix);
                if (i + 1 + len > n) continue;
                bool same = true;
                for (size_t k = 0; k < len; ++k)
                    same &= std::tolower((unsigned char)text[i + 1 + k]) == suffix[k];
                if (same) { i += 1 + len; break; }
            }
            if (i != start) { out.push_back(text.substr(start, i - start)); continue; }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        {
            size_t j = i;
            if (!is_nl(c.cp) && c.cat != 'A' && c.cat != 'N') j += c.len;
            if (j < n) {
                const Char next = char_at(text, j);
                if (next.cat == 'A') {
                    do { j += char_at(text, j).len; }
                    while (j < n && char_at(text, j).cat == 'A');
                    out.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
            }
        }

        // \p{N}: exactly one numeric codepoint, including non-ASCII digits.
        if (c.cat == 'N') {
            i += c.len;
            out.push_back(text.substr(start, i - start));
            continue;
        }

        //  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            size_t j = i + (c.cp == ' ' ? 1 : 0);
            const size_t punct_start = j;
            while (j < n) {
                const Char x = char_at(text, j);
                if (is_space(x.cp) || x.cat == 'A' || x.cat == 'N') break;
                j += x.len;
            }
            if (j > punct_start) {
                while (j < n && is_nl(char_at(text, j).cp)) j += char_at(text, j).len;
                out.push_back(text.substr(start, j - start));
                i = j;
                continue;
            }
        }

        // \s*[\r\n]+: greedily consumes the whole whitespace run, then
        // backtracks to the last CR/LF. Whitespace after it remains for the
        // next alternative at the next position.
        {
            size_t j = i, last_nl_end = i;
            while (j < n && is_space(char_at(text, j).cp)) {
                const Char x = char_at(text, j);
                j += x.len;
                if (is_nl(x.cp)) last_nl_end = j;
            }
            if (last_nl_end > i) {
                out.push_back(text.substr(start, last_nl_end - start));
                i = last_nl_end;
                continue;
            }
        }

        // \s+(?!\S) first, then \s+. Before non-space text the first
        // branch backtracks one whole whitespace codepoint for the next token.
        if (is_space(c.cp)) {
            size_t j = i, last_start = i;
            while (j < n && is_space(char_at(text, j).cp)) {
                last_start = j;
                j += char_at(text, j).len;
            }
            if (j < n && last_start > i) j = last_start;
            out.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // An unmatched codepoint is isolated by Split.
        i += c.len;
        out.push_back(text.substr(start, c.len));
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special) const {
    std::vector<int32_t> out;
    if (add_special && add_bos_token_ && bos_>=0) out.push_back(bos_);

    size_t i = 0;
    while (i < text.size()) {
        // Special tokens bypass the pre-tokenizer and BPE entirely.
        size_t best_len = 0;
        int32_t best_id = -1;
        for (const auto& kv : special_by_text_) {
            const size_t L = kv.first.size();
            if (L > best_len && text.compare(i, L, kv.first) == 0) {
                best_len = L;
                best_id  = kv.second;
            }
        }
        if (best_id >= 0) {
            out.push_back(best_id);
            i += best_len;
            continue;
        }

        // Find where the next special token begins, and pre-tokenize the
        // plain run up to it.
        size_t stop = text.size();
        for (const auto& kv : special_by_text_) {
            if (kv.first.empty()) continue;
            const size_t p2 = text.find(kv.first, i);
            if (p2 != std::string::npos && p2 < stop) stop = p2;
        }

        const std::string run = text.substr(i, stop - i);
        if (sentencepiece_) {
            auto ids=spm_word(run);out.insert(out.end(),ids.begin(),ids.end());
            i=stop;continue;
        }
        for (const std::string& piece : pre_tokenize(run)) {
            std::string mapped;
            for (unsigned char ch : piece) mapped += byte_to_uni_[ch];

            for (const std::string& sym : bpe_word(mapped)) {
                auto it = tok_to_id_.find(sym);
                if (it != tok_to_id_.end()) { out.push_back(it->second); continue; }
                // Unknown merge result: fall back to single characters
                // rather than dropping input.
                for (size_t k = 0; k < sym.size();) {
                    const int L = u8len((unsigned char)sym[k]);
                    auto it2 = tok_to_id_.find(sym.substr(k, size_t(L)));
                    if (it2 != tok_to_id_.end()) out.push_back(it2->second);
                    k += size_t(L);
                }
            }
        }
        i = stop;
    }
    if (add_special && add_eos_token_ && eos_>=0) out.push_back(eos_);
    return out;
}

std::string Tokenizer::decode_one(int32_t id) const {
    if (id < 0 || size_t(id) >= id_to_tok_.size()) return {};
    const std::string& t = id_to_tok_[size_t(id)];
    if (special_ids_.count(id)) return {};        // don't print control tokens

    if (sentencepiece_) {
        if (token_type_[size_t(id)]==6 && t.size()==6 && t.substr(0,3)=="<0x" && t[5]=='>') {
            char* end=nullptr;long value=std::strtol(t.substr(3,2).c_str(),&end,16);
            if (end && *end==0) return std::string(1,char(value));
        }
        std::string out=t;const std::string marker="\xE2\x96\x81";
        for(size_t p=0;(p=out.find(marker,p))!=std::string::npos;) {out.replace(p,marker.size()," ");++p;}
        return out;
    }

    std::string out;
    for (size_t i = 0; i < t.size();) {
        const unsigned char c = t[i];
        size_t len = 1;
        if ((c & 0xF8) == 0xF0) len = 4;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xE0) == 0xC0) len = 2;
        auto it = uni_to_byte_.find(t.substr(i, len));
        if (it != uni_to_byte_.end()) out += char(it->second);
        else out += t.substr(i, len);
        i += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) out += decode_one(id);
    if (sentencepiece_ && spm_add_prefix_ && !out.empty() && out[0]==' ') out.erase(0,1);
    return out;
}

std::string Tokenizer::apply_chat_template(const std::string& user,
                                           const std::string& system) const {
    std::vector<ChatMessage> messages;
    if (!system.empty()) messages.push_back({"system", system});
    messages.push_back({"user", user});
    return apply_chat_template(messages);
}

std::string Tokenizer::apply_chat_template(
        const std::vector<ChatMessage>& messages) const {
    // Muse-Glimmer's current tokenizer is Harmony-style, not ChatML.  Keep
    // this branch keyed to a checkpoint token so Qwen/Ornith retain their
    // existing template.  This mirrors transformers/vLLM's default template,
    // including the date-dependent default system message.
    if (special_by_text_.count("<|begin_of_text|>")) {
        char date[11] = {};
        const std::time_t now = std::time(nullptr);
        std::tm tm{};
#if defined(_WIN32)
        localtime_s(&tm, &now);
#else
        localtime_r(&now, &tm);
#endif
        std::strftime(date, sizeof(date), "%Y-%m-%d", &tm);
        const std::string default_system =
            "You are a helpful AI assistant.\n"
            "Knowledge cutoff: 2026-01-04.\n"
            "Current date: " + std::string(date) + ".\n\n"
            "Reasoning strength: high.\n\n"
            "# Valid recipients: \"self\", \"user\".";
        const std::string meta =
            "\n\nReasoning strength: high.\n\n"
            "# Valid recipients: \"self\", \"user\".";
        std::string out = "<|begin_of_text|>";
        bool has_system = false;
        for (const auto& message : messages)
            if (message.role == "system") has_system = true;
        if (!has_system)
            out += "<|start|>system<|message|>" + default_system + "<|eot|>";
        for (const auto& message : messages) {
            if (message.role == "system") {
                out += "<|start|>system<|message|>" + message.content + meta +
                       "<|eot|>";
            } else if (message.role == "user") {
                out += "<|start|>user<|message|>" + message.content + "<|eot|>";
            } else if (message.role == "assistant") {
                // OpenAI history contains the user-visible answer; reasoning
                // was intentionally filtered from the preceding response.
                out += "<|start|>assistant to=user<|message|>" + message.content +
                       "<|eot|>";
            }
        }
        out += "<|start|>assistant";
        return out;
    }
    // K2-Horizon (IFM): ChatML-like with its own markers, the BOS token
    // first, NO newline between turns, and an assistant turn that opens a
    // <ifm|think> block (reasoning_effort "high", the template's default).
    // Byte-identical to the checkpoint's Jinja template for system / user /
    // assistant turns (rendered with jinja2, 2026-10-06); an earlier assistant
    // turn is replayed with an empty think block, as the template does when
    // no reasoning is supplied.
    if (special_by_text_.count("<|ifm|im_start|>")) {
        std::string out = "<|ifm|begin_of_text|>";
        for (const auto& message : messages) {
            if (message.role == "system" || message.role == "user")
                out += "<|ifm|im_start|>" + message.role + "\n" + message.content + "<|ifm|im_end|>";
            else if (message.role == "assistant")
                out += "<|ifm|im_start|>assistant<ifm|think>\n</ifm|think>" + message.content +
                       "<|ifm|im_end|>";
        }
        out += "<|ifm|im_start|>assistant\n<ifm|think>\n";
        return out;
    }
    std::string out;
    for (const auto& message : messages) {
        if (message.role == "system" || message.role == "user" ||
            message.role == "assistant")
            out += "<|im_start|>" + message.role + "\n" + message.content +
                   "<|im_end|>\n";
    }
    out += "<|im_start|>assistant\n<think>\n";
    return out;
}

namespace {
std::string tmpl_trim(const std::string& s) {
    const char* ws = " \t\n\r\f\v";
    const size_t b = s.find_first_not_of(ws);
    if (b == std::string::npos) return {};
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}
const char* kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
    "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
    "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n"
    "Reminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> "
    "block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be "
    "specified\n- You may provide optional reasoning for your function call in natural language "
    "BEFORE the function call, but NOT after\n- If there is no function call available, answer the "
    "question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";
const char* kEffortXhigh =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";
const char* kEffortLow =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
    "conclusion without unnecessary elaboration.";
} // namespace

// The server's renderer.  For the Qwen3.5-family XML-tool template it is that
// template, line for line (tools block, reasoning-effort system text, past
// turns with their <think> block, tool calls, grouped tool responses);
// anything else keeps the legacy rendering above.
std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages,
                                           const ChatOptions& opt) const {
    if (special_by_text_.count("<|begin_of_text|>") || !tmpl_xml_tools_) {
        if (!opt.tools.empty())
            throw std::invalid_argument("this model's chat template has no tool-call format GRIMOIRE renders");
        std::string out = apply_chat_template(messages);
        if (!opt.enable_thinking && !special_by_text_.count("<|begin_of_text|>")) {
            const std::string open = "<think>\n";
            if (out.size() >= open.size() && out.compare(out.size() - open.size(), open.size(), open) == 0)
                out += "\n</think>\n\n";
        }
        return out;
    }
    if (messages.empty()) throw std::invalid_argument("No messages provided.");
    std::string ri;
    if (tmpl_effort_ && opt.enable_thinking) {
        std::string effort = opt.reasoning_effort.empty() ? "xhigh" : opt.reasoning_effort;
        if (effort == "high") effort = "xhigh";
        if (effort != "xhigh" && effort != "medium" && effort != "low")
            throw std::invalid_argument("Unexpected reasoning effort " + effort +
                                        ". Supported types are xhigh (default), medium, and low.");
        ri = effort == "xhigh" ? kEffortXhigh : effort == "low" ? kEffortLow : "";
    }
    std::string out;
    const bool first_system = messages[0].role == "system";
    if (!opt.tools.empty()) {
        out += "<|im_start|>system\n";
        if (!ri.empty()) out += ri + "\n\n";
        out += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (const auto& t : opt.tools) out += "\n" + t;
        out += "\n</tools>";
        out += kToolInstructions;
        if (first_system) {
            const std::string c = tmpl_trim(messages[0].content);
            if (!c.empty()) out += "\n\n" + c;
        }
        out += "<|im_end|>\n";
    } else if (first_system) {
        const std::string c = tmpl_trim(messages[0].content);
        if (!tmpl_effort_) out += "<|im_start|>system\n" + c + "<|im_end|>\n";
        else if (!c.empty()) out += "<|im_start|>system\n" + (ri.empty() ? "" : ri + "\n\n") + c + "<|im_end|>\n";
        else if (!ri.empty()) out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    } else if (!ri.empty()) {
        out += "<|im_start|>system\n" + ri + "<|im_end|>\n";
    }
    bool have_query = false;
    for (size_t i = messages.size(); i-- > 0;) {
        if (messages[i].role != "user") continue;
        const std::string c = tmpl_trim(messages[i].content);
        const std::string a = "<tool_response>", b = "</tool_response>";
        if (!(c.compare(0, a.size(), a) == 0 && c.size() >= b.size() &&
              c.compare(c.size() - b.size(), b.size(), b) == 0)) { have_query = true; break; }
    }
    if (!have_query) throw std::invalid_argument("No user query found in messages.");
    for (size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& m = messages[i];
        std::string c = tmpl_trim(m.content);
        if (m.role == "system") {
            if (i != 0) throw std::invalid_argument("System message must be at the beginning.");
        } else if (m.role == "user") {
            out += "<|im_start|>user\n" + c + "<|im_end|>\n";
        } else if (m.role == "assistant") {
            std::string r;
            if (m.has_reasoning) r = m.reasoning;
            else if (tmpl_think_split_ && c.find("</think>") != std::string::npos) {
                std::string head = c.substr(0, c.find("</think>"));
                while (!head.empty() && head.back() == '\n') head.pop_back();
                const size_t o = head.rfind("<think>");
                if (o != std::string::npos) head = head.substr(o + 7);
                r = head.substr(std::min(head.find_first_not_of('\n'), head.size()));
                c = c.substr(c.rfind("</think>") + 8);
                c = c.substr(std::min(c.find_first_not_of('\n'), c.size()));
            }
            r = tmpl_trim(r);
            out += "<|im_start|>assistant\n<think>\n" + r + "\n</think>\n\n" + c;
            for (size_t j = 0; j < m.tool_calls.size(); ++j) {
                const ChatToolCall& tc = m.tool_calls[j];
                if (j == 0) out += std::string(tmpl_trim(c).empty() ? "" : "\n\n") +
                                   "<tool_call>\n<function=" + tc.name + ">\n";
                else out += "\n<tool_call>\n<function=" + tc.name + ">\n";
                for (const auto& kv : tc.args)
                    out += "<parameter=" + kv.first + ">\n" + kv.second + "\n</parameter>\n";
                out += "</function>\n</tool_call>";
            }
            out += "<|im_end|>\n";
        } else if (m.role == "tool") {
            if (i > 0 && messages[i - 1].role != "tool") out += "<|im_start|>user";
            out += "\n<tool_response>\n" + c + "\n</tool_response>";
            if (i + 1 == messages.size() || messages[i + 1].role != "tool") out += "<|im_end|>\n";
        } else {
            throw std::invalid_argument("Unexpected message role.");
        }
    }
    out += "<|im_start|>assistant\n";
    out += opt.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return out;
}

} // namespace b70
