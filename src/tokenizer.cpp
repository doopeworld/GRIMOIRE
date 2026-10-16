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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <ctime>
#include <stdexcept>

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
    build_byte_tables();
    add_bos_token_ = false;
    // Raw-text scoring follows the checkpoint's tokenizer setting, not the
    // presence of a BOS id. Qwen-family tokenizers normally set this false.
    const std::string tokenizer_cfg = read_file(dir + "/tokenizer_config.json");
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

    const std::string js = read_file(dir + "/tokenizer.json");
    if (js.empty()) { err = "cannot read " + dir + "/tokenizer.json"; return false; }

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
        std::string tmpl = read_file(dir + "/chat_template.jinja");
        if (tmpl.empty()) tmpl = read_file(dir + "/tokenizer_config.json");
        tmpl_xml_tools_ = tmpl.find("<function=example_function_name>") != std::string::npos &&
                          tmpl.find("do not tell the user about function calls") != std::string::npos &&
                          tmpl.find("multi_step_tool=true") != std::string::npos;
        tmpl_effort_ = tmpl_xml_tools_ &&
                       tmpl.find("Reasoning effort is set to xhigh.") != std::string::npos;
        tmpl_think_split_ = tmpl_xml_tools_ &&
                            tmpl.find("content.split('</think>')") != std::string::npos;
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

// ---------------------------------------------------------------------
// Pre-tokenizer. The checkpoint specifies this Split regex:
//
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   |[^\r\n\p{L}\p{N}]?\p{L}+
//   |\p{N}{1,3}
//   | ?[^\s\p{L}\p{N}]+[\r\n]*
//   |\s*[\r\n]+
//   |\s+(?!\S)
//   |\s+
//
// Splitting on spaces alone (the previous behaviour) leaves things like
// "user\nWrite" as ONE pre-token. No merge rule matches a chunk that
// long, so BPE falls all the way back to single bytes: 47 tokens where
// the reference produces 18. Every downstream kernel then receives
// shredded input regardless of whether it is correct.
//
// Implemented directly rather than with std::regex, which does not
// support Unicode properties and would be far too slow on a prompt.
namespace {

inline bool is_ascii_letter(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}
inline bool is_digit(unsigned char c) { return c >= '0' && c <= '9'; }
inline bool is_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}
inline bool is_nl(unsigned char c) { return c == '\n' || c == '\r'; }

// UTF-8 sequence length from the lead byte.
inline int u8len(unsigned char c) {
    if ((c & 0x80) == 0)    return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Decode a UTF-8 sequence to its codepoint. Returns 0 on a malformed
// lead byte, which callers treat as "not a letter".
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

// Unicode punctuation/symbol blocks that are NOT \p{L}. Qwen's
// pre-tokenizer groups these with the punctuation run
// (' ?[^\s\p{L}\p{N}]+'), not with adjacent words.
inline bool is_punct_cp(uint32_t cp) {
    if (cp >= 0x00A0 && cp <= 0x00BF) return true;  // Latin-1 punct/symbols
    if (cp == 0x00D7 || cp == 0x00F7) return true;  // multiplication/division
    if (cp >= 0x2000 && cp <= 0x206F) return true;  // General Punctuation
    if (cp >= 0x2E00 && cp <= 0x2E7F) return true;  // Supplemental Punctuation
    if (cp >= 0x3000 && cp <= 0x303F) return true;  // CJK Symbols & Punctuation
    return false;
}

// \p{L}: ASCII letters plus non-ASCII codepoints that are not punctuation.
//
// This USED to return true for EVERY multi-byte codepoint, with a comment
// claiming that was a safe approximation because "the punctuation that is
// not gets handled by the byte-level fallback without changing the token
// count for normal text". That is false, and it was expensive: a curly
// apostrophe, smart quote or em dash was glued onto the adjacent word, so
// the chunk handed to BPE ("\u201cHe) never appears in the merge table,
// nothing merged, and the whole run fell back to raw bytes.
//
// MEASURED 2026-09-05 against vLLM serving the same Qwen3.8-27B tokenizer,
// identical strings through both /v1/completions (prompt_tokens):
//     ascii only     vLLM 16   GRIMOIRE 16    <- why this was never caught;
//     smart quotes   vLLM 17   GRIMOIRE 34       the in-tree verifier's
//     em dash        vLLM 15   GRIMOIRE 27       reference string is pure
//     apostrophes    vLLM 18   GRIMOIRE 38       ASCII and passes regardless
// Real prose is ~2x over-tokenised: 24051 chars of Conan Doyle gave vLLM
// 5983 tokens and GRIMOIRE 7475. That inflates every PP tok/s figure (more
// tokens counted for the same text), makes TG genuinely worse (each token
// carries less text), and feeds the model byte-fallback sequences it never
// saw in training -- which is a plausible contributor to the MTP draft
// acceptance collapse on real text (12-26%) versus ASCII-ish synthetic
// prompts (74%).
inline bool is_letter_at(const std::string& t, size_t i, int& len) {
    const unsigned char c = t[i];
    len = u8len(c);
    if (len > 1) {
        const uint32_t cp = u8cp(t, i, len);
        if (cp == 0) return false;
        return !is_punct_cp(cp);
    }
    return is_ascii_letter(c);
}

} // namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = text.size();

    while (i < n) {
        const size_t start = i;
        const unsigned char c = text[i];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c == '\'' && i + 1 < n) {
            static const char* conts[] = {"s","t","m","d","re","ve","ll"};
            bool hit = false;
            for (const char* k : conts) {
                const size_t kl = std::strlen(k);
                if (i + kl < n) {
                    bool eq = true;
                    for (size_t z = 0; z < kl; ++z) {
                        const char a = char(std::tolower((unsigned char)text[i + 1 + z]));
                        if (a != k[z]) { eq = false; break; }
                    }
                    if (eq) { i += 1 + kl; hit = true; break; }
                }
            }
            if (hit) { out.push_back(text.substr(start, i - start)); continue; }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+   -- optional single non-letter, then letters
        {
            size_t j = i;
            int len = 0;
            if (!is_nl(text[j]) && !is_letter_at(text, j, len) && !is_digit(text[j])) {
                const size_t save = j;
                j += u8len(text[j]);
                if (j < n && is_letter_at(text, j, len)) {
                    while (j < n && is_letter_at(text, j, len)) j += len;
                    out.push_back(text.substr(start, j - start));
                    i = j;
                    continue;
                }
                j = save;
            }
            if (j < n && is_letter_at(text, j, len)) {
                while (j < n && is_letter_at(text, j, len)) j += len;
                out.push_back(text.substr(start, j - start));
                i = j;
                continue;
            }
        }

        // \p{N}{1,3} -- the Muse tokenizer groups decimal runs in chunks of
        // at most three digits.  Splitting every digit changes prompt IDs.
        if (is_digit(c)) {
            size_t j = i;
            while (j < n && is_digit((unsigned char)text[j]) && j - i < 3) ++j;
            out.push_back(text.substr(i, j - i));
            i = j;
            continue;
        }

        // ' ?[^\s\p{L}\p{N}]+[\r\n]*'  -- optional space, punctuation run
        {
            size_t j = i;
            if (text[j] == ' ') ++j;
            int len = 0;
            size_t k = j;
            while (k < n && !is_space(text[k]) && !is_letter_at(text, k, len) && !is_digit(text[k]))
                k += u8len(text[k]);
            if (k > j) {
                while (k < n && is_nl(text[k])) ++k;
                out.push_back(text.substr(start, k - start));
                i = k;
                continue;
            }
        }

        // \s*[\r\n]+
        {
            size_t j = i;
            while (j < n && is_space(text[j]) && !is_nl(text[j])) ++j;
            if (j < n && is_nl(text[j])) {
                while (j < n && is_nl(text[j])) ++j;
                out.push_back(text.substr(start, j - start));
                i = j;
                continue;
            }
        }

        // \s+(?!\S) and \s+
        if (is_space(c)) {
            size_t j = i;
            while (j < n && is_space(text[j])) ++j;
            // (?!\S): if more text follows, the last space starts the
            // next token instead of ending this one.
            if (j < n && j - i > 1) --j;
            out.push_back(text.substr(start, j - start));
            i = j;
            continue;
        }

        // anything left: emit one character
        out.push_back(text.substr(i, u8len(c)));
        i += u8len(c);
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool) const {
    std::vector<int32_t> out;

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
    return out;
}

std::string Tokenizer::decode_one(int32_t id) const {
    if (id < 0 || size_t(id) >= id_to_tok_.size()) return {};
    const std::string& t = id_to_tok_[size_t(id)];
    if (special_ids_.count(id)) return {};        // don't print control tokens

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
