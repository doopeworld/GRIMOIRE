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
//  b70/tokenizer.hpp  --  byte-level BPE, reading tokenizer.json
//
//  Qwen3.5 uses GPT-2 style byte-level BPE with a 248320-token vocab.
//  The pieces that matter for correctness:
//
//    1. BYTE-LEVEL. Input is UTF-8 bytes, and every byte maps to a
//       printable placeholder character before merging (the classic
//       bytes_to_unicode table). This is why vocabulary entries look
//       like "Ġhello" rather than " hello": Ġ is byte 0x20.
//
//    2. MERGE ORDER IS THE ALGORITHM. Merges are applied in the order
//       they appear in the file, lowest rank first, repeatedly, until
//       none apply. Applying them in any other order gives a different
//       tokenization that still decodes to the same text -- so it looks
//       fine, but every token id is wrong and the model sees noise.
//
//    3. SPECIAL TOKENS bypass BPE entirely and must be matched before
//       any byte processing.
// =====================================================================
#ifndef B70_TOKENIZER_HPP
#define B70_TOKENIZER_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace b70 {

// One function call of an assistant turn, arguments in their order with each
// value as the template prints it (a string as is, anything else as JSON).
struct ChatToolCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args;
};
struct ChatMessage {
    std::string role;
    std::string content;
    std::string reasoning;                 // assistant reasoning_content
    bool has_reasoning = false;
    std::vector<ChatToolCall> tool_calls;  // assistant tool_calls
};
// Request-level template inputs: tool definitions (already tojson-printed),
// thinking on/off and the reasoning effort ("" = the template default).
struct ChatOptions {
    std::vector<std::string> tools;
    bool enable_thinking = true;
    std::string reasoning_effort;
};

class Tokenizer {
public:
    bool load(const std::string& dir, std::string& err);

    std::vector<int32_t> encode(const std::string& text, bool add_special = false) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string decode_one(int32_t id) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    size_t  vocab_size() const { return id_to_tok_.size(); }
    const std::string& raw_token(size_t id) const { return id_to_tok_.at(id); }
    size_t  merge_count() const { return merge_rank_.size(); }
    size_t  bad_merges()  const { return bad_merges_; }
    bool    has_merge(const std::string& a, const std::string& b) const {
        return merge_rank_.count(a + " " + b) != 0;
    }
    bool    is_special(int32_t id) const {
        return special_ids_.count(id) != 0;
    }
    // Id of a special token by its literal text, or -1. Harmony models end an
    // assistant turn with <|eot|>, which is NOT eos_ -- a server that stops
    // only on eos() runs past the answer and keeps talking to itself.
    int32_t special_id(const std::string& text) const {
        auto it = special_by_text_.find(text);
        return it == special_by_text_.end() ? -1 : it->second;
    }
    // The second token that ends an answer, or -1: Harmony's <|eot|>, K2's
    // <|ifm|endoftext|>, ChatML's <|endoftext|> (generation_config.json lists
    // it next to the end-of-turn token).  A server that stops on eos() alone
    // runs past the answer on any model whose turn ends with the other one.
    int32_t eos2() const { return eos2_; }

    // Apply the chat template. Qwen uses the ChatML form; the exact
    // strings come from tokenizer_config.json when present.
    std::string apply_chat_template(const std::string& user,
                                    const std::string& system = "") const;
    // Render an ordered conversation without dropping prior turns. Harmony
    // models use recipient channels; Qwen/Ornith retain ChatML formatting.
    std::string apply_chat_template(const std::vector<ChatMessage>& messages) const;
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    const ChatOptions& opt) const;
    // The checkpoint's chat_template.jinja is the Qwen3.5-family template with
    // XML tool calls (<tool_call><function=...>): Qwen3.8 / Agnes / Ornith.
    // Only then are tools rendered and parsed; other templates refuse them.
    bool supports_tools() const { return tmpl_xml_tools_; }
    bool thinking_template() const { return special_by_text_.count("</think>") != 0; }

private:
    std::vector<std::string>                     id_to_tok_;
    std::unordered_map<std::string, int32_t>     tok_to_id_;
    std::unordered_map<std::string, int32_t>     merge_rank_;   // "a b" -> rank
    std::unordered_map<int32_t, int32_t>         special_ids_;
    std::unordered_map<std::string, int32_t>     special_by_text_;

    int32_t bos_ = -1, eos_ = -1, eos2_ = -1;
    size_t  bad_merges_ = 0;
    // read from the checkpoint's chat template (chat_template.jinja or
    // tokenizer_config.json), see apply_chat_template
    bool tmpl_xml_tools_ = false;      // the Qwen3.5-family tool block
    bool tmpl_effort_ = false;         // Qwen3.8: reasoning-effort system text
    bool tmpl_think_split_ = false;    // Ornith: reasoning recovered from content

    // byte <-> placeholder-codepoint tables
    std::string byte_to_uni_[256];
    std::unordered_map<std::string, uint8_t> uni_to_byte_;

    void build_byte_tables();
    std::vector<std::string> bpe_word(const std::string& word) const;

    // Splits text the way the checkpoint's Split regex does. Without it
    // whole phrases arrive at BPE as one chunk, no merge matches, and
    // everything degrades to single bytes.
    std::vector<std::string> pre_tokenize(const std::string& text) const;
};

} // namespace b70
#endif
