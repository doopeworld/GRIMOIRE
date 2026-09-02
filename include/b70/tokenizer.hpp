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

class Tokenizer {
public:
    bool load(const std::string& dir, std::string& err);

    std::vector<int32_t> encode(const std::string& text, bool add_special = false) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    std::string decode_one(int32_t id) const;

    int32_t bos() const { return bos_; }
    int32_t eos() const { return eos_; }
    size_t  vocab_size() const { return id_to_tok_.size(); }
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

    // Apply the chat template. Qwen uses the ChatML form; the exact
    // strings come from tokenizer_config.json when present.
    std::string apply_chat_template(const std::string& user,
                                    const std::string& system = "") const;

private:
    std::vector<std::string>                     id_to_tok_;
    std::unordered_map<std::string, int32_t>     tok_to_id_;
    std::unordered_map<std::string, int32_t>     merge_rank_;   // "a b" -> rank
    std::unordered_map<int32_t, int32_t>         special_ids_;
    std::unordered_map<std::string, int32_t>     special_by_text_;

    int32_t bos_ = -1, eos_ = -1;
    size_t  bad_merges_ = 0;

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
