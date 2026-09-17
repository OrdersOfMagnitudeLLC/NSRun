#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "gguf_parser.h"

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Load tokenizer data from GGUF parser
    bool load(const GGUFParser& parser);

    // Load BPE merges from a text file (one "a b" merge per line)
    bool load_merges_file(const std::string& path);

    // Encode text to token ids
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token ids back to text
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Special token ids
    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    size_t vocab_size() const { return id_to_token_.size(); }
    bool is_stop_token(int32_t id) const;

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    bool is_gpt2_ = false;
    int32_t bos_id_;
    int32_t eos_id_;
    std::vector<int32_t> stop_token_ids_;

    // GPT-2 byte encoding utilities
    static std::string byte_encode(const std::string& text);
    static std::string byte_decode(const std::string& encoded);
};
