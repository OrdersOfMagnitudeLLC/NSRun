#include "tokenizer.h"
#include <algorithm>
#include <sstream>
#include <cctype>
#include <fstream>
#include <locale>
#include <codecvt>
#include <iostream>
#include <array>
#include <climits>

// Byte encoder/decoder for GPT-2 style byte-level tokenization
static std::array<std::string, 256> byte_encoder_;
static std::unordered_map<std::string, uint8_t> byte_decoder_;

// Initialize GPT-2 byte encoding
static void bytes_to_unicode() {
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    // Visible bytes that map to themselves
    auto is_visible = [](int b) {
        return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    };

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        // Visible bytes map to their own codepoint; non-visible map to 256+
        int codepoint = is_visible(b) ? b : 256 + n++;

        // Convert codepoint to UTF-8
        std::string utf8;
        if (codepoint < 0x80) {
            utf8 += static_cast<char>(codepoint);
        } else if (codepoint < 0x800) {
            utf8 += static_cast<char>(0xC0 | (codepoint >> 6));
            utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
        } else if (codepoint < 0x10000) {
            utf8 += static_cast<char>(0xE0 | (codepoint >> 12));
            utf8 += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F));
            utf8 += static_cast<char>(0x80 | (codepoint & 0x3F));
        }
        byte_encoder_[b] = utf8;

        // Build decoder
        byte_decoder_[byte_encoder_[b]] = static_cast<uint8_t>(b);
    }

    // Explicitly verify space (32) maps to U+0120 (Ġ)
    // U+0120 = 288 in decimal, which is 256 + 32 (since 32 is not in visible set)
    // 32 should be the 33rd non-visible byte (0-32 are all non-visible)
    // So it should map to 256 + 32 = 288 = U+0120
}

BPETokenizer::BPETokenizer()
    : bos_id_(151643), eos_id_(151645) {
}

BPETokenizer::~BPETokenizer() {
}

bool BPETokenizer::load(const GGUFParser& parser) {
    // Initialize byte encoder/decoder
    bytes_to_unicode();

    const std::vector<std::string>& tokens = parser.tokens();
    const std::vector<std::string>& merges = parser.merges();

    if (tokens.empty()) {
        std::cerr << "Error: No tokens found in GGUF file" << std::endl;
        return false;
    }

    // Build token_to_id_ and id_to_token_
    token_to_id_.clear();
    id_to_token_.clear();
    id_to_token_.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        token_to_id_[tokens[i]] = static_cast<int32_t>(i);
        id_to_token_.push_back(tokens[i]);
    }

    // Detect gpt2/tiktoken vocab (Qwen2.5, etc.)
    std::string model;
    if (parser.get_string("tokenizer.ggml.model", model)) {
        is_gpt2_ = (model == "gpt2" || model == "gpt-2");
    }

    // Parse merges
    merges_.clear();
    merge_rank_.clear();
    merges_.reserve(merges.size());

    for (size_t i = 0; i < merges.size(); ++i) {
        std::string merge = merges[i];
        // Split on space
        size_t space_pos = merge.find(' ');
        if (space_pos != std::string::npos) {
            std::string token_a = merge.substr(0, space_pos);
            std::string token_b = merge.substr(space_pos + 1);
            merges_.push_back({token_a, token_b});
            merge_rank_[token_a + " " + token_b] = static_cast<int32_t>(i);
        }
    }

    // Find special tokens from metadata first, then common names
    uint32_t meta_bos = 0, meta_eos = 0;
    if (parser.get_uint32("tokenizer.ggml.bos_token_id", meta_bos)) bos_id_ = static_cast<int32_t>(meta_bos);
    if (parser.get_uint32("tokenizer.ggml.eos_token_id", meta_eos)) eos_id_ = static_cast<int32_t>(meta_eos);

    auto bos_it = token_to_id_.find("<s>");
    if (bos_it != token_to_id_.end()) bos_id_ = bos_it->second;
    auto eos_it = token_to_id_.find("</s>");
    if (eos_it != token_to_id_.end()) eos_id_ = eos_it->second;

    std::cout << "Loaded tokenizer: " << tokens.size() << " tokens, " << merges.size() << " merges" << std::endl;
    std::cout << "BOS token id: " << bos_id_ << ", EOS token id: " << eos_id_ << std::endl;

    // Register Qwen2.5 ChatML end-of-turn token '07' as a stop token
    auto stop_it = token_to_id_.find("07");
    if (stop_it != token_to_id_.end()) {
        stop_token_ids_.push_back(stop_it->second);
        std::cout << "Stop token '07' id: " << stop_it->second << std::endl;
    }

    return true;
}

bool BPETokenizer::is_stop_token(int32_t id) const {
    if (id == eos_id_) return true;
    for (int32_t stop_id : stop_token_ids_) {
        if (id == stop_id) return true;
    }
    return false;
}

bool BPETokenizer::load_merges_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Failed to open merges file: " << path << std::endl;
        return false;
    }
    std::string line;
    merges_.clear();
    merge_rank_.clear();
    size_t rank = 0;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t sp = line.find(' ');
        if (sp == std::string::npos) continue;
        std::string a = line.substr(0, sp);
        std::string b = line.substr(sp + 1);
        // strip any trailing CR
        if (!b.empty() && b.back() == '\r') b.pop_back();
        merges_.push_back({a, b});
        merge_rank_[a + " " + b] = static_cast<int32_t>(rank++);
    }
    std::cout << "Loaded " << rank << " merges from " << path << std::endl;
    return true;
}

std::string BPETokenizer::byte_encode(const std::string& text) {
    std::string result;
    for (unsigned char c : text) {
        result += byte_encoder_[c];
    }
    return result;
}

std::string BPETokenizer::byte_decode(const std::string& encoded) {
    std::string result;
    size_t i = 0;
    while (i < encoded.size()) {
        // Try to match the longest possible UTF-8 sequence in decoder
        bool found = false;
        for (size_t len = 4; len >= 1 && !found; --len) {
            if (i + len <= encoded.size()) {
                std::string substr = encoded.substr(i, len);
                auto it = byte_decoder_.find(substr);
                if (it != byte_decoder_.end()) {
                    result += static_cast<char>(it->second);
                    i += len;
                    found = true;
                }
            }
        }
        if (!found) {
            // Unknown character, skip
            i++;
        }
    }
    return result;
}

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    // Step 1: Build initial word vector directly from byte encoder
    std::vector<std::string> word;
    for (unsigned char byte : text) {
        word.push_back(byte_encoder_[byte]);
    }

    // Step 2: BPE merge loop
    while (word.size() > 1) {
        int32_t best_rank = INT_MAX;
        int best_i = -1;

        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::string key = word[i] + " " + word[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i = static_cast<int>(i);
            }
        }

        if (best_i == -1) {
            break;
        }

        // Merge the pair
        std::string merged = word[best_i] + word[best_i + 1];
        word.erase(word.begin() + best_i, word.begin() + best_i + 2);
        word.insert(word.begin() + best_i, merged);
    }

    // Step 3: Map tokens to ids
    std::vector<int32_t> token_ids;
    for (const std::string& token : word) {
        auto it = token_to_id_.find(token);
        if (it != token_to_id_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Unknown token - use 0 as fallback
            token_ids.push_back(0);
        }
    }

    return token_ids;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string concatenated;

    if (is_gpt2_) {
        // GPT-2/tiktoken BPE: token strings are already byte-encoded
        // (e.g. Ġ for spaces, Ċ for newlines). Do not apply SentencePiece ▁ handling.
        for (int32_t id : tokens) {
            if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
                concatenated += id_to_token_[id];
            }
        }
    } else {
        // SentencePiece: word-initial ▁ (U+2581) → Ġ, then byte decode to space
        static const std::string SP_SPACE = byte_encoder_[32];
        for (int32_t id : tokens) {
            if (id >= 0 && static_cast<size_t>(id) < id_to_token_.size()) {
                const std::string& tok = id_to_token_[id];
                if (tok.size() >= 3 &&
                    tok[0] == '\xE2' && tok[1] == '\x96' && tok[2] == '\x81') {
                    concatenated += SP_SPACE;
                    concatenated += tok.substr(3);
                } else {
                    concatenated += tok;
                }
            }
        }
    }

    // Step 2: Iterate codepoint by codepoint and decode bytes
    std::string result;
    size_t i = 0;
    while (i < concatenated.size()) {
        // Extract UTF-8 codepoint as string
        unsigned char c = concatenated[i];
        size_t char_len = 1;
        if ((c & 0x80) == 0) {
            // ASCII: 1 byte
            char_len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            // 2-byte sequence
            char_len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            // 3-byte sequence
            char_len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            // 4-byte sequence
            char_len = 4;
        }

        if (i + char_len <= concatenated.size()) {
            std::string codepoint = concatenated.substr(i, char_len);
            auto it = byte_decoder_.find(codepoint);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
            } else {
                if (is_gpt2_) {
                    // gpt2/tiktoken stores plain UTF-8 token strings; pass through unknown chars.
                    result += codepoint;
                } else {
                    // Unknown codepoint, skip
                }
            }
            i += char_len;
        } else {
            // Incomplete sequence, skip
            i++;
        }
    }

    return result;
}
