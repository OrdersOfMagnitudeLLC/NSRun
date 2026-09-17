#pragma once

#include "ns_repack.h"
#include "nsm.h"

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>

// GGUF Type enum (as defined in GGUF spec)
enum class GGUFType : uint32_t {
    UINT8 = 0,
    INT8 = 1,
    UINT16 = 2,
    INT16 = 3,
    UINT32 = 4,
    INT32 = 5,
    FLOAT32 = 6,
    BOOL = 7,
    STRING = 8,
    ARRAY = 9,
    UINT64 = 10,
    INT64 = 11,
    FLOAT64 = 12
};

// GGML Type enum (tensor data types)
enum class GGMLType : uint32_t {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    BF16 = 30
};

// GGUFValue variant covering all scalar types + string
using GGUFValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t,
    float, bool, std::string, uint64_t, int64_t, double
>;

// Tensor descriptor
struct GGUFTensor {
    std::string name;
    GGMLType type;
    std::vector<uint64_t> shape;
    uint64_t offset;
    size_t cl_offset = 0;
    uint32_t nsm_n_clusters = 0;
    uint64_t nsm_data_bytes = 0;
    bool on_disk = false;
    std::vector<NSMCluster> clusters;
};

// Model parameters extracted from metadata
struct ModelParams {
    std::string architecture;
    std::string key_prefix;
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t n_embd;
    uint32_t n_ff;
    uint32_t n_vocab;
    uint32_t head_dim;
    uint32_t max_seq_len;
    float rope_freq_base;
    float rope_freq_scale;
};

class GGUFParser {
public:
    GGUFParser();
    ~GGUFParser();

    // Load and parse GGUF file
    bool load(const std::string& path);

    // Read only header/metadata/tensor-descriptors via fseek/fread.
    // Avoids mmap of the full file; use this for .gguf companions in .nsm mode.
    bool load_pread(const std::string& path, size_t head_bytes = 128 * 1024 * 1024);

    // Accessors
    const ModelParams& params() const { return params_; }
    const std::vector<GGUFTensor>& tensors() const { return tensors_; }
    const std::unordered_map<std::string, GGUFValue>& metadata() const { return metadata_; }
    const std::vector<std::string>& tokens() const { return tokens_; }
    const std::vector<std::string>& merges() const { return merges_; }

    // Metadata getters
    bool get_uint32(const std::string& key, uint32_t& out) const;
    bool get_uint64(const std::string& key, uint64_t& out) const;
    bool get_float32(const std::string& key, float& out) const;
    bool get_string(const std::string& key, std::string& out) const;

    // Data section access
    const uint8_t* data_ptr() const { return data_ptr_; }
    size_t data_offset() const { return data_offset_; }
    size_t full_file_size() const { return full_file_size_; }

    // Read a tensor's raw bytes by offset. Works for both mmap and pread modes.
    bool read_tensor(const GGUFTensor& t, uint8_t* out, size_t n) const;

    bool is_nsm() const { return is_nsm_; }

    // Access the raw GGUF metadata+tensor-descriptor bytes captured by load_pread()
    const std::vector<uint8_t>& head_buffer() const { return head_buf_; }
    size_t head_data_offset() const { return data_offset_; }

    // Load .nsm weights; .nsm is self-contained (gguf_path kept for call-site compatibility)
    bool load_nsm(const std::string& nsm_path, const std::string& gguf_path = "");

    // NSM lazy-load state populated by nsm_loader (inference loop reads these)
    std::vector<size_t> nsm_layer_offsets;
    std::vector<size_t> nsm_layer_bytes;
    std::vector<NSMLayerIndex> nsm_layer_index;
    int nsm_fd = -1;

private:
    // Parsing stages
    bool parse_header();
    bool parse_metadata();
    bool parse_tensors();
    bool extract_model_params();

    // File mapping
    int fd_;
    uint8_t* mapped_;
    size_t file_size_;
    bool mapped_is_mmap_ = false;
    std::vector<uint8_t> head_buf_;

    // Parsed data
    std::unordered_map<std::string, GGUFValue> metadata_;
    std::vector<GGUFTensor> tensors_;
    ModelParams params_;
    size_t data_offset_;
    const uint8_t* data_ptr_;
    size_t full_file_size_ = 0;
    AlignedVectorU8 q4k_data_;  // owned packed Q4_K panel data when loaded from .nsm
    bool is_nsm_ = false;
    std::vector<std::string> tokens_;
    std::vector<std::string> merges_;

    // Helper to read values from mapped memory
    template<typename T>
    T read_value(size_t offset) const {
        return *reinterpret_cast<const T*>(mapped_ + offset);
    }
};
