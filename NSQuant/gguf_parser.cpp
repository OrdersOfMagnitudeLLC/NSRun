#include "gguf_parser.h"
#include "nsm_loader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cstring>

static const uint32_t GGUF_MAGIC = 0x46554747; // 'GGUF'
static const uint32_t GGUF_VERSION = 3;

GGUFParser::GGUFParser()
    : fd_(-1), mapped_(nullptr), file_size_(0), data_offset_(0), data_ptr_(nullptr) {
    params_ = {};
}

GGUFParser::~GGUFParser() {
    if (mapped_ != nullptr && mapped_is_mmap_) {
        munmap(mapped_, file_size_);
    }
    if (fd_ != -1) {
        close(fd_);
    }
    if (nsm_fd >= 0) {
        close(nsm_fd);
    }
}

bool GGUFParser::load(const std::string& path) {
    // Open file
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        std::cerr << "Failed to get file size" << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
    }
    full_file_size_ = st.st_size;
    file_size_ = st.st_size;

    // mmap the file
    mapped_ = static_cast<uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    mapped_is_mmap_ = true;
    if (mapped_ == MAP_FAILED) {
        mapped_is_mmap_ = false;
        mapped_ = nullptr;
        std::cerr << "Failed to mmap file" << std::endl;
        close(fd_);
        fd_ = -1;
        mapped_ = nullptr;
        return false;
    }

    // Parse header
    if (!parse_header()) {
        return false;
    }

    // Parse metadata
    if (!parse_metadata()) {
        return false;
    }

    // Parse tensors
    if (!parse_tensors()) {
        return false;
    }

    // Extract model parameters
    if (!extract_model_params()) {
        return false;
    }

    return true;
}

bool GGUFParser::load_pread(const std::string& path, size_t head_bytes) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) == -1) {
        std::cerr << "Failed to get file size" << std::endl;
        close(fd_);
        fd_ = -1;
        return false;
    }
    full_file_size_ = st.st_size;
    file_size_ = std::min<size_t>(st.st_size, head_bytes);

    head_buf_.resize(file_size_);
    size_t got = 0;
    off_t off = 0;
    while (got < file_size_) {
        ssize_t r = ::pread(fd_, head_buf_.data() + got, file_size_ - got, off + (off_t)got);
        if (r <= 0) {
            std::cerr << "Failed to read GGUF metadata header (got " << got
                      << " of " << file_size_ << ")" << std::endl;
            close(fd_);
            fd_ = -1;
            return false;
        }
        got += (size_t)r;
    }

    mapped_ = head_buf_.data();
    mapped_is_mmap_ = false;

    if (!parse_header()) return false;
    if (!parse_metadata()) return false;
    if (!parse_tensors()) return false;
    if (!extract_model_params()) return false;

    if (data_offset_ > file_size_) {
        std::cerr << "GGUF metadata+tensor descriptors exceed head_bytes (" << head_bytes
                  << "), use full load() instead" << std::endl;
        return false;
    }

    // Release the mapped pointer so that tensor data reads use pread() later.
    // head_buf_ still owns the metadata buffer.
    mapped_ = nullptr;
    data_ptr_ = nullptr;
    return true;
}

bool GGUFParser::parse_header() {
    size_t offset = 0;

    // Verify magic
    uint32_t magic = read_value<uint32_t>(offset);
    if (magic != GGUF_MAGIC) {
        std::cerr << "Invalid GGUF magic: 0x" << std::hex << magic << std::endl;
        return false;
    }
    offset += sizeof(uint32_t);

    // Verify version
    uint32_t version = read_value<uint32_t>(offset);
    if (version != GGUF_VERSION) {
        std::cerr << "Unsupported GGUF version: " << version << " (expected " << GGUF_VERSION << ")" << std::endl;
        return false;
    }
    offset += sizeof(uint32_t);

    // Skip tensor count and kv count (will be read later)
    offset += sizeof(uint64_t) * 2;

    return true;
}

bool GGUFParser::parse_metadata() {
    size_t offset = 12; // After magic (4) + version (4) + tensor_count (8) is wrong, need to recalculate

    // Re-read from start
    offset = 0;
    offset += sizeof(uint32_t); // magic
    offset += sizeof(uint32_t); // version

    uint64_t tensor_count = read_value<uint64_t>(offset);
    (void)tensor_count; // Suppress unused warning
    offset += sizeof(uint64_t);

    uint64_t kv_count = read_value<uint64_t>(offset);
    offset += sizeof(uint64_t);

    // Parse all KV pairs
    for (uint64_t i = 0; i < kv_count; ++i) {
        // Read key name
        uint64_t key_len = read_value<uint64_t>(offset);
        offset += sizeof(uint64_t);
        std::string key(reinterpret_cast<char*>(mapped_ + offset), key_len);
        offset += key_len;

        // Read value type
        uint32_t value_type = read_value<uint32_t>(offset);
        offset += sizeof(uint32_t);

        // Parse value based on type
        switch (static_cast<GGUFType>(value_type)) {
            case GGUFType::UINT8: {
                uint8_t val = read_value<uint8_t>(offset);
                offset += sizeof(uint8_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::INT8: {
                int8_t val = read_value<int8_t>(offset);
                offset += sizeof(int8_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::UINT16: {
                uint16_t val = read_value<uint16_t>(offset);
                offset += sizeof(uint16_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::INT16: {
                int16_t val = read_value<int16_t>(offset);
                offset += sizeof(int16_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::UINT32: {
                uint32_t val = read_value<uint32_t>(offset);
                offset += sizeof(uint32_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::INT32: {
                int32_t val = read_value<int32_t>(offset);
                offset += sizeof(int32_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::FLOAT32: {
                float val = read_value<float>(offset);
                offset += sizeof(float);
                metadata_[key] = val;
                break;
            }
            case GGUFType::BOOL: {
                bool val = read_value<uint8_t>(offset) != 0;
                offset += sizeof(uint8_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::STRING: {
                uint64_t str_len = read_value<uint64_t>(offset);
                offset += sizeof(uint64_t);
                if (offset + str_len > file_size_) {
                    std::cerr << "String length exceeds file size for key: " << key << std::endl;
                    return false;
                }
                std::string val(reinterpret_cast<char*>(mapped_ + offset), str_len);
                offset += str_len;
                metadata_[key] = val;
                break;
            }
            case GGUFType::ARRAY: {
                // Read array type and length
                uint32_t array_type = read_value<uint32_t>(offset);
                offset += sizeof(uint32_t);
                uint64_t array_len = read_value<uint64_t>(offset);
                offset += sizeof(uint64_t);

                // Store only first element for numeric types, skip entirely for strings
                if (array_len > 0) {
                    switch (static_cast<GGUFType>(array_type)) {
                        case GGUFType::UINT8: {
                            uint8_t val = read_value<uint8_t>(offset);
                            offset += sizeof(uint8_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::INT8: {
                            int8_t val = read_value<int8_t>(offset);
                            offset += sizeof(int8_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::UINT16: {
                            uint16_t val = read_value<uint16_t>(offset);
                            offset += sizeof(uint16_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::INT16: {
                            int16_t val = read_value<int16_t>(offset);
                            offset += sizeof(int16_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::UINT32: {
                            uint32_t val = read_value<uint32_t>(offset);
                            offset += sizeof(uint32_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::INT32: {
                            int32_t val = read_value<int32_t>(offset);
                            offset += sizeof(int32_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::FLOAT32: {
                            float val = read_value<float>(offset);
                            offset += sizeof(float);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::UINT64: {
                            uint64_t val = read_value<uint64_t>(offset);
                            offset += sizeof(uint64_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::INT64: {
                            int64_t val = read_value<int64_t>(offset);
                            offset += sizeof(int64_t);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::FLOAT64: {
                            double val = read_value<double>(offset);
                            offset += sizeof(double);
                            metadata_[key] = val;
                            break;
                        }
                        case GGUFType::STRING: {
                            // Store string arrays for tokenizer data
                            if (key == "tokenizer.ggml.tokens") {
                                tokens_.reserve(array_len);
                                for (uint64_t j = 0; j < array_len; ++j) {
                                    uint64_t str_len = read_value<uint64_t>(offset);
                                    offset += sizeof(uint64_t);
                                    std::string val(reinterpret_cast<char*>(mapped_ + offset), str_len);
                                    offset += str_len;
                                    tokens_.push_back(val);
                                }
                            } else if (key == "tokenizer.ggml.merges") {
                                merges_.reserve(array_len);
                                for (uint64_t j = 0; j < array_len; ++j) {
                                    uint64_t str_len = read_value<uint64_t>(offset);
                                    offset += sizeof(uint64_t);
                                    std::string val(reinterpret_cast<char*>(mapped_ + offset), str_len);
                                    offset += str_len;
                                    merges_.push_back(val);
                                }
                            } else {
                                // Skip other string arrays
                                for (uint64_t j = 0; j < array_len; ++j) {
                                    uint64_t str_len = read_value<uint64_t>(offset);
                                    offset += sizeof(uint64_t) + str_len;
                                }
                                std::cerr << "Skipping string array " << key << " (" << array_len << " elements)" << std::endl;
                            }
                            break;
                        }
                        default:
                            std::cerr << "Skipping array with unsupported type: " << array_type << std::endl;
                            offset += array_len * 8; // Skip entire array (rough estimate)
                            break;
                    }
                    // Skip remaining elements for numeric types
                    if (array_len > 1 && static_cast<GGUFType>(array_type) != GGUFType::STRING) {
                        std::cerr << "Skipping array " << key << " (stored first element only, " << (array_len - 1) << " elements skipped)" << std::endl;
                        // Calculate element size and skip
                        size_t elem_size = 0;
                        switch (static_cast<GGUFType>(array_type)) {
                            case GGUFType::UINT8:
                            case GGUFType::INT8:
                                elem_size = 1;
                                break;
                            case GGUFType::UINT16:
                            case GGUFType::INT16:
                                elem_size = 2;
                                break;
                            case GGUFType::UINT32:
                            case GGUFType::INT32:
                            case GGUFType::FLOAT32:
                                elem_size = 4;
                                break;
                            case GGUFType::UINT64:
                            case GGUFType::INT64:
                            case GGUFType::FLOAT64:
                                elem_size = 8;
                                break;
                            default:
                                break;
                        }
                        offset += (array_len - 1) * elem_size;
                    }
                }
                break;
            }
            case GGUFType::UINT64: {
                uint64_t val = read_value<uint64_t>(offset);
                offset += sizeof(uint64_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::INT64: {
                int64_t val = read_value<int64_t>(offset);
                offset += sizeof(int64_t);
                metadata_[key] = val;
                break;
            }
            case GGUFType::FLOAT64: {
                double val = read_value<double>(offset);
                offset += sizeof(double);
                metadata_[key] = val;
                break;
            }
            default:
                std::cerr << "Unknown GGUF type: " << value_type << " for key: " << key << std::endl;
                return false;
        }
    }

    return true;
}

bool GGUFParser::parse_tensors() {
    size_t offset = 12; // After magic + version + tensor_count + kv_count

    // Re-read tensor count and kv count
    offset = 0;
    offset += sizeof(uint32_t); // magic
    offset += sizeof(uint32_t); // version

    uint64_t tensor_count = read_value<uint64_t>(offset);
    offset += sizeof(uint64_t);

    uint64_t kv_count = read_value<uint64_t>(offset);
    offset += sizeof(uint64_t);

    // Skip metadata section
    for (uint64_t i = 0; i < kv_count; ++i) {
        // Skip key
        uint64_t key_len = read_value<uint64_t>(offset);
        offset += sizeof(uint64_t) + key_len;

        // Skip value
        uint32_t value_type = read_value<uint32_t>(offset);
        offset += sizeof(uint32_t);

        switch (static_cast<GGUFType>(value_type)) {
            case GGUFType::UINT8:
            case GGUFType::INT8:
            case GGUFType::BOOL:
                offset += sizeof(uint8_t);
                break;
            case GGUFType::UINT16:
            case GGUFType::INT16:
                offset += sizeof(uint16_t);
                break;
            case GGUFType::UINT32:
            case GGUFType::INT32:
            case GGUFType::FLOAT32:
                offset += sizeof(uint32_t);
                break;
            case GGUFType::UINT64:
            case GGUFType::INT64:
            case GGUFType::FLOAT64:
                offset += sizeof(uint64_t);
                break;
            case GGUFType::STRING: {
                uint64_t str_len = read_value<uint64_t>(offset);
                offset += sizeof(uint64_t) + str_len;
                break;
            }
            case GGUFType::ARRAY: {
                uint32_t array_type = read_value<uint32_t>(offset);
                offset += sizeof(uint32_t);
                uint64_t array_len = read_value<uint64_t>(offset);
                offset += sizeof(uint64_t);
                // Skip array elements based on type
                switch (static_cast<GGUFType>(array_type)) {
                    case GGUFType::UINT8:
                    case GGUFType::INT8:
                        offset += array_len * sizeof(uint8_t);
                        break;
                    case GGUFType::UINT16:
                    case GGUFType::INT16:
                        offset += array_len * sizeof(uint16_t);
                        break;
                    case GGUFType::UINT32:
                    case GGUFType::INT32:
                    case GGUFType::FLOAT32:
                        offset += array_len * sizeof(uint32_t);
                        break;
                    case GGUFType::UINT64:
                    case GGUFType::INT64:
                    case GGUFType::FLOAT64:
                        offset += array_len * sizeof(uint64_t);
                        break;
                    case GGUFType::STRING:
                        // Skip string array
                        for (uint64_t j = 0; j < array_len; ++j) {
                            uint64_t str_len = read_value<uint64_t>(offset);
                            offset += sizeof(uint64_t) + str_len;
                        }
                        break;
                    default:
                        offset += array_len * 8; // Fallback
                        break;
                }
                break;
            }
            default:
                std::cerr << "Unknown type in metadata skip: " << value_type << std::endl;
                return false;
        }
    }

    // Parse tensor descriptors
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensor tensor;

        // Read name
        uint64_t name_len = read_value<uint64_t>(offset);
        offset += sizeof(uint64_t);
        tensor.name = std::string(reinterpret_cast<char*>(mapped_ + offset), name_len);
        offset += name_len;

        // Read n_dims
        uint32_t n_dims = read_value<uint32_t>(offset);
        offset += sizeof(uint32_t);

        // Read shape
        tensor.shape.resize(n_dims);
        for (uint32_t j = 0; j < n_dims; ++j) {
            tensor.shape[j] = read_value<uint64_t>(offset);
            offset += sizeof(uint64_t);
        }

        // Read type
        uint32_t type_val = read_value<uint32_t>(offset);
        offset += sizeof(uint32_t);
        tensor.type = static_cast<GGMLType>(type_val);

        // Read offset
        tensor.offset = read_value<uint64_t>(offset);
        offset += sizeof(uint64_t);

        tensors_.push_back(tensor);
    }

    // Data section starts at next 32-byte alignment boundary
    data_offset_ = offset;
    if (data_offset_ % 32 != 0) {
        data_offset_ = ((data_offset_ / 32) + 1) * 32;
    }
    data_ptr_ = mapped_ + data_offset_;

    return true;
}

bool GGUFParser::extract_model_params() {
    // Get architecture
    if (!get_string("general.architecture", params_.architecture)) {
        std::cerr << "Warning: general.architecture not found" << std::endl;
        params_.architecture = "";
    }

    // Build key_prefix dynamically
    params_.key_prefix = params_.architecture + ".";

    // Helper to get uint32 with default
    auto get_uint32_def = [this](const std::string& key, uint32_t def) -> uint32_t {
        uint32_t val;
        if (get_uint32(key, val)) {
            return val;
        }
        std::cerr << "Warning: " << key << " not found, using default: " << def << std::endl;
        return def;
    };

    // Helper to get float with default
    auto get_float_def = [this](const std::string& key, float def, bool silent = false) -> float {
        float val;
        if (get_float32(key, val)) {
            return val;
        }
        if (!silent) {
            std::cerr << "Warning: " << key << " not found, using default: " << def << std::endl;
        }
        return def;
    };

    // Extract parameters using key_prefix with architecture-specific suffixes
    params_.n_layers = get_uint32_def(params_.key_prefix + "block_count", 0);
    params_.n_heads = get_uint32_def(params_.key_prefix + "attention.head_count", 0);
    params_.n_kv_heads = get_uint32_def(params_.key_prefix + "attention.head_count_kv", params_.n_heads);
    params_.n_embd = get_uint32_def(params_.key_prefix + "embedding_length", 0);
    params_.n_ff = get_uint32_def(params_.key_prefix + "feed_forward_length", 0);
    params_.max_seq_len = get_uint32_def(params_.key_prefix + "context_length", 0);
    params_.rope_freq_base = get_float_def(params_.key_prefix + "rope.freq_base", 10000.0f);
    params_.rope_freq_scale = get_float_def(params_.key_prefix + "rope_freq_scale", 1.0f, true);

    // n_vocab: try multiple sources in order
    params_.n_vocab = 0;
    // 1. Try metadata key "general.vocab_size"
    if (get_uint32("general.vocab_size", params_.n_vocab) && params_.n_vocab != 0) {
        // Found in metadata
    } else {
        // 2. Try token_embd.weight tensor shape[1]
        for (const auto& tensor : tensors_) {
            if (tensor.name == "token_embd.weight") {
                if (tensor.shape.size() >= 2) {
                    params_.n_vocab = static_cast<uint32_t>(tensor.shape[1]);
                }
                break;
            }
        }
        // 3. Try output.weight tensor shape[0]
        if (params_.n_vocab == 0) {
            for (const auto& tensor : tensors_) {
                if (tensor.name == "output.weight") {
                    if (tensor.shape.size() >= 1) {
                        params_.n_vocab = static_cast<uint32_t>(tensor.shape[0]);
                    }
                    break;
                }
            }
        }
        // 4. Warn if still 0
        if (params_.n_vocab == 0) {
            std::cerr << "Warning: n_vocab not found in metadata or tensors, set to 0" << std::endl;
        }
    }

    // Compute head_dim
    if (params_.n_heads > 0) {
        params_.head_dim = params_.n_embd / params_.n_heads;
    } else {
        params_.head_dim = 0;
    }

    return true;
}

bool GGUFParser::get_uint32(const std::string& key, uint32_t& out) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return false;
    }
    if (std::holds_alternative<uint32_t>(it->second)) {
        out = std::get<uint32_t>(it->second);
        return true;
    }
    return false;
}

bool GGUFParser::get_uint64(const std::string& key, uint64_t& out) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return false;
    }
    if (std::holds_alternative<uint64_t>(it->second)) {
        out = std::get<uint64_t>(it->second);
        return true;
    }
    return false;
}

bool GGUFParser::get_float32(const std::string& key, float& out) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return false;
    }
    if (std::holds_alternative<float>(it->second)) {
        out = std::get<float>(it->second);
        return true;
    }
    return false;
}

bool GGUFParser::get_string(const std::string& key, std::string& out) const {
    auto it = metadata_.find(key);
    if (it == metadata_.end()) {
        return false;
    }
    if (std::holds_alternative<std::string>(it->second)) {
        out = std::get<std::string>(it->second);
        return true;
    }
    return false;
}

bool GGUFParser::read_tensor(const GGUFTensor& t, uint8_t* out, size_t n) const {
    if (mapped_ != nullptr) {
        if (data_offset_ + t.offset + n > file_size_) {
            std::cerr << "read_tensor: request exceeds mapped file size for " << t.name << std::endl;
            return false;
        }
        std::memcpy(out, mapped_ + data_offset_ + t.offset, n);
        return true;
    }
    if (fd_ == -1) {
        std::cerr << "read_tensor: no open file for " << t.name << std::endl;
        return false;
    }
    off_t file_off = static_cast<off_t>(data_offset_ + t.offset);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::pread(fd_, out + got, n - got, file_off + static_cast<off_t>(got));
        if (r <= 0) {
            std::cerr << "read_tensor: pread failed for " << t.name
                      << " (got " << got << " of " << n << ")" << std::endl;
            return false;
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

static bool read_exact_fd(int fd, void* dst, size_t n) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(fd, p + got, n - got);
        if (r <= 0) return false;
        got += (size_t)r;
    }
    return true;
}

bool GGUFParser::load_nsm(const std::string& nsm_path, const std::string& /*gguf_path*/) {
    int fd = ::open(nsm_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "Failed to open .nsm: " << nsm_path << std::endl;
        return false;
    }

    NSMHeader hdr;
    if (!read_exact_fd(fd, &hdr, sizeof(hdr))) {
        std::cerr << "Failed to read NSM header" << std::endl;
        ::close(fd);
        return false;
    }
    if (hdr.magic != NSM_MAGIC) {
        std::cerr << "Invalid NSM magic" << std::endl;
        ::close(fd);
        return false;
    }
    if (hdr.version != 2) {
        std::cerr << "Unsupported NSM version: " << hdr.version << " (expected 2)" << std::endl;
        ::close(fd);
        return false;
    }
    if (hdr.gguf_metadata_bytes == 0) {
        std::cerr << "NSM file missing embedded GGUF metadata" << std::endl;
        ::close(fd);
        return false;
    }

    // Read embedded GGUF metadata+tensor-descriptor header.
    head_buf_.resize(hdr.gguf_metadata_bytes);
    size_t got = 0;
    while (got < hdr.gguf_metadata_bytes) {
        ssize_t r = ::pread(fd, head_buf_.data() + got,
                            hdr.gguf_metadata_bytes - got,
                            (off_t)hdr.gguf_metadata_offset + (off_t)got);
        if (r <= 0) {
            std::cerr << "Failed to read embedded GGUF metadata" << std::endl;
            ::close(fd);
            return false;
        }
        got += (size_t)r;
    }
    ::close(fd);

    mapped_ = head_buf_.data();
    file_size_ = head_buf_.size();
    mapped_is_mmap_ = false;

    if (!parse_header() || !parse_metadata() || !parse_tensors() || !extract_model_params()) {
        mapped_ = nullptr;
        data_ptr_ = nullptr;
        return false;
    }

    mapped_ = nullptr;
    data_ptr_ = nullptr;

    // Load the .nsm weight payload using the shapes parsed above.
    q4k_data_.clear();
    std::vector<GGUFTensor> shapes = tensors_;
    std::vector<GGUFTensor> new_tensors;
    if (!nsm::load_nsm_weights(nsm_path, shapes, q4k_data_, new_tensors, this)) {
        return false;
    }
    tensors_.swap(new_tensors); // take ownership of the nsm_loader output
    data_ptr_ = q4k_data_.data();
    is_nsm_ = true;
    return true;
}
