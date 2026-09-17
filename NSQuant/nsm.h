// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#ifndef NSM_H
#define NSM_H

#include <cstdint>

#define NSM_MAGIC 0x314D534E  // "NSM1"

typedef struct {
    uint32_t magic;
    uint32_t version;       // 2
    uint32_t n_layers;
    uint32_t n_tensors;
    uint64_t data_offset;   // byte offset to first tensor data
    uint64_t layer_table_offset; // byte offset to per-layer packed-data index
    uint64_t gguf_metadata_offset; // v2: offset to embedded GGUF header
    uint64_t gguf_metadata_bytes;  // v2: length of embedded GGUF header
} NSMHeader;

typedef struct {
    uint64_t data_offset;  // absolute offset in NSM file where this layer's packed bytes start
    uint64_t data_bytes;   // total packed bytes for this layer
} NSMLayerIndex;

typedef struct {
    char     name[64];
    uint32_t n_clusters;
    uint32_t quant_type;    // 2, 4, or 8 per cluster (variable)
    uint64_t data_offset;
    uint64_t data_bytes;
} NSMTensor;

typedef struct {
    float    activation_freq;
    uint8_t  quant_bits;    // 2, 4, or 8
    uint32_t data_offset;   // offset within tensor data
    uint32_t data_bytes;
    float    scale;
    uint8_t  compressed;    // 1 = zlib compressed, 0 = raw packed bytes
} NSMCluster;

#endif // NSM_H
