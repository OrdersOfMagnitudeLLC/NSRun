#ifndef NSM_LOADER_H
#define NSM_LOADER_H

#include "gguf_parser.h"
#include "ns_repack.h"
#include "nsm.h"
#include <string>
#include <vector>

namespace nsm {

// Dequantize one NSM tensor (clusters at cl_offset) into out_buf.
// out_buf must be sized to the packed or F32 form implied by shape.type:
//   - shape.type == F32  => n * sizeof(float)
//   - Q4_K 2D non-preserve => n_panels * sizeof(ns_q4_Kx8)
//   - Q4_K 1D              => full_blocks * sizeof(block_q4_K)
bool nsm_dequant_tensor(int fd, const NSMTensor& nt, size_t cl_offset,
                        const GGUFTensor& shape, const uint8_t* prev_base,
                        GGMLType prev_type, size_t prev_n,
                        uint8_t* out_buf, size_t out_bytes,
                        const uint8_t* layer_base = nullptr,
                        size_t layer_base_offset = 0);

// Load a .nsm file and pack Q4_K panels into an aligned Q4_K buffer.
// 'shapes' provides the expected tensor names and shapes (usually from the
// original .gguf metadata file).  The returned 'out_tensors' copies those
// shapes but updates offset/type to point into the produced Q4_K panel buffer.
bool load_nsm_weights(const std::string& nsm_path,
                      const std::vector<GGUFTensor>& shapes,
                      AlignedVectorU8& out_q4k,
                      std::vector<GGUFTensor>& out_tensors,
                      GGUFParser* model);

    void validate_panel_dequant(const ns_q4_Kx8* panel);
}

#endif // NSM_LOADER_H
