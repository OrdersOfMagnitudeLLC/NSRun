//
// NSInfer CUDA backend — sparse MLP activation masking
// MIT license
//
#pragma once

#include "common.cuh"

void ggml_cuda_op_ns_infer(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
