//
// NSAttend CUDA backend — faithful port of the CPU ns_attention op
// MIT license
//
#pragma once

#include "common.cuh"

void ggml_cuda_op_ns_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
