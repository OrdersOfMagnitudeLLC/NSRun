# Copyright (c) 2026 Orders of Magnitude LLC
# Licensed under the OOM Commercial License v1.0
# See LICENSE.md in the repository root or ofmagnitude.com

#!/usr/bin/env python3
import struct
import random
import os

random.seed(7)

n_vocab = 64
n_embd = 64
n_layers = 2
n_heads = 4
n_kv_heads = 4
n_ff = 128
context = 128

def align32(x):
    return ((x + 31) // 32) * 32

def pack_f32(vals):
    return struct.pack('<' + 'f' * len(vals), *vals)

def kvp_string(key, val):
    b = val.encode('utf-8')
    return struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', 8) + struct.pack('<Q', len(b)) + b

def kvp_uint32(key, val):
    return struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', 4) + struct.pack('<I', val)

def kvp_float(key, val):
    return struct.pack('<Q', len(key)) + key.encode() + struct.pack('<I', 6) + struct.pack('<f', val)

metadata = b''
metadata += kvp_string("general.architecture", "llama")
metadata += kvp_uint32("general.vocab_size", n_vocab)
metadata += kvp_uint32("llama.block_count", n_layers)
metadata += kvp_uint32("llama.attention.head_count", n_heads)
metadata += kvp_uint32("llama.attention.head_count_kv", n_kv_heads)
metadata += kvp_uint32("llama.embedding_length", n_embd)
metadata += kvp_uint32("llama.feed_forward_length", n_ff)
metadata += kvp_uint32("llama.context_length", context)
metadata += kvp_float("llama.rope.freq_base", 10000.0)

# Build tensors
weights = {}

def rand2d(rows, cols, scale=0.02):
    return [random.gauss(0, scale) for _ in range(rows * cols)]

def rand1d(n, scale=0.02):
    return [1.0 + random.gauss(0, scale) for _ in range(n)]

weights['token_embd.weight'] = (n_vocab, n_embd), rand2d(n_vocab, n_embd)
weights['output.weight'] = (n_vocab, n_embd), rand2d(n_vocab, n_embd)
weights['output_norm.weight'] = (n_embd,), rand1d(n_embd)

for li in range(n_layers):
    pre = f"blk.{li}."
    weights[pre + "attn_norm.weight"] = (n_embd,), rand1d(n_embd)
    weights[pre + "ffn_norm.weight"] = (n_embd,), rand1d(n_embd)
    weights[pre + "attn_q.weight"] = (n_embd, n_embd), rand2d(n_embd, n_embd)
    weights[pre + "attn_k.weight"] = (n_embd, n_embd), rand2d(n_embd, n_embd)
    weights[pre + "attn_v.weight"] = (n_embd, n_embd), rand2d(n_embd, n_embd)
    weights[pre + "attn_output.weight"] = (n_embd, n_embd), rand2d(n_embd, n_embd)
    weights[pre + "ffn_gate.weight"] = (n_ff, n_embd), rand2d(n_ff, n_embd)
    weights[pre + "ffn_up.weight"] = (n_ff, n_embd), rand2d(n_ff, n_embd)
    weights[pre + "ffn_down.weight"] = (n_embd, n_ff), rand2d(n_embd, n_ff)

tensor_names = list(weights.keys())

# Header
header = struct.pack('<I', 0x46554747)  # 'GGUF'
header += struct.pack('<I', 3)
header += struct.pack('<Q', len(tensor_names))
header += struct.pack('<Q', 9)  # kv count

# Tensor info
tinfo = b''
data = b''
for name in tensor_names:
    shape, vals = weights[name]
    bname = name.encode('utf-8')
    tinfo += struct.pack('<Q', len(bname)) + bname
    tinfo += struct.pack('<I', len(shape))
    for d in shape:
        tinfo += struct.pack('<Q', d)
    tinfo += struct.pack('<I', 0)  # F32
    off = len(data)
    tinfo += struct.pack('<Q', off)
    data += pack_f32(vals)

# Pad data start to 32 bytes
info_end = len(header) + len(metadata) + len(tinfo)
pad = align32(info_end) - info_end
# data offset in tensor descriptors must include this padding
# adjust offsets
padded_data = b'\x00' * pad + data
# update tinfo offsets (rebuild for simplicity)
tinfo2 = b''
cursor = 0
for name in tensor_names:
    shape, vals = weights[name]
    bname = name.encode('utf-8')
    tinfo2 += struct.pack('<Q', len(bname)) + bname
    tinfo2 += struct.pack('<I', len(shape))
    for d in shape:
        tinfo2 += struct.pack('<Q', d)
    tinfo2 += struct.pack('<I', 0)
    tinfo2 += struct.pack('<Q', pad + cursor)
    cursor += len(pack_f32(vals))

out = header + metadata + tinfo2 + padded_data
with open("synth_f32.gguf", "wb") as f:
    f.write(out)
print(f"Wrote synth_f32.gguf ({len(out)} bytes)")
