# NSQuant

Variable-rate GGUF quantizer with activation-aware (Wanda-style) cluster
scoring. Reads a float GGUF (F32 / F16 / BF16 tensors) and writes a standard
.gguf readable by NSRun / llama.cpp.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./nsquant [--dequant-input] [--hot-budget N] [--warm-budget N] [--cold-tier TYPE] <input.gguf> <output.gguf> [n_prompts [prompts.txt]]
```

## Parameters

| Argument | Description |
|----------|-------------|
| `--dequant-input` | Dequantize Q4_K/Q5_K/Q6_K/Q8_0 input tensors to float before profiling |
| `--hot-budget N` | Element budget for the Q8_0 (hot) tier. Default: 5% of params |
| `--warm-budget N` | Element budget for the Q4_K (warm) tier. Default: 25% of params |
| `--cold-tier TYPE` | Output type for remaining (coldest) clusters: `IQ2_XXS` (default), `IQ1_S`, `Q2_K`, `Q4_K`, `Q8_0` |
| `input.gguf` | Input model file |
| `output.gguf` | Output quantized GGUF |
| `n_prompts` | Calibration sample count (default 500, `0` disables profiling) |
| `prompts.txt` | Calibration prompt file (default `./prompts.txt`) |

## Wanda-style profiling

Enabled whenever `n_prompts > 0` with a prompts file. Prompts are embedded via
`token_embd`; per input channel j, `act_norm[j] = sum over samples |x_j|`.
Each 256-element cluster's score = mean `|W_ij| * act_norm[j]` over its
elements. Clusters are sorted globally across all tensors: top `hot_budget`
-> Q8_0, next `warm_budget` -> Q4_K, remainder -> cold tier.

There is no `--wanda` flag and no fixed calibration sample count — the number
of samples is exactly `n_prompts` (e.g. `500 prompts.txt` = 500 samples).
Website copy referencing "128 calibration samples" describes the Wanda method
generically, not a hardcoded preset.

## Cold-tier coherence floor

Empirical on a 31B dense model: `IQ2_XXS` assigned to ~65% of parameters
produced incoherent output. For 30B+ models the cold tier must not dominate —
use `--cold-tier Q4_K` or raise `--warm-budget`. **IQ3_S is not implemented in
this codebase** (no quantizer); Q4_K is the highest-fidelity cold tier
available today.

## Recommended invocation, large models

```bash
./nsquant --cold-tier Q4_K input.gguf output.gguf 500 prompts.txt
```

## Bug fixes (Oct 2026)

- **BF16 input tensors bypassed quantization.** BF16 was missing from the
  `float_path` gate, the `any_float` check, the `token_embd.weight` lookup,
  `elem_offset_bytes`, and `read_tensor_floats` — tensors fell into raw
  passthrough (<2x compression) or hit "unsupported type". Added `bf16_to_f32`
  and BF16 cases throughout.
- **BOOL metadata arrays desynced the parser.** GGUF BOOL array elements are
  1 byte; the array-skip fallback assumed 8 bytes/element, over-skipping and
  corrupting subsequent keys -> `std::bad_alloc` (trigger:
  `gemma4.attention.sliding_window_pattern` bool[60]). BOOL cases added to all
  three array switches in `gguf_parser.cpp`. The same `*8` fallback remains
  for any future non-8-byte unhandled array element type.

## Third-party

- **zlib**: zlib License (system-installed)
- Headers from deprecated NSRun: `ns_q4k_quant.h`, `ns_repack.h`, `ns_dequant.h`

## Build dependencies

`nsm.h`, `nsm_loader.cpp`, `gguf_parser.cpp`, `tokenizer.cpp` are required to build. The NSM output format is deprecated; these files now serve GGUF parsing and tokenization only. Do not remove them.
