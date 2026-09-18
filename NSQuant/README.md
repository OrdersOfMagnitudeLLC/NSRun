# NSQuant

Neural network quantization tool supporting Q4_K, Q6_K, Q8_0, and Q8_K formats with repacking for GEMV inference.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./build/nsquant <input.gguf> <output.nsm> [n_prompts [prompts.txt]]
```

## Parameters

| Argument | Description |
|----------|-------------|
| `input.gguf` | Input GGUF model file |
| `output.nsm` | Output NSM quantized model file |
| `n_prompts` | Optional: number of prompts to process |
| `prompts.txt` | Optional: prompt file |

## Results

Requires a GGUF input file to run. No benchmark data available without a model file.

## Third-party

- **zlib**: zlib License (system-installed)
- Headers from deprecated NSRun: `ns_q4k_quant.h`, `ns_repack.h`, `ns_dequant.h`

## Build dependencies

`nsm.h`, `nsm_loader.cpp`, `gguf_parser.cpp`, `tokenizer.cpp` are required to build. The NSM output format is deprecated; these files now serve GGUF parsing and tokenization only. Do not remove them.
