// NSKVCache — OOM LLC Commercial License — see /NS/LICENSING.md
// Copyright (c) 2026 Orders of Magnitude LLC. All rights reserved.
//
// kv_box.cpp — Bare KV allocator math validation (no inference, no LLM)
//
// Computes raw KV cache size for Qwen3-4B at 1M tokens, then applies
// compression steps in sequence and prints a table. Pure math — no
// memory is allocated.
//
// Build:  g++ -std=c++17 -O2 -o kv_box kv_box.cpp
// Run:    ./kv_box

#include <cstdint>
#include <cstdio>
#include <cassert>
#include <string>

// ─── Qwen3-4B config (hardcoded) ────────────────────────────────────
static constexpr uint32_t N_LAYERS   = 32;
static constexpr uint32_t N_KV_HEADS = 8;
static constexpr uint32_t HEAD_DIM   = 128;
static constexpr uint32_t FP16_BYTES = 2;          // sizeof(ggml_fp16_t)
static constexpr uint64_t N_TOKENS   = 1'000'000; // 1M tokens

// ─── Compression parameters ────────────────────────────────────────
static constexpr uint32_t CLG_GROUPS         = 4;   // cross-layer grouping: 32 → 4
static constexpr uint32_t INT4_FACTOR        = 4;   // fp16 → INT4 = 4× shrink
static constexpr double   EVICT_KEEP_FRAC    = 0.4; // keep top 40% → 1/0.4 = 2.5×
static constexpr double   FILLER_KEEP_FRAC   = 0.60; // drop 40% filler → 1/0.60 = 1.67×
static constexpr double   SEMANTIC_DEDUP_FACTOR = 2.0; // merge near-duplicate KV entries
static constexpr uint64_t TARGET_MAX_BYTES   = 512ULL * 1024 * 1024; // 512 MiB

// ─── Helpers ───────────────────────────────────────────────────────

static double to_mib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

static double to_gib(uint64_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

struct StepResult {
    std::string name;
    uint64_t    bytes;
    double      ratio_vs_raw;   // cumulative compression vs raw
    double      step_ratio;     // this step's compression factor
};

// ─── Main ──────────────────────────────────────────────────────────

int main() {
    // Per token, per layer: K + V = 2 * n_kv_heads * head_dim * fp16_bytes
    const uint64_t bytes_per_token_per_layer =
        2ULL * N_KV_HEADS * HEAD_DIM * FP16_BYTES;

    // Raw KV size
    const uint64_t raw_bytes =
        static_cast<uint64_t>(N_LAYERS) * N_TOKENS * bytes_per_token_per_layer;

    // Step 1: Cross-layer grouping (32 → 4 groups)
    // Layers sharing a group reuse a single KV slot → N_LAYERS / CLG_GROUPS × reduction
    const double clg_ratio = static_cast<double>(N_LAYERS) / CLG_GROUPS; // 8×
    const uint64_t after_clg = static_cast<uint64_t>(raw_bytes / clg_ratio);

    // Step 2: INT4 quantization (fp16 2 bytes → INT4 0.5 bytes = 4× shrink)
    const double int4_ratio = static_cast<double>(INT4_FACTOR); // 4×
    const uint64_t after_int4 = static_cast<uint64_t>(after_clg / int4_ratio);

    // Step 3: Eviction — keep top 40% by information contribution
    // Compression = 1 / keep_fraction = 1 / 0.4 = 2.5×
    const double evict_ratio = 1.0 / EVICT_KEEP_FRAC; // 2.5×
    const uint64_t after_evict = static_cast<uint64_t>(after_int4 / evict_ratio);

    // Step 4: Filler elimination — keep 60% of remaining tokens (drop 40%)
    // Compression = 1 / keep_fraction = 1 / 0.60 ≈ 1.67×
    const double filler_ratio = 1.0 / FILLER_KEEP_FRAC; // 1.67×
    const uint64_t after_filler = static_cast<uint64_t>(after_evict / filler_ratio);

    // Step 5: Semantic dedup — merge near-duplicate KV entries
    const double dedup_ratio = SEMANTIC_DEDUP_FACTOR; // 2.0×
    const uint64_t after_dedup = static_cast<uint64_t>(after_filler / dedup_ratio);

    // Collect steps
    StepResult steps[] = {
        {"Raw KV (fp16)",              raw_bytes,    1.0,                    1.0},
        {"After cross-layer grouping", after_clg,    clg_ratio,              clg_ratio},
        {"After INT4 quantization",    after_int4,   clg_ratio * int4_ratio, int4_ratio},
        {"After eviction (40% keep)",  after_evict,  clg_ratio * int4_ratio * evict_ratio, evict_ratio},
        {"After filler elimination",   after_filler, clg_ratio * int4_ratio * evict_ratio * filler_ratio, filler_ratio},
        {"After semantic dedup",       after_dedup,  clg_ratio * int4_ratio * evict_ratio * filler_ratio * dedup_ratio, dedup_ratio},
    };

    // ─── Print config ──────────────────────────────────────────────
    std::printf("=== KV Box — Bare Allocator Math Validation ===\n\n");
    std::printf("Model: Qwen3-4B\n");
    std::printf("  layers     : %u\n", N_LAYERS);
    std::printf("  kv heads   : %u\n", N_KV_HEADS);
    std::printf("  head_dim   : %u\n", HEAD_DIM);
    std::printf("  dtype      : fp16 (%u bytes)\n", FP16_BYTES);
    std::printf("  tokens     : %llu\n", (unsigned long long)N_TOKENS);
    std::printf("  bytes/token/layer : %llu\n",
                (unsigned long long)bytes_per_token_per_layer);
    std::printf("\n");

    // ─── Print compression table ──────────────────────────────────
    std::printf("%-30s  %14s  %10s  %10s  %10s\n",
                "Step", "Bytes", "MiB", "GiB", "Cumul. ×");
    std::printf("%-30s  %14s  %10s  %10s  %10s\n",
                "----", "-----", "---", "---", "-------");
    for (const auto &s : steps) {
        std::printf("%-30s  %14llu  %10.2f  %10.4f  %10.1f\n",
                    s.name.c_str(),
                    (unsigned long long)s.bytes,
                    to_mib(s.bytes),
                    to_gib(s.bytes),
                    s.ratio_vs_raw);
    }

    // ─── Step-level detail ────────────────────────────────────────
    std::printf("\n--- Step detail ---\n");
    std::printf("1. Cross-layer grouping: %u layers → %u groups = %.1fx\n",
                N_LAYERS, CLG_GROUPS, clg_ratio);
    std::printf("2. INT4 quantization: fp16 → INT4 = %.1fx\n", int4_ratio);
    std::printf("3. Eviction: keep top %.0f%% → %.1fx\n",
                EVICT_KEEP_FRAC * 100.0, evict_ratio);
    std::printf("4. Filler elimination: keep %.0f%% → %.2fx\n",
                FILLER_KEEP_FRAC * 100.0, filler_ratio);
    std::printf("5. Semantic dedup: %.1fx\n", dedup_ratio);
    std::printf("   Total cumulative compression: %.1fx\n",
                clg_ratio * int4_ratio * evict_ratio * filler_ratio * dedup_ratio);

    // ─── Assertion ────────────────────────────────────────────────
    std::printf("\n--- Assertion ---\n");
    std::printf("Target: final size < %llu bytes (%.0f MiB)\n",
                (unsigned long long)TARGET_MAX_BYTES,
                to_mib(TARGET_MAX_BYTES));
    std::printf("Final:  %llu bytes (%.2f MiB, %.4f GiB)\n",
                (unsigned long long)after_dedup,
                to_mib(after_dedup),
                to_gib(after_dedup));

    if (after_dedup < TARGET_MAX_BYTES) {
        std::printf("Result: PASS ✓\n");
    } else {
        std::printf("Result: FAIL ✗  (exceeds target by %.2f MiB)\n",
                    to_mib(after_dedup) - to_mib(TARGET_MAX_BYTES));
    }

    assert(after_dedup < TARGET_MAX_BYTES && "KV size exceeds 512 MiB budget");

    return (after_dedup < TARGET_MAX_BYTES) ? 0 : 1;
}
