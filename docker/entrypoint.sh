#!/bin/bash
# NSRun KVBox entrypoint: fetch model if absent, verify SHA256, serve.
# Exits non-zero with a clear message if the model hash does not match —
# no requests are served on a corrupt or wrong file.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-14B-Q4_K_M.gguf}"
MODEL_URL="${MODEL_URL:-https://modelscope.cn/models/Qwen/Qwen3-14B-GGUF/resolve/master/Qwen3-14B-Q4_K_M.gguf}"
EXPECTED_SHA256="${EXPECTED_SHA256:-500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0}"

mkdir -p "$(dirname "$MODEL_PATH")"

if [ ! -s "$MODEL_PATH" ]; then
    echo "[nsrun] downloading model: $MODEL_URL"
    curl -fL --retry 3 --retry-delay 5 -o "${MODEL_PATH}.part" "$MODEL_URL"
    mv "${MODEL_PATH}.part" "$MODEL_PATH"
fi

echo "[nsrun] verifying SHA256 of $MODEL_PATH ..."
ACTUAL="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
if [ "$ACTUAL" != "$EXPECTED_SHA256" ]; then
    echo "[nsrun] FATAL: model SHA256 mismatch — refusing to serve" >&2
    echo "  file:     $MODEL_PATH" >&2
    echo "  expected: $EXPECTED_SHA256" >&2
    echo "  actual:   $ACTUAL" >&2
    exit 1
fi
echo "[nsrun] SHA256 OK ($ACTUAL)"

exec /usr/local/bin/llama-server \
    -m "$MODEL_PATH" \
    -c "${CTX:-3145728}" \
    -b 1024 \
    --kv-box \
    -ngl 999 \
    --parallel "${PARALLEL:-3}" \
    --host 0.0.0.0 \
    --port "${PORT:-8080}" \
    "$@"
