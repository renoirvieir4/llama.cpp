#!/usr/bin/env bash
# Build this branch, download round2-1.7b-loop-preview and serve it with the recommended settings.
# Model card: https://huggingface.co/notfakerv/round2-1.7b-loop-preview
# usage: scripts/round2-loop-preview.sh [PORT]
# env:   CUDA_ARCH (default 86), MODEL_DIR (default models), CMAKE_ARGS (extra cmake options)
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-8080}
arch=${CUDA_ARCH:-86}
model_dir=${MODEL_DIR:-models}
gguf=$model_dir/round2-1.7b-loop-preview-BF16.gguf
sha=9c9e3d24bc59ce6ec3e9c0d9663801120822c594b7212f27dff498eaa1ebed54
url=https://huggingface.co/notfakerv/round2-1.7b-loop-preview/resolve/main/round2-1.7b-loop-preview-BF16.gguf

if [ ! -x build/bin/llama-server ]; then
    # shellcheck disable=SC2086 # CMAKE_ARGS is a list of options
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
        -DLLAMA_BUILD_UI=OFF -DCMAKE_CUDA_ARCHITECTURES="$arch" ${CMAKE_ARGS:-}
    cmake --build build --config Release --target llama-server -j
fi

if [ ! -f "$gguf" ]; then
    mkdir -p "$model_dir"
    if command -v hf > /dev/null; then
        hf download notfakerv/round2-1.7b-loop-preview round2-1.7b-loop-preview-BF16.gguf --local-dir "$model_dir"
    else
        curl -L --fail -o "$gguf" "$url"
    fi
fi
echo "$sha  $gguf" | sha256sum -c -

exec ./build/bin/llama-server -m "$gguf" -ngl 99 -fa on -c 65536 \
    --temp 1.0 --top-k 20 --top-p 0.95 --repeat-penalty 1.1 --port "$port"
