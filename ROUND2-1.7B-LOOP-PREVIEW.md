# Running round2-1.7b-loop-preview with this llama.cpp

**Model card, weights and results:** https://huggingface.co/notfakerv/round2-1.7b-loop-preview

`round2-1.7b-loop-preview` is an experimental looped version of [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B) by Liquid AI. Liquid AI's network runs once, unchanged; then layers 5-20 run a second time through 44M trained parameters, and a learned gate blends the result back in. Stock llama.cpp cannot load the model's GGUF. This branch can.

## What this branch is

It is llama.cpp commit [`ff0d396e3`](https://github.com/ggml-org/llama.cpp/commit/ff0d396e3dda0f33e7ce5ad4267fce4f7690cfb1) plus three commits:

1. **LFM3 loop runtime prerequisites:**
   - [llama.cpp PR #31306](https://github.com/ggml-org/llama.cpp/pull/31306), Clover, by Maphist0 and contributors;
   - native SenseNova Loop support;
   - the earlier LFM3 Loop (mHC) and NVFP4 support;
   - [llama.cpp PR #30306](https://github.com/ggml-org/llama.cpp/pull/30306), Nemotron Dory.
2. **The LFM3 refine loop:** the loop this model uses, read from the `lfm3.loop_refine.*` GGUF keys.
3. **This recipe:** this file, `scripts/round2-loop-preview.sh` and a note at the top of `README.md`.

The first two commits match `runtime/01-lfm3-loop-prerequisites.patch` and `runtime/02-lfm3-refine-loop.patch` in the model repository. Everything upstream llama.cpp `ff0d396e3` runs still works. The second commit changes no other architecture: llama.cpp's architecture test suite gives identical results before and after it.

## Requirements

- **System:** Linux x86_64 and an NVIDIA GPU with at least 6 GB of free VRAM. Tested on an RTX 3090.
- **CUDA:** the CUDA toolkit 12.x, with a host compiler it supports. CUDA 12.8 supports GCC up to 14.
- **Build tools:** CMake and a C++17 compiler.
- **Download tool:** `hf` (from `pip install huggingface_hub`) or `curl`.

## Recipe

```bash
git clone --branch round2-1.7b-loop-preview https://github.com/renoirvieir4/llama.cpp.git llama.cpp-round2
cd llama.cpp-round2

# 1. Build (set CMAKE_CUDA_ARCHITECTURES for your GPU: 86 = RTX 30xx, 89 = RTX 40xx, 120 = RTX 50xx)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_UI=OFF -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --target llama-server -j

# 2. Download the model (3.64 GB) and check it
hf download notfakerv/round2-1.7b-loop-preview round2-1.7b-loop-preview-BF16.gguf --local-dir models
echo "9c9e3d24bc59ce6ec3e9c0d9663801120822c594b7212f27dff498eaa1ebed54  models/round2-1.7b-loop-preview-BF16.gguf" | sha256sum -c -

# 3. Serve it with the recommended settings
./build/bin/llama-server -m models/round2-1.7b-loop-preview-BF16.gguf \
  -ngl 99 -fa on -c 65536 --temp 1.0 --top-k 20 --top-p 0.95 --repeat-penalty 1.1 --port 8080
```

In another terminal:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is 17 multiplied by 23?"}], "max_tokens": 8192}'
```

The answer should end with **391**, after a short reasoning trace.

`scripts/round2-loop-preview.sh` does all three steps: it builds if needed, downloads and checks the model, then starts the server.

```bash
CUDA_ARCH=86 scripts/round2-loop-preview.sh 8080
```

Build notes:
- **Compiler newer than CUDA supports:** add `-DCMAKE_CUDA_HOST_COMPILER=/path/to/g++-14`, or pass it through `CMAKE_ARGS` to the script.
- **OpenMP link error against a static `libgomp`:** add `-DGGML_OPENMP=OFF`.
- **Web UI:** `-DLLAMA_BUILD_UI=OFF` skips downloading the server's web UI bundle, which this branch's build number does not match. The OpenAI-compatible API used above does not need it.
- **Without `hf`:** download from `https://huggingface.co/notfakerv/round2-1.7b-loop-preview/resolve/main/round2-1.7b-loop-preview-BF16.gguf` with `curl -L`.

## Recommended settings

These come from the model card.

| Use | Temperature | Top-k | Top-p | Repetition penalty |
|---|---|---|---|---|
| **Math, science, logic, multi-step reasoning (recommended)** | 1.0 | 20 | 0.95 | 1.1 |
| Chat, coding, tool use (Liquid AI's general setting) | 0.1 | 50 | 1.0 | 1.1 |

- **Thinking:** keep it on, which is the chat template's default.
- **max_tokens:** use **8,192**, the budget the model was trained for and where it beats stock LFM3-1.7B. Above 16k tokens it matches stock.
- **Flash attention:** keep it on (`-fa on`). The 65,536-token context then needs about 5.2 GB of VRAM. Without flash attention it took 17.7 GB across two GPUs.
- **Speed:** about 89 tokens/s on one RTX 3090.

## What to expect

With an 8,192-token budget, the model answered more questions right than stock LFM3-1.7B on all four evaluation sets, by 1.5 to 5.6 points.
- **Mechanism:** it gains mostly by finishing its reasoning sooner.
- **Untouched final split:** the gain was +1.5 ± 0.9 points, which is not statistically significant.
- **Longer budgets:** with 16k-32k tokens it ties stock.
- **Weak spots:** it is slightly worse than stock on coding and cybersecurity questions.

Full results, method and limitations are on the [model card](https://huggingface.co/notfakerv/round2-1.7b-loop-preview).

## Tested

- **This recipe:** from this branch, on Linux with an RTX 3090, CUDA 12.8 and GCC 14 (`-DGGML_OPENMP=OFF`, see the build notes). It built `llama-server`, downloaded and checked the model, served it and answered 391.
- **Native runtime against PyTorch:** the GGUF matches the PyTorch reference as closely as stock LFM3-1.7B does: 99.3-100% top-1 agreement and mean KL at most 0.0003. See the model's `validation/` folder.

## License and attribution

- **This repository:** llama.cpp, MIT ([LICENSE](LICENSE)).
- **The model:** LFM Open License v1.0, inherited from Liquid AI's LFM3-1.7B; see the model card.
- **Credits:** the base model is Liquid AI's. The loop design follows SenseNova-Loop-9B-Preview and mHC, and training used ReST-EM; the model card lists all credits.
