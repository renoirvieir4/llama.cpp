# Running Inkling 2 Mini with this llama.cpp (LM Studio and llama-server)

**Model used for the measurements:** [notfakerv/Inkling-2-Mini-Abliterated-WSP-GGUF](https://huggingface.co/notfakerv/Inkling-2-Mini-Abliterated-WSP-GGUF)
(Q4_K_M, with vision). Any Inkling 2 Mini GGUF in the ggml-org layout works, including
[ggml-org/Inkling-2-Mini-GGUF](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF).

[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) by Thinking Machines is a looped MoE (128 experts, 8
active; 24 layers, of which layers 6-17 run three times, so 48 layer passes per token) with a vision encoder. Its
attention has no RoPE: it adds a per-head relative-position bias, with sliding-window layers and a log-scaled position
term on the full-attention layers. LM Studio's bundled runtimes cannot run it. This branch is built as the `libllama.so`
and `libmtmd.so` of LM Studio's stock Linux CUDA 12 runtime 2.66.0, and the same code runs in llama-server.

## What this branch adds for Inkling 2

| Commit | Change |
|---|---|
| `437152d86` | Inkling 2 Mini, `inkling2` and `inkling2_loop`: a src-only port of ggml-org/llama.cpp 5c008e000 (Inkling 2), 8af68d486 (attention-mask fixes) and 29b52d398 (ggml-org/llama.cpp#32694, `inkling2_loop`) onto this tree. Per-head masks take the soft_max path, because the flash attention kernels of this ggml read one mask per stream. |
| `4a7537bbc` | Inkling fixture in `test-llama-archs`, Inkling keys in the model saver |
| `26961a9fe` | the `inkling2v` vision encoder in `libmtmd`, plus the two entry points LM Studio's server imports from its own libmtmd |
| `0c759c95b` | optional per-expert `.scale` tensors (NVFP4 `scale_2`, as written by llama.cpp b12633+), e.g. for orcarouter's Mix-16G |
| `282b2cb81` | answers no longer end with a visible `<|end_message|>` in LM Studio: inside model text the `<|end_message|>` logit is folded into the end-of-sampling token (same stop probability); thinking and tool calls are unchanged |
| `044d2b7a8` | exact speedups for long contexts, and an optional flash-attention path (below) |
| `0280b55a2` | DFlash2 speculative decoding with [incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (below), and the `<|end_message|>` fold state tracked by position |

All of it is Inkling-only (the DFlash changes act only for an Inkling target or behind new GGUF keys). Other
architectures build the same graphs as before: a 64-token greedy regression on 30 models is identical.

## Speedups (`044d2b7a8`)

- **Relative bias, window form** (prompts): a ubatch with consecutive positions only has `extent + n_tokens` window
  positions below the extent. Their bias is gathered once, then one long row per cell, instead of the dense
  `[n_kv, n_tokens]` index (CPU fill and copy) and one 24-float gather per cell and token.
- **Relative bias, near form** (generation): the far bucket is repeated over all cells, then `set_rows` writes the cells
  of the token's sequence at a distance below the extent (at most 1024).
- **Tiled transpose:** the bias goes key-first through the copy kernel's tiled transpose path; the old strided copy took
  44% of the prompt time at 26.7K.
- **Chunked attention:** the queries are split so one dense F32 `[n_kv, chunk, n_head]` tensor stays under
  `LLAMA_INKLING_ATTN_CHUNK_MIB` (default 512). Ubatches up to 32 tokens are never split and larger ones are split into
  chunks of at least 34, so the KQ/KQV products keep their CUDA kernel (cuBLAS above 16 columns; the small-batch kernels
  round differently).
- **One K read per kv head** (one token, at least 20480 cells): the 3 q heads of a kv head are the columns of one KQ
  product instead of 3 broadcast reads.
- **Fewer GPU ops per token:** the clip thresholds, the zero sinks and the padded bias table are built once per graph;
  depth mixing keeps a running stack of the blocks; one-token mixing uses the query directly.
- **Optional flash attention** (only with flash attention on): q head `g + 3k` runs as sequence `k` of call `g`, so each
  head gets its own mask. The bias then goes through an F16 mask (KLD 0.013 against the exact path) and generation is
  slower, so **keep flash attention off** for Inkling.

**Exactness:** with flash attention off, the logits are bit-identical to `282b2cb81` at the same settings, at the end of
the prompt and token by token: 3K tokens (micro-batch 512), 26.7K (micro-batch 32 and 128, also with forced 32-token
chunks) and 98K (micro-batch 32, small SWA cache as in LM Studio).

## DFlash2 speculative decoding (`0280b55a2`)

[incoai/Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0, by Inco AI; DFlash by
z-lab) is a 0.5B block-diffusion drafter for Inkling 2 Mini: it drafts 7 tokens in one pass, and a candidate selector
picks a path through the top-16 candidates of each position. The target verifies every draft, so the drafter changes
speed, not what the model may say. Its Q8_0 GGUF (538 MB) is in
[notfakerv/Inkling-2-Mini-Abliterated-WSP-GGUF](https://huggingface.co/notfakerv/Inkling-2-Mini-Abliterated-WSP-GGUF) as
`dflash-Inkling-2-Mini-DFlash2-Q8_0.gguf`; `convert_inkling2_dflash2_to_gguf.py` makes it from the drafter repo and an
Inkling 2 GGUF (whose vocabulary it copies).

What `0280b55a2` adds:

- Inkling 2 records the drafter's features: the residual state entering each layer visit, in execution order (prelude,
  the three passes of layers 6-17, tail), and the final-norm state at index 48. These are pointer records; without a
  drafter the graph is unchanged.
- DFlash drafts with an Inkling target embed through the target's token norm (as SGLang does), use the 200058
  real-vocabulary rows of the target's padded head, and read `<arch>.selector_softcap_after_correction` (the softcap is
  applied after the pairwise correction, as the drafter was trained).
- `<arch>.target_visits` holds the exact visits [5, 23, 41, 48]; `<arch>.target_layers` then holds the physical layer
  of each visit, [5, 11, 17, 24], which is what LM Studio compares with the target's `block_count` (24) before it offers
  the drafter.
- The `<|end_message|>` fold now takes each sequence's state from the positions before the batch, so rejected drafts
  no longer leave stale state (they made some answers stop at the end of the reasoning).

Measurements, same machine and settings as above, LM Studio, temperature 1.0 (top-k 50), 8 short prompts of up to 400
tokens:

| | tok/s | |
|---|---|---|
| DFlash off | 55.4 | |
| DFlash, draft tokens 3 | 78.0 | 1.41x |
| DFlash, draft tokens 7 | 81.4 | 1.47x |

In the runtime's llama-server at 16K context: greedy 56.2 -> 87.3 tok/s (153 tok/s on a code prompt), and about 3.8
tokens per verification step at temperature 1.0. LM Studio's draft tokens setting is also the drafter's block length
minus one, and the drafter was trained on blocks of 8, so use 7. The gain is small at long contexts: after a 26.7K-token
prompt, generation went from 46.0 to 48.7 tok/s (draft tokens 3) and prompt processing stayed the same (342 vs 338
tok/s).

**Exactness:** without a drafter, the logits are bit-identical to `044d2b7a8` (3K, 26.7K and 98K tokens) and the 30-model
regression is identical. With a drafter, the target decodes each draft in one batch of 8. Batched and single-token
decoding round differently, and Inkling amplifies that (8-bit activation rounding, MoE routing over 48 layer visits): the
same text decoded with different micro-batch sizes differs by a mean KLD of about 0.1 in every build since Inkling
support was added, with or without DFlash. Generations therefore differ from DFlash off, but are not worse: over 1024
positions of text, the mean next-token NLL with DFlash-style verification (blocks of 8, rejected tails rolled back) was
-0.006 +/- 0.031 nats against normal decoding. Rollback and feature capture alone are bit-exact.

**Abliterated targets:** the drafter was trained on stock Inkling. With the WSP v3 target it proposed refusal-like text
that the target rejected 3 times in 11,260 draft steps on 100 harmful prompts (none an actual refusal), and its acceptance
on those answers was within the noise of harmless ones, so it needs no abliteration of its own.

## Measurements

Inkling 2 Mini WSP v3 Q4_K_M, 2x RTX 3090 (24 GB, PCIe 4.0 x8), 256K context, GPU split 6:43, micro-batch 32, flash
attention off, F16 KV; LM Studio runtime 2.66.111 (`282b2cb81`) vs 2.66.113 (`044d2b7a8`), in LM Studio or the runtime's
llama-server with the same settings:

| | 2.66.111 | 2.66.113 | |
|---|---|---|---|
| Generation, short prompt | 50.0 tok/s | 56.4 tok/s | 1.13x |
| 24K-token prompt | 273 tok/s | 372 tok/s | 1.36x |
| Generation after the 24K prompt | 41.0 tok/s | 47.2 tok/s | 1.15x |
| 98K-token prompt | 195.5 tok/s | 291.3 tok/s | 1.49x |
| Generation at 98K | 27.7 tok/s | 36.8 tok/s | 1.33x |

A 229,826-token prompt ran in LM Studio at 195 tok/s on average, then generated at 24.3 tok/s, with GPU 1 at 23.2 GB at
its peak.

## Build and install into LM Studio (Linux, CUDA 12)

```bash
# 1. build the two libraries (CPU options; CUDA comes from LM Studio's own ggml)
git clone -b inkling2-scale https://github.com/renoirvieir4/llama.cpp && cd llama.cpp
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON
cmake --build build --target llama mtmd

# 2. get LM Studio's stock runtime 2.66.0 (pick llama.cpp-linux-x86_64-nvidia-cuda12-avx2), copy it to a new version
#    and put the libraries in it, linked to LM Studio's ggml
lms runtime get 'llama.cpp@2.66.0'
B=~/.lmstudio/extensions/backends; R=$B/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.66.114
cp -a $B/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.66.0 $R
cp -L build/bin/libllama.so build/bin/libmtmd.so $R/ && cd $R
patchelf --set-soname libllama.so --set-rpath '$ORIGIN' \
  --replace-needed libggml.so.0 libggml_llamacpp.so --replace-needed libggml-base.so.0 libggml-base.so libllama.so
patchelf --set-soname libmtmd.so --set-rpath '$ORIGIN' --replace-needed libllama.so.0 libllama.so \
  --replace-needed libggml.so.0 libggml_llamacpp.so --replace-needed libggml-base.so.0 libggml-base.so libmtmd.so

# 3. set "version": "2.66.114" in backend-manifest.json, then select it (twice if it is not listed yet)
lms runtime select llama.cpp-linux-x86_64-nvidia-cuda12-avx2@2.66.114
```

The libraries must match the ggml of the LM Studio runtime they go into (2.66.0 here).

**Settings** (the model repo has them as `lmstudio/inkling-2-mini-wsp-v3.config.json`, and the template override
`lmstudio/template-lmstudio.jinja`):

| Setting | Value | Why |
|---|---|---|
| Flash attention | off | the exact path; on, the bias goes through an F16 mask |
| KV cache | F16 | a quantized V cache needs flash attention |
| Context (two 24 GB GPUs) | 256000 | the F16 KV cache is about 8 GB at 256K |
| GPU split | custom 6:43 | GPU 0 holds layers 0-5, GPU 1 the looped layers 6-17 and 18-23: one GPU hop per token |
| Micro-batch / eval batch | 32 / 512 | the tested setting; in llama.cpp a different micro-batch can change the last bits of prompt processing |
| Sampling | temperature 1.0, top-k 50, top-p 1.0, min-p 0 | the base model's card |
| Speculative decoding | DFlash Sidecar on, draft tokens 7 | with `dflash-Inkling-2-Mini-DFlash2-Q8_0.gguf` next to the model (above) |

On one 24 GB GPU, use a 16K context.

## llama-server

The LM Studio runtime folder has its own `llama-server`, which uses the installed libraries (all measurements above
were also taken with it). With flash attention off and an F16 cache, for example:

```bash
R=~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-nvidia-cuda12-avx2-2.66.114
V=~/.lmstudio/extensions/backends/vendor/linux-llama-cuda12-vendor-v1
LD_LIBRARY_PATH=$R:$V $R/llama-server -m Inkling-2-Mini-Abliterated-WSP-v3-Q4_K_M.gguf \
  --mmproj mmproj-Inkling-2-Mini-F16.gguf -c 16384 -fa off -ctk f16 -ctv f16 --jinja \
  --temp 1.0 --top-k 50 --top-p 1.0 --min-p 0
```

With the drafter, add `-md dflash-Inkling-2-Mini-DFlash2-Q8_0.gguf --spec-type draft-dflash --spec-draft-n-max 7`. Leave
the drafter on the default devices: it uses the target's output head, which sits on the last GPU (`-devd` with another
GPU fails at load).

For upstream llama.cpp without LM Studio, use upstream b12597 or later, which has Inkling 2 support (ggml-org/llama.cpp#32694).

## Limits

- DFlash speeds up short and medium contexts most; after tens of thousands of tokens the gain is a few percent.
- LM Studio's chat library has no Inkling tool-call parser: tool calls are generated correctly but not parsed into
  `tool_calls`.
- Large images on a nearly full GPU can fail in `clip_encode` (allocation); keep some VRAM free for vision.
