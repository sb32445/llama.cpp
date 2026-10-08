# prism-addon: Ternary Bonsai (PTQ1_0) on top of current ggml-org/llama.cpp

This branch (`prism-addon`) is `ggml-org/llama.cpp` master plus the pieces needed to run the
ternary Bonsai 2 27B model in the `PTQ1_0` format on an NVIDIA RTX 4070.

## Why this branch exists

The goal is to stay close to the ggml project. If the Bonsai support sits on top of current upstream,
improvements in upstream (kernels, model support, fixes) arrive with a plain rebase instead of waiting
until a fork catches up. The branch is rebased onto upstream master from time to time.

## Scope: what is supported

- **Hardware:** NVIDIA RTX 4070 (Ada, sm_89, 12 GB), CUDA, Linux. Nothing else is tested.
- **Format and model:** `PTQ1_0` (1.75 bpw, ternary) with Ternary-Bonsai-2-27B (Qwen3.5 based), optionally with the MTP head
  (`--spec-type draft-mtp`).
- **Everything else is unsupported and untested:** other GPUs (including other Ada, Ampere, Hopper, Blackwell cards),
  other backends (Vulkan, Metal, HIP, SYCL; the CPU path is a plain generic implementation and slow), other models.
- **`PQ2_0`:** the type is registered (it is part of the ported code) but is **not supported**. Two `test-backend-ops`
  cases with `PQ2_0` fail (`MUL_MAT` m=16 n=1 k=128 and a `MUL_MAT_ID` case); they fail the same way before the
  last rebase.

The branch is experimental, provided as is (MIT), and not meant to be merged into another project as it is.

## What is on the branch

Compared with upstream master `08246a28f` (2026-10-08) it changes 76 files (about +5600 / -350 lines):

- Types `PQ2_0` (142) and `PTQ1_0` (143), CPU dot products, the CUDA decode path (mat-vec, dequantization, `get_rows`,
  FWHT) and the Hadamard weight fold in the loader, graph and context. These parts are ported from the PrismML fork.
- A CUDA MMQ path (quantized matrix-matrix kernels) for `PTQ1_0` prefill.
- Flash attention: the MMA kernel reads a `q4_0` K/V cache in place, with a kernel choice for GQA above 4
  and an 8-column MMA configuration.
- Gated delta net: raw gates folded into the op, fused recurrent-state gather, four state columns per warp from four
  tokens on (the kernel itself is upstream's), MTP deferred catch-up.
- `--kv-mean-center` (K-cache mean-centering with a bias file).
- Fused gate/up SwiGLU mat-vec for 2 to 4 columns, L2 prefetch of the next mat-vec's weights (capped at 8 MiB on Ada),
  and a few host-side changes (transposing concat kernel, bounded `seq_rm`, reuse of evicted checkpoint buffers, top-k
  straight from the logits).

`git diff <upstream base>..prism-addon` shows everything; the commit messages describe each step.

## Build and run

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench llama-perplexity test-backend-ops
```

Tested with CUDA 13.4, GCC 16.2, NVIDIA driver 615.71 on Linux 6.18.

- Use a `PTQ1_0` GGUF of the model. For MTP the GGUF needs the MTP head; a recipe to build one is on the `ktrain` branch of this
  fork (`docs/ktrain/mtp`).
- Typical server options: `-fa on --cache-type-k q4_0 --cache-type-v q4_0`, with MTP additionally
  `--spec-type draft-mtp --spec-draft-n-max 2`.
- Upstream rotates the K cache by default for quantized K. The optional `--kv-mean-center FILE` bias file was
  calibrated with the rotation off, so it needs `LLAMA_ATTN_ROT_DISABLE=1`; the loader refuses the file otherwise.

## Checks

On an RTX 4070, `test-backend-ops test -b CUDA0` passes 19332 of 19334 cases; the two failures are the `PQ2_0` cases above.
Perplexity (16 x 512 tokens of a German text, `q4_0` K/V) is unchanged within noise when switching between kernel variants.
The outputs of this build are not bit-identical to other builds of the same model (different kernels sum in a different
order, and the model amplifies such rounding differences); judge a build by perplexity, not by identical text.

## AI assistance

The changes on this branch were developed with Claude Code (Anthropic's coding agent): it wrote the code, the measurement
scripts and the first drafts of the commit messages. I decided what to work on and I maintain the changes. The
measurements were run in the Claude Code sessions; I did not re-run them independently. All commits on the branch carry
a `Co-Authored-By` trailer.

## Credits and license

MIT, as llama.cpp. The base is the work of the ggml authors; the gated delta net four-column kernel is upstream's
(ggml-org/llama.cpp#30087). The `PTQ1_0` / `PQ2_0` types, their kernels and the Hadamard weight fold are ported from the
PrismML fork (PrismML-Eng/llama.cpp), which is also MIT licensed.
