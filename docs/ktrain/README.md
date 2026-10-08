# kTrain patches for the Prism llama.cpp fork

Decode-speed patches for PTQ1_0 / PQ2_0 ternary models (tested with Ternary-Bonsai-2-27B plus an MTP draft head) on NVIDIA Ada (RTX 4070, cc 8.9).
They are meant to be taken one by one: every patch is also available as its own branch off `prism` (`pr/*`), applies on `prism` on its own and in any order
(textual conflicts were checked), and is offered upstream as a separate pull request. This branch (`ktrain`) merges all of them for
people who do not want to wait for the merges.

**Layout:** every `pr/*` branch starts with the measured version, which contains an environment switch for A/B tests (the commit that was benchmarked), and ends with a commit that removes the switch(es) (and, where noted in the commit, fixes the constants). This stack merges each branch with `--no-ff`, so the single commits stay visible in the history and can be cherry-picked; the tip of the stack contains no switches.

Base: `prism` at `2459f68b5` (measured on `88c4bc60b`; the commits in between do not touch these code paths).

**Numerics:** 0004 to 0010 do not change results (identical outputs in every A/B run and at every depth I checked). **0001 (PQ2_0 kernel), 0002 and 0003 (attention kernel choice and tile size) change the arithmetic order**, so results differ at rounding level; for single-query decode with quantized K/V (no speculative decoding) greedy text diverges after some tokens (4 short prompts x 512 tokens: 3 of 4 differ, first difference at 13 % to 77 % of the text; with MTP and 3 queries there is no difference). I did not measure a task-level quality metric for these three.

## Patches

| # | Branch | What | Measured effect (RTX 4070) | Measurement switch (first commit of the branch only) |
|---|---|---|---|---|
| 0001 | `pr/pq2_0-multicol` | PQ2_0 mat-vec kernel for 3-8 columns (raw 2-bit codes into `dp4a` with an integer correction) | 4 columns 105.6 -> 54.0 us, 8 columns 136.8 -> 78.1 us (m=5120, k=17408); +17 % / +25 % / +29 % decode with 4 / 6 / 8 parallel sequences | `GGML_CUDA_PQ2_MULTICOL=0` |
| 0002 | `pr/fattn-gqa-mma` | MMA flash attention for GQA > 4 with quantized K/V (instead of the vector kernel), only when MMA can read that K/V type in place | 120k context, no MTP: 25.3 -> 41.2 tok/s | `GGML_CUDA_FATTN_GQA_MMA=0`, `GGML_CUDA_FATTN_VEC_MAXQ` |
| 0003 | `pr/fattn-mma-tile` | smaller KV tile (`nbatch_fa` 32) for the 8-column MMA config, head size 256 | 120k context, no MTP: 41.2 -> 46.0 tok/s (+12 %); no effect with MTP | - |
| 0004 | `pr/concat-transpose-all-gpus` | transposing concat kernel (was GB10-only) on all GPUs | 8.8 -> 2.5 us per call; +1.2 % decode with MTP (n-max 2) | `GGML_CUDA_CONCAT_TRANSPOSE=0` |
| 0005 | `pr/ptq1-gate-up-fuse-mc` (1/2) | gate + up + SwiGLU fused PTQ1_0 mat-vec for 2-4 columns (speculative verify steps; needs a contiguous bias, covered by new test-backend-ops cases) | +1.01 % decode | `GGML_CUDA_PTQ1_FUSE_MC=0` |
| 0006 | `pr/ptq1-gate-up-fuse-mc` (2/2) | keep that fusion when the GLU output overlaps the q8 rows (pool block) | +0.33 % (0005 + 0006: +1.28 %) | `GGML_CUDA_FWHT_GLU_POOL=0` |
| 0007 | `pr/kv-seq-rm-bound` | `llama_kv_cache::seq_rm` only over the used cell range | +0.77 % at 114688 context | - |
| 0008 | `pr/server-ckpt-buffer-reuse` | server: reuse the buffers of evicted prompt checkpoints (no zero-fill of ~150 MiB each) | -20 ms per turn (646 -> 626 ms) in a growing multi-turn conversation; decode speed unchanged | `LLAMA_CKPT_REUSE=0` |
| 0009 | `pr/sampler-topk-from-logits` | take the top-k straight from the logits when top-k starts the sampler chain (same heap steps as `std::partial_sort`, same order for equal logits) | +1.9 % (greedy benchmark), +1.85 % (thinking sampling 1.0 / 0.95 / 20 / 0.05, reasoning budget) | `LLAMA_SAMPLER_FAST_TOPK=0` |
| 0010 | `pr/ptq1-l2-prefetch` | the last 46 CTAs of a PTQ1_0 mat-vec prefetch 50 % (capped at 8 MiB on Ada, 2 MiB elsewhere) of the next mat-vec's weights into L2 | +2.0 % (greedy), +2.1 % (Hermes-like setup), +1.7 to +2.2 % at depth 0 to 65k | `GGML_CUDA_L2_PREFETCH_PCT=0` (also `_CTAS`, `_MAX_KB`) |

The switches exist only in the first commit(s) of each branch; the last commit of the branch removes them (for 0010 it fixes `_PCT` and `_CTAS` to 50 % / 46 and the cap to 8 MiB on Ada, 2 MiB elsewhere; the gains in the table were measured with the earlier 16 MiB cap on the PR branch).

All 10 together against the stack without 0008-0010 (`patched-fb`): **+2.83 %** (Hermes-like setup: 114688 context, thinking sampling, reasoning budget 16384, MTP n-max 2, q4_0 K/V),
+3.2 % on the greedy benchmark. The gains of 0009 and 0010 overlap and do not add up fully.

## How it was measured and checked
- Decode speed: the same binary, one switch flipped through an environment variable (build the first commit of a branch to reproduce a measurement), runs alternating A B B A, 4 greedy prompts x 256 tokens (or the sampling setup above), 95 % bootstrap interval of the paired difference. Run-to-run noise about 0.04 to 0.12 %, so effects above ~0.3 % are reliable; sessions differ by up to ~1 %, only paired numbers count.
- Outputs (patches 0004 to 0010): identical text in every A/B run (also with sampling and a fixed seed), identical token hashes at depth 0 / 16k / 65k, and 80 of 80 identical answers (same pass/fail) of a greedy functional test suite for the stack up to 0009 compared with the stack before 0008 (which already contained 0001 to 0007).
- `test-backend-ops` (MUL_MAT, MUL_MAT_ID, MUL_MAT_VEC_FUSION, GLU, SWIGLU, RMS_NORM, FLASH_ATTN_EXT, CONCAT) on the full stack: 6373 of 6373 cases pass on CUDA0 (CUDA against CPU).
- 0009: the heap selection was compared with `std::partial_sort` on 120000 random arrays (many ties, NaN, +-inf, sorted inputs): no differences; 8 request types (tool calls with grammar, thinking, penalties, top_k 1 and 200, logprobs) give identical outputs with the switch on and off. It only takes effect when everything before top-k in the chain does nothing (no penalties / DRY / top-n-sigma / logit bias / mirostat / backend sampling, k <= 128, grammar not applied first, reasoning budget not forcing); otherwise the old path runs.

## Limits
- One GPU (RTX 4070, cc 8.9) and one model family. Other GPUs and models are untested; 0010 in particular is tuned on this GPU (184 CTAs instead of 46 is 2.3 % slower here).
- `llm-testsuite`-style checks were not repeated for 0010 alone (outputs were identical in all A/B runs and depth hashes).
- Multiple server slots, 262k context and vision are untested.

## Using the stack
Build as usual (`-DGGML_CUDA=ON`); there are no new build options and no environment switches at the tip. Individual patches can be taken from the `pr/*` branches or cherry-picked from the history of this branch.

## AI assistance
The patches were developed with Claude Code (Anthropic's coding agent): it wrote the code,
the measurement scripts and the first drafts of the commit messages and PR texts.
I decided what to work on (which kernels and host paths to optimise, based on profiles of
my own decode setup). The measurements and checks listed in the PR texts were run in the
Claude Code sessions; I did not re-run them independently. I will maintain the changes.
Commits where Claude Code was used carry a `Co-Authored-By` trailer.
