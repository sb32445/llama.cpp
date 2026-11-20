# kTrain patches for the Prism llama.cpp fork

Decode-speed patches for PTQ1_0 / PQ2_0 ternary models (tested with Ternary-Bonsai-2-27B plus an MTP draft head) on NVIDIA Ada (RTX 4070, cc 8.9).
They are offered to PrismML as separate pull requests. This branch (`ktrain`) is `prism` plus those that are still open ([#310](https://github.com/PrismML-Eng/llama.cpp/pull/310), [#314](https://github.com/PrismML-Eng/llama.cpp/pull/314), [#410](https://github.com/PrismML-Eng/llama.cpp/pull/410), [#411](https://github.com/PrismML-Eng/llama.cpp/pull/411)), for people who do not want to wait for the merges, one prefill patch and one sampler patch that are not submitted, PrismML's open pull requests #319 and #320 (tiered KV cache and draft window, by Cary Palmer) and eight commits ported from his bonsai-ada-surgery series (see "Tiered KV and ported patches" below).

Base: `prism` at `3f035ba66` (2026-11-16, includes #338 `--checkpoint-every-nt`). Earlier bases were `5628c7338` (build 12130, 2026-10-30), `c96baf4ab` and `9d9efeadb`; the measurements below name the base they were taken on. Seven earlier patches of this stack are already part of `prism` (#306 to #309, #311 to #313; the table below lists them).

## Tiered KV and ported patches (2026-11-19 to 2026-11-20)

PrismML's open pull requests #319 (`--kv-vram-cells N`: the first N cells of the KV cache live in VRAM, the rest in pinned host memory in one VMM range) and #320 (`--spec-draft-window`, MTP drafting at every depth) are in this branch unchanged, with Cary Palmer as author (three commits for #320, two for #319). On top of them eight commits port patches from his [bonsai-ada-surgery](https://github.com/professorpalmer/bonsai-ada-surgery) series (MIT); each commit message names the patch number, and where a hunk or call site had to be adapted to this tree it says so:

| Commit subject (short) | Patch | What |
|---|---|---|
| route quantized-KV GQA decode to the in-place MMA flash-attention kernel | 29 | decode with more than 4 queries per KV head and at most 8 queries (decode, speculative verify) uses the MMA kernel instead of the vector kernel for q4_0/q8_0 K/V from 256 KV cells on (`GGML_CUDA_FA_MMA_DECODE_MIN_KV`, 0 = never) |
| tiered KV prefill writes new rows through the VRAM alias and prefetches host rows | 41 | prefill graphs write new rows to VRAM first and copy them to the host pages in whole 16-byte stores; the host rows of the next attention op are staged on a second stream (`GGML_CUDA_KV_TIER_REDIRECT=0`, `GGML_CUDA_KV_TIER_PREFETCH=0` turn this off) |
| run graphs that read the host tail without CUDA graphs | 53 | the prefetch pipeline only runs on the non-graph path (`GGML_CUDA_KV_TIER_GRAPHS=1` restores the old behaviour) |
| partial staging, in-place reads for small attention ops, quieter graph log | 47, 71 | `GGML_CUDA_KV_TIER_STAGING_FRAC` and `GGML_CUDA_KV_TIER_STAGE_MIN_Q`, both off by default |
| `--spec-draft-n-max-tail` | 31 (rest) | draft size once the sequence reaches the `--kv-vram-cells` line, where a wider verify batch reads the host tail once for all columns; 0 = `--spec-draft-n-max` |
| report reasoning tokens, stop drafts at EOG, no EOG in an open think block, F16 prefill only for large batches | 35, 67, 73, 43 | server and reasoning-budget fixes; `GGML_CUDA_FA_PREFILL_F16_MIN_Q` (default 64 queries) keeps speculative verify batches on the in-place quantized read |
| dequantize K/V tiles as packed halves in the MMA loader | 57 | q4_0 and q8_0 only (the q3_0 part is left out, this tree has no q3_0 KV type), tile bit-identical to a `to_fp16` pass |
| load both tiles up front (FA MMA, quantized K/V, no pipeline stages) | 72 | loads of the K and V tile words start at the beginning of the tile, each tile is dequantized when needed |

What was measured (RTX 4070 12 GB, Ternary-Bonsai-2-27B variant with PTQ1_0 + MTP head, q4_0 K/V, single runs, not repeated):
- **262k window:** with `--kv-vram-cells 218000`, `-ub 256` and `--spec-draft-n-max-tail 4` (the value at the time of the run) the server took a prompt of 257,654 tokens without error (VRAM peak 11,434 MiB, no checkpoint dropped). Decode 86.6 / 58.2 / 53.9 / 37.4 / 36.6 / 36.1 tok/s at 32k / 131k / 210k / 230k / 245k / 258k of context with prose prompts and short answers (draft acceptance 21 to 26 % there); a code question behind 218k decoded at 68 to 74 tok/s. A needle test with six needles found five; the one at about 100k tokens was wrong at 258k and right at 131k, the cause is not known.
- **Decode, this branch against the same branch without the last three ported commits** (the two kernel commits for patches 57 and 72 and the server/reasoning-budget commit; one run per depth): decode +1.5 % / +4.2 % / +6.2 % at 32k / 128k / 192k of context; output hashes equal.
- `test-backend-ops` (`FLASH_ATTN_EXT`) 4492 of 4492 cases on this state; the full operator suite (20797 of 20797 cases) was run on the state before the tiered-KV and ported commits (the branch at `611730568`), not again afterwards.
- Not measured: other GPUs, more than one server slot with the tiered cache, vision input, quality (perplexity) of the tiered-KV commits.

## Open patches (the first commits of this stack)

| PR | Commit | What | Effect on the RTX 4070 (MTP `n-max 2`, 8 interleaved A/B pairs, outputs identical) |
|---|---|---|---|
| [#310](https://github.com/PrismML-Eng/llama.cpp/pull/310) | gate (SwiGLU) fused PTQ1_0 mat-vec for 2-4 columns | gate + up + SwiGLU run as one kernel in every FFN block of a 2-4 token verify step (speculative decoding). Includes keeping the fusion when the GLU output overlaps the q8 rows (pool block), a contiguous-bias guard, the per-cc `rows_per_item` guard and new `test-backend-ops` cases. On the current base the 2-4 column kernel is not used for the `SWIGLU_CLAMP` GLU op that `prism` fuses now (the graph runs unfused; the kernel computed it wrongly before this guard, found by `test-backend-ops`) | removing it from this branch costs **0.65 %** at depth 0 and 0.37 % at 114688 tokens |
| [#314](https://github.com/PrismML-Eng/llama.cpp/pull/314) | prefetch the next PTQ1_0 mat-vec's weights into L2 | the last CTAs of a PTQ1_0 mat-vec (one per SM) prefetch 50 % of the next mat-vec's weights, capped at 8 MiB on Ada and 2 MiB elsewhere | removing it costs **0.97 %** at depth 0 and 0.55 % at 114688 tokens |

The pull-request branches `pr/ptq1-gate-up-fuse-mc` and `pr/ptq1-l2-prefetch` carry the same patches as separate branches, based on an older `prism`; this branch has them squashed to one commit each and ported to `prism`'s current `ggml_cuda_try_fwht_q8`.

## Prefill patches (#410 and #411, open)

| Commit | What | Effect on the RTX 4070 (prefill, q4_0 K/V) |
|---|---|---|
| flash attention, GQA ratio 6 | prompt batches (>= 64 query tokens) use `ncols2 = 2` instead of 8, which removes 25 % padding in the MMA tiles; decode and MTP verify keep 8 | with the next commit: `llama-bench` pp2048 (`-ub 512`) +10.3 % at depth 32768 and +17.1 % at 98304 |
| PTQ1_0 MMQ scale | the block scale is applied once per 128 weights instead of per 32-wide MMA step | +3.9 % pp2048 at depth 0 (both commits) |

Prefill speed without a server (`llama-bench`, pp2048, `-ub 512`, q4_0 K/V, 2 repetitions, run order A B B A, both commits together; `ktrain` without them against with them):

| Context depth | without ↑ | with ↑ | change |
|---|---|---|---|
| 0 | 1368.1 tok/s | 1422.0 tok/s | +3.9 % |
| 32768 | 963.9 tok/s | 1063.6 tok/s | +10.3 % |
| 98304 | 577.7 tok/s | 676.3 tok/s | +17.1 % |

↑ more is better. The gain at depth 0 matches the MMQ scale commit; the growth with depth matches the attention commit (this split is an inference, the two commits were measured together only).

In `llama-server` the prefill of 114688 tokens takes 139.9 s instead of 157.7 s (-11.3 %); -5.9 % at 2048, -7.7 % at 32768, -8.9 % at 65536. These two commits are **not bit-identical** to `prism`: perplexity 14.5129 / 12.7715 against 14.5316 / 12.7576 (within the error of 0.62 / 0.38), and the generated text of a greedy run changes, which moves the MTP acceptance in either direction (4 prompts: 74.0 -> 72.0 %, decode -0.84 %; one code question per depth: -1.7 % to +3.2 %). They are measured on the GPU above only; the decode kernels are unchanged.

**Numerics of the two open patches:** on the earlier base `c96baf4ab` both left the results unchanged (perplexity and all generated texts bit-identical to `prism`, at every depth tested). That was not repeated on `5628c7338` separately; the whole branch is not bit-identical because of the prefill commits.

**Not submitted: SwiGLU + Hadamard fusion for prompt batches** (two commits, "SwiGLU + sign flip + Hadamard transform in one launch for prompt batches" and its q8_1 follow-up for the PTQ1_0 MMQ). No measurement of it is published here.

## Sampler patch (one commit)

`common : keep the fast top-k path when penalties only lower logits`: with repetition/frequency/presence penalties that can only lower logits, the sampler keeps the straight top-k path of #313 instead of sorting the whole vocabulary. Greedy and penalty-free sampling are unchanged. Measured earlier on a `prism-addon` build (2026-10-28, decode +10 % with the penalty settings of my server, texts bit-identical); not re-measured on the current base.

## Measured against `prism` `5628c7338` (2026-11-02)

RTX 4070 12 GB, Ternary-Bonsai-2-27B PTQ1_0 + MTP head, MTP `n-max 2`, q4_0 K/V, greedy, no K-mean-centering file, same build settings; `prism` `5628c7338` against this branch (all five code commits). Single A/B runs this time (8 interleaved pairs for the server decode, one run per depth):

| Measurement | `prism` | this branch | change |
|---|---|---|---|
| decode, 4 prompts, 8 pairs, depth 0, median ↑ | 115.90 tok/s | 117.68 tok/s | **+1.5 %** (first run +1.52 %, CI [+1.47, +1.57] %; in the second run one outlier of the branch lowers the mean to +1.13 %) |
| decode, code question, depth 0 ↑ | 125.2 | 128.2 | +2.4 % |
| decode, depth 32768 ↑ | 105.5 | 107.2 | +1.6 % |
| decode, depth 114688 ↑ | 80.2 | 81.6 | +1.7 % |
| server prefill time, depth 32768 ↓ | 29.5 s | 27.5 s | -7 % |
| server prefill time, depth 65536 ↓ | 70.0 s | 63.4 s | -9 % |
| server prefill time, 114688 tokens ↓ | 152.2 s | 134.0 s | **-12 %** |
| perplexity 16 x 512 ↓ | 14.5423 ± 0.62 | 14.5345 | within the error |
| perplexity 4 x 4096 ↓ | 12.7692 ± 0.38 | 12.7657 | within the error |

↑ more is better, ↓ less is better. Not bit-identical to `prism` (the prefill commits change rounding): draft acceptance 72.2 % against 73.5 % in the 4-prompt run. The new `prism` commits since `9d9efeadb` (#417 to #424: MMQ rework, Hopper paths, MoE epilogue fusions, float4 RMS norm) give a dense PTQ1_0 model nothing measurable without these patches. The `llama-bench` prefill table above was measured on the old base and not repeated. `test-backend-ops` on CUDA against the CPU: `MUL_MAT_VEC_FUSION` 1632/1632 (after the `SWIGLU_CLAMP` guard), `FLASH_ATTN_EXT` 4492, `MUL_MAT` 2943, `MUL_MAT_ID` 1097, no failure.

## Measured against `prism` `c96baf4ab` (2026-10-27, earlier base)

RTX 4070 12 GB, Ternary-Bonsai-2-27B PTQ1_0 + MTP head, MTP `n-max 2`, q4_0 K/V, greedy, no K-mean-centering file, same build settings; `prism` `c96baf4ab` against this branch **without the two prefill commits** (measured before they were added; their effect on decode is in the section above):

| Context depth | `prism` ↑ | first two commits ↑ | gain |
|---|---|---|---|
| 0 | 126.30 tok/s | 128.70 tok/s | +1.9 % |
| 8192 | 117.55 | 119.75 | +1.9 % |
| 32768 | 105.15 | 106.70 | +1.5 % |
| 65536 | 93.00 | 94.15 | +1.2 % |
| 114688 | 80.70 | 81.65 | +1.2 % |

Four prompts at depth 0 with 8 interleaved A/B pairs: 116.49 -> 118.80 tok/s, **+2.0 %** (95 % CI [+1.97, +2.03] %). Prefill time and draft acceptance (74.0 %) are the same. Every other number depends on the day (the same binary moved by 1.6 % between two days), so only numbers measured in one session next to each other count.

↑ more is better.

## Patches that are already in `prism`

| PR | What | Effect on the RTX 4070 |
|---|---|---|
| [#306](https://github.com/PrismML-Eng/llama.cpp/pull/306) | PQ2_0 mat-vec kernel for 3-8 columns | +17 / +25 / +29 % decode with 4 / 6 / 8 parallel sequences |
| [#307](https://github.com/PrismML-Eng/llama.cpp/pull/307) | MMA flash attention for GQA > 4 with quantized K/V | 120k context, no MTP: 25.3 -> 41.2 tok/s |
| [#308](https://github.com/PrismML-Eng/llama.cpp/pull/308) | smaller KV tile for the 8-column MMA config (head size 256) | 120k context, no MTP: 41.2 -> 46.0 tok/s (with #307) |
| [#309](https://github.com/PrismML-Eng/llama.cpp/pull/309) | transposing concat kernel on all GPUs | +1.2 % decode with MTP |
| [#311](https://github.com/PrismML-Eng/llama.cpp/pull/311) | `seq_rm` only over the used cell range | +0.8 % at 114688 context |
| [#312](https://github.com/PrismML-Eng/llama.cpp/pull/312) | reuse the buffers of evicted prompt checkpoints | -20 ms per turn in a long multi-turn chat |
| [#313](https://github.com/PrismML-Eng/llama.cpp/pull/313) | top-k straight from the logits | +1.9 % decode with MTP |

#306, #307 and #308 change the arithmetic order, so results differ at rounding level against the code before them; the others give identical outputs.

## How it was measured and checked
- Speed: paired runs of two binaries, A B A B ..., 4 greedy prompts (`tools/ab_bench.py`: 8 pairs, bootstrap interval of the paired difference), and one code question per depth with a filled context (`tools/mtp_depth.py`, two rounds in reversed order).
- Ablation on this branch (one group removed, 8 pairs): without the prefetch -0.97 %, without the gate fusion (including the pool block and its guards) -0.65 % at depth 0.
- Quality: perplexity on a German text (16 x 512 and a large batch) is bit-identical to `prism` for the first two commits (for the prefill commits see above); `test-backend-ops` against the CPU: `MUL_MAT_VEC_FUSION` 546, `GATED_DELTA_NET` 74, `FLASH_ATTN_EXT` 4347, `MUL_MAT` 1669 and `MUL_MAT_ID` 1035 cases, no failure (with the prefill commits `FLASH_ATTN_EXT` has 54 more cases, 4401, no failure).

## Limits
- One GPU (RTX 4070, cc 8.9) and one model family; other hardware is untested, the L2 prefetch in particular is tuned on this GPU (the 2 MiB cap for other GPUs is not measured).
- Greedy decoding only in the tables from the earlier bases; no `llm-testsuite`-style run for this state; multiple server slots with the tiered cache and vision are untested. The 262k results above are single runs.

## Using the branch
Build as usual (`-DGGML_CUDA=ON`); there are no new build options. New server options: `--kv-vram-cells`, `--spec-draft-window` (both from PrismML's #319/#320) and `--spec-draft-n-max-tail`. The environment switches listed in the table above are all optional and keep their defaults unless set.

## AI assistance
The patches were developed with Claude Code (Anthropic's coding agent): it wrote the code,
the measurement scripts and the first drafts of the commit messages and PR texts.
I decided what to work on (which kernels and host paths to optimise, based on profiles of
my own decode setup). The measurements and checks listed in the PR texts were run in the
Claude Code sessions; I did not re-run them independently. I will maintain the changes.
Commits where Claude Code was used carry a `Co-Authored-By` trailer. That includes the eight commits ported from the bonsai-ada-surgery series; the original code in them is Cary Palmer's, the porting to this tree was done with Claude Code. Commits of #319 and #320 are his and unchanged.
