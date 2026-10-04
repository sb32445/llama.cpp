# PTQ1_0 + MTP head for Ternary Bonsai 2 27B (recipe)

Part of the [kTrain patches](../README.md). Run the result with this fork (`ktrain` branch) or with PrismML's `prism`.

Unofficial recipe. It builds `Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q8_0.gguf` (5.96 GiB) from two files you download yourself:
the Prism ML **PTQ1_0** model and the community **MTP head** by ProCreations (shipped there inside a PQ2_0 file).
No weights are hosted here. See `NOTICE` for licenses and attribution.

## Why
The PQ2_0 and PTQ1_0 files of Bonsai 2 27B decode to bit-identical weights; PTQ1_0 is simply smaller (5.53 vs 6.70 GiB) and faster at one token.
The MTP head (one extra transformer layer, `blk.64.*`, Q8_0) speeds up decoding with speculative decoding, but is only published inside the PQ2_0 file.
This script moves the 15 head tensors onto the PTQ1_0 trunk. The trunk is copied byte for byte.

## Steps
```bash
pip install gguf numpy
# 1. download (check against SHA256SUMS): Ternary-Bonsai-2-27B-PTQ1_0.gguf, Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
sha256sum -c --ignore-missing SHA256SUMS
# 2. build
python3 make_ptq1_mtp.py Ternary-Bonsai-2-27B-PTQ1_0.gguf Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q8_0.gguf
# 3. verify: every tensor SHA-256 against its source (expect 866 tensors, no deviations)
python3 verify_merge.py Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q8_0.gguf Ternary-Bonsai-2-27B-PTQ1_0.gguf Ternary-Bonsai-2-27B-PQ2_0-MTP-Q8_0.gguf
```
Needs about 20 GiB of free disk space (sources 13.6 GB, output 6.4 GB); verifying took about 35 s on the author's machine. Use `GGUF_PY=<llama.cpp>/gguf-py` instead of pip if you want the `gguf` module of your llama.cpp checkout.

## Run
Use this fork or the PrismML llama.cpp fork (stock llama.cpp binaries fail to create the MTP context for these files):
```bash
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-MTP-Q8_0.gguf -ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 2
```
Loading the file without `--spec-type` ignores the head; plain decoding is unchanged (tg128 63.6 tok/s, same as the original file).

## Measurements (one RTX 4070 12 GB, one run per cell, greedy, 4 prompts; PrismML fork `prism` + the kTrain patches in an earlier state)
| `--spec-draft-n-max` | decode tok/s | vs. no MTP |
|---|---|---|
| none | 62.5 | |
| 1 | 92.2 | +48 % |
| 2 | 106.8 | +71 % |
| 3 | 112.7 | +80 % |
VRAM: +1.26 GiB. In a separate run with n-max 2 at context depth 0, 87 % of the drafted tokens were accepted (acceptance falls with deeper context and differs by prompt). Your numbers will differ with GPU, prompt and context depth.

## Checks done (2026-10-02)
- `SHA256SUMS` lists the source hashes as published by the two repos (PTQ1_0: 5946648928 bytes, MTP file: 7657489728 bytes). The author's own build has SHA-256 `cc8c830d1cb086c7508a4ec43415c09dc5105d787d5c73d376c9c3e2a3d8d9e7` (6397969728 bytes); other `gguf` versions may write the file differently, so `verify_merge.py` is the real check.
- Built file: 866 of 866 tensors byte-identical to their source; only metadata differs (`general.name`, `qwen35.block_count` 64 -> 65, `qwen35.nextn_predict_layers` 1).
- The head was not retrained here; whether it helps on your hardware is something to measure, not assume.
- The build script was last run for the original file with the llama.cpp `gguf-py` (0.19.0); the pip version was not tested. `verify_merge.py` was re-run today with the head source replaced by the built file, so the head comparison itself dates from the original build (866/866).
