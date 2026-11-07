# Swift-1.5-Qwen3.8-27B INT4 on 2x RTX A4500 (SM86, NVLink NV4), TP2

All numbers are from `bench.sh` against the server presets in `serve.sh`, on branch `ampere-opt`.
They are single-user (`--max-running-requests 1`) unless noted.

| Scenario | Workload |
|---|---|
| S1 | random (ShareGPT text) 1k in / 1k out, c1 |
| S2b / S2 | 8k / 32k prefill, 64 out, c1, flushed cache (TTFT) |
| S5 | ShareGPT chat, 512 out, c1 |
| S6 | HumanEval + MBPP chat, 512 out, c1 |

## Decode (output tok/s, accept length in parentheses)

| Config | S1 | S5 | S6 |
|---|---|---|---|
| A: reference cmd (MTP 3/1/4, BF16 draft), upstream code | 95.3 (2.99) | 82.8 (2.60) | - |
| + tiny_gemm for skinny BF16 linears on SM8x | 102.9 (3.06) | 87.5 (2.61) | 106.1 (3.09) |
| + `--speculative-token-map` 32k (TP-aware fix) | 120.5 (2.90) | 102.8 (2.48) | 129.8 (3.05) |
| + `--speculative-draft-model-quantization fp8` (MTP) | 125.9 (2.91) | 107.4 (2.48) | 134.5 (3.05) |
| same, MTP 4 steps / 5 draft tokens | 129.1 (3.19) | 106.0 (2.60) | 137.6 (3.33) |
| DFLASH2 (BF16 draft), 8 draft tokens | 148.7 (4.24) | 118.5 (3.38) | 151.0 (4.22) |
| DFLASH2, FP8 draft | 153.9 (4.16) | 123.4 (3.35) | 152.8 (4.06) |
| DFLASH2, FP8 draft, 32k hot vocab | 149.7 (3.75) | 125.8 (3.17) | 157.9 (3.90) |
| **DFLASH2, FP8 draft, 64k hot vocab** | **160.9 (4.11)** | **129.7 (3.33)** | **166.2 (4.15)** |

Token maps for MTP (FP8 off), output tok/s: full 102.9/87.5/106.1, 16k 115.9/100.3/127.1,
32k 120.5/102.8/129.8, 64k 121.3/103.2/124.1.

## Prefill

| Config | S2b TTFT | S2 TTFT |
|---|---|---|
| A | 3.88 s (2.1k tok/s) | 17.78 s (1.84k tok/s) |
| DFLASH2 FP8 + 64k | 3.80 s | 17.67 s |
| + custom AR v2 forced to 64 MB (no gain) | 3.86 s | 17.78 s |

## Findings

- **Decode.** Marlin W4A16 runs at 430-550 GB/s on the decode shapes. Measured achievable
  read bandwidth is about 570-577 GB/s, so the target GEMMs are near the ceiling.
  The remaining decode cost was in the draft and lm_head:
  - Every MTP draft step read the full 248k-vocab BF16 head (2.15 ms each). The token map
    fixes this.
  - The MTP layer itself is BF16. FP8 Marlin W8A16 fixes this.
  - The GDN `in_proj_ba` (N=48) cuBLAS GEMM took 28-61 us; `tiny_gemm` takes 3.6 us.
- **Prefill.** Prefill is about 80% Marlin. Under sustained load the A4500 sits at its 200 W
  power cap, with SM clock 1.36-1.43 GHz against a 2.1 GHz max. Sustained Marlin reaches
  about 65 TFLOPS and cuBLAS BF16 about 75. Slicing M into smaller calls does not help once
  throttled. A larger custom all-reduce does not beat NCCL for 40 MB prefill all-reduces,
  because both are NVLink-bandwidth-bound.
- **FP8 KV scales.** The checkpoint ships per-layer `k_scale` (about 0.03-0.04) and
  `v_scale` (about 0.02-0.03). `QWEN3_5_KV_SCALE_MAPPER` maps them onto `attn.k_scale`
  and `attn.v_scale`, which the compressed-tensors KV method creates at -1. The load log
  has no "not found in params_dict" warnings, and the BF16-KV bug below confirms that the
  scales reach the attention kernels. The "Defaulting to scaling factors of 1.0" warning only
  refers to `--quantization-param-path`.

## Multi-user

Presets `M` (MTP) and `MD` (DFlash2), with `USERS=4`. Both use `extra_buffer_lazy`,
`SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`, `--max-mamba-cache-size 12`, 2048-token prefill chunks,
and `--disable-prefill-cuda-graph`. Prefill graphs OOM at 0.90 with 4 users.

| Config | KV pool | S1 c1 | S1 c2 | S1 c4 |
|---|---|---|---|---|
| MTP 3/1/4, FP8 draft, 32k map | 333k tokens | 125.6 | 209.9 (TPOT 8.2 ms) | 330.1 (TPOT 9.8 ms) |
| DFlash2, FP8 draft, 64k map | 128k tokens | 161.2 | 229.1 (TPOT 6.9 ms) | 308.2 (TPOT 10.1 ms) |

- DFlash2 is fastest for 1-2 users. Its verify intermediates (2.8 GB for 4 x 8 tokens) and
  draft weights leave only 128k KV tokens, so 4 x 80k users only fit with MTP.
- `--enable-linear-replayssm-spec` is KDA-only for DFLASH, so it does not apply to this
  GDN model.
- **Multi-turn shared prefix (S4)** is limited by the mamba state slots. A hit needs a saved
  state at the prefix boundary; those exist at chunk boundaries and at prompt end.
  - Sequential probes do hit: a shared 16k prefix gets 14,336 cached, a multi-turn
    continuation gets 16,256.
  - With 4 running requests and 12 slots, most S4 turns re-prefill (mean TTFT 11.2 s).
    Going to 24 slots roughly doubles the hits (mean TTFT 8.9 s) but costs 56k KV tokens.

## Accuracy (GSM8K, 250 examples, greedy, thinking off, 4 concurrent)

| Server | Score | Eval wall time | Output tok/s |
|---|---|---|---|
| upstream `main`, MTP 4-user config | 0.968 | 240.9 s | 378 |
| this branch, MTP + FP8 draft + 32k map | 0.968 (rerun 0.972) | 206.5 s | 442 |

Upstream vs branch: 214/250 responses byte-identical. The branch against itself: 244/250.
The extra divergence comes from BF16 reduction order (tiny_gemm for `in_proj_ba`) and from
different verify batch shapes. Accuracy is unchanged.

## ReplaySSM, text-only mode, dense prefill, agent turns

**`--enable-linear-replayssm-spec` (GDN):** keeps verify drafts in a ring, so the intermediate
SSM cache goes to 0. Before this branch, DFLASH rejected it on GDN models; this branch routes
the DFLASH commit through the generic replay commit.

| Config | KV pool | c1 | c2 | c4 |
|---|---|---|---|---|
| MTP 4-user + replay | 333k -> 423k | 122.9 | - | 330.1 -> 367.1 |
| DFlash2 4-user + replay | 128k -> 269k | 161.2 -> 173.0 | 229.1 -> 262.8 | 308.2 -> 372.4 |

- DFlash2 + replay at c1 measured S1 172.8, ShareGPT chat 136.9, code 157.3 tok/s.
- Accept length moves by about +/-5-10% on fixed prompts, because greedy sequences diverge.
- GSM8K score is 0.964, against 0.968 / 0.972 without replay.

**`--language-model-only`:** this branch adds Qwen3.5 to the allow-list.
- Weights drop from 9.07 to 8.62 GB per GPU, and the KV pool grows from 422k to 461k tokens.
  Prefill speed is unchanged.
- GSM8K is 0.960. It is not bit-identical to the mrope path (129/250 responses match).

**Dequant + cuBLAS for M >= 2048 (SM8x default):**
- 8k prefill TTFT 3.80 -> 3.48 s; 32k prefill TTFT 17.67 -> 16.24 s (about 8%).
- The scratch buffer adds 0.21 GB per GPU.
- Logprob deviation is below this model's own chunk-size noise (0.13 vs 0.19 nats).
- 8192-token chunks OOM at 0.90.

**Agent turn latency** (cached 30k prefix plus a new delta, MTP, `turn_latency.py`):

| Delta | No prefill graph | Prefill graph <= 1024 tokens (mem 0.86) |
|---|---|---|
| 64 | 119 ms | 96 ms |
| 256 | 234 ms | 237 ms |
| 1024 | 677 ms | 679 ms |
| 2048 | 1312 ms | 1331 ms |

- The prefix hits are exact: 29,952 of 30,000 tokens.
- Long-context attention with fp8 KV is compute-bound on SM86 (FlashInfer about 42-45 TFLOPS,
  against 57-62 TFLOPS with bf16 KV), because the fp8 to bf16 dequant sits in the inner loop.

## Output quality A/B (sampling + thinking, multi-turn with cache hits)

`quality_ab.py` ran 32 single-turn prompts (ShareGPT chat and code) and 6 three-turn conversations, with thinking on and up to 1024 output tokens. Turns 2 and 3 hit the mamba prefix cache.
- **Sampled runs** use the model's defaults: temperature 1.0, top_k 20, top_p 0.95.
- **Scoring:** every output is teacher-forced on upstream `main` without speculation. NLL is the mean negative log-likelihood per output token. Top-1 is the share of greedy tokens equal to the reference argmax.

| Config | Sampled NLL/token [95% CI] | Greedy NLL/token | Greedy top-1 |
|---|---|---|---|
| upstream, no spec (reference) | 0.504 [0.452, 0.555] | 0.233 | 99.00% (self-scored) |
| upstream MTP 3/1/4, BF16 draft | 0.479 [0.429, 0.531] | 0.221 | 99.09% |
| upstream DFlash2, BF16 draft | 0.476 [0.433, 0.518] | 0.237 | 99.01% |
| branch `best1` (MTP, FP8 draft, 32k map, ReplaySSM) | 0.471 [0.422, 0.517] | 0.242 | 98.99% |
| ...with `--disable-radix-cache` | - | 0.240 | 98.96% |
| branch `dflash1` with `REPLAY=0` | 0.496 [0.449, 0.541] | 0.245 | 98.96% |
| branch `dflash1` (DFlash2, FP8 draft, 64k map, ReplaySSM) | 0.482 [0.432, 0.531] | 0.234 | 99.04% |
| ...with `--disable-radix-cache` | - | 0.235 | 98.94% |

- No config differs from the reference beyond noise, including after prefix-cache hits.
- **Greedy decoding is not reproducible run to run on this setup.** The same config with the cache on and off first diverges at a median of about 125 tokens, and as early as token 7, on identical uncached prompts. Early divergence between configs is therefore not evidence of a bug.

**`best1` is now MTP.** Single user, output tok/s with accept length:

| Config | KV pool | S1 | S5 | S6 |
|---|---|---|---|---|
| MTP, FP8 draft, 32k map, ReplaySSM, 8 slots | 432k tokens | 123.5 (2.76) | 110.9 (2.48) | 137.5 (3.02) |
| `dflash1`, with 39 default slots (no `--max-mamba-cache-size`) | 165k tokens | - | - | - |
| `dflash1`, 5 slots (previous `best1`) | not recorded | 173.0 (4.29) | 138.0 (3.39) | 157.5 (3.76) |

## FP8 checkpoint (`Qwen/Qwen3.8-27B-FP8`)

The official FP8 checkpoint and `orcarouter/Qwen3.8-27B-Uncensored-FP8` (abliterated, same layout) are
block-FP8: e4m3 weights with 128x128 scale blocks and dynamic activations. Their MTP head is FP8 too.
`lm_head`, embeddings, vision, norms and `in_proj_a/b` stay BF16, and there are no `k_scale`/`v_scale`
tensors, so fp8 KV runs at scale 1.0. On SM8x, sglang serves the weights as FP8 Marlin W8A16.

Branch optimizations that apply unchanged: the token map, the shared draft embed/lm_head, `tiny_gemm` for
`in_proj_ba`, ReplaySSM, `--language-model-only`, and the DFlash hot vocabulary. The checkpoint's own MTP
is already FP8, so `fp8x1` does not pass `--speculative-draft-model-quantization`.

Preset `fp8x1` (`LM_ONLY=1`, TP2, MTP 3/1/4, 32k map, ReplaySSM, 8 mamba slots, mem 0.92):

| | KV pool | S1 | S5 | S6 | 8k TTFT | 32k TTFT |
|---|---|---|---|---|---|---|
| INT4 `best1` (reference) | 437k | 123.6 (2.76) | 110.9 (2.48) | 137.5 (3.02) | 3.61 s | 16.2 s |
| FP8, Marlin prefill | 125k | 76.9 (2.61) | 73.4 (2.48) | 87.6 (2.91) | 4.67 s | 21.0 s |
| FP8, dequant + cuBLAS prefill (default) | 114k | - | - | - | 3.58 s | 16.6 s |

- **Decode** is 0.62-0.66x of INT4, the ratio of weight bytes. Acceptance is the same.
- **KV room:** without `expandable_segments`, the pool was 61k tokens. The draft's embed and lm_head copies
  (2.5 GB per GPU) are freed when they are shared with the target, but the caching allocator cannot release
  segments that FP8-repacked MTP weights still occupy. 1.4 GB stayed reserved. `serve.sh` now sets
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; INT4 `best1` gains 432k -> 437k with S1 unchanged.
- **Prefill:** FP8 Marlin reaches 51-63 TFLOPS at M >= 2048, against 72-79 for cuBLAS BF16
  (`bench_marlin_shapes.py --wtype fp8`). Dequant + cuBLAS (`marlin_fp8_dequant`, shared with the W4A16
  path and `SGLANG_MARLIN_DEQUANT_GEMM_MIN_M`) cuts TTFT by 21-23%. The dequantized weight is
  bit-identical to fp8 x bf16 scale. Prompt logprobs move by a mean of 0.137 nats against the Marlin
  path, inside the chunk-size noise floor (Marlin 4096 vs 2048-token chunks: 0.116).

## BF16 KV with checkpoint KV scales (bug, fixed in `c34b472525`)

With `--kv-cache-dtype bf16`, `best1` on the INT4 checkpoint answered 64k/128k-token prompts with
fluent text unrelated to the prompt, ending in random tokens. The same config with fp8 KV quoted
the right code from the first 10% of a 128k context.

Cause, also in upstream `main`:
- The checkpoint's fp8 calibration scales (k about 0.03) are loaded whatever the cache dtype is.
- The write path divides by them only when it casts to fp8.
- FlashInfer multiplies them back in on every read.
- A BF16 cache therefore had its attention logits scaled by about 0.03.

Fix: `configure_kv_cache_dtype` resets the scales to 1.0 for a 16-bit cache. A regression test
fails on the old code. `fp8_e4m3` and `auto` (which resolves to fp8 here) were never affected.

## Long-context and full-thinking quality

`quality_ab.py --set long`: 4 documents at 64k and 4 at 128k tokens, cut from a pinned snapshot
of the repo sources (`--src-dir`). Each asks about a function defined in the first 10% and has two
follow-up turns at depth.

Generation is sampled at xhigh thinking (temperature 1.0, top_p 0.95, top_k 20) with up to 16k
output tokens. Scoring is on upstream `main` + the fix above, no spec, bf16 KV.

| Config | 64k NLL/tok | 128k NLL/tok | Overall [95% CI] | top-1 | stop% | Short set, 16k tokens |
|---|---|---|---|---|---|---|
| ref: upstream, no spec, bf16 KV | 0.283 | 0.285 | 0.284 [0.261, 0.310] | 89.1% | 100 | 0.326 [0.284, 0.367], 88.0%, stop 84% |
| `best1`, fp8 KV | 0.297 | 0.298 | 0.297 [0.272, 0.324] | 88.8% | 96 | 0.332 [0.291, 0.377], 87.7%, stop 86% |
| `best1`, bf16 KV | 0.268 | 0.322 | 0.289 [0.263, 0.319] | 88.9% | 92 | - |
| upstream MTP, fp8 KV | 0.285 | 0.296 | 0.290 [0.274, 0.311] | 88.9% | 96 | - |

All four are inside the reference CI at both lengths, with no trend from 64k to 128k. 24 turns per
config resolve about 0.04 nats/token.

### FP8 checkpoint at 32k: fp8 KV at scale 1.0 vs bf16 KV

`fp8x1` (MTP) on `Qwen/Qwen3.8-27B-FP8`. Long set is 6 docs x 3 turns at 32k; short set is 16k
output tokens. Scored on an FP8 no-spec bf16-KV server (`--chunked-prefill-size 256`).

| KV cache | Long 32k NLL/tok | top-1 | Short NLL/tok | top-1 | RULER2 32k (mk/mv_niah_hard, qa_medium/hard) |
|---|---|---|---|---|---|
| fp8, scale 1.0 | 0.448 [0.429, 0.466] | 83.6% | 0.479 [0.442, 0.518] | 82.9% | 87.0% (92/100/80/76) |
| bf16 | 0.453 [0.431, 0.478] | 83.4% | 0.480 [0.445, 0.517] | 82.8% | 85.6% (88/92/84/78.4) |

No measurable loss from uncalibrated fp8 KV, so no calibration is needed. 128k-250k on FP8 weights
does not fit here and needs an A100 run. These NLL values come from a different scorer, so they are
not comparable with the INT4 table.

## Single-user recommendation, re-measured 2026-10-13 (INT4, current branch)

Same session and same `bench.sh` seeds. S7 is a 100k-token prompt, 256 out.

| Preset | KV pool | S1 | S5 | S6 | 8k / 32k TTFT |
|---|---|---|---|---|---|
| `dflash1` (5 slots), vision on | 294k | 172.8* | 137.4 (3.38) | 158.2 (3.76) | - |
| `dflash1`, `LM_ONLY=1` | 323k | 176.3 (4.28) | 142.0 (3.47) | 138.3 (3.26), reproduced | 3.59 s / 16.2 s |
| `best1` (MTP, 8 slots), `LM_ONLY=1` | 476k | 133.4 (3.00) | 112.1 (2.51) | 141.6 (3.11) | - |

\* from the ReplaySSM section above.

- The earlier 165k pool for DFlash2 came from leaving the default 39 mamba slots. With 5 slots,
  DFlash2 holds the full 262k context. `dflash1` with vision on is now the single-user
  recommendation. A 100k-token prompt runs (65 s TTFT).
- `--language-model-only` lowers DFlash2 acceptance on code (3.76 -> 3.26), reproducibly. It does
  not hurt MTP. The cause has not been investigated. The text-only path uses plain RoPE positions
  instead of mrope, which may shift the target hidden states the draft conditions on.

## After the upstream sync, 2026-11-07 (branch rebased onto `ba5906471c`)

Same `bench.sh` scenarios. "Upstream" is upstream `main` at `ba5906471c` with the reference MTP
command (preset `A`), so it is today's baseline, not the October one. Accept length in parentheses.

| Config | KV pool | S1 | S5 | S6 | 8k / 32k TTFT |
|---|---|---|---|---|---|
| upstream, preset `A` | 283k | 94.9 (2.99) | 82.6 (2.60) | 102.3 (3.18) | 4.02 s / 17.96 s |
| branch `best1`, `LM_ONLY=1` | 476k | 133.1 (3.00) | 111.5 (2.51) | 141.4 (3.11) | 3.57 s / 16.44 s (vision on) |
| branch `dflash1` | 294k | 173.8 (4.29) | 137.2 (3.34) | 158.8 (3.76) | - |

- **The branch's numbers are unchanged by the sync** (pre-sync: 133.4 / 112.1 / 141.6 and
  172.8 / 137.4 / 158.2). Upstream's own MTP baseline has not moved either (95.3 / 82.8 in October).
- **Accuracy:** GSM8K, 250 examples, greedy, thinking off, `best1` with `LM_ONLY=1`: 0.980. `sgl-eval`
  moved from 0.1.2 to 0.1.3 in the sync, so this is not directly comparable with the 0.96-0.97 above.
- **Fixes the sync needed:**
  - `tiny_gemm` lost its `max_m` argument and now picks its kernel by batch size, so the SM8x
    skinny-linear route crashed on its first decode call.
  - The SM8x fallback in `_set_mha_kv_buffer_prefix_valid` now divides out of place, like the fused
    kernel it stands in for. The legacy eager writer keeps dividing in place, as upstream's tests require.
- **Known test failures on SM86:** `TestDflashFp8PrefixCommit::test_dflash_constructor_initializes_fp8_prefix_scales`
  asserts the fused fp8 Triton kernel ran, which is impossible below SM89. Upstream fails all 21 tests in
  `test_prefix_valid_fp8_pool.py` on SM86 (`fp8e4nv not supported`); the branch fails only those 2.

