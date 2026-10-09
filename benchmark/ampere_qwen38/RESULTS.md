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
  has no "not found in params_dict" warnings, so the scales appear to load. This is
  inferred from the load path; the values were not read back at runtime. The
  "Defaulting to scaling factors of 1.0" warning only refers to
  `--quantization-param-path`.

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
