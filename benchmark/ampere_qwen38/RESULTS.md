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
