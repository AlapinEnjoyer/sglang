# Handoff: SGLang on Ampere for Swift-1.5-Qwen3.8-27B (INT4)

- **Branch:** `ampere-opt`, forked from `main` at `14e3414513`. It has 15 commits and has not been pushed.
- **Environment:** `/home/ubuntu/experiments/sglang/.venv`, built with uv: `uv pip install --prerelease=allow -e "python[dev]"`. It contains torch 2.14.1+cu130, flashinfer 0.7.0.post1 and sglang-kernel 0.4.9.
- **Hardware:** 2x RTX A4500 (20 GB, SM86, 46 SMs), joined by an NVLink bridge (NV4).
- **Model:** `/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4`.
  - Qwen3.5 hybrid architecture: 48 GDN layers and 16 gated full-attention layers.
  - W4A16 compressed-tensors weights, group 128, symmetric.
  - BF16 MTP head, FP8 KV-cache scales.
- **DFlash2 draft:** `/home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2`.

All numbers below were measured on this machine with the scripts in `benchmark/ampere_qwen38/`.
Results from the old `../inference/sglang-inf` folder came from an older sglang and were ignored.

---

## 1. Headline results

**Single user** (output tok/s; accept length in parentheses; TTFT for the prefill rows):

| Scenario | Your original command (MTP 3/1/4, BF16 draft) | `best1` (DFlash2 + FP8 draft + 64k hot vocab + ReplaySSM) |
|---|---|---|
| S1: 1k in / 1k out | 95.3 (2.99) | **173.0** (4.29), +82% |
| S5: ShareGPT chat, 512 out | 82.8 (2.60) | **138.0** (3.39), +67% |
| S6: HumanEval/MBPP chat, 512 out | 106.1 (3.09)¹ | **157.5** (3.76), +48% |
| S2b: 8k prefill TTFT | 3.88 s | **3.61 s**, -7% |
| S2: 32k prefill TTFT | 17.78 s | **16.29 s**, -8% |

¹ This S6 baseline already includes the first fix (tiny_gemm). S6 did not exist yet when the pure-upstream baseline was measured.

**Multi-user** (`best4`: MTP + FP8 draft + 32k map + ReplaySSM, 4 slots):

| Metric | Value |
|---|---|
| KV pool | 411k tokens, enough for 4 users x 80k |
| 4 concurrent users, 1k / 1k | 360.6 tok/s total, TPOT 9.2 ms |
| 4 concurrent users, 16k prompts / 1k out | 86.2 tok/s total (prefill-dominated) |

**Accuracy** (GSM8K, 250 examples, greedy, thinking off):

| Run | Score |
|---|---|
| upstream `main` | 0.968 |
| branch, MTP config | 0.968 |
| branch, MTP config (rerun) | 0.972 |
| branch, DFlash2 + ReplaySSM | 0.964 |
| branch, `--language-model-only` | 0.960 |

All of these are within noise; one question is 0.4%.

---

## 2. How to run

```bash
cd /home/ubuntu/experiments/sglang
benchmark/ampere_qwen38/serve.sh best1          # single user, best decode (DFlash2)
benchmark/ampere_qwen38/serve.sh best4          # 4 users x ~80k context (MTP)
benchmark/ampere_qwen38/serve.sh A              # your original command (reference)
# Add --language-model-only to any preset for text-only serving (frees 0.45 GB/GPU).

benchmark/ampere_qwen38/restart.sh <logname> <preset> [extra args]   # stop, relaunch, wait for ready
benchmark/ampere_qwen38/bench.sh <tag> S1 S5 S6 S2b S2               # results appended to results/<tag>.jsonl
```

Scenarios defined in `bench.sh`:

| Scenario | Workload |
|---|---|
| S1 | 1k / 1k decode, concurrency 1 |
| S1c2, S1c4 | the same at concurrency 2 and 4 |
| S2b, S2 | 8k and 32k prefill |
| S3c2, S3c4 | 16k / 1k |
| S4 | multi-turn shared prefix |
| S5 | ShareGPT chat |
| S6 | code chat |
| S7 | 100k context |

The `best1` launch command, spelled out:

```bash
sglang serve --model-path $MODEL --served-model-name local --tp-size 2 --context-length 262144 \
  --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 4096 --mamba-radix-cache-strategy extra_buffer \
  --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 5 \
  --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1 \
  --speculative-algorithm DFLASH --speculative-num-draft-tokens 8 \
  --speculative-draft-model-path /home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2 \
  --speculative-draft-model-quantization fp8 \
  --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_64k.pt \
  --enable-linear-replayssm-spec \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
```

---

## 3. What was done, change by change

Each item below is one commit on `ampere-opt`. The commit message has the before and after numbers.

### 3.1 Skinny BF16 linears go to `tiny_gemm` on SM8x (`c551c2e65e`)

- **File:** `python/sglang/srt/layers/quantization/unquant.py`. Adds `_use_sm8x_tiny_gemm` and `_sm8x_tiny_gemm`.
- **Problem:** the GDN `in_proj_ba` projection (BF16, N=48 per rank, K=5120) ran through cuBLAS at 28-61 us per call during decode and verify. cuBLAS picks a single-CTA kernel for that shape. Across 48 layers this was about 2 ms of a 29 ms MTP cycle.
- **Fix:** on SM8x, use the existing JIT `tiny_gemm_bf16` when N <= 256, M <= 8, bf16 and contiguous. It takes 3.6 us.
- **Result:** S1 95.3 -> 102.9 tok/s, ShareGPT chat 82.8 -> 87.5.

### 3.2 `--speculative-token-map` works under TP > 1 (`e712a95930`, `7ee66f5af9`)

- **Files:**
  - `python/sglang/srt/speculative/spec_utils.py`: new `shard_hot_token_head`.
  - `python/sglang/srt/speculative/eagle_worker_v2.py`: `init_lm_head`.
- **Problem:** each MTP draft step projected through the full 248k-vocab BF16 lm_head, costing 2.15 ms per step, three times per cycle. A reduced hot vocabulary (FR-spec) is the standard fix. But the existing token-map code indexed the vocab-sharded TP head with global ids, which triggered a device-side assert on every rank.
- **Fix:**
  - Hot rows are gathered once at init, in 4096-row chunks so the transient stays small; it runs after the KV pool is sized.
  - The hot vocabulary is re-sharded evenly across ranks. Hot tokens cluster in low ids, so a per-shard slice would have left one rank with almost the whole head.
  - `hot_token_id` becomes the rank-major order that the logits all-gather produces.
  - The target still verifies against the full vocabulary, so outputs are lossless.
- **Token maps** are built by `benchmark/ampere_qwen38/build_token_map.py` from ShareGPT text with this tokenizer, and stored in `benchmark/ampere_qwen38/token_maps/` (16k, 32k and 64k). Coverage on held-out ShareGPT text:

  | Map size | Coverage |
  |---|---|
  | 16k | 92.7% |
  | 32k | 96.9% |
  | 64k | 99.4% |

- **Result for MTP** (output tok/s on S1 / S5 / S6):

  | Map | S1 | S5 | S6 |
  |---|---|---|---|
  | full vocab | 102.9 | 87.5 | 106.1 |
  | 16k | 115.9 | 100.3 | 127.1 |
  | 32k | 120.5 | 102.8 | 129.8 |
  | 64k | 121.3 | 103.2 | 124.1 |

  32k is the recommended size for MTP.

### 3.3 FP8 draft for the in-checkpoint MTP (`3beac253c1`)

- **File:** `python/sglang/srt/models/qwen3_5_mtp.py`: `_mtp_quant_config`.
- **Problem:** compressed-tensors checkpoints keep `mtp.*` in BF16 through their ignore list. `--speculative-draft-model-quantization fp8` was overridden on CUDA by the checkpoint's detected method, so the flag did nothing.
- **Fix:** when that case is detected, return an online `Fp8Config`. On SM8x the draft then runs as Marlin W8A16.
- **Result:** S1 120.5 -> 125.9, ShareGPT chat 102.8 -> 107.4, code chat 129.8 -> 134.5 tok/s. Acceptance is unchanged.

### 3.4 Eager FP8 KV fallback on SM < 89 (`740498f917`)

- **File:** `python/sglang/srt/mem_cache/memory_pool.py`: `set_kv_buffer_prefix_valid` and the new `_triton_has_fp8e4nv`.
- **Problem:** the fused quantize-and-write Triton kernel uses `fp8e4nv`, which Triton only lowers on SM89 and newer. Any draft whose attention layers carry KV scales crashed on Ampere. DFlash2 with an FP8 draft is one such case.
- **Fix:** SM8x quantizes eagerly, using the same layer scales the fused kernel would use.
- **Result:** the DFlash2 FP8 draft now runs (+1-4%) and its weights drop from 2.18 GB to 1.65 GB per GPU.

### 3.5 DFlash2 selector uses the hot vocabulary (`90b2aed2b6`)

- **Files:**
  - `python/sglang/srt/models/dflash.py`: `hot_candidate_topk` and `DFlash2DraftModel.set_hot_candidate_head`.
  - `python/sglang/srt/speculative/dflash_worker_v2.py`: `_maybe_bind_hot_candidate_head`.
- **Problem:** the DFlash2 candidate selector projected every drafted position through the full target lm_head, a second full-vocabulary GEMM per cycle at 2.18 ms.
- **Fix:** with `--speculative-token-map`, candidate top-k runs over an evenly re-sharded hot head built with the same helper as 3.2.
- **Result with the 64k map:** S1 153.9 -> 160.9, ShareGPT chat 123.4 -> 129.7, code chat 152.8 -> 166.2 tok/s. Acceptance is preserved. A 32k map costs too much acceptance over 7 drafted positions.

### 3.6 Draft embed/lm_head shared before memory sizing (`1acc7459ea`)

- **File:** `python/sglang/srt/speculative/eagle_worker_v2.py`. `init_token_map` and `init_lm_head` move from `alloc_memory_pool` to `__init__`.
- **Problem:** the MTP draft loads its own copies of the 248k embedding and lm_head. They used to be freed only after the scheduler had measured free memory, so about 1.9 GB per GPU was never given to the KV pool.
- **Result:** at the same `--mem-fraction-static 0.90`, the KV pool grows from 277,031 to 422,311 tokens (+52%). Speed is unchanged.

### 3.7 GDN ReplaySSM for DFLASH (`ab15f0d1b3`)

- **Files:**
  - `python/sglang/srt/mem_cache/kv_cache_configurator.py`: the configurator guard.
  - `python/sglang/srt/speculative/dflash_worker_v2.py`: `_uses_gdn_replayssm`, plus routing in `_update_target_mamba_state_after_verify`.
- **Problem:** `--enable-linear-replayssm-spec` was rejected for DFLASH on non-KDA models. DFLASH committed verify states through a backend fold that only KDA implements. Without ReplaySSM, DFlash2 allocates 1.12 GB per GPU for one user and 2.8 GB for four users, as per-draft-token intermediate SSM states.
- **Fix:** GDN pools route the DFLASH commit through the generic `commit_mamba_states_after_verify`, which is the ring advance/fold MTP already uses.
- **Result at 4 users:** KV pool 128k -> 269k tokens, and throughput rises at every concurrency:

  | Concurrency | Output tok/s |
  |---|---|
  | 1 | 161 -> 173 |
  | 2 | 229 -> 263 |
  | 4 | 308 -> 372 |

  GSM8K is 0.964. With MTP, ReplaySSM also works out of the box: at 4 users the KV pool goes from 333k to 423k tokens and throughput from 330 to 367 tok/s.

### 3.8 `--language-model-only` for Qwen3.5 (`5e57753d8d`)

- **Files:**
  - `python/sglang/srt/server_args.py`: `LANGUAGE_MODEL_ONLY_ARCHITECTURES`.
  - `python/sglang/srt/models/qwen3_5.py`: skips `visual.*` weights in both `load_weights` loops.
- **Result:** weights 9.07 -> 8.62 GB per GPU, and the KV pool grows from 422k to 461k tokens. GSM8K is 0.960.
- **Caveat:** output is not bit-identical to the multimodal (mrope) path; 129 of 250 greedy answers match. Text-only therefore stays opt-in.

### 3.9 Dequant + cuBLAS for large-M W4A16 GEMMs (`b632da97ad`)

- **Files:**
  - New `python/sglang/kernels/ops/quantization/marlin_dequant.py`: a Triton kernel that inverts the Marlin weight and scale permutations, bit-exact, at about 400 GB/s.
  - `python/sglang/kernels/ops/quantization/__init__.py`: registration.
  - `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`: dispatch, and a shared scratch buffer sized at load time.
  - `python/sglang/srt/environ.py`: the new `SGLANG_MARLIN_DEQUANT_GEMM_MIN_M`. Unset means 2048 on SM8x and off elsewhere; 0 disables.
  - New `test/registered/kernels/ops/quantization/test_marlin_dequant.py`: 16 cases, exact match against the Marlin reference.
- **Reasoning:** prefill is about 80% Marlin. Under sustained load the A4500 is pinned at its 200 W power cap, with SM clock 1.36-1.43 GHz instead of 2.1. At M=4096 Marlin sustains about 65 TFLOPS and cuBLAS BF16 about 77. Dequantizing one layer at a time costs about 60 ms per forward, while cuBLAS saves about 220 ms per 4096-token chunk.
- **Result:** 8k TTFT 3.80 -> 3.48 s and 32k TTFT 17.67 -> 16.24 s (about 8%). The scratch adds 0.21 GB per GPU.
- **Numerics:** tail prompt logprobs differ from Marlin by 0.13 nats on average. The model's own noise floor, Marlin with prefill chunk 2048 against 4096, is 0.19.

### 3.10 Docs, scripts and results

- `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`: an Ampere (SM80/SM86) entry under Configuration Tips, with the measured recipe. This follows the repo's cookbook-first rule.
- `benchmark/ampere_qwen38/`:

  | File | Purpose |
  |---|---|
  | `serve.sh` | presets A, nospec, B, dflash, M, MD, best1, best4; env knobs `DRAFT_QUANT`, `USERS`, `CHUNK`, `MEM_FRAC`, `PREFILL_GRAPH` |
  | `restart.sh`, `wait_ready.sh` | server lifecycle |
  | `bench.sh` | the scenarios listed in section 2 |
  | `bench_marlin_shapes.py` | Marlin vs cuBLAS on every per-rank TP2 shape |
  | `trace_breakdown.py` | per-kernel profile tables |
  | `turn_latency.py` | agent-turn TTFT |
  | `build_token_map.py`, `build_code_prompts.py` | token maps and the S6 prompt set |
  | `RESULTS.md` | full result tables |

---

## 4. Measurements that shaped the decisions

These are negative or neutral results, recorded so they don't get re-tried blindly.

- **Decode is close to the memory-bandwidth ceiling.**
  - Achievable read bandwidth on an A4500 is about 570-577 GB/s, not the 640 GB/s spec.
  - Marlin reaches 430-550 GB/s on the decode shapes.
  - Per MTP cycle, the target verify GEMMs take about 12.2 ms, which is near optimal. The rest of the cycle was draft cost and lm_head, which sections 3.2-3.5 address.
- **Marlin tile configs:** the default `{64,256,256}` large-batch config is the fastest on SM86. The alternatives are 13-34% slower.
- **Slicing M into smaller Marlin calls:** no gain once the GPU is power-throttled.
- **Larger custom all-reduce for prefill:** forcing the v2 cap to 64 MB gives no gain. 40 MB all-reduces are NVLink-bandwidth-bound either way.
- **MTP steps 4 or 5:** at most +2%. The default stays at 3/1/4.
- **8192-token prefill chunks:** OOM at mem 0.90. The expected gain was only about 1.4%.
- **Prefill CUDA graphs:** they only help very small turns. A 64-token turn drops from 119 to 96 ms; 256 tokens and up are unchanged. Graphs up to 1024 tokens need `--mem-fraction-static 0.86`.
- **The FP8 KV scales are loaded.** The checkpoint values are k about 0.03-0.04 and v about 0.02-0.03. The "Defaulting to scaling factors of 1.0" warning only refers to `--quantization-param-path`. This is inferred from the load path and a log with no missing parameters; the values were not read back at runtime.
- **DFlash2 vs MTP:** DFlash2 is fastest for 1-2 users. At 4 users it has less KV room than MTP: 269k against 411-423k tokens.
- **Multi-turn prefix caching works when requests arrive one after another.**
  - A 16k shared prefix hits 14,336 tokens, and a multi-turn continuation hits 16,256.
  - With 4 concurrent users the hit rate is bounded by mamba state slots. Going from 12 to 24 slots roughly doubles the hits.

---

## 5. Open opportunities (measured, not implemented)

1. **FP8-KV attention at long context.** FlashInfer on SM86 reaches about 42-45 TFLOPS with fp8 KV against 57-62 TFLOPS with bf16 KV, because it dequantizes in the inner loop. At 100k context, attention is about 30% of prefill. An fp8 -> bf16 workspace for prefill would save an estimated 4-10% of TTFT at long context.
2. **Verify attention at 100k+ context.** With 8 queries and 6-way GQA, the 48 rows are padded to a 64-row tile. It runs at 329 GB/s against 496 for q=1, about 5 ms per DFlash cycle, and about 2 ms is recoverable.
3. **Small agent turns.** A 64-token turn takes about 100 ms. About 40 ms of that is GPU idle time (CPU scheduling, one 14.7 ms gap), and about 15 ms is attention over the 30k prefix.
4. **All-reduce + residual + RMSNorm fusion on SM8x.** The existing fusion paths are SM90/100 only. About 1.2 ms per cycle is spent on all-reduce plus add/norm.
5. **Opt-in INT8 verify lm_head.** The verify lm_head is 2.2 ms per cycle; INT8 would save about 5% of decode. Validate with GSM8K.
6. **Mamba slot policy for multi-turn agents,** and an opt-in bf16 SSM state to halve slot cost.
7. **A100 validation.** All code paths are gated on SM8x and keyed to SM count or shared memory where relevant, but only SM86 was measured.

---

## 6. File index (`git diff --stat main..ampere-opt`)

**Runtime code**

| File | Change |
|---|---|
| `python/sglang/srt/layers/quantization/unquant.py` | SM8x tiny_gemm route |
| `python/sglang/srt/speculative/spec_utils.py` | `shard_hot_token_head` |
| `python/sglang/srt/speculative/eagle_worker_v2.py` | TP-aware hot head; embed/head shared before pool sizing |
| `python/sglang/srt/speculative/dflash_worker_v2.py` | hot candidate head; GDN ReplaySSM commit |
| `python/sglang/srt/models/dflash.py` | `hot_candidate_topk` |
| `python/sglang/srt/models/qwen3_5_mtp.py` | FP8 draft for BF16 MTP in compressed-tensors |
| `python/sglang/srt/models/qwen3_5.py` | skip vision weights in LM-only mode |
| `python/sglang/srt/server_args.py` | LM-only allow-list |
| `python/sglang/srt/mem_cache/memory_pool.py` | eager FP8 KV fallback on SM < 89 |
| `python/sglang/srt/mem_cache/kv_cache_configurator.py` | allow DFLASH + GDN ReplaySSM |
| `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | dequant + cuBLAS dispatch |
| `python/sglang/srt/environ.py` | `SGLANG_MARLIN_DEQUANT_GEMM_MIN_M` |
| `python/sglang/kernels/ops/quantization/marlin_dequant.py` (new) | Marlin W4 -> dense dequant kernel |
| `python/sglang/kernels/ops/quantization/__init__.py` | kernel registration |

**Tests:** `test/registered/kernels/ops/quantization/test_marlin_dequant.py` (new).

**Docs:** `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`.

**Benchmarks and results:** `benchmark/ampere_qwen38/*`, including the scripts, the token maps and `RESULTS.md`.
