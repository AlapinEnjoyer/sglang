# Handoff: SGLang on Ampere for Qwen3.8-27B (Swift-1.5 INT4 and Qwen FP8)

- **Branch:** `ampere-opt`, rebased on 2026-11-07 onto upstream `main` at `ba5906471c` (originally forked at `14e3414513`). It has 25 commits and has not been pushed. The pre-sync branch is kept as `ampere-opt-pre-sync-20261107`.
  - Remote `upstream` is `sgl-project/sglang`. To sync again: `git fetch upstream`, fast-forward `main` in the `../sglang-upstream` worktree, `git rebase main`, then reinstall (`uv pip install --prerelease=allow -e "python[dev]"`) and copy the rebuilt `*.so` files into the upstream worktree.
- **Environment:** `/home/ubuntu/experiments/sglang/.venv`, built with uv: `uv pip install --prerelease=allow -e "python[dev]"`. It contains torch 2.14.1+cu130, flashinfer 0.7.0.post1 and sglang-kernel 0.4.9. RULER2 also needs `inflect nltk wonderwords`, which are installed.
  - The login shell exports `VIRTUAL_ENV` pointing to the old sglang-inf env. Always `export VIRTUAL_ENV=$PWD/.venv` and use `uv run --active`; `serve.sh` and `bench.sh` already do.
- **Hardware:** 2x RTX A4500 (20 GB, SM86, 46 SMs), joined by an NVLink bridge (NV4).
- **Model:** `/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4`.
  - Qwen3.5 hybrid architecture: 48 GDN layers and 16 gated full-attention layers.
  - W4A16 compressed-tensors weights, group 128, symmetric.
  - BF16 MTP head, FP8 KV-cache scales.
- **FP8 model:** `Qwen/Qwen3.8-27B-FP8` (HF cache). It is block-FP8 (128x128), its MTP head is FP8, and it ships no KV scales. `orcarouter/Qwen3.8-27B-Uncensored-FP8` (abliterated, HF-gated) has the same layout, so everything below applies to it too.
- **DFlash2 draft:** `/home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2`.

All numbers below were measured on this machine with the scripts in `benchmark/ampere_qwen38/`.
Results from the old `../inference/sglang-inf` folder came from an older sglang and were ignored.

- **Benchmark studio:** `benchmark/capacity/studio.py` (`plan` / `run` / `report`) is the one
  tool for measuring a setup on any machine. See section 7 and `benchmark/capacity/README.md`.

---

## 1. Headline results

**Single user** (output tok/s; accept length in parentheses; TTFT for the prefill rows):

| Scenario | Your original command (MTP 3/1/4, BF16 draft) | `best1` (MTP + FP8 draft + 32k hot vocab + ReplaySSM) | `dflash1` (DFlash2 + FP8 draft + 64k hot vocab + ReplaySSM) |
|---|---|---|---|
| S1: 1k in / 1k out | 95.3 (2.99) | **123.5** (2.76), +30% | 173.0 (4.29), +82% |
| S5: ShareGPT chat, 512 out | 82.8 (2.60) | **110.9** (2.48), +34% | 138.0 (3.39), +67% |
| S6: HumanEval/MBPP chat, 512 out | 106.1 (3.09)¹ | **137.5** (3.02), +30% | 157.5 (3.76), +48% |
| KV pool (tokens) | - | **432k** | 165k with default mamba slots |
| S2b: 8k prefill TTFT | 3.88 s | 3.61 s, -7% (both) | |
| S2: 32k prefill TTFT | 17.78 s | 16.29 s, -8% (both) | |

`best1` is the single-user recommendation: its KV pool covers the full 262k context with room to spare. `dflash1` decodes faster but leaves far less KV. Its draft weights take 1.6 GB per GPU, and its draft KV is sized for the full context although the draft only attends to the last 2048 tokens.

¹ This S6 baseline already includes the first fix (tiny_gemm). S6 did not exist yet when the pure-upstream baseline was measured.

**Multi-user** (`best4`: MTP + FP8 draft + 32k map + ReplaySSM, 4 slots):

| Metric | Value |
|---|---|
| KV pool | 411k tokens, enough for 4 users x 80k |
| 4 concurrent users, 1k / 1k | 360.6 tok/s total, TPOT 9.2 ms |
| 4 concurrent users, 16k prompts / 1k out | 86.2 tok/s total (prefill-dominated) |

**FP8 checkpoint** (`fp8x1`: same recipe on `Qwen/Qwen3.8-27B-FP8`, native FP8 MTP, `LM_ONLY=1`; section 3.10):

| Metric | FP8, upstream code paths | `fp8x1` |
|---|---|---|
| KV pool (tokens) | 61k | 114k (125k without the prefill scratch) |
| S1 / S5 / S6 output tok/s | - | 76.9 / 73.4 / 87.6 (accept 2.61 / 2.48 / 2.91) |
| 8k / 32k prefill TTFT | 4.67 s / 21.0 s | 3.58 s / 16.6 s |

FP8 decode is 0.62-0.66x of INT4 because the weights are twice the bytes. On A100, the ~3x higher HBM bandwidth more than covers that.

**Output quality, sampled and multi-turn:** see section 4b. No config differs from non-speculative upstream decoding. The long-context and full-thinking checks are in section 4c.

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
benchmark/ampere_qwen38/serve.sh best1          # single user (MTP), 432k-token KV pool
benchmark/ampere_qwen38/serve.sh dflash1        # single user, fastest decode (DFlash2), small KV pool
benchmark/ampere_qwen38/serve.sh best4          # 4 users x ~80k context (MTP)
benchmark/ampere_qwen38/serve.sh A              # your original command (reference)
LM_ONLY=1 benchmark/ampere_qwen38/serve.sh fp8x1   # best1 on the FP8 checkpoint (FP8_MODEL=... to override)
# LM_ONLY=1 adds --language-model-only (text only, frees 0.45 GB/GPU); REPLAY=0 drops ReplaySSM.
# serve.sh sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True for every preset (section 3.12).
uv run --active python benchmark/ampere_qwen38/quality_ab.py --help   # output-quality A/B (sections 4b, 4c)

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
  --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 8 \
  --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1 \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --speculative-draft-model-quantization fp8 \
  --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt \
  --enable-linear-replayssm-spec \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
```

Always set `--max-mamba-cache-size`. Left unset with `--max-running-requests 1`, sglang sized 39 mamba state slots, which took 2.8 GB per GPU out of the KV pool.

For the FP8 checkpoint, use the same command with `--model-path Qwen/Qwen3.8-27B-FP8`, without `--speculative-draft-model-quantization` (its MTP is already FP8), plus `--language-model-only` for text-only, and launch it with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

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

### 3.10 Dequant + cuBLAS for large-M FP8 Marlin GEMMs (`be93d84363`)

- **Files:**
  - `python/sglang/kernels/ops/quantization/marlin_dequant.py`: new `marlin_fp8_dequant`. It inverts the 8-bit Marlin layout and decodes e4m3 with integer ops, because Triton has no fp8e4nv casts below SM89. Marlin folds a 2^120 exponent bias into its scales; the kernel undoes it.
  - `python/sglang/srt/layers/quantization/fp8.py`: `Fp8LinearMethod._apply_marlin_dense` and the scratch sized at load.
  - `python/sglang/srt/layers/quantization/marlin_utils.py`: `marlin_dense_gemm_min_m`, `use_marlin_dense_gemm`, `marlin_dense_scratch`. These moved here from the W4A16 scheme so both paths share them.
  - `test_marlin_dequant.py`: 8 new FP8 cases (block and channel-wise, every e4m3 code including subnormals), all bit-exact.
  - `bench_marlin_shapes.py --wtype fp8`.
- **Problem:** block-FP8 checkpoints run as Marlin W8A16 on SM8x. At M >= 2048 that reaches 51-63 TFLOPS, against 72-79 for cuBLAS BF16.
- **Result:** with the same `SGLANG_MARLIN_DEQUANT_GEMM_MIN_M` (2048 on SM8x), 8k TTFT 4.67 -> 3.58 s and 32k TTFT 21.0 -> 16.6 s. The scratch costs 0.18 GB per GPU at TP2.
- **Numerics:** prompt logprobs move by 0.137 nats on average against Marlin. The chunk-size noise floor is 0.116.

### 3.11 Checkpoint KV scales ignored for a 16-bit KV cache (`c34b472525`)

- **Files:** `python/sglang/srt/mem_cache/kv_cache_dtype.py` (`_reset_kv_scales_for_unquantized_cache`), plus a regression test in `test/registered/unit/layers/quantization/test_compressed_tensors_kv_cache.py`.
- **Bug (also in upstream `main`):** the Swift INT4 checkpoint ships fp8 calibration scales (k about 0.03), and sglang loads them whatever `--kv-cache-dtype` is.
  - The write path divides by the scales only when it casts to fp8.
  - FlashInfer and the other backends always multiply them back in on read.
  - With `--kv-cache-dtype bf16`, the attention logits were therefore scaled by about 0.03. Attention flattened, and the model wrote fluent text that ignored the prompt. It was found because the 64k/128k BF16-KV quality run returned unrelated answers.
- **Fix:** once the cache dtype is resolved and it is 16-bit, the scales are reset to 1.0. The log says "Unquantized KV cache: ignoring checkpoint k/v scales on 16 attention layers".
  - `auto` resolves to fp8 on this checkpoint and keeps the scales.
  - The recommended presets (`fp8_e4m3`) were never affected.
  - The FP8 checkpoints ship no scales, so they were never affected either.
- **Test:** the new case fails on the old code and passes with the fix.
- **Upstream worktree:** the same patch is applied, uncommitted, in `/home/ubuntu/experiments/sglang-upstream`, so that it can serve as the BF16-KV reference.

### 3.12 `expandable_segments` in `serve.sh`

- **Problem:** with the FP8 checkpoint, the KV pool was only 61k tokens. The draft's embed and lm_head copies (2.5 GB per GPU) are freed when they are shared with the target (3.6), but their blocks sit in allocator segments that the FP8-repacked MTP weights still occupy. That left 1.4 GB reserved and not allocated.
- **Fix:** `serve.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, overridable.
- **Result:** FP8 pool 61k -> 125k tokens. INT4 `best1` 432k -> 437k tokens, with S1 unchanged (123.6 tok/s).
- **Possible follow-up:** load the draft without its own embed and lm_head instead.

### 3.13 Quality tooling: full thinking and long context

- **`quality_ab.py`:**
  - It now passes `enable_thinking=True` and `reasoning_effort` explicitly (default xhigh), and sends min_p, presence and repetition at 0 / 0 / 1.
  - Default `--max-tokens` is 16384, and the report has a stop% column.
  - `--set long --ctx 65536,131072` cuts real source files into long documents and asks about a function defined in the first 10%, then two follow-up turns at depth.
  - `--src-dir` pins the corpus (for example a `git archive`), so every config sees byte-identical prompts.
  - `report --by-kind` breaks results down per context bucket.
- **sgl-eval:** RULER2 and MMLU-Pro run with `--reasoning-effort xhigh --temperature 1.0 --top-p 0.95`; top_k 20 comes from `generation_config.json`. GPQA is gated on HF.

### 3.14 Docs, scripts and results

- `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`: an Ampere (SM80/SM86) entry under Configuration Tips, with the measured recipe. This follows the repo's cookbook-first rule.
- `benchmark/ampere_qwen38/`:

  | File | Purpose |
  |---|---|
  | `serve.sh` | presets A, nospec, B, dflash, M, MD, best1, dflash1, best4, fp8x1; env knobs `MODEL`, `FP8_MODEL`, `DRAFT_QUANT`, `USERS`, `CHUNK`, `MEM_FRAC`, `PREFILL_GRAPH`, `LM_ONLY`, `REPLAY` |
  | `quality_ab.py` | output-quality A/B: generate on a config, teacher-force score on a reference server (NLL, greedy top-1, divergence, stop%); short and long (64k/128k) sets |
  | `restart.sh`, `wait_ready.sh` | server lifecycle |
  | `bench.sh` | the scenarios listed in section 2 |
  | `bench_marlin_shapes.py` | Marlin (W4A16 or FP8, `--wtype`) vs cuBLAS on every per-rank TP shape |
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
- **The FP8 KV scales are loaded and applied.** The checkpoint values are k about 0.03-0.04 and v about 0.02-0.03. The broken BF16-KV run (3.11) shows they reach the attention kernels. The "Defaulting to scaling factors of 1.0" warning only refers to `--quantization-param-path`. For the FP8 checkpoints the same warning is real: they ship no scales, so fp8 KV runs at scale 1.0.
- **DFlash2 vs MTP:** DFlash2 is fastest for 1-2 users. At 4 users it has less KV room than MTP: 269k against 411-423k tokens.
- **Multi-turn prefix caching works when requests arrive one after another.**
  - A 16k shared prefix hits 14,336 tokens, and a multi-turn continuation hits 16,256.
  - With 4 concurrent users the hit rate is bounded by mamba state slots. Going from 12 to 24 slots roughly doubles the hits.

### 4b. Output quality A/B

This was prompted by a report that `dflash1` felt worse with sampling, thinking and multi-turn chat.
- **Setup:** `benchmark/ampere_qwen38/quality_ab.py` ran 32 single-turn prompts and 6 three-turn conversations, with thinking on. Turns 2 and 3 hit the prefix cache.
- **Scoring:** outputs were teacher-forced on upstream `main` without speculation. NLL is per output token; top-1 is the share of greedy tokens equal to the reference argmax.

| Config | Sampled NLL/token [95% CI] | Greedy top-1 |
|---|---|---|
| upstream, no spec (reference) | 0.504 [0.452, 0.555] | 99.00% (self) |
| upstream MTP / upstream DFlash2 | 0.479 / 0.476 | 99.09% / 99.01% |
| `best1` / same with radix cache off | 0.471 / - | 98.99% / 98.96% |
| `dflash1` without / with ReplaySSM | 0.496 / 0.482 | 98.96% / 99.04% |
| `dflash1` with ReplaySSM, radix cache off | - | 98.94% |

- **Nothing is lossy, beyond noise.** That includes the DFlash + ReplaySSM routing added on this branch, and the outputs after prefix-cache hits.
- **Greedy decoding is not run-to-run reproducible here.** The same config with the cache on and off first diverges at a median of about 125 tokens on identical uncached prompts. Differences between single sampled chats are expected and do not indicate a quality loss.
- **Caveat:** this run used xhigh thinking, the template default, at the recommended sampling params, but with `max_tokens` 1024. Outputs were cut off inside the thinking, so only thinking prefixes were scored. Prompts were at most about 10k tokens. Section 4c closes both gaps.

### 4c. Long-context and full-thinking quality

**Setup:**
- `quality_ab.py --set long`: 4 documents at 64k and 4 at 128k tokens, 3 turns each. Turns 2 and 3 hit the cache at 64k-135k depth.
- The short set is the 4b prompts with up to 16k output tokens.
- Everything is sampled at xhigh, temperature 1.0, top_p 0.95, top_k 20.
- Scoring: teacher-forced on upstream `main` + the 3.11 fix, no spec, BF16 KV.
- Driver and outputs: `$SCRATCH/quality5_driver.sh`, `$SCRATCH/quality2/`.

NLL/token of the sampled outputs under the reference model [95% bootstrap CI], top-1 agreement, and the share of turns that stopped before the token cap:

| Config | Long 64k | Long 128k | Long overall | top-1 | stop% | Short (16k) |
|---|---|---|---|---|---|---|
| ref: upstream, no spec, BF16 KV (exact sampler) | 0.283 | 0.285 | 0.284 [0.261, 0.310] | 89.1% | 100 | 0.326 [0.284, 0.367], stop 84% |
| `best1`, FP8 KV | 0.297 | 0.298 | 0.297 [0.272, 0.324] | 88.8% | 96 | 0.332 [0.291, 0.377], stop 86% |
| `best1`, BF16 KV | 0.268 | 0.322 | 0.289 [0.263, 0.319] | 88.9% | 92 | - |
| upstream MTP, FP8 KV | 0.285 | 0.296 | 0.290 [0.274, 0.311] | 88.9% | 96 | - |

- **Reading:** every config sits inside the reference CI, at 64k and 128k alike. The branch with FP8 KV is 0.013 nats/token above ref and upstream MTP with FP8 KV is 0.006 above, both well inside noise. There is no trend with context length.
- **Resolution:** 24 turns per config resolve differences of about 0.04 nats/token; smaller losses would need more documents.

**FP8 checkpoint** (`Qwen/Qwen3.8-27B-FP8`, `fp8x1` with MTP):
- **Setup:** a 32k long set (6 docs x 3 turns, turns 2-3 at 32k-50k cached depth) and the short set, both at 16k output tokens.
- **Scoring:** on an FP8 no-spec BF16-KV server.
- **Comparison:** FP8 KV with no checkpoint scales (scale 1.0) against BF16 KV. Speculation is the same in both, so the difference isolates the KV cache.

| FP8 checkpoint | Long 32k NLL/tok [95% CI] | top-1 | stop% | Short (16k) NLL/tok [95% CI] | top-1 | stop% | RULER2 32k |
|---|---|---|---|---|---|---|---|
| FP8 KV, scale 1.0 | 0.448 [0.429, 0.466] | 83.6% | 89 | 0.479 [0.442, 0.518] | 82.9% | 80 | 87.0% |
| BF16 KV | 0.453 [0.431, 0.478] | 83.4% | 83 | 0.480 [0.445, 0.517] | 82.8% | 76 | 85.6% |

- **RULER2:** mk_niah_hard / mv_niah_hard / qa_medium / qa_hard, 25 each, at xhigh. Per task, FP8 KV scored 92/100/80/76 and BF16 KV 88/92/84/78.4. All 100 answers stopped before the cap.
- **Verdict:** FP8 KV at scale 1.0 is not measurably worse than BF16 KV on the FP8 checkpoint. On RULER the standard error is about 3.5 points. **No KV-scale calibration is needed.**
- **NLL is not comparable with the INT4 table.** It is scored by a different model on different outputs.
- **Scorer memory:** scoring needs `--chunked-prefill-size 256` next to FP8 weights on 20 GB cards. At 1024, the full-vocab logits (about 1 GB per chunk) OOM.

A BF16 pool next to FP8 weights holds only about 57k tokens on 2x A4500, so FP8 at 128k-250k has to be validated on the A100.

- **Qualitative so far:** `best1` with FP8 KV answers 128k-token questions with exact code quotes from the first 10% of the context, and follow-up turns reuse 129k-134k cached tokens.

---

## 5. Open opportunities (measured, not implemented)

1. **FP8-KV attention at long context.** FlashInfer on SM86 reaches about 42-45 TFLOPS with fp8 KV against 57-62 TFLOPS with bf16 KV, because it dequantizes in the inner loop. At 100k context, attention is about 30% of prefill. An fp8 -> bf16 workspace for prefill would save an estimated 4-10% of TTFT at long context.
2. **Verify attention at 100k+ context.** With 8 queries and 6-way GQA, the 48 rows are padded to a 64-row tile. It runs at 329 GB/s against 496 for q=1, about 5 ms per DFlash cycle, and about 2 ms is recoverable.
3. **Small agent turns.** A 64-token turn takes about 100 ms. About 40 ms of that is GPU idle time (CPU scheduling, one 14.7 ms gap), and about 15 ms is attention over the 30k prefix.
4. **All-reduce + residual + RMSNorm fusion on SM8x.** The existing fusion paths are SM90/100 only. About 1.2 ms per cycle is spent on all-reduce plus add/norm.
5. **Opt-in INT8 verify lm_head.** The verify lm_head is 2.2 ms per cycle; INT8 would save about 5% of decode. Validate with GSM8K.
6. **Mamba slot policy for multi-turn agents,** and an opt-in bf16 SSM state to halve slot cost.
7. **A100 validation.** All code paths are gated on SM8x and keyed to SM count or shared memory where relevant, but only SM86 was measured.
   - Capacity estimates for the FP8 checkpoint at mem 0.92, 8 users x 3 mamba slots:

     | Setup | KV pool, fp8 | KV pool, bf16 |
     |---|---|---|
     | 2x A100 80GB, TP2 (about 14.5 GB weights per GPU) | about 3.5M tokens | about 1.75M tokens |
     | 1x A100 80GB, TP1 (about 28.7 GB weights) | about 1.25M tokens | about 0.6M tokens |

   - So 8 x 250k fits at TP2 with fp8 KV only. bf16 KV would hold about 7 x 250k, but its attention is faster on SM8x.
   - To run on the A100s: the FP8 long-context quality check (4c) at 128k-250k, the dequant threshold (`bench_marlin_shapes.py --wtype fp8`), and RULER2 at 128k.
8. **KV scales for the FP8 checkpoints (not needed so far).** 4c shows that scale-1.0 fp8 KV matches BF16 KV at 32k. If an A100 run at 128k-250k shows a loss, add `load_kv_cache_scales` to `qwen3_5.py` (as in `qwen3.py`) plus a small calibration script for `--quantization-param-path`.

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
| `python/sglang/srt/environ.py` | `SGLANG_MARLIN_DEQUANT_GEMM_MIN_M` (W4A16 and FP8) |
| `python/sglang/kernels/ops/quantization/marlin_dequant.py` (new) | Marlin W4 and FP8 -> dense dequant kernels |
| `python/sglang/srt/layers/quantization/fp8.py` | FP8 Marlin dense prefill dispatch |
| `python/sglang/srt/layers/quantization/marlin_utils.py` | shared dense-GEMM helpers |
| `python/sglang/srt/mem_cache/kv_cache_dtype.py` | ignore checkpoint KV scales for a 16-bit cache |
| `python/sglang/kernels/ops/quantization/__init__.py` | kernel registration |

**Tests:** `test/registered/kernels/ops/quantization/test_marlin_dequant.py` (new; W4 and FP8), `test/registered/unit/layers/quantization/test_compressed_tensors_kv_cache.py` (KV-scale regression).

**Docs:** `docs/cookbook/autoregressive/Qwen/Qwen3.8-27B.mdx`.

**Benchmarks and results:** `benchmark/ampere_qwen38/*`, including the scripts, the token maps and `RESULTS.md`.

---

## 7. Eight agent users, HiCache, the storage tier and the benchmark studio (2026-11-08)

Full tables: `benchmark/ampere_qwen38/RESULTS.md`, "Eight agent users with HiCache".
Reproduce with `benchmark/capacity/configs/studio-swift15-2xa4500.toml`.

### Findings
- **8 users at 120k context, TP2.**
  - The fork with an FP8 MTP draft and the 32k map decodes 107 / 308 / 462 tok/s total at
    1 / 4 / 8 users of 32k. Vanilla (0.5.23.dev) gets 83 / 272 and cannot fit 8 users at 32k:
    its 264k pool holds 7.
  - GSM8K is 97.6-98.0% everywhere.
- **Keep `--mem-fraction-static` at 0.87 or lower on the fork.**
  - Section 3.6 (the shared draft embed and lm_head) hands vanilla's hidden headroom to the
    KV pool. At 0.90, a prefill chunk next to 7 decoding users ran out of memory in
    `_draft_extend_for_prefill`.
  - An upstream PR for 3.6 must reserve that activation memory explicitly.
- **GDN state slots decide multi-turn hit rates.** A returning session needs its saved state,
  not just its KV.
  - With 20 rotating sessions: 32 slots give 0.42 hits, 48 give 0.64, and 64 give 0.68,
    against a 0.738 ceiling.
  - `--hicache-size` splits host RAM between KV and state in proportion to the GPU pools.
- **The HiCache `file` storage tier works for this hybrid model, GDN state included.**
  - A 30k session comes back after a flush in 0.78 s, and after a server restart in 9 s on
    this VM's 131 MB/s disk. Cold prefill is about 16 s.
  - It needs `libssl-dev`, `--page-size 64` (page 1 writes a file per token) and
    `--hicache-storage-prefetch-policy wait_complete` (the default drops slow restores).
- **The decode gain is mostly the token map.** About three quarters of it is the map plus the
  FP8 draft. The map is lossless, but it was built from English ShareGPT. Build one from your
  own traffic (`build_token_map.py`) when the text differs, and compare acceptance.

### Upstreaming status
- **pre-commit:** clean after the formatting commit on top of `085978f852`.
- **Rebase:** none of the 50 newer upstream commits touch the branch's files.
- **To fix first:**
  - `test_dflash_logits.py` (5 tests): the stubs lack `hot_candidate_head`.
  - `test_fp8_lm_head_mxfp8.py`: the FP8 dense-prefill path reads `input_size_per_partition`,
    which an FP8 `ParallelLMHead` lacks. That is a real bug.
- **PRs in order of value.** One change each, with accuracy and speed results per the PR
  template:
  1. Draft embed/lm_head sharing (3.6), with explicit activation headroom.
  2. The bf16-KV checkpoint-scale fix (3.11).
  3. Token map under TP > 1 (3.2).
  4. FP8 draft for compressed-tensors MTP (3.3).
  5. Eager FP8 KV fallback on SM<89 (3.4).
  6. `--language-model-only` for Qwen3.5 (3.8).
  7. `tiny_gemm` on SM8x (3.1).
  8. Dense prefill on SM8x (3.9, 3.10; follow `kernel-organization`).
  9. DFlash hot vocab and GDN ReplaySSM (3.5, 3.7).

### Benchmark studio
- **`studio.py plan`:** users at a context and the serve flags, for this machine or
  `--gpu NAME:MEM_GB:COUNT:SM`. Calibrated here to 1.5% on the KV pool.
- **Estimates at 128k per user:**

  | GPU | Users at once |
  |---|---|
  | A100 or H100 80 GB, TP1 | 10 |
  | 2x H100, TP2 | 26 |
  | L40S | 4 |

  Verify each with `plan --write-suite` and `run`.
- **`studio.py run`:** launch, cache check, sgl-eval GSM8K, burst, multi-turn (optionally with
  `/v1/decisions` load and session ids or priorities) and storage-tier restore. Results go to
  `benchmark/capacity/results/<run>/`.
- **`studio.py report`:** compares runs from several machines.
