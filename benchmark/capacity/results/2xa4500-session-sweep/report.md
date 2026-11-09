# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-session-sweep | 2x NVIDIA RTX A4500 NVLink | 53 GB | - MB/s | ampere-opt@906ab743e9 | 2026-11-08T23:02:52 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| s64-path2-gpu-only | 2x NVIDIA RTX A4500 NVLink | ok | 121 s | 286604 | 8 | 64 / None | None | - | - |
| s64-path2-hicache | 2x NVIDIA RTX A4500 NVLink | ok | 126 s | 286604 | 8 | 64 / 113 | 436704 | - | - |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|
| s64-path2-gpu-only | 10 | 0.730 (0.738) | 3.6 / 7.6 | 34 | 0.32 (0.5) | - |
| s64-path2-gpu-only | 15 | 0.713 (0.738) | 3.6 / 12.5 | 37 | 0.30 (0.5) | - |
| s64-path2-gpu-only | 20 | 0.606 (0.738) | 7.4 / 35.0 | 38 | 0.25 (0.5) | - |
| s64-path2-gpu-only | 25 | 0.467 (0.738) | 12.2 / 43.3 | 48 | 0.20 (0.5) | - |
| s64-path2-gpu-only | 30 | 0.419 (0.738) | 14.3 / 57.6 | 51 | 0.18 (0.5) | - |
| s64-path2-gpu-only | 40 | 0.334 (0.738) | 18.0 / 62.0 | 65 | 0.16 (0.5) | - |
| s64-path2-hicache | 10 | 0.730 (0.738) | 3.9 / 9.6 | 31 | 0.33 (0.5) | - |
| s64-path2-hicache | 15 | 0.730 (0.738) | 3.4 / 7.4 | 37 | 0.31 (0.5) | - |
| s64-path2-hicache | 20 | 0.716 (0.738) | 5.2 / 13.1 | 32 | 0.31 (0.5) | - |
| s64-path2-hicache | 25 | 0.674 (0.738) | 6.0 / 18.2 | 35 | 0.28 (0.5) | - |
| s64-path2-hicache | 30 | 0.626 (0.738) | 7.0 / 27.5 | 36 | 0.25 (0.5) | - |
| s64-path2-hicache | 40 | 0.529 (0.738) | 10.1 / 39.6 | 40 | 0.21 (0.5) | - |

## Session restore from the storage tier

Each restore is checked against the same tokens recomputed from scratch (a fresh cache_salt): a wrong or mismatched state changes the top token or moves logprobs by far more than numerical noise (about 0.1).

| Config | Session | Recomputed (reference) | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|

## Chat clients: reuse through /v1/chat/completions

The template re-renders history; a client that drops reasoning_content changes the previous turn, so only the part before it can match. Cached needs --enable-cache-report.

| Config | Client | Turn | Prompt | Cached | Reusable (prev prompt + output) |
|---|---|---|---|---|---|

## Server commands

- **s64-path2-gpu-only** (2xa4500-session-sweep): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --max-mamba-cache-size 64 --mamba-max-states-per-path 2 --port 30000`
- **s64-path2-hicache** (2xa4500-session-sweep): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --max-mamba-cache-size 64 --mamba-max-states-per-path 2 --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --port 30000`
