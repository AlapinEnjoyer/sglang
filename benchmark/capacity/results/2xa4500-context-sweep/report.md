# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-context-sweep | 2x NVIDIA RTX A4500 NVLink | 53 GB | - MB/s | ampere-opt@9c41ea18de | 2026-11-10T11:10:14 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| s64-path2-hicache | 2x NVIDIA RTX A4500 NVLink | ok | 136 s | 286604 | 8 | 64 / 113 | 436704 | - | - |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|
| s64-path2-hicache | 32768 | 1 | 142.8 | 142.8 | 16.7 | 1967 | 3.48 |
| s64-path2-hicache | 32768 | 2 | 81.2 | 162.3 | 36.0 | 1942 | 2.25 |
| s64-path2-hicache | 32768 | 4 | 76.5 | 305.9 | 76.2 | 1882 | 2.80 |
| s64-path2-hicache | 32768 | 8 | 61.8 | 494.7 | 160.0 | 1882 | 4.01 |
| s64-path2-hicache | 65536 | 1 | 108.7 | 108.7 | 40.1 | 1636 | 2.96 |
| s64-path2-hicache | 65536 | 2 | 100.3 | 200.7 | 85.0 | 1638 | 3.23 |
| s64-path2-hicache | 65536 | 4 | 63.7 | 254.6 | 178.3 | 1637 | 2.79 |
| s64-path2-hicache | 98304 | 1 | 87.2 | 87.2 | 67.9 | 1448 | 2.59 |
| s64-path2-hicache | 98304 | 2 | 92.9 | 185.8 | 144.7 | 1448 | 3.39 |
| s64-path2-hicache | 120314 | 2 | 69.6 | 139.3 | 191.2 | 1338 | 3.05 |
| s64-path2-hicache | 122362 | 1 | 75.0 | 75.0 | 91.8 | 1333 | 2.37 |
| s64-path2-hicache | 64k x 8: L=65,536 fits 4 users (--include-unfit to run) | | | | | | |
| s64-path2-hicache | 96k x 4: L=98,304 fits 2 users (--include-unfit to run) | | | | | | |
| s64-path2-hicache | 96k x 8: L=98,304 fits 2 users (--include-unfit to run) | | | | | | |
| s64-path2-hicache | full x 4: L=120,314 fits 2 users (--include-unfit to run) | | | | | | |
| s64-path2-hicache | full x 8: L=120,314 fits 2 users (--include-unfit to run) | | | | | | |

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|

## Session restore from the storage tier

Each restore is checked against the same tokens recomputed from scratch (a fresh cache_salt): a wrong or mismatched state changes the top token or moves logprobs by far more than numerical noise (about 0.1).

| Config | Session | Recomputed (reference) | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|

## Chat clients: reuse through /v1/chat/completions

The template re-renders history; a client that drops reasoning_content changes the previous turn, so only the part before it can match. Cached needs --enable-cache-report.

| Config | Client | Turn | Prompt | Cached | Reusable (prev prompt + output) |
|---|---|---|---|---|---|

## Server commands

- **s64-path2-hicache** (2xa4500-context-sweep): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --max-mamba-cache-size 64 --mamba-max-states-per-path 2 --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --port 30000`
