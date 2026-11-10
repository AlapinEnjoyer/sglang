# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-20261110-rebase | 2x NVIDIA RTX A4500 NVLink | 53 GB | 122 MB/s | ampere-opt@2cfc570e15 | 2026-11-10T08:36:18 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| fork-best-ram | 2x NVIDIA RTX A4500 NVLink | ok | 366 s | 286604 | 8 | 64 / 113 | 436704 | - | ok |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|
| fork-best-ram | 32768 | 1 | 108.6 | 108.6 | 16.8 | 1956 | 2.65 |
| fork-best-ram | 32768 | 4 | 84.0 | 336.1 | 76.1 | 1899 | 2.90 |
| fork-best-ram | 32768 | 8 | 54.9 | 439.4 | 159.8 | 1878 | 2.86 |

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|
| fork-best-ram | 10 | 0.730 (0.738) | 4.5 / 11.4 | 33 | 0.32 (0.5) | - |
| fork-best-ram | 20 | 0.670 (0.738) | 5.7 / 24.4 | 37 | 0.28 (0.5) | - |

## Session restore from the storage tier

Each restore is checked against the same tokens recomputed from scratch (a fresh cache_salt): a wrong or mismatched state changes the top token or moves logprobs by far more than numerical noise (about 0.1).

| Config | Session | Recomputed (reference) | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|

## Chat clients: reuse through /v1/chat/completions

The template re-renders history; a client that drops reasoning_content changes the previous turn, so only the part before it can match. Cached needs --enable-cache-report.

| Config | Client | Turn | Prompt | Cached | Reusable (prev prompt + output) |
|---|---|---|---|---|---|

## Server commands

- **fork-best-ram** (2xa4500-20261110-rebase): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
