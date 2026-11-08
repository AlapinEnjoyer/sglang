# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-gdn-states | 2x NVIDIA RTX A4500 NVLink | 53 GB | 125 MB/s | ampere-opt@60c5e60c6b | 2026-11-08T21:51:56 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| s32-int8 | 2x NVIDIA RTX A4500 NVLink | launch failed: server exited with code 1: ValueError: --enab | - s | - | - | - / - | - | - | - |
| s32-path2 | 2x NVIDIA RTX A4500 NVLink | ok | 111 s | 363212 | 8 | 32 / 61 | 553432 | - | ok |
| s64-disk-checks | 2x NVIDIA RTX A4500 NVLink | ok | 121 s | 286592 | 8 | 64 / 113 | 436736 | - | ok |
| s64-path2 | 2x NVIDIA RTX A4500 NVLink | ok | 126 s | 286604 | 8 | 64 / 113 | 436704 | - | ok |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|
| s32-path2 | 20 | 0.535 (0.738) | 9.3 / 28.5 | 45 | 0.22 (0.5) | - |
| s64-path2 | 20 | 0.710 (0.738) | 5.1 / 16.6 | 32 | 0.29 (0.5) | - |

## Session restore from the storage tier

Each restore is checked against the same tokens recomputed from scratch (a fresh cache_salt): a wrong or mismatched state changes the top token or moves logprobs by far more than numerical noise (about 0.1).

| Config | Session | Recomputed (reference) | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|
| s64-disk-checks | 30000 | 16.12 s, 0 cached, same top-1, max dlogp 0.000 | 0.27 s, 29952 cached, same top-1, max dlogp 0.052 | 13.76 s, 30272 cached, same top-1, max dlogp 0.031 | 9.02 s, 30272 cached, same top-1, max dlogp 0.031 |

## Chat clients: reuse through /v1/chat/completions

The template re-renders history; a client that drops reasoning_content changes the previous turn, so only the part before it can match. Cached needs --enable-cache-report.

| Config | Client | Turn | Prompt | Cached | Reusable (prev prompt + output) |
|---|---|---|---|---|---|
| s64-disk-checks | keep_reasoning=True | 1 | 677 | 0 | 0 |
| s64-disk-checks | keep_reasoning=True | 2 | 2406 | 1536 | 1771 |
| s64-disk-checks | keep_reasoning=True | 3 | 3259 | 2560 | 2621 |
| s64-disk-checks | keep_reasoning=True | 4 | 4080 | 3328 | 3444 |
| s64-disk-checks | keep_reasoning=False | 1 | 679 | 0 | 0 |
| s64-disk-checks | keep_reasoning=False | 2 | 1357 | 640 | 1519 |
| s64-disk-checks | keep_reasoning=False | 3 | 2044 | 1344 | 1794 |
| s64-disk-checks | keep_reasoning=False | 4 | 2729 | 1984 | 2375 |

## Server commands

- **s32-int8** (2xa4500-gdn-states): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --max-mamba-cache-size 32 --enable-int8-mamba-checkpoint --port 30000`
- **s32-path2** (2xa4500-gdn-states): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --max-mamba-cache-size 32 --mamba-max-states-per-path 2 --port 30000`
- **s64-disk-checks** (2xa4500-gdn-states): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --max-mamba-cache-size 64 --page-size 64 --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete --port 30000`
- **s64-path2** (2xa4500-gdn-states): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-cache-report --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --max-mamba-cache-size 64 --mamba-max-states-per-path 2 --port 30000`
