# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-ab-sync-20261110 | 2x NVIDIA RTX A4500 NVLink | 53 GB | - MB/s | ampere-opt@f74126aef2 | 2026-11-10T09:23:46 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| presync-1 | 2x NVIDIA RTX A4500 NVLink | ok | 246 s | 286604 | 8 | 64 / 113 | 436704 | - | - |
| presync-2 | 2x NVIDIA RTX A4500 NVLink | ok | 176 s | 286604 | 8 | 64 / 113 | 436704 | - | - |
| synced-1 | 2x NVIDIA RTX A4500 NVLink | ok | 176 s | 286604 | 8 | 64 / 113 | 436704 | - | - |
| synced-2 | 2x NVIDIA RTX A4500 NVLink | ok | 171 s | 286604 | 8 | 64 / 113 | 436704 | - | - |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|
| presync-1 | 32768 | 8 | 58.3 | 466.1 | 159.6 | 1881 | 2.90 |
| presync-2 | 32768 | 8 | 59.9 | 478.9 | 159.6 | 1882 | 3.00 |
| synced-1 | 32768 | 8 | 57.0 | 455.9 | 159.8 | 1882 | 2.94 |
| synced-2 | 32768 | 8 | 60.5 | 484.4 | 159.7 | 1881 | 2.97 |

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|
| presync-1 | 20 | 0.679 (0.738) | 5.9 / 20.0 | 36 | 0.28 (0.5) | - |
| presync-2 | 20 | 0.681 (0.738) | 5.6 / 15.3 | 36 | 0.28 (0.5) | - |
| synced-1 | 20 | 0.657 (0.738) | 5.9 / 24.3 | 36 | 0.27 (0.5) | - |
| synced-2 | 20 | 0.679 (0.738) | 5.7 / 22.1 | 34 | 0.28 (0.5) | - |

## Session restore from the storage tier

Each restore is checked against the same tokens recomputed from scratch (a fresh cache_salt): a wrong or mismatched state changes the top token or moves logprobs by far more than numerical noise (about 0.1).

| Config | Session | Recomputed (reference) | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|

## Chat clients: reuse through /v1/chat/completions

The template re-renders history; a client that drops reasoning_content changes the previous turn, so only the part before it can match. Cached needs --enable-cache-report.

| Config | Client | Turn | Prompt | Cached | Reusable (prev prompt + output) |
|---|---|---|---|---|---|

## Server commands

- **presync-1** (2xa4500-ab-sync-20261110): `/home/ubuntu/experiments/sglang-presync/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
- **presync-2** (2xa4500-ab-sync-20261110): `/home/ubuntu/experiments/sglang-presync/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
- **synced-1** (2xa4500-ab-sync-20261110): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
- **synced-2** (2xa4500-ab-sync-20261110): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
