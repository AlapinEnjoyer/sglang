# Benchmark studio report

## Machines

| Run | GPUs | RAM | Storage read | Repo | Date |
|---|---|---|---|---|---|
| 2xa4500-20261108 | 2x NVIDIA RTX A4500 NVLink | 53 GB | 125 MB/s | ampere-opt@f3e888b236 | 2026-11-08T19:56:44 |

## Configs

| Config | GPUs | Status | Start | KV pool | Running | GDN slots GPU / host | Host KV | GSM8K | Cache reuse |
|---|---|---|---|---|---|---|---|---|---|
| fork-best-disk | 2x NVIDIA RTX A4500 NVLink | ok | 201 s | 286592 | 8 | 64 / 113 | 436736 | - | ok |
| fork-best-ram | 2x NVIDIA RTX A4500 NVLink | ok | 121 s | 286604 | 8 | 64 / 113 | 436704 | - | ok |
| fork-fp8-map | 2x NVIDIA RTX A4500 NVLink | ok | 126 s | 363212 | 8 | 32 / 61 | 553432 | 98.0% | ok |
| fork-user-cmd | 2x NVIDIA RTX A4500 NVLink | ok | 141 s | 370892 | 8 | 32 / 60 | 555701 | 98.0% | ok |
| vanilla | 2x NVIDIA RTX A4500 NVLink | ok | 81 s | 263889 | 8 | 32 / None | 515201 | 98.4% | ok |

## Burst: N users arrive together

| Config | Context | Users | Decode tok/s per user | Total | TTFT last (s) | Prefill tok/s | Accept len |
|---|---|---|---|---|---|---|---|
| fork-best-ram | 32768 | 1 | 105.8 | 105.8 | 17.0 | 1932 | 2.58 |
| fork-best-ram | 32768 | 4 | 78.7 | 314.7 | 76.5 | 1884 | 2.81 |
| fork-best-ram | 32768 | 8 | 58.8 | 470.4 | 159.7 | 1879 | 3.48 |
| fork-fp8-map | 32768 | 1 | 110.7 | 110.7 | 17.3 | 1890 | 2.73 |
| fork-fp8-map | 32768 | 4 | 78.4 | 313.8 | 76.6 | 1875 | 2.78 |
| fork-fp8-map | 32768 | 8 | 59.4 | 475.1 | 159.8 | 1880 | 3.00 |
| fork-user-cmd | 32768 | 1 | 87.8 | 87.8 | 17.3 | 1897 | 2.75 |
| fork-user-cmd | 32768 | 4 | 69.2 | 276.8 | 77.9 | 1878 | 2.87 |
| fork-user-cmd | 32768 | 8 | 51.4 | 411.1 | 162.9 | 1879 | 2.99 |
| vanilla | 32768 | 1 | 78.7 | 78.7 | 17.9 | 1826 | 2.58 |
| vanilla | 32768 | 4 | 68.5 | 274.0 | 81.6 | 1790 | 2.92 |
| vanilla | 32k x 8: L=32,768 fits 7 users (--include-unfit to run) | | | | | | |

## Multi-turn sessions

Ceiling: the hit rate if every turn's history were cached; new user tokens never are.

| Config | Sessions | Hit rate (ceiling) | Avg / P99 TTFT (s) | Median TPOT (ms) | Req/s done (offered) | Decisions ok, median s |
|---|---|---|---|---|---|---|
| fork-best-disk | 10 | 0.730 (0.738) | 4.2 / 11.9 | 34 | 0.32 (0.5) | - |
| fork-best-disk | 20 | 0.699 (0.738) | 6.7 / 29.5 | 33 | 0.29 (0.5) | - |
| fork-best-ram | 10 | 0.730 (0.738) | 4.3 / 11.5 | 34 | 0.33 (0.5) | - |
| fork-best-ram | 20 | 0.691 (0.738) | 5.3 / 17.4 | 38 | 0.30 (0.5) | - |
| fork-fp8-map | 10 | 0.701 (0.738) | 4.8 / 14.5 | 32 | 0.30 (0.5) | - |
| fork-fp8-map | 20 | 0.487 (0.738) | 10.9 / 36.2 | 50 | 0.20 (0.5) | - |
| fork-user-cmd | 10 | 0.716 (0.738) | 4.2 / 16.9 | 36 | 0.30 (0.5) | - |
| fork-user-cmd | 20 | 0.437 (0.738) | 14.2 / 55.7 | 54 | 0.18 (0.5) | - |
| vanilla | 10 | 0.730 (0.738) | 4.1 / 10.9 | 37 | 0.30 (0.5) | - |
| vanilla | 20 | 0.425 (0.738) | 15.6 / 59.1 | 53 | 0.17 (0.5) | - |

## Session restore from the storage tier

| Config | Session | Cold prefill | GPU hit | After flush (storage) | After restart |
|---|---|---|---|---|---|
| fork-best-disk | 30000 | 15.58 s, 0 cached | 0.27 s, 29952 cached | 0.60 s, 29952 cached | 9.34 s, 29952 cached |

## Server commands

- **fork-best-disk** (2xa4500-20261108): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --page-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete --port 30000`
- **fork-best-ram** (2xa4500-20261108): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 64 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
- **fork-fp8-map** (2xa4500-20261108): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 32 --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --port 30000`
- **fork-user-cmd** (2xa4500-20261108): `/home/ubuntu/experiments/sglang/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.87 --max-mamba-cache-size 32 --port 30000`
- **vanilla** (2xa4500-20261108): `/home/ubuntu/experiments/inference/sglang-inf/.venv/bin/sglang serve --model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.90 --max-mamba-cache-size 32 --port 30000`
