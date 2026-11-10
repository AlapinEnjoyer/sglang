# SGLang capacity benchmark

- **Date:** 2026-11-10 11:32
- **Server:** `http://127.0.0.1:30000`, sglang 0.5.24.dev1123+gabd7532b5
- **Model:** `/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4`
- **Parallelism:** TP 2, DP 1, PP 1, EP 1
- **KV cache:** 286,604 tokens, dtype `fp8_e4m3`, page size 1
- **Max running requests:** 8 per DP rank
- **Context length:** 122,880 (max input 122,874)
- **Speculative:** `EAGLE` (draft tokens 4)
- **Prefill:** chunked prefill size 2048, attention backend `flashinfer`
- **Workload:** prompt `text` from `sharegpt`, 512 decode tokens measured per user, sampling `{"temperature": 1.0, "top_p": 0.95, "top_k": 20}`, repeats 1
- **GPUs on this host:** NVIDIA RTX A4500 (20470 MiB, 200.00 W), NVIDIA RTX A4500 (20470 MiB, 200.00 W)

Launch command: `--model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --mem-fraction-static 0.87 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --speculative-draft-model-quantization fp8 --speculative-token-map benchmark/ampere_qwen38/token_maps/swift15_sharegpt_32k.pt --max-mamba-cache-size 64 --mamba-max-states-per-path 2 --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --port 30000`

## How many users fit

Estimated from the KV pool and the running-request cap (hybrid / mamba models fold their state slots into the cap). Each user reserves its prompt plus the burst's max_new_tokens. The burst run confirms it: 'all concurrent = no' means some users queued.

| Context (input tokens) | Users | KV tokens per user | Users that fit | Limited by |
|---:|---:|---:|---:|---|
| 32,768 | 1 | 33,284 | 8 | max running requests |
| 32,768 | 2 | 35,332 | 8 | max running requests |
| 32,768 | 4 | 35,332 | 8 | max running requests |
| 32,768 | 8 | 35,332 | 8 | max running requests |
| 65,536 | 1 | 66,052 | 4 | KV pool |
| 65,536 | 2 | 68,100 | 4 | KV pool |
| 65,536 | 4 | 68,100 | 4 | KV pool |
| 98,304 | 1 | 98,820 | 2 | KV pool |
| 98,304 | 2 | 100,868 | 2 | KV pool |
| 122,362 | 1 | 122,878 | 2 | KV pool |
| 120,314 | 2 | 122,878 | 2 | KV pool |

## Burst: N users arrive together

- **TTFT** is time to first token from the common start. The last user waits for every prefill.
- **Prefill, 1st user** is that user's prompt tokens over its own TTFT. With N = 1 it is the single-user prefill speed.
- **Prefill, all** is all prompt tokens over the last user's TTFT.
- **Decode per user / total / slowest** come from the window where all N users decode together, at a depth of about L tokens.
- **Decode while others prefill** is the rate early users get while later prompts are still prefilling.

| Context | Users | All concurrent | TTFT 1st / last (s) | Prefill tok/s, 1st user | Prefill tok/s, all | Decode tok/s per user | Decode tok/s total | Slowest user tok/s | Decode tok/s while others prefill | Accept len |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 1 | yes | 16.66 / 16.66 | 1,967 | 1,967 | 142.8 | 142.8 | 142.8 | - | 3.48 |
| 32,768 | 2 | yes | 16.87 / 36.05 | 1,942 | 1,818 | 81.2 | 162.3 | 77.8 | 8.6 | 2.25 |
| 32,768 | 4 | yes | 17.41 / 76.16 | 1,882 | 1,721 | 76.5 | 305.9 | 70.7 | 10.6 | 2.80 |
| 32,768 | 8 | yes | 17.41 / 159.97 | 1,882 | 1,639 | 61.8 | 494.7 | 40.3 | 10.3 | 4.01 |
| 65,536 | 1 | yes | 40.07 / 40.07 | 1,636 | 1,636 | 108.7 | 108.7 | 108.7 | - | 2.96 |
| 65,536 | 2 | yes | 40.01 / 85.03 | 1,638 | 1,542 | 100.3 | 200.7 | 95.2 | 9.7 | 3.23 |
| 65,536 | 4 | yes | 40.04 / 178.29 | 1,637 | 1,470 | 63.7 | 254.6 | 55.2 | 8.2 | 2.79 |
| 98,304 | 1 | yes | 67.89 / 67.89 | 1,448 | 1,448 | 87.2 | 87.2 | 87.2 | - | 2.59 |
| 98,304 | 2 | yes | 67.89 / 144.68 | 1,448 | 1,359 | 92.9 | 185.8 | 81.5 | 8.6 | 3.39 |
| 122,362 | 1 | yes | 91.82 / 91.82 | 1,333 | 1,333 | 75.0 | 75.0 | 75.0 | - | 2.37 |
| 120,314 | 2 | yes | 89.90 / 191.20 | 1,338 | 1,259 | 69.6 | 139.3 | 67.7 | 8.2 | 3.05 |
