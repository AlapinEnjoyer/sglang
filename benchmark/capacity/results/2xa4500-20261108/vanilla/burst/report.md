# SGLang capacity benchmark

- **Date:** 2026-11-08 18:03
- **Server:** `http://127.0.0.1:30000`, sglang 0.5.23.dev20261018+g14ad64c411
- **Model:** `/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4`
- **Parallelism:** TP 2, DP 1, PP 1, EP 1
- **KV cache:** 263,889 tokens, dtype `fp8_e4m3`, page size 1
- **Max running requests:** 8 per DP rank
- **Context length:** 122,880 (max input 122,874)
- **Speculative:** `EAGLE` (draft tokens 4)
- **Prefill:** chunked prefill size 2048, attention backend `flashinfer`
- **Workload:** prompt `text` from `sharegpt`, 512 decode tokens measured per user, sampling `{"temperature": 1.0, "top_p": 0.95, "top_k": 20}`, repeats 2
- **GPUs on this host:** NVIDIA RTX A4500 (20470 MiB, 200.00 W), NVIDIA RTX A4500 (20470 MiB, 200.00 W)

Launch command: `--model-path /home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4 --served-model-name local --tp 2 --host 127.0.0.1 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser qwen3_coder --chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy --disable-prefill-cuda-graph --context-length 122880 --max-running-requests 8 --cuda-graph-max-bs-decode 8 --enable-mixed-chunk --prefill-decode-interval 4 --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-linear-replayssm-spec --enable-hierarchical-cache --hicache-size 12 --hicache-write-policy write_through --mem-fraction-static 0.90 --max-mamba-cache-size 32 --port 30000`

## How many users fit

Estimated from the KV pool and the running-request cap (hybrid / mamba models fold their state slots into the cap). Each user reserves its prompt plus the burst's max_new_tokens. The burst run confirms it: 'all concurrent = no' means some users queued.

| Context (input tokens) | Users | KV tokens per user | Users that fit | Limited by |
|---:|---:|---:|---:|---|
| 32,768 | 1 | 33,284 | 7 | KV pool |
| 32,768 | 4 | 35,332 | 7 | KV pool |

## Burst: N users arrive together

- **TTFT** is time to first token from the common start. The last user waits for every prefill.
- **Prefill, 1st user** is that user's prompt tokens over its own TTFT. With N = 1 it is the single-user prefill speed.
- **Prefill, all** is all prompt tokens over the last user's TTFT.
- **Decode per user / total / slowest** come from the window where all N users decode together, at a depth of about L tokens.
- **Decode while others prefill** is the rate early users get while later prompts are still prefilling.

| Context | Users | All concurrent | TTFT 1st / last (s) | Prefill tok/s, 1st user | Prefill tok/s, all | Decode tok/s per user | Decode tok/s total | Slowest user tok/s | Decode tok/s while others prefill | Accept len |
|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,768 | 1 | yes | 17.95 / 17.95 | 1,826 | 1,826 | 78.7 | 78.7 | 78.7 | - | 2.58 |
| 32,768 | 4 | yes | 18.30 / 81.60 | 1,790 | 1,606 | 68.5 | 274.0 | 56.0 | 9.8 | 2.92 |
