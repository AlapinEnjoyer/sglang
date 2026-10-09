#!/usr/bin/env bash
# Launch Swift-1.5-Qwen3.8-27B-INT4 at TP2 on Ampere through the repo's uv env.
# Usage: serve.sh <preset> [extra sglang args...]
#   preset A: the reference single-user MTP command; B: cookbook-style variant; nospec: A without speculation;
#   dflash: A with the DFlash2 block-diffusion draft instead of MTP;
#   M / MD: multi-user (USERS, default 4) with MTP / DFlash2;
#   best1 / best4: recommended single-user (DFlash2) and 4-user (MTP) configs on 2x A4500.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export VIRTUAL_ENV="$REPO/.venv"
MODEL="${MODEL:-/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4}"
PORT="${PORT:-8000}"
preset="${1:-A}"; shift || true

common=(
  --model-path "$MODEL" --served-model-name local --tp-size 2
  --context-length 262144 --kv-cache-dtype fp8_e4m3
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
  --host 0.0.0.0 --port "$PORT"
)
mtp=(
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4 --speculative-draft-model-quantization "${DRAFT_QUANT:-unquant}"
)
case "$preset" in
  A) args=(--chunked-prefill-size 4096 --mamba-radix-cache-strategy extra_buffer
           --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 5
           $([ "${PREFILL_GRAPH:-0}" = 1 ] || echo --disable-prefill-cuda-graph) --cuda-graph-max-bs-decode 1 "${mtp[@]}") ;;
  nospec) args=(--chunked-prefill-size 4096 --mamba-radix-cache-strategy extra_buffer
           --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 5
           --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1) ;;
  dflash) args=(--chunked-prefill-size 4096 --mamba-radix-cache-strategy extra_buffer
           --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 5
           --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1
           --speculative-algorithm DFLASH --speculative-num-draft-tokens 8
           --speculative-draft-model-path "${DFLASH_PATH:-/home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2}"
           --speculative-draft-model-quantization "${DRAFT_QUANT:-unquant}") ;;
  M|MD) users="${USERS:-4}"
         args=(--chunked-prefill-size "${CHUNK:-2048}" --mamba-radix-cache-strategy extra_buffer_lazy
           --mem-fraction-static "${MEM_FRAC:-0.90}" --max-running-requests "$users"
           --max-mamba-cache-size "$((users * 3))" --cuda-graph-max-bs-decode "$users")
         export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
         if [ "$preset" = M ]; then args+=("${mtp[@]}")
         else args+=(--speculative-algorithm DFLASH --speculative-num-draft-tokens 8
           --speculative-draft-model-path "${DFLASH_PATH:-/home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2}"
           --speculative-draft-model-quantization "${DRAFT_QUANT:-unquant}"); fi ;;
  best1) maps="$REPO/benchmark/ampere_qwen38/token_maps"
         args=(--chunked-prefill-size 4096 --mamba-radix-cache-strategy extra_buffer
           --mem-fraction-static 0.90 --max-running-requests 1 --max-mamba-cache-size 5
           --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1
           --speculative-algorithm DFLASH --speculative-num-draft-tokens 8
           --speculative-draft-model-path "${DFLASH_PATH:-/home/ubuntu/experiments/inference/models/Qwen3.8-27B-DFlash2}"
           --speculative-draft-model-quantization fp8
           --speculative-token-map "$maps/swift15_sharegpt_64k.pt"
           --enable-linear-replayssm-spec) ;;
  best4) maps="$REPO/benchmark/ampere_qwen38/token_maps"
         export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
         args=(--chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy
           --mem-fraction-static 0.90 --max-running-requests 4 --max-mamba-cache-size 12
           --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 4
           "${mtp[@]/unquant/fp8}" --speculative-token-map "$maps/swift15_sharegpt_32k.pt"
           --enable-linear-replayssm-spec) ;;
  B) args=(--chunked-prefill-size 2048 --mamba-radix-cache-strategy extra_buffer_lazy
           --attention-backend flashinfer --mem-fraction-static 0.90 --max-running-requests 1
           "${mtp[@]}") ;;
  *) echo "unknown preset $preset" >&2; exit 1 ;;
esac
exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" uv run --active sglang serve "${common[@]}" "${args[@]}" "$@"
