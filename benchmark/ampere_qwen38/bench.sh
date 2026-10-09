#!/usr/bin/env bash
# Scenario runner against a live server. Usage: bench.sh <tag> <scenario...>
# Scenarios: S1 (decode, 1k/1k c1), S2 (prefill, 32k/64 c1), S2b (8k/64 c1),
#            S3c2/S3c4 (16k/1k), S4 (multi-turn shared prefix c4), S5 (sharegpt chat c1).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export VIRTUAL_ENV="$REPO/.venv"
PORT="${PORT:-8000}"
OUT="${OUT:-$REPO/benchmark/ampere_qwen38/results}"
mkdir -p "$OUT"
tag="$1"; shift

run() {
  local name="$1"; shift
  echo "=== $tag/$name"
  uv run --active python -m sglang.benchmark.serving --host 127.0.0.1 --port "$PORT" \
    --flush-cache --warmup-requests 1 --seed 1 \
    --output-file "$OUT/${tag}.jsonl" --tag "$name" "$@" 2>&1 \
    | grep -E "Successful requests|Benchmark duration|Input token throughput|Output token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Median ITL|Accept length|Max ReqConcurrency|Concurrency" || true
}

rnd=(--backend sglang --dataset-name random --random-range-ratio 1)
for s in "$@"; do
  case "$s" in
    S1)   run S1 "${rnd[@]}" --random-input-len 1024 --random-output-len 1024 --num-prompts 4 --max-concurrency 1 ;;
    S2)   run S2 "${rnd[@]}" --random-input-len 32768 --random-output-len 64 --num-prompts 3 --max-concurrency 1 ;;
    S2b)  run S2b "${rnd[@]}" --random-input-len 8192 --random-output-len 64 --num-prompts 4 --max-concurrency 1 ;;
    S3c2) run S3c2 "${rnd[@]}" --random-input-len 16384 --random-output-len 1024 --num-prompts 6 --max-concurrency 2 ;;
    S3c4) run S3c4 "${rnd[@]}" --random-input-len 16384 --random-output-len 1024 --num-prompts 8 --max-concurrency 4 ;;
    S4)   run S4 --backend sglang --dataset-name generated-shared-prefix --gsp-num-groups 4 \
            --gsp-prompts-per-group 2 --gsp-system-prompt-len 16000 --gsp-question-len 512 \
            --gsp-output-len 512 --gsp-num-turns 4 --max-concurrency 4 ;;
    S5)   run S5 --backend sglang-oai-chat --dataset-name sharegpt --sharegpt-output-len 512 \
            --num-prompts 12 --max-concurrency 1 ;;
    *) echo "unknown scenario $s" >&2 ;;
  esac
done
