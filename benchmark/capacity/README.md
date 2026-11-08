# Capacity benchmark

How many concurrent users can a running SGLang server hold at a given context length,
and what prefill and decode speed does each of them get?

`capacity_bench.py` is an HTTP client. It reads the KV pool size, running-request cap,
context length, speculative settings and launch command from the server
(`/server_info`, `/model_info`, `/v1/models`), so it needs no model- or GPU-specific
configuration. Launch the server however you deploy it. For a model's recommended
deployment, see its page under `docs/cookbook/`. Then point the tool at it.

## Benchmark studio

`studio.py` is the single entry point for comparing serving setups across machines.
It plans, launches, measures and reports, and it uses the tools below internally.
Run it from the repository root with the interpreter of the sglang you want to test:
`sglang serve` in a suite resolves to that interpreter's `sglang`.

```bash
# 1. How many users of a given context fit, and the serve flags for them
python benchmark/capacity/studio.py plan --model /models/Swift-1.5-Qwen3.8-27b-INT4 \
    --context 131072                      # this machine's GPUs and RAM
python benchmark/capacity/studio.py plan --model <dir> --gpu H100:80:2:9.0 --ram-gb 512 \
    --write-suite h100.toml               # another machine; writes a suite to verify it

# 2. A 15-minute check of a new machine, then a full suite
MODEL=<dir> python benchmark/capacity/studio.py run benchmark/capacity/configs/studio-smoke.toml
MODEL=<dir> VANILLA=<pip venv> STUDIO_STORAGE=/nvme/hicache \
python benchmark/capacity/studio.py run benchmark/capacity/configs/studio-swift15-2xa4500.toml

# 3. One report over runs from several machines
python benchmark/capacity/studio.py report benchmark/capacity/results/<run> <run2> --out cmp.md
```

**Planner.** It reads the checkpoint's safetensors headers and `config.json`, then
computes per GPU:
- **Weights and per-token KV:** from the headers and the attention config.
- **GDN state per slot:** from the linear-attention config.
- **Memory fraction:** leaves room for the prefill logits (chunk x vocab / TP, fp32).

Its constants were calibrated on 2x RTX A4500. Against measured pools it is within
1.5% for KV and within 5% for host state slots. The plan is an estimate: run the
written suite on the target machine, because the burst at the planned user count is
what proves the memory fraction.

**Suite format.** It extends the `config_sweep.py` format below with steps:

```toml
launch = "sglang serve"           # a config may override: launch = "$VANILLA/bin/sglang serve"
port = 30000
server_args = "--model-path $MODEL ..."   # shared; $VARS come from the environment
[env]
SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR = "$STUDIO_STORAGE"

[steps.cache]                     # each table enables a step for every config
[steps.gsm8k]                     # sgl-eval, pinned version via uvx; num_examples = 250
[steps.burst]
args = "--context-lens 32k --users 1,4,8 --repeats 2"   # capacity_bench.py flags
[steps.multiturn]                 # bench_multiturn sessions, one run per client count
clients = [10, 20]                # also: rounds, request_length, output_length,
decision_rate = 0.5               # request_rate, max_parallel, session_ids, priority
[steps.disk]                      # restore one session from the storage tier
session_len = 30000               # after a flush, and after a restart (restart = true)

[[configs]]
name = "fork-best-disk"
server_args = "--page-size 64 --hicache-storage-backend file ..."
steps = ["cache", "multiturn", "disk"]   # optional subset
```

**Output.** Each run writes `results/<run>/`, with one directory per config:
- `machine.json`: GPUs, NVLink, RAM, storage read speed, and the repo revision.
- `<config>/summary.json`: pools, startup time and every step's numbers.
- `report.md`: rebuilt after every config.
- The server and step logs, which git ignores.

`--resume` skips configs already marked ok.

**Steps.**

| Step | What it answers |
|---|---|
| `cache` | Do turns reuse the prefix cache (`cache_check.py`)? |
| `gsm8k` | Is accuracy unchanged (sgl-eval, greedy, thinking off by default)? |
| `burst` | Prefill and decode speed with N users at once (`capacity_bench.py`) |
| `multiturn` | Hit rate against its ceiling and TTFT/TPOT with S rotating sessions, optionally under `/v1/decisions` load |
| `disk` | First-token time of a session restored from the storage tier, against a cold prefill |

The multi-turn ceiling is what full reuse would give: every turn's new user tokens are
uncached by definition. 6 rounds of 4000 new and 500 output tokens give 0.738.

## Quick start

```bash
# 1. Launch the server with the command you deploy (cookbook page for the model)
sglang serve --model-path <model> --tp-size <n> ... --port 30000

# 2. Sweep: every context length x every user count that fits
python benchmark/capacity/capacity_bench.py --base-url http://127.0.0.1:30000 \
    --context-lens 8k,32k,full --users auto --output-len 512 --repeats 3

# "Two users at full context: what prefill and decode speed?"
python benchmark/capacity/capacity_bench.py --context-lens full --users 1,2

# The most context each of 1, 2 or 4 users can have, and the speed at that size
python benchmark/capacity/capacity_bench.py --context-lens fit --users 1,2,4

# Add the realistic steady-state run (sglang.bench_serving, 5N prompts at concurrency N)
python benchmark/capacity/capacity_bench.py --context-lens 32k --users 1,2,4 --steady

# Max users that keep >= 20 tok/s per user and TTFT <= 60 s
python benchmark/capacity/capacity_bench.py --users auto --slo-decode 20 --slo-ttft 60
```

Each run writes `benchmark/capacity/results/<timestamp>/` (override with `--out-dir`):

| File | Contents |
|---|---|
| `report.md` | Server and GPU summary, capacity table, burst table, steady-state table, SLO answer |
| `results.jsonl` | One raw burst result per repeat, including per-user TTFT, decode rate and token counts |
| `server_info.json` | The full `/server_info` snapshot and the local GPU list |
| `bench_serving.jsonl`, `bench_serving_*.log` | Raw `sglang.bench_serving` output from `--steady` runs |

The report is rewritten after every cell, so an interrupted sweep still leaves one.

## What is measured

**Capacity estimate.**
- Users that fit = `min(KV pool // (L + max_new_tokens + draft tokens), max running
  requests)`, times the DP size. `max_new_tokens` is the burst's per-request budget
  (`--output-len`, plus `--decode-headroom` when N > 1).
- Hybrid models (Mamba, GDN, KDA) already fold their state slots into the server's
  effective running-request cap, so that number is used.
- Cells beyond the estimate are skipped unless you pass `--include-unfit`. Running them
  shows what queueing looks like.
- The estimate is an upper bound. The scheduler can still hold a request back for its own
  reasons, such as mamba-slot gating on hybrid models. The burst run is the ground truth:
  `all concurrent = no` means the N users never ran together.

**Burst (default).** For each context length L and user count N:
1. The tool flushes the cache (`/flush_cache`).
2. N users each send a distinct L-token prompt at the same instant, as separate
   streaming `/generate` requests with `ignore_eos`.
3. The decode window opens when the last user receives its first token, so all N are
   decoding. It closes once every user has produced `--output-len` more tokens.
4. The tool then aborts the requests (`/abort_request`).

Early users keep decoding, slowly, while later prompts prefill. Each request is therefore
allowed `--output-len + --decode-headroom` tokens, so no one finishes before the window
opens.

It reports:

| Metric | Definition |
|---|---|
| TTFT 1st / last | Time to first token from the common start. The last user waits for every prefill. |
| Prefill tok/s, 1st user | The first user's uncached prompt tokens / its TTFT. With N = 1 this is the single-user prefill speed at length L. |
| Prefill tok/s, all | All uncached prompt tokens / the last TTFT: aggregate prefill throughput. |
| Decode tok/s per user | Tokens generated in the all-decoding window / window / N. This is decode with batch N at depth ~L, the same full-batch window `bench_one_batch_server` reports. |
| Decode tok/s total | The same window, all users together. |
| Slowest user | The slowest user's tokens in the window / window. |
| Decode while others prefill | Mean decode rate of the early users between their own first token and the window opening. It shows how much a running user stalls when someone else's long prompt arrives. |
| All concurrent | `no` when some user finished before the last one started, meaning the N users never ran together (queued). |
| Accept len | Completion tokens / verify steps, with speculative decoding only. |

**Steady state (`--steady`).** For each cell the tool runs
`python -m sglang.bench_serving --backend sglang --dataset-name random
--random-range-ratio 1 --max-concurrency N --num-prompts 5N --flush-cache`.
The 5N follows the docs' rule `num-prompts >= 5 * max-concurrency`. In this mode
users arrive at different times, so a user's decode stalls while others prefill. The
TTFT and TPOT percentiles are therefore what a user sees under sustained load. Per-user
decode is reported as 1000 / median TPOT.

Use burst numbers to compare configs and hardware: they are clean, separated phases.
Use steady-state numbers for user-facing latency promises.

## Options worth knowing

- `--context-lens`: plain ints, `k` suffix (x1024), `full` or `fit`.
  - `full` is the longest prompt the server accepts with room left for the request's
    `max_new_tokens`.
  - `fit` picks, for each user count, the longest prompt at which that many users fit in
    the KV pool together. `--context-lens fit --users 1,2,4` answers "how much context
    does each user get with 1, 2 or 4 users, and how fast is it?"
- `--users`: a comma list, or `auto` for 1, 2, 4, ... up to the users that fit, plus the
  exact maximum.
- `--sampling-params '{"temperature":0.6,"top_p":0.95,"top_k":20}'`: use the model's
  recommended sampling. It barely moves speed without speculative decoding, but it sets
  the acceptance length (and so decode speed) with MTP / EAGLE / DFlash. The default `{}`
  takes the server's defaults.
- `--prompt text|random`.
  - `text` (default) tokenizes a corpus on the server through `/tokenize`, so no local
    tokenizer is needed. Each prompt is cut from a random offset and starts with a random
    nonce, so no two prompts share a prefix and the radix cache cannot help.
  - The default corpus is ShareGPT, the same text `bench_serving --dataset-name random`
    uses, so burst and steady runs share a text distribution. `--corpus <files or dirs>`
    replaces it. Use your own traffic's text when speculative acceptance matters.
  - `random` uses random token ids.
- `--repeats K` repeats each burst cell and reports medians.
- `--no-burst --steady` runs only `bench_serving`.

## Interpreting the numbers

- Decode speed falls with context depth even at N = 1, because attention reads the whole
  KV cache every step. Measure at the depth you care about, not 1k.
- At long contexts a burst of N users is prefill-bound. The last user's TTFT is roughly
  N x the single-user prefill time, because chunked prefill serves one long prompt at a
  time.
- Speculative acceptance depends on the content. Random-token prompts and repetitive
  greedy outputs both distort it.
- With speculative decoding, per-user decode = accept length x verify steps per second.
  To see what batching itself costs, compare `decode tok/s per user / accept len` across
  user counts: that ratio moves only with the weight and KV bytes read per step, while
  accept length moves with the text. A decode window of a few seconds is noisy, so use
  `--repeats 3`.
- By default a prefill chunk runs alone, so running users stop decoding while a new
  prompt prefills ("Decode while others prefill" near 0).
  - `--enable-mixed-chunk` adds one decode token per user to each prefill chunk: about
    1 / (chunk time), so ~1.5 tok/s with 2048-token chunks on a 27B model. EAGLE, EAGLE3
    and DFLASH accept it; a mixed step decodes without speculation.
  - `--prefill-decode-interval K` runs K decode steps after every prefill chunk. Running
    users keep a useful rate, and long prompts take longer.
- Power-capped cards (for example RTX A-series) clock down under sustained prefill.
  Prefill speed from a short sweep can be higher than in production.

## Comparing server configurations

One `capacity_bench.py` run measures one launch. `config_sweep.py` runs it against several
launches in turn, from a TOML file. For each `[[configs]]` entry it:
1. Starts the server.
2. Waits for `/health_generate`.
3. Runs `capacity_bench.py`.
4. Stops the server and moves on.

A config that fails to launch (OOM, a rejected flag) is recorded and skipped.

```bash
MODEL=/models/Qwen3.8-27B DFLASH=/models/Qwen3.8-27B-DFlash2 \
python benchmark/capacity/config_sweep.py benchmark/capacity/configs/qwen3.8-27b-1gpu.toml \
    --slo-decode 30 --slo-ttft 120

# Print the commands only
python benchmark/capacity/config_sweep.py <file>.toml --dry-run
# Realistic arrivals (sglang.bench_serving) for the two best configs only
python benchmark/capacity/config_sweep.py <file>.toml \
    --only mtp-120k-c4,nospec-120k-c8 --bench-args "--no-burst --steady --context-lens 32k"
# Continue an interrupted sweep: configs already marked ok are skipped
python benchmark/capacity/config_sweep.py <file>.toml --out-dir <sweep dir> --resume
```

```toml
launch = "sglang serve"            # the sweep appends --port <port>
port = 30000
server_args = "--model-path $MODEL --kv-cache-dtype fp8_e4m3"   # shared; $VARS from env
bench_args = "--context-lens 8k,32k,full --users auto --repeats 2"
[env]
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

[[configs]]
name = "mtp-c4"
server_args = "--max-running-requests 4 --speculative-algorithm EAGLE ..."
bench_args = "--users 1,2,4"         # optional, overrides the shared bench_args
env = { SOME_VAR = "1" }             # optional
```

Config arguments come after the shared ones, so a repeated flag takes the config's value.

Output, in `results/sweep-<timestamp>/`:
- One `capacity_bench` directory per config, with its `server.log` and `bench.log`.
- `summary.md`, which lists every config's KV pool, running cap and launch status.
- In `summary.md`, a table of every (context, users) cell with each config side by side,
  the best total decode in bold.
- With `--slo-*`, the most users each config holds within the SLO.

The summary is rebuilt after every config. Use `--summary-only --out-dir <dir>` to rebuild
it by hand.

## Multi-turn sessions and the prefix cache

The burst and steady runs flush the cache and use fresh prompts, so they never measure
cache reuse. For chat or agent traffic, where each turn resends the whole conversation,
check reuse first, then load the server with many sessions:

```bash
# One minute: turn 2 must report most of turn 1 as cached
python benchmark/capacity/cache_check.py --base-url http://127.0.0.1:30000
# S sessions, at most P running, each turn adding request-length new tokens
python benchmark/hicache/bench_multiturn.py --model-path <model> --port 30000 \
    --num-clients 20 --max-parallel 8 --num-rounds 8 \
    --request-length 8000 --output-length 500 --request-rate 0.5 \
    --disable-auto-run --log-file multiturn.jsonl --tag <name>
```

- The highest possible hit rate is (total prompt tokens - new tokens) / total prompt
  tokens; 0.79 for the command above.
- Sessions take turns, so once their total context exceeds what the cache holds, the
  least-recently-used one evicted is the one coming back next. The hit rate then drops
  close to 0 rather than in proportion.
- `--enable-hierarchical-cache --hicache-size <GB>` keeps evicted KV (and GDN state on
  hybrid models) in host RAM. Size the host pool from the total context of the sessions
  you expect to be active.
- Hybrid models also need a saved state slot at the match point, so idle sessions
  compete for `--max-mamba-cache-size` as well as for KV. When hits are near 0, run
  `cache_check.py` first: if it passes, the sessions simply do not fit.

## Profiling a cell

To see *why* a cell is slow, profile the same shape with the tools from the
[benchmark and profiling guide](../../docs/docs/developer_guide/benchmark_and_profiling.mdx):

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/profiles   # on the server too
python -m sglang.bench_serving --backend sglang --dataset-name random \
    --random-input-len 32768 --random-output-len 64 --random-range-ratio 1 \
    --num-prompts 2 --max-concurrency 2 --profile
# or capture N decode steps of a live run:
curl -X POST http://127.0.0.1:30000/start_profile -H 'Content-Type: application/json' \
    -d '{"start_step": 3, "num_steps": 10, "detailed_annotations": true}'
```
