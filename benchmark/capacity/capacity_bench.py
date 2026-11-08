"""Capacity benchmark for a running SGLang server.

Answers "with N concurrent users at context length L, what prefill and decode speed does
each user get, and how many users fit at all?" for whatever model, GPUs and launch flags
the server was started with. It only talks HTTP, so the same command works for any model.

For every (context length L, users N) cell it can run two measurements:

  burst   (default) N users send a distinct L-token prompt at the same instant, with the
          cache flushed first. Reports each user's TTFT and the prefill throughput. Decode
          is measured in the window where all N users decode together: from the last
          user's first token until every user has produced --output-len more tokens, after
          which the requests are aborted. That is decode at depth ~L with batch N, the
          full-batch window `sglang.bench_one_batch_server` also reports.
  steady  (--steady) runs `sglang.bench_serving` with --max-concurrency N and
          --num-prompts 5N (the docs' rule for steady state). Arrivals overlap, so one
          user's decode is interleaved with other users' prefill chunks. These are the
          TTFT/TPOT a real user sees under that load.

Usage (server already running, launched with the cookbook command for the model):
  python benchmark/capacity/capacity_bench.py --base-url http://127.0.0.1:30000 \
      --context-lens 8k,32k,full --users auto --output-len 512
"""

import argparse
import asyncio
import datetime
import json
import math
import random
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import msgspec
import requests

from sglang.benchmark.stream_metrics import validate_finish_reason

CORPUS_SUFFIXES = (".md", ".mdx", ".txt", ".rst")
# ~2M tokens: enough distinct text for several full-context prompts without repeating.
SHAREGPT_CORPUS_CHARS = 8_000_000
TOKENIZE_CHUNK_CHARS = 200_000
NONCE_TOKENS = 16


# ---------------------------------------------------------------- server info


class ServerCapacity(msgspec.Struct, frozen=True):
    model_path: str
    tokenizer_path: Optional[str]
    context_len: int
    max_input_len: int
    kv_pool_tokens: int
    max_running_requests: int
    dp_size: int
    page_size: int
    spec_draft_tokens: int
    raw_server_info: Dict[str, Any]

    def full_input_len(self, max_new_tokens: int) -> int:
        # The server also counts a few reserved tokens against the context; the gap
        # between context_len and max_req_input_len covers them.
        reserved = self.context_len - self.max_input_len
        return min(self.max_input_len, self.context_len - max_new_tokens - reserved)

    def tokens_per_user(self, input_len: int, max_new_tokens: int) -> int:
        need = input_len + max_new_tokens + self.spec_draft_tokens
        return math.ceil(need / self.page_size) * self.page_size

    def users_that_fit(self, input_len: int, max_new_tokens: int) -> int:
        """Upper bound from the KV pool and the running-request cap (per DP rank)."""
        by_kv = self.kv_pool_tokens // self.tokens_per_user(input_len, max_new_tokens)
        return self.dp_size * min(by_kv, self.max_running_requests)

    def max_input_for_users(self, users: int, max_new_tokens: int) -> int:
        """Longest prompt at which `users` requests fit the KV pool together; 0 if none."""
        per_dp = math.ceil(users / self.dp_size)
        if per_dp > self.max_running_requests:
            return 0
        tokens = (self.kv_pool_tokens // per_dp) // self.page_size * self.page_size
        fit = tokens - max_new_tokens - self.spec_draft_tokens
        return max(0, min(fit, self.full_input_len(max_new_tokens)))

    def limited_by(self, input_len: int, max_new_tokens: int) -> str:
        by_kv = self.kv_pool_tokens // self.tokens_per_user(input_len, max_new_tokens)
        return (
            "KV pool" if by_kv < self.max_running_requests else "max running requests"
        )


def fetch_server_capacity(base_url: str) -> ServerCapacity:
    si = requests.get(f"{base_url}/server_info", timeout=60).json()
    mi = requests.get(f"{base_url}/model_info", timeout=60).json()
    models = requests.get(f"{base_url}/v1/models", timeout=60).json()
    context_len = models["data"][0].get("max_model_len") or si.get("context_length")

    states = si.get("internal_states") or [{}]
    per_dp_mrr = [
        s["effective_max_running_requests_per_dp"]
        for s in states
        if s.get("effective_max_running_requests_per_dp")
    ]
    max_running = min(per_dp_mrr) if per_dp_mrr else si.get("max_running_requests")
    spec = si.get("speculative_algorithm")
    return ServerCapacity(
        model_path=mi.get("model_path") or si.get("model_path"),
        tokenizer_path=mi.get("tokenizer_path"),
        context_len=int(context_len),
        max_input_len=int(si.get("max_req_input_len") or context_len),
        kv_pool_tokens=int(si["max_total_num_tokens"]),
        max_running_requests=int(max_running or 1),
        dp_size=int(si.get("dp_size") or 1),
        page_size=int(si.get("page_size") or 1),
        spec_draft_tokens=int(si.get("speculative_num_draft_tokens") or 0)
        if spec
        else 0,
        raw_server_info=si,
    )


def local_gpus() -> List[Dict[str, str]]:
    """GPUs visible on this machine; only meaningful when the server runs here too."""
    if not shutil.which("nvidia-smi"):
        return []
    fields = "index,name,memory.total,power.limit,clocks.max.sm"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    ).stdout
    keys = fields.split(",")
    return [
        dict(zip(keys, (v.strip() for v in line.split(","))))
        for line in out.splitlines()
    ]


def flush_cache(base_url: str, timeout: float = 300) -> None:
    """Flush the radix cache; this only succeeds once the server is idle."""
    deadline = time.time() + timeout
    while True:
        r = requests.post(f"{base_url}/flush_cache", timeout=60)
        if r.status_code == 200:
            return
        if time.time() > deadline:
            raise RuntimeError(f"/flush_cache kept failing: {r.status_code} {r.text}")
        time.sleep(1)


# ---------------------------------------------------------------- prompts


class PromptSource:
    """Exact-length prompts that share no prefix, so the radix cache cannot help.

    `text` tokenizes a corpus on the server (/tokenize, so no local tokenizer is needed)
    and cuts each prompt from a random offset. The default corpus is ShareGPT, the text
    `bench_serving --dataset-name random` tiles, so burst and steady runs see the same
    distribution. `random` samples ids from the corpus vocabulary. Each prompt starts
    with a random nonce, so no two share a prefix.
    """

    def __init__(self, base_url: str, kind: str, corpus: List[str], seed: int):
        self.kind = kind
        self.rng = random.Random(seed)
        self.pool = self._tokenize(
            base_url=base_url, text=self._load_text(corpus, seed)
        )
        self.vocab = sorted(set(self.pool))
        print(
            f"prompt pool: {len(self.pool):,} tokens, {len(self.vocab):,} distinct ids"
        )

    @staticmethod
    def _load_text(corpus: List[str], seed: int) -> str:
        if corpus == ["sharegpt"]:
            from sglang.benchmark.datasets.common import (
                SHAREGPT_FILENAME,
                SHAREGPT_REPO_ID,
            )
            from sglang.benchmark.utils import download_and_cache_hf_file

            path = download_and_cache_hf_file(
                repo_id=SHAREGPT_REPO_ID, filename=SHAREGPT_FILENAME
            )
            with open(path) as f:
                conversations = json.load(f)
            random.Random(seed).shuffle(conversations)
            parts: List[str] = []
            num_chars = 0
            for conv in conversations:
                for turn in conv["conversations"]:
                    parts.append(turn["value"])
                    num_chars += len(turn["value"])
                if num_chars >= SHAREGPT_CORPUS_CHARS:
                    break
            return "\n\n".join(parts)
        files: List[Path] = []
        for p in map(Path, corpus):
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.suffix in CORPUS_SUFFIXES)
            else:
                files.append(p)
        return "\n\n".join(f.read_text(errors="ignore") for f in files)

    @staticmethod
    def _tokenize(base_url: str, text: str) -> List[int]:
        if not text.strip():
            raise ValueError("prompt corpus is empty")
        tokens: List[int] = []
        for i in range(0, len(text), TOKENIZE_CHUNK_CHARS):
            r = requests.post(
                f"{base_url}/tokenize",
                json={
                    "prompt": text[i : i + TOKENIZE_CHUNK_CHARS],
                    "add_special_tokens": False,
                },
                timeout=300,
            )
            r.raise_for_status()
            tokens += r.json()["tokens"]
        return tokens

    def make(self, length: int) -> List[int]:
        nonce = [self.rng.choice(self.vocab) for _ in range(min(NONCE_TOKENS, length))]
        body_len = length - len(nonce)
        if self.kind == "random":
            return nonce + [self.rng.choice(self.vocab) for _ in range(body_len)]
        body: List[int] = []
        start = self.rng.randrange(len(self.pool))
        while len(body) < body_len:
            body += self.pool[start : start + body_len - len(body)]
            start = 0
        return nonce + body


# ---------------------------------------------------------------- burst

# (arrival time, cumulative completion tokens) of one streamed chunk.
_Obs = Tuple[float, int]


def _tokens_at(obs: List[_Obs], t: float) -> float:
    """Cumulative tokens of one stream at time t, interpolated between chunks."""
    if not obs or t <= obs[0][0]:
        return 0.0 if not obs or t < obs[0][0] else float(obs[0][1])
    for (t0, c0), (t1, c1) in zip(obs, obs[1:]):
        if t <= t1:
            return c0 + (c1 - c0) * (t - t0) / (t1 - t0) if t1 > t0 else float(c1)
    return float(obs[-1][1])


class _BurstTracker:
    """Shared progress of the N streams; decides when the decode window is complete."""

    def __init__(self, users: int, output_len: int):
        self.output_len = output_len
        self.obs: List[List[_Obs]] = [[] for _ in range(users)]
        self.finished = [False] * users
        self.metas: List[Dict[str, Any]] = [{} for _ in range(users)]
        # (completion tokens, verify steps) from the last chunk that carried spec counters;
        # not every streamed chunk does.
        self.spec: List[Tuple[int, int]] = [(0, 0)] * users
        self.window_start: Optional[float] = None
        self.window_end: Optional[float] = None
        # A user finished before the last one started: they never all ran together.
        self.queued = False

    def on_chunk(
        self, index: int, now: float, meta: Dict[str, Any], finished: bool
    ) -> None:
        self.metas[index] = meta
        if meta.get("spec_verify_ct"):
            self.spec[index] = (meta["completion_tokens"], meta["spec_verify_ct"])
        if meta["completion_tokens"] > 0:
            self.obs[index].append((now, meta["completion_tokens"]))
        self.finished[index] |= finished
        if self.window_start is None:
            if finished:
                self.queued = True
            elif all(self.obs):
                self.window_start = now
            return
        if self.window_end is None and (finished or self._all_reached_target()):
            self.window_end = now

    def _all_reached_target(self) -> bool:
        return all(
            o[-1][1] - _tokens_at(o, self.window_start) >= self.output_len
            for o in self.obs
        )

    def user_done(self, index: int) -> bool:
        if self.finished[index]:
            return True
        # Keep reading until this stream has a chunk at or after the window end, so the
        # end-of-window token count is interpolated rather than floored.
        return (
            not self.queued
            and self.window_end is not None
            and self.obs[index][-1][0] >= self.window_end
        )


class UserResult(msgspec.Struct, frozen=True):
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    ttft: float
    window_tokens: Optional[float]
    decode_tps_while_others_prefill: Optional[float]
    spec_tokens: int
    spec_verify_ct: int


class BurstResult(msgspec.Struct, frozen=True):
    context_len: int
    users: int
    repeat: int
    max_new_tokens: int
    all_concurrent: bool
    ttft_first: float
    ttft_last: float
    ttft_mean: float
    prefill_tps_first_user: float
    prefill_tps_all: float
    decode_window_s: Optional[float]
    decode_tps_per_user: Optional[float]
    decode_tps_total: Optional[float]
    decode_tps_slowest_user: Optional[float]
    decode_tps_while_others_prefill: Optional[float]
    accept_length: Optional[float]
    per_user: List[UserResult]


async def _stream_user(
    session: aiohttp.ClientSession,
    base_url: str,
    index: int,
    rid: str,
    input_ids: List[int],
    sampling_params: Dict[str, Any],
    tracker: _BurstTracker,
) -> None:
    payload = {
        "rid": rid,
        "input_ids": input_ids,
        "sampling_params": sampling_params,
        "stream": True,
    }
    async with session.post(f"{base_url}/generate", json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f"user {index}: HTTP {resp.status}: {await resp.text()}")
        async for raw in resp.content:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            body = line[5:].strip()
            if body == b"[DONE]":
                break
            msg = json.loads(body)
            if "error" in msg:
                raise RuntimeError(f"user {index}: {msg['error']}")
            meta = msg["meta_info"]
            finish = meta.get("finish_reason")
            if finish is not None:
                validate_finish_reason(finish, ignore_eos=True)
            tracker.on_chunk(
                index=index,
                now=time.perf_counter(),
                meta=meta,
                finished=finish is not None,
            )
            if tracker.user_done(index):
                break


async def _run_burst_async(
    base_url: str,
    prompts: List[List[int]],
    sampling_params: Dict[str, Any],
    output_len: int,
    timeout: float,
) -> Tuple[float, _BurstTracker]:
    tracker = _BurstTracker(users=len(prompts), output_len=output_len)
    rids = [uuid.uuid4().hex for _ in prompts]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout),
        connector=aiohttp.TCPConnector(limit=0),
        read_bufsize=1 << 22,
    ) as session:
        t0 = time.perf_counter()
        await asyncio.gather(
            *(
                _stream_user(session, base_url, i, rid, p, sampling_params, tracker)
                for i, (rid, p) in enumerate(zip(rids, prompts))
            )
        )
        # Requests still decoding past the window are no longer needed.
        for i, rid in enumerate(rids):
            if not tracker.finished[i]:
                async with session.post(f"{base_url}/abort_request", json={"rid": rid}):
                    pass
    return t0, tracker


def _summarize_burst(
    t0: float,
    tracker: _BurstTracker,
    context_len: int,
    repeat: int,
    max_new_tokens: int,
) -> BurstResult:
    n = len(tracker.obs)
    first_t = [o[0][0] for o in tracker.obs]
    concurrent = (
        not tracker.queued
        and tracker.window_start is not None
        and tracker.window_end is not None
    )
    window_tokens: List[Optional[float]] = [None] * n
    if concurrent:
        window_tokens = [
            _tokens_at(o, tracker.window_end) - _tokens_at(o, tracker.window_start)
            for o in tracker.obs
        ]
    # Users that started early keep decoding, slowly, while the later prompts prefill.
    early_rates: List[Optional[float]] = [None] * n
    if tracker.window_start is not None and n > 1:
        for i, o in enumerate(tracker.obs):
            span = tracker.window_start - first_t[i]
            if span > 1.0:
                early_rates[i] = (_tokens_at(o, tracker.window_start) - o[0][1]) / span

    per_user = [
        UserResult(
            prompt_tokens=m["prompt_tokens"],
            cached_tokens=m.get("cached_tokens", 0),
            completion_tokens=m["completion_tokens"],
            ttft=first_t[i] - t0,
            window_tokens=window_tokens[i],
            decode_tps_while_others_prefill=early_rates[i],
            spec_tokens=tracker.spec[i][0],
            spec_verify_ct=tracker.spec[i][1],
        )
        for i, m in enumerate(tracker.metas)
    ]
    ttfts = [u.ttft for u in per_user]
    first = min(per_user, key=lambda u: u.ttft)
    verify = sum(u.spec_verify_ct for u in per_user)
    duration = tracker.window_end - tracker.window_start if concurrent else None
    early = [r for r in early_rates if r is not None]
    return BurstResult(
        context_len=context_len,
        users=n,
        repeat=repeat,
        max_new_tokens=max_new_tokens,
        all_concurrent=concurrent,
        ttft_first=min(ttfts),
        ttft_last=max(ttfts),
        ttft_mean=statistics.mean(ttfts),
        prefill_tps_first_user=(first.prompt_tokens - first.cached_tokens) / first.ttft,
        prefill_tps_all=sum(u.prompt_tokens - u.cached_tokens for u in per_user)
        / max(ttfts),
        decode_window_s=duration,
        decode_tps_per_user=sum(window_tokens) / n / duration if duration else None,
        decode_tps_total=sum(window_tokens) / duration if duration else None,
        decode_tps_slowest_user=min(window_tokens) / duration if duration else None,
        decode_tps_while_others_prefill=statistics.mean(early) if early else None,
        accept_length=sum(u.spec_tokens for u in per_user) / verify if verify else None,
        per_user=per_user,
    )


def run_burst(
    base_url: str,
    source: PromptSource,
    context_len: int,
    users: int,
    output_len: int,
    max_new_tokens: int,
    sampling_params: Dict[str, Any],
    repeat: int,
    timeout: float,
) -> BurstResult:
    flush_cache(base_url)
    prompts = [source.make(context_len) for _ in range(users)]
    sp = {**sampling_params, "max_new_tokens": max_new_tokens, "ignore_eos": True}
    t0, tracker = asyncio.run(
        _run_burst_async(
            base_url=base_url,
            prompts=prompts,
            sampling_params=sp,
            output_len=output_len,
            timeout=timeout,
        )
    )
    result = _summarize_burst(
        t0=t0,
        tracker=tracker,
        context_len=context_len,
        repeat=repeat,
        max_new_tokens=max_new_tokens,
    )
    if result.accept_length is None:
        # Aborted streams in a multi-request batch may never carry the spec counters;
        # fall back to the server's recent average, as bench_one_batch_server does.
        states = requests.get(f"{base_url}/server_info", timeout=60).json()[
            "internal_states"
        ]
        accept = [
            s["avg_spec_accept_length"]
            for s in states
            if s.get("avg_spec_accept_length")
        ]
        if accept:
            result = msgspec.structs.replace(
                result, accept_length=statistics.mean(accept)
            )
    return result


# ---------------------------------------------------------------- steady


def run_steady(
    base_url: str,
    cap: ServerCapacity,
    context_len: int,
    users: int,
    output_len: int,
    sampling_params: Dict[str, Any],
    args: argparse.Namespace,
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    """One `sglang.bench_serving` run; returns its JSONL result row."""
    out_file = out_dir / "bench_serving.jsonl"
    tag = f"L{context_len}_N{users}"
    # bench_serving's own sampling_params would be replaced wholesale by this, so it
    # carries max_new_tokens and ignore_eos too (every request has the same output len).
    extra = {
        "sampling_params": {
            **sampling_params,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        }
    }
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--base-url", base_url,
        "--dataset-name", args.steady_dataset,
        "--random-input-len", str(context_len),
        "--random-output-len", str(output_len),
        "--random-range-ratio", "1",
        "--num-prompts", str(max(args.steady_prompts_factor * users, users)),
        "--max-concurrency", str(users),
        "--request-rate", "inf",
        "--flush-cache",
        "--warmup-requests", "1",
        "--seed", str(args.seed),
        "--extra-request-body", json.dumps(extra),
        "--output-file", str(out_file),
        "--tag", tag,
    ]  # fmt: skip
    if cap.tokenizer_path:
        cmd += ["--tokenizer", cap.tokenizer_path]
    print("  $", " ".join(cmd[:3]), "...", f"--max-concurrency {users}")
    log = out_dir / f"bench_serving_{tag}.log"
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"  bench_serving failed (exit {proc.returncode}), see {log}")
        return None
    rows = [
        json.loads(line) for line in out_file.read_text().splitlines() if line.strip()
    ]
    row = next((r for r in reversed(rows) if r.get("tag") == tag), None)
    if row is None:
        print(f"  no bench_serving result row tagged {tag}, see {log}")
    return row


# ---------------------------------------------------------------- report


def fmt(v: Optional[float], spec: str = ".1f", none: str = "-") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return none
    return format(v, spec)


def median_burst(results: List[BurstResult]) -> Dict[str, Any]:
    """Median over repeats of each numeric field."""
    out: Dict[str, Any] = {
        "context_len": results[0].context_len,
        "users": results[0].users,
        "max_new_tokens": results[0].max_new_tokens,
        "all_concurrent": all(r.all_concurrent for r in results),
    }
    for key in (
        "ttft_first", "ttft_last", "prefill_tps_first_user", "prefill_tps_all",
        "decode_tps_per_user", "decode_tps_total", "decode_tps_slowest_user",
        "decode_tps_while_others_prefill", "accept_length",
    ):  # fmt: skip
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        out[key] = statistics.median(vals) if vals else None
    return out


def _report_header(
    args: argparse.Namespace,
    cap: ServerCapacity,
    gpus: List[Dict[str, str]],
    sampling_params: Dict[str, Any],
) -> List[str]:
    si = cap.raw_server_info
    lines = [
        "# SGLang capacity benchmark",
        "",
        f"- **Date:** {datetime.datetime.now():%Y-%m-%d %H:%M}",
        f"- **Server:** `{args.base_url}`, sglang {si.get('version', '?')}",
        f"- **Model:** `{cap.model_path}`",
        f"- **Parallelism:** TP {si.get('tp_size')}, DP {cap.dp_size}, PP {si.get('pp_size')}, "
        f"EP {si.get('ep_size')}",
        f"- **KV cache:** {cap.kv_pool_tokens:,} tokens, dtype `{si.get('kv_cache_dtype')}`, "
        f"page size {cap.page_size}",
        f"- **Max running requests:** {cap.max_running_requests} per DP rank",
        f"- **Context length:** {cap.context_len:,} (max input {cap.max_input_len:,})",
        f"- **Speculative:** `{si.get('speculative_algorithm')}` "
        f"(draft tokens {si.get('speculative_num_draft_tokens')})",
        f"- **Prefill:** chunked prefill size {si.get('chunked_prefill_size')}, "
        f"attention backend `{si.get('attention_backend')}`",
        f"- **Workload:** prompt `{args.prompt}` from `{' '.join(map(str, args.corpus))}`, "
        f"{args.output_len} decode tokens measured per user, sampling "
        f"`{json.dumps(sampling_params)}`, repeats {args.repeats}",
    ]
    if gpus:
        names = ", ".join(
            f"{g['name']} ({g['memory.total']}, {g['power.limit']})" for g in gpus
        )
        lines.append(f"- **GPUs on this host:** {names}")
    lines += ["", f"Launch command: `{si.get('launch_command', '?')}`", ""]
    return lines


def _report_capacity(cap: ServerCapacity, cells: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "## How many users fit",
        "",
        "Estimated from the KV pool and the running-request cap (hybrid / mamba models fold "
        "their state slots into the cap). Each user reserves its prompt plus the burst's "
        "max_new_tokens. The burst run confirms it: 'all concurrent = no' means some users "
        "queued.",
        "",
        "| Context (input tokens) | Users | KV tokens per user | Users that fit | Limited by |",
        "|---:|---:|---:|---:|---|",
    ]
    for c in cells:
        L, budget = c["context_len"], c["max_new_tokens"]
        lines.append(
            f"| {L:,} | {c['users']} | {cap.tokens_per_user(L, budget):,} | "
            f"{cap.users_that_fit(L, budget)} | {cap.limited_by(L, budget)} |"
        )
    return lines


def _report_burst(bursts: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "",
        "## Burst: N users arrive together",
        "",
        "- **TTFT** is time to first token from the common start. The last user waits for "
        "every prefill.",
        "- **Prefill, 1st user** is that user's prompt tokens over its own TTFT. With N = 1 "
        "it is the single-user prefill speed.",
        "- **Prefill, all** is all prompt tokens over the last user's TTFT.",
        "- **Decode per user / total / slowest** come from the window where all N users "
        "decode together, at a depth of about L tokens.",
        "- **Decode while others prefill** is the rate early users get while later prompts "
        "are still prefilling.",
        "",
        "| Context | Users | All concurrent | TTFT 1st / last (s) | Prefill tok/s, 1st user "
        "| Prefill tok/s, all | Decode tok/s per user | Decode tok/s total "
        "| Slowest user tok/s | Decode tok/s while others prefill | Accept len |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in bursts:
        b = c["burst"]
        lines.append(
            f"| {b['context_len']:,} | {b['users']} | {'yes' if b['all_concurrent'] else '**no**'} "
            f"| {fmt(b['ttft_first'], '.2f')} / {fmt(b['ttft_last'], '.2f')} "
            f"| {fmt(b['prefill_tps_first_user'], ',.0f')} | {fmt(b['prefill_tps_all'], ',.0f')} "
            f"| {fmt(b['decode_tps_per_user'])} | {fmt(b['decode_tps_total'])} "
            f"| {fmt(b['decode_tps_slowest_user'])} | {fmt(b['decode_tps_while_others_prefill'])} "
            f"| {fmt(b['accept_length'], '.2f')} |"
        )
    return lines


def _report_steady(
    args: argparse.Namespace, steadies: List[Dict[str, Any]]
) -> List[str]:
    lines = [
        "",
        "## Steady state: `sglang.bench_serving`, concurrency N",
        "",
        f"Each row uses `--num-prompts {args.steady_prompts_factor}N --max-concurrency N "
        f"--dataset-name {args.steady_dataset}` with {args.output_len} output tokens. "
        "Per-user decode is 1000 / median TPOT. TPOT here includes stalls while other users "
        "prefill. P99 ITL is the gap between two tokens that 1% of gaps exceed: a stall.",
        "",
        "| Context | Users | Median TTFT (s) | P99 TTFT (s) | Median TPOT (ms) | P99 TPOT (ms) "
        "| P99 ITL (ms) | Decode tok/s per user | Output tok/s | Input tok/s | Accept len |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in steadies:
        s = c["steady"]
        tpot = s.get("median_tpot_ms")
        lines.append(
            f"| {c['context_len']:,} | {c['users']} | {fmt(s['median_ttft_ms'] / 1e3, '.2f')} "
            f"| {fmt(s['p99_ttft_ms'] / 1e3, '.2f')} | {fmt(tpot, '.2f')} "
            f"| {fmt(s.get('p99_tpot_ms'), '.2f')} | {fmt(s.get('p99_itl_ms'), '.0f')} "
            f"| {fmt(1e3 / tpot if tpot else None)} "
            f"| {fmt(s.get('output_throughput'))} | {fmt(s.get('input_throughput'), ',.0f')} "
            f"| {fmt(s.get('accept_length'), '.2f')} |"
        )
    return lines


def _report_slo(args: argparse.Namespace, bursts: List[Dict[str, Any]]) -> List[str]:
    lines = [
        "",
        "## Users within the SLO",
        "",
        f"SLO: per-user decode >= {args.slo_decode or 0} tok/s and last-user TTFT <= "
        f"{args.slo_ttft or 'inf'} s (burst numbers).",
        "",
    ]
    for L in sorted({c["context_len"] for c in bursts}):
        ok = [
            c["burst"]["users"]
            for c in bursts
            if c["context_len"] == L
            and c["burst"]["all_concurrent"]
            and (c["burst"]["decode_tps_per_user"] or 0) >= (args.slo_decode or 0)
            and (not args.slo_ttft or c["burst"]["ttft_last"] <= args.slo_ttft)
        ]
        lines.append(f"- {L:,} tokens: **{max(ok) if ok else 0}** users")
    return lines


def write_report(
    out_dir: Path,
    args: argparse.Namespace,
    cap: ServerCapacity,
    gpus: List[Dict[str, str]],
    sampling_params: Dict[str, Any],
    cells: List[Dict[str, Any]],
) -> Path:
    lines = _report_header(
        args=args, cap=cap, gpus=gpus, sampling_params=sampling_params
    )
    lines += _report_capacity(cap=cap, cells=cells)
    bursts = [c for c in cells if c.get("burst")]
    if bursts:
        lines += _report_burst(bursts)
    steadies = [c for c in cells if c.get("steady")]
    if steadies:
        lines += _report_steady(args=args, steadies=steadies)
    if bursts and (args.slo_decode or args.slo_ttft):
        lines += _report_slo(args=args, bursts=bursts)
    path = out_dir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------- plan


def users_for(spec: str, top: int, max_users: int) -> List[int]:
    if spec != "auto":
        return [int(u) for u in spec.split(",")]
    top = max(1, min(top, max_users))
    out = [1 << i for i in range(top.bit_length()) if (1 << i) <= top]
    return out + ([top] if top not in out else [])


def build_plan(
    args: argparse.Namespace, cap: ServerCapacity, budget: Callable[[int], int]
) -> List[Tuple[int, int]]:
    """(context_len, users) cells to run, skipping those estimated not to fit."""
    plan: List[Tuple[int, int]] = []
    max_concurrent = cap.dp_size * cap.max_running_requests
    for tok in (t.strip().lower() for t in args.context_lens.split(",")):
        if tok == "fit":
            resolve = lambda n: cap.max_input_for_users(n, budget(n))
            top = max_concurrent
        elif tok == "full":
            resolve = lambda n: cap.full_input_len(budget(n))
            top = max(1, cap.users_that_fit(resolve(2), budget(2)))
        else:
            L = int(float(tok[:-1]) * 1024) if tok.endswith("k") else int(tok)
            resolve = lambda n, L=L: L
            top = max(1, cap.users_that_fit(L, budget(2)))
        for n in users_for(args.users, top, args.max_users):
            L = resolve(n)
            longest = cap.full_input_len(budget(n))
            if L > longest:
                print(
                    f"skip {tok} N={n}: the server accepts prompts up to {longest:,} tokens "
                    f"with {budget(n):,} new tokens (use 'full' for the longest)"
                )
                continue
            fit = cap.users_that_fit(L, budget(n)) if L > 0 else 0
            if n > fit and not args.include_unfit:
                print(
                    f"skip {tok} N={n}: L={L:,} fits {fit} users (--include-unfit to run)"
                )
                continue
            if L > 0:
                plan.append((L, n))
    return list(dict.fromkeys(plan))


# ---------------------------------------------------------------- main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument(
        "--context-lens",
        default="1k,8k,32k,full",
        help="Prompt lengths in tokens: plain ints, 'k' suffix (x1024), 'full' (longest "
        "prompt the server accepts), or 'fit' (for each user count, the longest prompt at "
        "which that many users fit in the KV pool together).",
    )
    p.add_argument(
        "--users",
        default="auto",
        help="Comma list of concurrent users, or 'auto': 1, 2, 4, ... up to what fits.",
    )
    p.add_argument("--max-users", type=int, default=64, help="Cap for --users auto.")
    p.add_argument(
        "--output-len",
        type=int,
        default=512,
        help="Decode tokens measured per user while all users decode together "
        "(burst), and the output length of --steady requests.",
    )
    p.add_argument(
        "--decode-headroom",
        type=int,
        default=2048,
        help="Extra max_new_tokens per burst request when N > 1. Early users keep decoding "
        "while later prompts prefill; this keeps them from finishing before the "
        "all-users window starts. Counted in the KV fit.",
    )
    p.add_argument(
        "--repeats", type=int, default=1, help="Burst repeats; report medians."
    )
    p.add_argument(
        "--sampling-params",
        default="{}",
        help='JSON merged into every request, e.g. \'{"temperature":0.6,"top_p":0.95}\'. '
        "Matters for speculative decoding (acceptance). Empty = server defaults.",
    )
    p.add_argument("--prompt", choices=["text", "random"], default="text")
    p.add_argument(
        "--corpus",
        nargs="+",
        default=["sharegpt"],
        help="'sharegpt' (default, HF-cached like bench_serving) or files / dirs "
        "(.md/.mdx/.txt/.rst) whose text feeds --prompt text.",
    )
    p.add_argument(
        "--no-burst", action="store_true", help="Skip the burst measurement."
    )
    p.add_argument(
        "--steady", action="store_true", help="Also run sglang.bench_serving."
    )
    p.add_argument("--steady-prompts-factor", type=int, default=5)
    p.add_argument(
        "--steady-dataset", default="random", help="bench_serving --dataset-name"
    )
    p.add_argument(
        "--include-unfit",
        action="store_true",
        help="Also run cells the capacity estimate says do not fit (shows queueing).",
    )
    p.add_argument(
        "--slo-decode", type=float, default=None, help="Min per-user decode tok/s."
    )
    p.add_argument(
        "--slo-ttft", type=float, default=None, help="Max last-user TTFT (s)."
    )
    p.add_argument(
        "--timeout", type=float, default=3 * 3600, help="Per burst, seconds."
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: benchmark/capacity/results/<timestamp>.",
    )
    args = p.parse_args()
    args.base_url = args.base_url.rstrip("/")
    return args


def run_cell(
    args: argparse.Namespace,
    cap: ServerCapacity,
    source: Optional[PromptSource],
    sampling_params: Dict[str, Any],
    context_len: int,
    users: int,
    max_new_tokens: int,
    out_dir: Path,
) -> Dict[str, Any]:
    cell: Dict[str, Any] = {
        "context_len": context_len,
        "users": users,
        "max_new_tokens": max_new_tokens,
    }
    if source is not None:
        reps = []
        for r in range(args.repeats):
            print(
                f"burst L={context_len:,} N={users} repeat {r + 1}/{args.repeats} ...",
                flush=True,
            )
            res = run_burst(
                base_url=args.base_url,
                source=source,
                context_len=context_len,
                users=users,
                output_len=args.output_len,
                max_new_tokens=max_new_tokens,
                sampling_params=sampling_params,
                repeat=r,
                timeout=args.timeout,
            )
            reps.append(res)
            print(
                f"  TTFT {res.ttft_first:.2f}/{res.ttft_last:.2f} s, "
                f"prefill {res.prefill_tps_all:,.0f} tok/s, decode/user "
                f"{fmt(res.decode_tps_per_user)} tok/s (while others prefill "
                f"{fmt(res.decode_tps_while_others_prefill)}), concurrent={res.all_concurrent}"
            )
            with open(out_dir / "results.jsonl", "a") as f:
                f.write(
                    json.dumps({"kind": "burst", **msgspec.to_builtins(res)}) + "\n"
                )
        cell["burst"] = median_burst(reps)
    if args.steady:
        print(f"steady L={context_len:,} N={users} (bench_serving) ...", flush=True)
        row = run_steady(
            base_url=args.base_url,
            cap=cap,
            context_len=context_len,
            users=users,
            output_len=args.output_len,
            sampling_params=sampling_params,
            args=args,
            out_dir=out_dir,
        )
        if row:
            cell["steady"] = row
            print(
                f"  median TTFT {row['median_ttft_ms'] / 1e3:.2f} s, median TPOT "
                f"{row['median_tpot_ms']:.2f} ms, output {row['output_throughput']:.1f} tok/s"
            )
    return cell


def main() -> None:
    args = parse_args()
    sampling_params = json.loads(args.sampling_params)
    out_dir = args.out_dir or (
        Path(__file__).parent
        / "results"
        / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = fetch_server_capacity(args.base_url)
    gpus = local_gpus()
    (out_dir / "server_info.json").write_text(
        json.dumps(
            {"server_info": cap.raw_server_info, "gpus": gpus}, indent=1, default=str
        )
    )
    print(
        f"model {cap.model_path}: KV pool {cap.kv_pool_tokens:,} tokens, "
        f"max running {cap.max_running_requests}/DP x {cap.dp_size} DP, "
        f"context {cap.context_len:,}"
    )

    def budget(users: int) -> int:
        return args.output_len + (args.decode_headroom if users > 1 else 0)

    plan = build_plan(args=args, cap=cap, budget=budget)
    if not plan:
        sys.exit("nothing to run: no (context, users) cell fits")
    print("cells:", ", ".join(f"L={L:,} N={n}" for L, n in plan))

    source = None
    if not args.no_burst:
        source = PromptSource(
            base_url=args.base_url, kind=args.prompt, corpus=args.corpus, seed=args.seed
        )
        # Warm-up so the first cell does not pay one-time costs.
        run_burst(
            base_url=args.base_url,
            source=source,
            context_len=256,
            users=1,
            output_len=16,
            max_new_tokens=16,
            sampling_params=sampling_params,
            repeat=-1,
            timeout=args.timeout,
        )

    cells: List[Dict[str, Any]] = []
    for L, n in plan:
        cells.append(
            run_cell(
                args=args,
                cap=cap,
                source=source,
                sampling_params=sampling_params,
                context_len=L,
                users=n,
                max_new_tokens=budget(n),
                out_dir=out_dir,
            )
        )
        # Rewrite after every cell so an interrupted sweep still leaves a report.
        report = write_report(
            out_dir=out_dir,
            args=args,
            cap=cap,
            gpus=gpus,
            sampling_params=sampling_params,
            cells=cells,
        )

    print(f"\nreport: {report}\nraw:    {out_dir / 'results.jsonl'}\n")
    print(report.read_text())


if __name__ == "__main__":
    main()
