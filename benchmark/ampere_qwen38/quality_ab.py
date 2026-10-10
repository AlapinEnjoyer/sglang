"""Output-quality A/B for serving configs: generate on a server under test, then
score the saved outputs on a reference server by teacher forcing.

An exact speculative sampler leaves the output distribution unchanged, so its
samples have the same expected NLL under the reference as reference samples do.
A state or verify bug shows up as a higher NLL; for greedy runs, as a lower
top-1 agreement with the reference argmax.

  quality_ab.py gen    --url http://127.0.0.1:8000 --mode sample --out runs/X.sample.jsonl
  quality_ab.py gen    --set long --ctx 65536,131072 --mode sample --out runs/X.long.jsonl
  quality_ab.py score  --url http://127.0.0.1:8000 runs/*.jsonl      # reference server
  quality_ab.py report --baseline runs/ref.sample.scored.jsonl runs/*.scored.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer

MODEL = "/home/ubuntu/experiments/inference/models/Swift-1.5-Qwen3.8-27b-INT4"
SHAREGPT = Path.home() / (
    ".cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/"
    "snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
CODE = Path(__file__).with_name("code_prompts.jsonl")
REPO_SRC = Path(__file__).resolve().parents[2] / "python/sglang/srt"
FOLLOW_UPS = [
    "Can you go deeper on the weakest part of your answer and fix anything that is wrong?",
    "Now summarize everything we discussed in five concise bullet points.",
]
# Qwen3.8 thinking-mode recommendation; min_p and the penalties are sent explicitly
# so a server-side generation_config default cannot change them.
SAMPLE_PARAMS = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}
# Room left in a long prompt for the chat template and the question.
LONG_PROMPT_MARGIN_TOKENS = 512


def _first_human_turns(*, min_chars: int, max_chars: int) -> list[str]:
    convs = json.loads(SHAREGPT.read_text())
    out = []
    for conv in convs:
        turns = conv.get("conversations") or []
        if not turns or turns[0].get("from") != "human":
            continue
        text = turns[0]["value"].strip()
        if min_chars <= len(text) <= max_chars and text.isascii():
            out.append(text)
    return out


def build_prompts(*, seed: int, n_chat: int, n_code: int, n_conv: int) -> dict:
    rng = random.Random(seed)
    chat = rng.sample(_first_human_turns(min_chars=200, max_chars=3000), n_chat)
    long_turns = rng.sample(_first_human_turns(min_chars=6000, max_chars=30000), n_conv)
    code_rows = [json.loads(line) for line in CODE.read_text().splitlines()]
    code = [r["conversations"][0]["content"] for r in rng.sample(code_rows, n_code)]
    single = [("chat", t) for t in chat] + [("code", t) for t in code]
    return {"single": single, "conv": long_turns}


def build_long_prompts(
    *, tok, src: Path, seed: int, n_docs: int, ctxs: list[int]
) -> list[tuple]:
    """Repo sources concatenated to each target length, with a question about a
    function defined in the first 10% of the context."""
    files = sorted(p for p in src.rglob("*.py") if p.stat().st_size > 4096)
    out = []
    for ctx in ctxs:
        for d in range(n_docs):
            rng = random.Random(seed * 1000 + ctx + d)
            order = files[:]
            rng.shuffle(order)
            parts, n_tok, early_defs = [], 0, []
            budget = ctx - LONG_PROMPT_MARGIN_TOKENS
            for f in order:
                text = f.read_text(errors="ignore")
                if not text.isascii():
                    continue
                block = f"### File: {f.relative_to(src.parents[1])}\n```python\n{text}\n```\n\n"
                block_tok = len(tok.encode(block, add_special_tokens=False))
                if n_tok + block_tok > budget:
                    continue
                if n_tok < ctx // 10:
                    rel = f.relative_to(src.parents[1])
                    defs = re.findall(r"^\s*def (\w{6,})\(", text, re.M)
                    early_defs += [(str(rel), name) for name in defs]
                parts.append(block)
                n_tok += block_tok
                if budget - n_tok < 2048:
                    break
            path, fn = rng.choice(early_defs)
            question = (
                f"The document above contains source files. In `{path}`, near the "
                f"beginning, what does `{fn}` do? Then name every other place in the "
                "document that calls it or depends on its behavior, and explain how."
            )
            out.append((ctx, "".join(parts) + question))
    return out


def _generate(url: str, *, input_ids: list[int], params: dict, max_tokens: int) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {**params, "max_new_tokens": max_tokens},
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=3600)
    r.raise_for_status()
    return r.json()


def _strip_thinking(text: str) -> str:
    return text.split("</think>")[-1].strip()


def cmd_gen(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.model)
    params = SAMPLE_PARAMS if args.mode == "sample" else {"temperature": 0.0}
    template_kwargs = {
        "enable_thinking": True,
        "reasoning_effort": args.reasoning_effort,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def run(sample_id: str, kind: str, turn: int, messages: list[dict]) -> str:
        prompt_text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        t0 = time.time()
        res = _generate(
            args.url, input_ids=prompt_ids, params=params, max_tokens=args.max_tokens
        )
        meta = res["meta_info"]
        row = {
            "id": sample_id,
            "kind": kind,
            "turn": turn,
            "prompt_hash": hashlib.sha1(str(prompt_ids).encode()).hexdigest()[:12],
            "prompt_ids": prompt_ids,
            "output_ids": res["output_ids"],
            "cached_tokens": meta.get("cached_tokens"),
            "finish_reason": meta.get("finish_reason"),
            "seconds": round(time.time() - t0, 2),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"{sample_id} turn={turn} prompt={len(prompt_ids)} out={len(res['output_ids'])} "
            f"cached={row['cached_tokens']} {row['seconds']}s",
            flush=True,
        )
        return res["text"]

    def run_conv(sample_id: str, kind: str, first: str) -> None:
        messages = [{"role": "user", "content": first}]
        for turn in range(1 + len(FOLLOW_UPS)):
            reply = run(sample_id, kind, turn, messages)
            if turn < len(FOLLOW_UPS):
                messages = messages + [
                    {"role": "assistant", "content": _strip_thinking(reply)},
                    {"role": "user", "content": FOLLOW_UPS[turn]},
                ]

    out_path.write_text("")
    if args.set == "long":
        ctxs = [int(c) for c in args.ctx.split(",")]
        long_prompts = build_long_prompts(
            tok=tok,
            src=Path(args.src_dir),
            seed=args.seed,
            n_docs=args.n_docs,
            ctxs=ctxs,
        )
        for i, (ctx, text) in enumerate(long_prompts):
            run_conv(f"l{i:02d}", f"long{ctx // 1024}k", text)
        return
    prompts = build_prompts(
        seed=args.seed, n_chat=args.n_chat, n_code=args.n_code, n_conv=args.n_conv
    )
    for i, (kind, text) in enumerate(prompts["single"]):
        run(f"s{i:02d}", kind, 0, [{"role": "user", "content": text}])
    for i, first in enumerate(prompts["conv"]):
        run_conv(f"c{i:02d}", "conv", first)


def _score_row(url: str, row: dict) -> dict:
    prompt, output = row["prompt_ids"], row["output_ids"]
    payload = {
        "input_ids": prompt + output,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
        "return_logprob": True,
        # The first returned position never has a logprob, so start one token early.
        "logprob_start_len": len(prompt) - 1,
        "top_logprobs_num": 1,
    }
    r = requests.post(f"{url}/generate", json=payload, timeout=3600)
    r.raise_for_status()
    meta = r.json()["meta_info"]
    lps = meta["input_token_logprobs"]
    tops = meta["input_top_logprobs"]
    # Align on token ids: entry j scores input position logprob_start_len + j.
    offset = len(lps) - len(output)
    assert offset >= 0, (len(lps), len(output))
    nll, agree = [], 0
    for j, tid in enumerate(output):
        lp, lp_tid, _ = lps[offset + j]
        assert lp_tid == tid, (row["id"], j, lp_tid, tid)
        nll.append(-lp)
        top = tops[offset + j]
        agree += int(top is not None and top[0][1] == tid)
    return {
        **{k: v for k, v in row.items() if k not in ("prompt_ids", "output_ids")},
        "n": len(output),
        "nll_sum": float(np.sum(nll)),
        "top1_agree": agree,
        "output_ids": output,
    }


def cmd_score(args) -> None:
    for path in map(Path, args.files):
        if path.name.endswith(".scored.jsonl"):
            continue
        out = path.with_name(path.name.replace(".jsonl", ".scored.jsonl"))
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        with out.open("w") as f:
            for row in rows:
                if not row["output_ids"]:
                    continue
                f.write(json.dumps(_score_row(args.url, row)) + "\n")
        print(f"scored {path} -> {out}", flush=True)


def _load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _bootstrap_ci(per_sample: np.ndarray, weights: np.ndarray, iters: int = 4000):
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(per_sample), size=(iters, len(per_sample)))
    stats = (per_sample[idx] * weights[idx]).sum(1) / weights[idx].sum(1)
    return np.percentile(stats, [2.5, 97.5])


def _first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _finished(row: dict) -> bool:
    reason = row.get("finish_reason")
    return isinstance(reason, dict) and reason.get("type") == "stop"


def cmd_report(args) -> None:
    base = (
        {(r["id"], r["turn"]): r for r in _load(args.baseline)} if args.baseline else {}
    )
    print(
        f"{'file':48s} {'n':>4s} {'tokens':>7s} {'nll/tok':>8s} {'95% CI':>17s} "
        f"{'top1%':>6s} {'cached':>8s} {'div@med':>8s} {'stop%':>6s}"
    )
    for path in args.files:
        all_rows = _load(path)
        groups = [("", all_rows)]
        if args.by_kind:
            kinds = sorted({r["kind"] for r in all_rows})
            groups += [
                (f"  {k}", [r for r in all_rows if r["kind"] == k]) for k in kinds
            ]
        for label, rows in groups:
            _report_rows(label or Path(path).name, rows, base)


def _report_rows(label: str, rows: list[dict], base: dict) -> None:
    n = np.array([r["n"] for r in rows], dtype=float)
    nll = np.array([r["nll_sum"] for r in rows]) / n
    lo, hi = _bootstrap_ci(nll, n)
    top1 = 100 * sum(r["top1_agree"] for r in rows) / n.sum()
    cached = sum(r["cached_tokens"] or 0 for r in rows if r["turn"] > 0)
    divs = [
        _first_divergence(r["output_ids"], base[(r["id"], r["turn"])]["output_ids"])
        for r in rows
        if (r["id"], r["turn"]) in base
        and base[(r["id"], r["turn"])]["prompt_hash"] == r["prompt_hash"]
    ]
    div = f"{int(np.median(divs))}" if divs else "-"
    stop = 100 * sum(_finished(r) for r in rows) / len(rows)
    print(
        f"{label:48s} {len(rows):4d} {int(n.sum()):7d} "
        f"{(nll * n).sum() / n.sum():8.4f} [{lo:.4f},{hi:.4f}] {top1:6.2f} "
        f"{cached:8d} {div:>8s} {stop:6.1f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--url", default="http://127.0.0.1:8000")
    g.add_argument("--mode", choices=["sample", "greedy"], required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", default=MODEL, help="tokenizer / chat template path")
    g.add_argument("--set", choices=["short", "long"], default="short")
    g.add_argument("--ctx", default="65536,131072", help="long set: prompt tokens")
    g.add_argument(
        "--src-dir",
        default=str(REPO_SRC),
        help="long set: python/sglang/srt tree the documents are cut from; pin it "
        "(e.g. a git archive) so every config sees identical prompts",
    )
    g.add_argument("--n-docs", type=int, default=4, help="long set: docs per length")
    g.add_argument(
        "--reasoning-effort", default="xhigh", choices=["xhigh", "medium", "low"]
    )
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n-chat", type=int, default=24)
    g.add_argument("--n-code", type=int, default=8)
    g.add_argument("--n-conv", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--url", default="http://127.0.0.1:8000")
    s.add_argument("files", nargs="+")
    r = sub.add_parser("report")
    r.add_argument(
        "--baseline", help="scored greedy file to measure divergence against"
    )
    r.add_argument("--by-kind", action="store_true", help="add a row per prompt kind")
    r.add_argument("files", nargs="+")
    args = p.parse_args()
    {"gen": cmd_gen, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
