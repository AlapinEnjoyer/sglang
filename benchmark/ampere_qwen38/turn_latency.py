"""Agent-turn latency: TTFT of (cached long prefix + small new delta), the common case in
tool-calling loops. Prefills the prefix once, then times deltas of several sizes.
Usage: turn_latency.py [--port 8000] [--prefix 30000] [--deltas 128,512,2048] [--reps 3]
"""

import argparse
import random
import time

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--prefix", type=int, default=30000)
    p.add_argument("--deltas", default="128,512,2048")
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}"
    rng = random.Random(0)
    tok = lambda n: [rng.randint(1000, 100000) for _ in range(n)]

    def ttft(ids):
        t = time.time()
        r = requests.post(
            url + "/generate",
            json={
                "input_ids": ids,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            },
        ).json()
        return time.time() - t, r["meta_info"].get("cached_tokens", 0)

    requests.post(url + "/flush_cache")
    prefix = tok(args.prefix)
    t, _ = ttft(prefix)
    print(f"cold prefix {args.prefix}: {t:.2f}s")
    for d in map(int, args.deltas.split(",")):
        times, cached = [], []
        for _ in range(args.reps):
            t, c = ttft(prefix + tok(d))
            times.append(t)
            cached.append(c)
        print(
            f"delta {d:5d}: ttft median {sorted(times)[len(times) // 2] * 1e3:7.1f} ms  cached {cached}"
        )


if __name__ == "__main__":
    main()
