"""Per-kernel breakdown of a torch-profiler trace, optionally restricted to a time window.

Usage: trace_breakdown.py TRACE.json.gz [--skip-ms N] [--top K]
Kernels longer than --max-kernel-us are dropped (filters out prefill chunks caught in a decode window).
"""
import argparse
import collections
import gzip
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace")
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--max-kernel-us", type=float, default=1e12)
    p.add_argument("--name-width", type=int, default=110)
    args = p.parse_args()
    with gzip.open(args.trace) as f:
        events = json.load(f)["traceEvents"]
    kernels = [
        e for e in events
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("dur", 0) <= args.max_kernel_us
    ]
    agg = collections.defaultdict(lambda: [0, 0.0])
    for e in kernels:
        agg[e["name"]][0] += 1
        agg[e["name"]][1] += e["dur"]
    total = sum(v[1] for v in agg.values())
    span = (max(e["ts"] + e["dur"] for e in kernels) - min(e["ts"] for e in kernels)) if kernels else 0
    print(f"kernels={len(kernels)} gpu_busy={total/1e3:.2f}ms span={span/1e3:.2f}ms")
    print(f"{'count':>6} {'total_ms':>9} {'avg_us':>8} {'share':>6}  name")
    for name, (cnt, dur) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: args.top]:
        print(f"{cnt:6d} {dur/1e3:9.2f} {dur/cnt:8.1f} {100*dur/total:5.1f}%  {name[:args.name_width]}")


if __name__ == "__main__":
    main()
