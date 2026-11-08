"""Background workloads for studio.py, run as subprocesses.

studio_workloads.py multiturn [--priority N] [--session-ids] <bench_multiturn.py args>
    bench_multiturn.py with a session_id (and optionally a priority) on every request.
studio_workloads.py decisions --rate R --stop FILE --out FILE [--words N]
    /v1/decisions requests with unique inputs at a Poisson rate until FILE exists.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

MULTITURN = Path(__file__).resolve().parents[1] / "hicache" / "bench_multiturn.py"
WORDS = (
    "account billing invoice payment refund order shipping delivery customer support "
    "integration error failure server network latency timeout database query index "
    "contract renewal pricing discount subscription upgrade downgrade cancel request "
    "report quarter revenue forecast budget expense audit compliance policy security "
    "password login token session browser mobile desktop update release version bug"
).split()


def run_multiturn(argv: list) -> None:
    priority = None
    if "--priority" in argv:
        i = argv.index("--priority")
        priority = int(argv[i + 1])
        del argv[i : i + 2]
    session_ids = "--session-ids" in argv
    if session_ids:
        argv.remove("--session-ids")
    sys.argv = [str(MULTITURN)] + argv
    sys.path.insert(0, str(MULTITURN.parent))
    import bench_multiturn

    original = bench_multiturn.gen_payload

    def gen_payload(input_ids, output_len, lora_path=""):
        payload = original(input_ids, output_len, lora_path)
        if session_ids:
            # A client's first 64 prompt tokens are unique to it and never change.
            key = ",".join(map(str, input_ids[:64])).encode()
            payload["session_id"] = "s-" + hashlib.sha1(key).hexdigest()[:12]
        if priority is not None:
            payload["priority"] = priority
        return payload

    bench_multiturn.gen_payload = gen_payload
    # Run the script's own main block in its module namespace, so the patched
    # gen_payload is the one its code looks up.
    main_src = MULTITURN.read_text().split('if __name__ == "__main__":', 1)[1]
    exec("if True:" + main_src, vars(bench_multiturn))


async def _one_decision(session, url, rng, words, latencies):
    import aiohttp

    text = " ".join(rng.choice(WORDS) for _ in range(words))
    body = {
        "input": f"Ticket {rng.getrandbits(64):x}: {text}",
        "questions": [
            {
                "id": "urgent",
                "type": "yes_no",
                "question": "The customer needs an answer today.",
            }
        ],
    }
    t0 = time.perf_counter()
    try:
        async with session.post(url, json=body) as r:
            await r.read()
            ok = r.status == 200
    except aiohttp.ClientError:
        ok = False
    latencies.append((time.perf_counter() - t0, ok))


async def _decisions(args) -> None:
    import aiohttp

    rng = random.Random(1)
    latencies, tasks = [], []
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while not os.path.exists(args.stop):
            tasks.append(
                asyncio.create_task(
                    _one_decision(session, args.url, rng, args.words, latencies)
                )
            )
            await asyncio.sleep(rng.expovariate(args.rate))
        await asyncio.gather(*tasks)
    ok = sorted(t for t, good in latencies if good)
    result = {
        "sent": len(latencies),
        "ok": len(ok),
        "median_s": statistics.median(ok) if ok else None,
        "p90_s": ok[int(0.9 * (len(ok) - 1))] if ok else None,
    }
    Path(args.out).write_text(json.dumps(result))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "multiturn":
        run_multiturn(sys.argv[2:])
        return
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decisions")
    d.add_argument("--url", default="http://127.0.0.1:30000/v1/decisions")
    d.add_argument("--rate", type=float, required=True)
    d.add_argument("--words", type=int, default=1500)
    d.add_argument("--stop", required=True)
    d.add_argument("--out", required=True)
    asyncio.run(_decisions(p.parse_args()))


if __name__ == "__main__":
    main()
