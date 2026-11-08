"""Check that a running SGLang server reuses cached prefixes across turns.

Sends a random-token prompt, the same prompt again, and a second turn (prompt + the
first output + new tokens), then prints how many prompt tokens each request found in
the cache. Hybrid models (Mamba / GDN) also need a saved state at the match point, so
a server can keep KV yet report 0 cached tokens.

    python benchmark/capacity/cache_check.py --base-url http://127.0.0.1:30000
"""

import argparse
import random

import requests


def _generate(base_url: str, input_ids: list, max_new_tokens: int) -> dict:
    body = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
    }
    resp = requests.post(f"{base_url}/generate", json=body, timeout=600)
    resp.raise_for_status()
    return resp.json()


def check_reuse(
    base_url: str, prompt_len: int = 4000, output_len: int = 200, new_len: int = 1000
) -> dict:
    """Cached prompt tokens for a new prompt, the same prompt again, and a second turn."""
    # Fresh random ids, so nothing from earlier traffic can match.
    rng = random.Random()
    prompt = [rng.randrange(1000, 100000) for _ in range(prompt_len)]
    new = [rng.randrange(1000, 100000) for _ in range(new_len)]

    turn1 = _generate(base_url, prompt, output_len)
    output_ids = turn1["output_ids"]
    again = _generate(base_url, prompt, 8)
    turn2 = _generate(base_url, prompt + output_ids + new, 8)
    rows = []
    for name, r, reusable in (
        ("turn 1 (new prompt)", turn1, 0),
        ("same prompt again", again, len(prompt)),
        ("turn 2 (prompt + output + new)", turn2, len(prompt) + len(output_ids)),
    ):
        m = r["meta_info"]
        rows.append(
            {
                "request": name,
                "prompt": m["prompt_tokens"],
                "cached": m.get("cached_tokens", 0),
                "reusable": reusable,
            }
        )
    return {"rows": rows, "ok": rows[2]["cached"] > 0.5 * len(prompt)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--prompt-len", type=int, default=4000)
    p.add_argument("--output-len", type=int, default=200)
    p.add_argument("--new-len", type=int, default=1000)
    args = p.parse_args()
    result = check_reuse(args.base_url, args.prompt_len, args.output_len, args.new_len)
    print(f"{'request':<32} {'prompt':>8} {'cached':>8} {'reusable':>9}")
    for row in result["rows"]:
        print(
            f"{row['request']:<32} {row['prompt']:>8} {row['cached']:>8} {row['reusable']:>9}"
        )
    print(
        "\nOK: turns reuse the cache."
        if result["ok"]
        else "\nNO REUSE: the second turn recomputed its history. Check the mamba radix "
        "cache strategy and that 'mamba usage' in the server log stays above 0 when idle."
    )


if __name__ == "__main__":
    main()
