"""Build a hot-token map (FR-spec style) for --speculative-token-map from ShareGPT text.

The draft lm_head is restricted to the K most frequent tokens; verification still uses the
full target vocab, so outputs are unchanged and only draft acceptance can move.
Usage: build_token_map.py --tokenizer PATH --out map.pt [--size 32768] [--max-convs 40000]
"""

import argparse
import collections
import json
import os

import torch
from transformers import AutoTokenizer

SHAREGPT = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/"
    "192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json"
)


def count_tokens(tok, convs):
    counts = collections.Counter()
    for conv in convs:
        for turn in conv.get("conversations", []):
            if turn.get("from") in ("gpt", "human"):
                counts.update(tok.encode(turn["value"], add_special_tokens=False))
    return counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=32768)
    p.add_argument("--max-convs", type=int, default=40000)
    p.add_argument("--dataset", default=SHAREGPT)
    args = p.parse_args()
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.load(open(args.dataset))
    # Second half builds the map, a disjoint slice of the first half measures coverage.
    build = data[len(data) // 2 :][: args.max_convs]
    held = data[:2000]
    counts = count_tokens(tok, build)
    held_counts = count_tokens(tok, held)
    special = set(tok.all_special_ids) | {i for i in tok.added_tokens_decoder}
    ranked = [t for t, _ in counts.most_common() if t not in special]
    held_total = sum(held_counts.values())
    for k in (8192, 16384, 32768, 49152, 65536):
        hot = set(ranked[: k - len(special)]) | special
        cov = sum(c for t, c in held_counts.items() if t in hot) / held_total
        print(f"size {k:6d}: held-out token coverage {cov:.4f}")
    hot_ids = sorted(set(ranked[: args.size - len(special)]) | special)
    torch.save(hot_ids, args.out)
    print(f"wrote {len(hot_ids)} ids ({len(special)} special) to {args.out}")


if __name__ == "__main__":
    main()
