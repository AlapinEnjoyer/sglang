"""Write HumanEval + MBPP prompts as a bench_serving `custom` JSONL (S6 code scenario)."""

import json
import sys

from datasets import load_dataset

out = sys.argv[1]
rows = []
for ex in load_dataset("openai/openai_humaneval", split="test"):
    rows.append(
        "Complete the following Python function. Explain briefly, then give the code.\n\n"
        + ex["prompt"]
    )
for ex in load_dataset("google-research-datasets/mbpp", "full", split="test"):
    rows.append(
        ex["text"]
        + "\nYour code should pass these tests:\n"
        + "\n".join(ex["test_list"])
    )
with open(out, "w") as f:
    for r in rows:
        f.write(
            json.dumps({"conversations": [{"content": r}, {"content": "x" * 64}]})
            + "\n"
        )
print(len(rows), "prompts ->", out)
