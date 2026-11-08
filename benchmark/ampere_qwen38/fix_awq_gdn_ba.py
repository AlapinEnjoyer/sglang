"""Copy a compressed-tensors W4A16 Qwen3.5/3.8 checkpoint with the GDN in_proj_a/in_proj_b
layers dequantized to bf16 and added to the quantization ignore list.

Those layers have num_v_heads (48) output rows; SGLang's Marlin repack needs multiples of 64,
so the original checkpoint cannot load.
    python fix_awq_ba.py <src dir> <dst dir>
"""

import json
import os
import shutil
import sys

import torch
from compressed_tensors.compressors.pack_quantized.helpers import unpack_from_int32
from safetensors import safe_open
from safetensors.torch import save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
cfg = json.load(open(f"{src}/config.json"))
q = cfg["quantization_config"]["config_groups"]["group_0"]["weights"]
assert q["num_bits"] == 4 and q["symmetric"] and q["strategy"] == "group"
group = q["group_size"]

out, fixed = {}, []
with safe_open(f"{src}/model.safetensors", "pt") as f:
    keys = list(f.keys())
    for k in keys:
        mod = k.rsplit(".", 1)[0]
        if not mod.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            out[k] = f.get_tensor(k)
            continue
        if not k.endswith(".weight_packed"):
            continue
        shape = torch.Size(f.get_tensor(f"{mod}.weight_shape").tolist())
        qw = unpack_from_int32(f.get_tensor(k), 4, shape).to(torch.float32)
        assert qw.min() >= -8 and qw.max() <= 7, (k, qw.min(), qw.max())
        scale = f.get_tensor(f"{mod}.weight_scale").to(torch.float32)
        assert (
            f"{mod}.weight_g_idx" not in keys and f"{mod}.weight_zero_point" not in keys
        )
        w = qw * scale.repeat_interleave(group, dim=1)[:, : shape[1]]
        out[f"{mod}.weight"] = w.to(torch.bfloat16)
        fixed.append(mod)

print(
    f"dequantized {len(fixed)} layers, e.g. {fixed[0]} {tuple(out[fixed[0] + '.weight'].shape)}"
)
save_file(out, f"{dst}/model.safetensors", metadata={"format": "pt"})
cfg["quantization_config"]["ignore"] += fixed
json.dump(cfg, open(f"{dst}/config.json", "w"), indent=2)
for name in os.listdir(src):
    if name not in ("model.safetensors", "config.json"):
        shutil.copy2(f"{src}/{name}", f"{dst}/{name}")
print("wrote", dst)
