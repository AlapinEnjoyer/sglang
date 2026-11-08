"""Capacity model for `studio.py plan`: how many users of a given context fit on a GPU setup.

Sizes come from the checkpoint (safetensors headers, config.json) and the serving flags.
The constants below were calibrated against server logs on 2x RTX A4500 (Swift-1.5
Qwen3.8-27B INT4, TP2); the model reproduced the measured KV pools within 1.5%.
"""

import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import msgspec

GB = 1e9
GIB = 1 << 30
DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "F8_E4M3": 1}
DTYPE_BYTES.update({"F8_E5M2": 1, "U8": 1, "I8": 1, "BOOL": 1})

# Per-GPU bytes in the static region beyond the checkpoint share (replicated tensors,
# Marlin workspaces, rotary caches). Measured on A4500 TP2: 9.10 GiB loaded vs 9.30 GB share.
STATIC_OVERHEAD_GB = 1.0
# Outside --mem-fraction-static: CUDA context, NCCL buffers, decode CUDA graphs (A4500 TP2).
NONSTATIC_FIXED_GIB = 1.25
# Free memory left after init must cover the prefill-time logits (fp32, chunk x vocab / tp)
# plus this margin. Your 8-user command crashed at 0.64 GiB free with a 1.1 GiB peak.
ACTIVATION_MARGIN_GIB = 0.6
# The CUDA driver reserves part of each card; torch sees total minus about this much.
DRIVER_RESERVED_GIB = 0.4
MAX_MEM_FRACTION = 0.95
# extra_buffer_lazy holds 3 GDN state slots per running request (cookbook ratio calculator).
SLOTS_PER_RUNNING = 3
# Saved states for idle sessions; 40 gave the best 20-session hit rate on 2x A4500.
DEFAULT_IDLE_SLOTS = 40
HOST_RAM_RESERVE_GB = 8.0


class ModelSizes(msgspec.Struct, frozen=True):
    target_bytes: int
    visual_bytes: int
    mtp_bytes: int
    vocab_size: int
    kv_layers: int
    kv_heads: int
    head_dim: int
    mtp_layers: int
    linear_layers: int
    ssm_elems: int
    conv_elems: int


class Gpu(msgspec.Struct, frozen=True):
    name: str
    mem_gib: float
    count: int
    sm: str


class ServeChoice(msgspec.Struct, frozen=True):
    # GDN slots one running request holds: 3 for extra_buffer_lazy with
    # SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1, 4 without it, 5 for extra_buffer (from
    # kv_cache_configurator._calculate_mamba_ratio).
    slots_per_running: int = SLOTS_PER_RUNNING
    kv_dtype: str = "fp8_e4m3"
    state_dtype: str = "bfloat16"
    draft_quant: str = "fp8"
    token_map_vocab: Optional[int] = 32768
    chunk: int = 2048
    language_model_only: bool = False
    idle_slots: int = DEFAULT_IDLE_SLOTS
    spec: bool = True


class Plan(msgspec.Struct, frozen=True):
    tp: int
    mem_fraction: float
    pool_gb_per_gpu: float
    kv_tokens: int
    max_running: int
    mamba_slots: int
    users_at_context: int
    hicache_gb_per_gpu: int
    host_kv_tokens: int
    host_state_slots: int
    notes: List[str]


def _safetensors_bytes(model_dir: Path) -> Dict[str, int]:
    out = {"target": 0, "visual": 0, "mtp": 0}
    for path in sorted(model_dir.glob("*.safetensors")):
        with open(path, "rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            size = DTYPE_BYTES[meta["dtype"]] * math.prod(meta["shape"])
            if name.startswith("mtp."):
                out["mtp"] += size
            elif ".visual." in name or name.startswith("visual."):
                out["visual"] += size
            else:
                out["target"] += size
    return out


def model_sizes(model_dir: Path) -> ModelSizes:
    cfg = json.loads((model_dir / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    layers = text["num_hidden_layers"]
    head_dim = (
        text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]
    )
    if "layer_types" in text:
        kv_layers = sum(t == "full_attention" for t in text["layer_types"])
    elif "full_attention_interval" in text:
        kv_layers = layers // text["full_attention_interval"]
    else:
        kv_layers = layers
    linear_layers = layers - kv_layers
    ssm = conv = 0
    if linear_layers:
        k_dim = text["linear_num_key_heads"] * text["linear_key_head_dim"]
        v_heads, v_dim = text["linear_num_value_heads"], text["linear_value_head_dim"]
        ssm = v_heads * text["linear_key_head_dim"] * v_dim
        conv = (2 * k_dim + v_heads * v_dim) * (text["linear_conv_kernel_dim"] - 1)
    sizes = _safetensors_bytes(model_dir)
    return ModelSizes(
        target_bytes=sizes["target"],
        visual_bytes=sizes["visual"],
        mtp_bytes=sizes["mtp"],
        vocab_size=text["vocab_size"],
        kv_layers=kv_layers,
        kv_heads=text["num_key_value_heads"],
        head_dim=head_dim,
        mtp_layers=text.get("mtp_num_hidden_layers", 0) if sizes["mtp"] else 0,
        linear_layers=linear_layers,
        ssm_elems=ssm,
        conv_elems=conv,
    )


def local_gpus() -> Optional[Gpu]:
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,compute_cap",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
            .splitlines()
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    name, mem_mib, sm = [x.strip() for x in out[0].split(",")]
    return Gpu(name=name, mem_gib=float(mem_mib) / 1024, count=len(out), sm=sm)


def parse_gpu(spec: str) -> Gpu:
    """NAME:MEM_GB[:COUNT[:SM]], e.g. H100:80:2:9.0 (memory as marketed, in GB)."""
    parts = spec.split(":")
    count = int(parts[2]) if len(parts) > 2 else 1
    sm = parts[3] if len(parts) > 3 else "9.0"
    return Gpu(name=parts[0], mem_gib=float(parts[1]) * GB / GIB, count=count, sm=sm)


def host_ram_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024 / GB
    return 0.0


def kv_bytes_per_token(m: ModelSizes, choice: ServeChoice) -> float:
    layers = m.kv_layers + (m.mtp_layers if choice.spec else 0)
    per_elem = 1 if choice.kv_dtype.startswith("fp8") else 2
    return 2 * layers * m.kv_heads * m.head_dim * per_elem


def state_bytes_per_slot(m: ModelSizes, choice: ServeChoice) -> float:
    ssm = 4 if choice.state_dtype == "float32" else 2
    return m.linear_layers * (m.ssm_elems * ssm + m.conv_elems * 2)


def _mem_fraction(gpu: Gpu, m: ModelSizes, choice: ServeChoice, tp: int) -> float:
    vocab = (
        choice.token_map_vocab
        if (choice.spec and choice.token_map_vocab)
        else m.vocab_size
    )
    logits_gib = choice.chunk * vocab / tp * 4 / GIB * 1.2
    usable = gpu.mem_gib - DRIVER_RESERVED_GIB
    reserve = NONSTATIC_FIXED_GIB + logits_gib + ACTIVATION_MARGIN_GIB
    return min(MAX_MEM_FRACTION, math.floor((1 - reserve / usable) * 100) / 100)


def _static_gb(m: ModelSizes, gpu: Gpu, choice: ServeChoice, tp: int) -> float:
    weights = m.target_bytes + (0 if choice.language_model_only else m.visual_bytes)
    draft = 0.0
    if choice.spec and m.mtp_bytes:
        draft = m.mtp_bytes * (0.5 if choice.draft_quant == "fp8" else 1.0)
    # The dense-prefill scratch (fork, SM8x only) is about 0.42 GB in total, split by tp.
    scratch = 0.42 * GB if gpu.sm.startswith("8.") else 0.0
    return (weights + draft + scratch) / tp + STATIC_OVERHEAD_GB * GB


def plan_for(
    m: ModelSizes,
    gpu: Gpu,
    choice: ServeChoice,
    tp: int,
    context: int,
    ram_gb: float,
    max_running_cap: int = 64,
    mem_fraction: Optional[float] = None,
    running_override: Optional[int] = None,
    slots_override: Optional[int] = None,
) -> Plan:
    mf = mem_fraction or _mem_fraction(gpu, m, choice, tp)
    pool = (gpu.mem_gib - DRIVER_RESERVED_GIB) * GIB * mf - _static_gb(
        m, gpu, choice, tp
    )
    kv_tok = kv_bytes_per_token(m, choice) / tp
    slot = state_bytes_per_slot(m, choice) / tp
    per_user = context * kv_tok + choice.slots_per_running * slot
    idle = choice.idle_slots if m.linear_layers else 0
    users = max(0, int((pool - idle * slot) // per_user)) if pool > 0 else 0
    running = running_override or max(1, min(users, max_running_cap))
    slots = (choice.slots_per_running * running + idle) if m.linear_layers else 0
    if slots_override is not None:
        slots = slots_override
    kv_tokens = int((pool - slots * slot) // kv_tok) if pool > 0 else 0

    gpus_per_host = gpu.count
    hicache = max(0, int((ram_gb - HOST_RAM_RESERVE_GB) * 0.8 / gpus_per_host))
    # sglang splits --hicache-size between KV and GDN state like the GPU pools.
    kv_share = kv_tokens * kv_tok / max(1.0, kv_tokens * kv_tok + slots * slot)
    host_kv = int(hicache * GB * kv_share // kv_tok) if hicache else 0
    host_slots = int(hicache * GB * (1 - kv_share) // slot) if (hicache and slot) else 0
    notes = []
    if users == 0:
        notes.append(f"not even one {context}-token user fits at tp={tp}")
    if users > max_running_cap:
        notes.append(f"{users} users fit; capped at {max_running_cap} running")
    return Plan(
        tp=tp,
        mem_fraction=mf,
        pool_gb_per_gpu=pool / GB,
        kv_tokens=kv_tokens,
        max_running=running,
        mamba_slots=slots,
        users_at_context=users,
        hicache_gb_per_gpu=hicache,
        host_kv_tokens=host_kv,
        host_state_slots=host_slots,
        notes=notes,
    )


def server_flags(plan: Plan, choice: ServeChoice, context: int, model: str) -> str:
    flags = [
        f"--model-path {model} --tp {plan.tp}",
        f"--kv-cache-dtype {choice.kv_dtype} --mem-fraction-static {plan.mem_fraction:.2f}",
        f"--context-length {context} --max-running-requests {plan.max_running}",
        f"--cuda-graph-max-bs-decode {plan.max_running} --chunked-prefill-size {choice.chunk}",
    ]
    if plan.mamba_slots:
        flags.append(
            f"--mamba-ssm-dtype {choice.state_dtype} --max-mamba-cache-size {plan.mamba_slots}"
            " --mamba-radix-cache-strategy extra_buffer_lazy"
        )
    if choice.language_model_only:
        flags.append("--language-model-only")
    if plan.hicache_gb_per_gpu:
        flags.append(
            f"--enable-hierarchical-cache --hicache-size {plan.hicache_gb_per_gpu}"
            " --hicache-write-policy write_through"
        )
    return " \\\n  ".join(flags)
