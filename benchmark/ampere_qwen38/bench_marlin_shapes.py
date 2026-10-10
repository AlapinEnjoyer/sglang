"""Microbench Marlin on the per-rank TP shapes of Qwen3.8-27B: W4A16 (Swift INT4) or
block-FP8 W8A16 (Qwen3.8-27B-FP8).

Reports GB/s (small M, bandwidth-bound) and TFLOPS (large M), vs cuBLAS bf16 dense.
Usage: bench_marlin_shapes.py [--ms 1,2,4,8,16,512,2048,4096] [--tp 2] [--wtype int4|fp8]
"""

import argparse
from types import SimpleNamespace

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.gemm.gptq_marlin import gptq_marlin_gemm
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.test.test_marlin_utils import marlin_quantize

HIDDEN, INTER, GDN_QKVZ, GDN_V, ATTN_QKVG, ATTN_O = (
    5120,
    17408,
    16384,
    6144,
    14336,
    6144,
)


def shapes(tp):
    # name: (N, K, count per forward)
    return {
        "gdn.in_proj_qkvz": (GDN_QKVZ // tp, HIDDEN, 48),
        "gdn.out_proj": (HIDDEN, GDN_V // tp, 48),
        "attn.qkv_gate": (ATTN_QKVG // tp, HIDDEN, 16),
        "attn.o_proj": (HIDDEN, ATTN_O // tp, 16),
        "mlp.gate_up": (2 * INTER // tp, HIDDEN, 64),
        "mlp.down": (HIDDEN, INTER // tp, 64),
    }


def _int4_copy(n, k, dev):
    w = torch.randn((k, n), dtype=torch.bfloat16, device=dev) * 0.02
    _, q, s, _, _, _ = marlin_quantize(w, scalar_types.uint4b8, 128, False)
    return q, s


def _fp8_copy(n, k, dev):
    # Block-FP8 [N, K] weight with [N/128, K/128] inverse scales, as in the checkpoint.
    w = (torch.randn((n, k), device=dev) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        weight=w,
        weight_scale_inv=torch.rand((n // 128, k // 128), device=dev) * 1e-3,
        output_size_per_partition=n,
        input_size_per_partition=k,
        weight_block_size=[128, 128],
        orig_dtype=torch.bfloat16,
        bias=None,
    )
    prepare_fp8_layer_for_marlin(layer, size_k_first=False)
    return layer.weight, layer.weight_scale


def bench(fn, iters=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(10):
            fn()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    g.replay()
    torch.cuda.synchronize()
    st.record()
    for _ in range(iters // 10):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1e3 / iters  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", default="1,2,4,8,16,512,2048,4096")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument(
        "--copies", type=int, default=4, help="weight copies rotated to defeat L2"
    )
    p.add_argument("--wtype", choices=["int4", "fp8"], default="int4")
    args = p.parse_args()
    ms = [int(m) for m in args.ms.split(",")]
    dev = torch.device("cuda")
    workspace = marlin_make_workspace(dev)
    qt = scalar_types.uint4b8
    total = {m: 0.0 for m in ms}
    total_ref = {m: 0.0 for m in ms}
    for name, (n, k, count) in shapes(args.tp).items():
        make = _fp8_copy if args.wtype == "fp8" else _int4_copy
        copies = [make(n, k, dev) for _ in range(args.copies)]
        dense = torch.randn((n, k), dtype=torch.bfloat16, device=dev)
        wbytes = n * k * (1 if args.wtype == "fp8" else 0.5) + n * (k // 128) * 2
        for m in ms:
            a = torch.randn((m, k), dtype=torch.bfloat16, device=dev)
            it = iter(range(10**9))

            def run():
                q, s = copies[next(it) % len(copies)]
                if args.wtype == "fp8":
                    apply_fp8_marlin_linear(
                        input=a,
                        weight=q,
                        weight_scale=s,
                        workspace=workspace,
                        size_n=n,
                        size_k=k,
                        bias=None,
                    )
                else:
                    gptq_marlin_gemm(
                        a,
                        None,
                        q,
                        s,
                        None,
                        None,
                        None,
                        None,
                        workspace,
                        qt,
                        m,
                        n,
                        k,
                        is_k_full=True,
                        use_atomic_add=False,
                        use_fp32_reduce=True,
                        is_zp_float=False,
                    )

            t = bench(run)
            t_ref = bench(lambda: torch.nn.functional.linear(a, dense))
            total[m] += t * count
            total_ref[m] += t_ref * count
            gbs = (wbytes + 2 * m * (n + k)) / t / 1e3
            tf = 2 * m * n * k / t / 1e6
            print(
                f"{name:18s} N={n:6d} K={k:5d} M={m:5d}  marlin {t:8.1f}us {gbs:6.0f}GB/s {tf:6.1f}TF"
                f"  | cublas-bf16 {t_ref:8.1f}us {2 * m * n * k / t_ref / 1e6:6.1f}TF"
            )
    for m in ms:
        print(
            f"per-forward linear time M={m:5d}: marlin {total[m] / 1e3:7.2f} ms  cublas-bf16 {total_ref[m] / 1e3:7.2f} ms"
        )


if __name__ == "__main__":
    main()
