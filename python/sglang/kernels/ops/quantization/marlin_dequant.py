"""Dequantize a GPTQ-Marlin packed W4A16 or FP8 W8A16 weight back to a dense [N, K] matrix.

Large-M prefill GEMMs run faster as dequant + cuBLAS than as Marlin on SM8x
(measured on RTX A4500: ~65 TFLOPS Marlin vs ~77 TFLOPS cuBLAS at M=4096), and
the dequant pass costs one write of the dense weight. Supports the symmetric
uint4b8 layout and fp8 e4m3 (block-FP8 checkpoints on SM8x), with group-wise or
channel-wise scales and no act-order (g_idx).

Layout inverted here (see sglang.test.test_marlin_utils.marlin_weights): the
[K, N] nibble matrix is cut into 16x16 tiles, each 16-row band is flattened
tile-major into a row of 16*N values, every 1024-value chunk is permuted by the
Marlin weight perm, and 8 consecutive values are packed per int32 (nibble i at
bits 4i); 8-bit packs 4 values per int32 (byte i at bits 8i). Scales are permuted in
64-wide chunks by the Marlin scale perm.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import triton
import triton.language as tl


def _marlin_weight_perm(num_bits: int) -> np.ndarray:
    perm_list = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in (0, 1):
            for row in (
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ):
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm_list.extend(p + 256 * j for p in perm1)
    perm = np.array(perm_list)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7] if num_bits == 4 else [0, 2, 1, 3])
    return perm.reshape((-1, len(interleave)))[:, interleave].ravel()


def _marlin_scale_perms() -> tuple[np.ndarray, np.ndarray]:
    grouped = [i + 8 * j for i in range(8) for j in range(8)]
    single = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    return np.array(grouped), np.array(single)


@lru_cache(maxsize=None)
def _inverse_tables(device: torch.device, num_bits: int) -> tuple[torch.Tensor, ...]:
    def inverse(p: np.ndarray) -> torch.Tensor:
        inv = np.empty_like(p)
        inv[p] = np.arange(p.size)
        return torch.from_numpy(inv.astype(np.int32)).to(device)

    grouped, single = _marlin_scale_perms()
    return inverse(_marlin_weight_perm(num_bits)), inverse(grouped), inverse(single)


@triton.jit
def _marlin_w4_dequant_kernel(
    q_ptr,  # [K // 16, 2 * N] int32, Marlin-packed
    s_ptr,  # [num_groups, N], Marlin-permuted scales
    out_ptr,  # [N, K] dense
    inv_wperm_ptr,  # [1024] int32
    inv_sperm_ptr,  # [SPERM] int32
    N,
    K,
    GROUP_SIZE: tl.constexpr,
    SPERM: tl.constexpr,  # scale perm width: 64 grouped, 32 channel-wise
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    pos_old = (n // 16) * 256 + (k % 16) * 16 + (n % 16)
    j = tl.load(inv_wperm_ptr + (pos_old % 1024))
    pos_new = (pos_old // 1024) * 1024 + j
    word = tl.load(q_ptr + (k // 16).to(tl.int64) * (2 * N) + pos_new // 8)
    val = ((word >> ((pos_new % 8) * 4)) & 0xF) - 8

    s_col = (n // SPERM) * SPERM + tl.load(inv_sperm_ptr + (n % SPERM))
    scale = tl.load(s_ptr + (k // GROUP_SIZE).to(tl.int64) * N + s_col)
    w = val.to(tl.float32) * scale.to(tl.float32)
    tl.store(out_ptr + n.to(tl.int64) * K + k, w.to(out_ptr.dtype.element_ty))


def marlin_w4_dequant(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense [size_n, size_k] weight (scales' dtype) from Marlin uint4b8 tensors.

    ``group_size`` is the quantization group along K, or -1 for channel-wise.
    ``out`` may be a reusable buffer viewed to [size_n, size_k].
    """
    assert size_k % 64 == 0 and size_n % 64 == 0
    assert q_weight.shape == (size_k // 16, size_n * 2), q_weight.shape
    inv_wperm, inv_grouped, inv_single = _inverse_tables(q_weight.device, 4)
    channelwise = group_size == -1 or group_size >= size_k
    if out is None:
        out = torch.empty((size_n, size_k), dtype=scales.dtype, device=q_weight.device)
    block_n, block_k = 64, 64
    grid = (size_k // block_k, size_n // block_n)
    _marlin_w4_dequant_kernel[grid](
        q_weight,
        scales,
        out,
        inv_wperm,
        inv_single if channelwise else inv_grouped,
        size_n,
        size_k,
        GROUP_SIZE=size_k if channelwise else group_size,
        SPERM=32 if channelwise else 64,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


@triton.jit
def _marlin_fp8_dequant_kernel(
    q_ptr,  # [K // 16, 4 * N] int32, Marlin-packed fp8 e4m3 bytes
    s_ptr,  # [num_groups, N], Marlin-permuted scales with 2^120 folded in
    out_ptr,  # [N, K] dense
    inv_wperm_ptr,  # [1024] int32
    inv_sperm_ptr,  # [SPERM] int32
    N,
    K,
    GROUP_SIZE: tl.constexpr,
    SPERM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]

    pos_old = (n // 16) * 256 + (k % 16) * 16 + (n % 16)
    j = tl.load(inv_wperm_ptr + (pos_old % 1024))
    pos_new = (pos_old // 1024) * 1024 + j
    word = tl.load(q_ptr + (k // 16).to(tl.int64) * (4 * N) + pos_new // 4)
    byte = (word >> ((pos_new % 4) * 8)) & 0xFF
    # e4m3 decoded with integer ops (no fp8 casts below SM89): |x| = mag * 2^-9.
    exp = (byte >> 3) & 0xF
    man = byte & 0x7
    mag = tl.where(exp == 0, man, (man + 8) << tl.maximum(exp - 1, 0))
    val = tl.where((byte & 0x80) != 0, -mag, mag).to(tl.float32)

    s_col = (n // SPERM) * SPERM + tl.load(inv_sperm_ptr + (n % SPERM))
    scale = tl.load(s_ptr + (k // GROUP_SIZE).to(tl.int64) * N + s_col)
    # 2^-120 undoes the exponent bias Marlin folds into the scales; both factors stay
    # fp32-normal (2^-129 in one constant would be subnormal and flush to zero).
    w = (val * 0.001953125) * (scale.to(tl.float32) * 7.52316384526264e-37)
    tl.store(out_ptr + n.to(tl.int64) * K + k, w.to(out_ptr.dtype.element_ty))


def marlin_fp8_dequant(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense [size_n, size_k] weight (scales' dtype) from FP8 Marlin tensors, as built by
    ``prepare_fp8_layer_for_marlin`` (scales carry the fused 2^120 exponent bias).

    ``group_size`` is the scale group along K (the block size for block-FP8), or -1
    for channel-wise. ``out`` may be a reusable buffer viewed to [size_n, size_k].
    """
    assert size_k % 64 == 0 and size_n % 64 == 0
    assert q_weight.shape == (size_k // 16, size_n * 4), q_weight.shape
    inv_wperm, inv_grouped, inv_single = _inverse_tables(q_weight.device, 8)
    channelwise = group_size == -1 or group_size >= size_k
    if out is None:
        out = torch.empty((size_n, size_k), dtype=scales.dtype, device=q_weight.device)
    block_n, block_k = 64, 64
    grid = (size_k // block_k, size_n // block_n)
    _marlin_fp8_dequant_kernel[grid](
        q_weight,
        scales,
        out,
        inv_wperm,
        inv_single if channelwise else inv_grouped,
        size_n,
        size_k,
        GROUP_SIZE=size_k if channelwise else group_size,
        SPERM=32 if channelwise else 64,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out
