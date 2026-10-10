import sys
from types import SimpleNamespace

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.quantization.marlin_dequant import (
    marlin_fp8_dequant,
    marlin_w4_dequant,
)
from sglang.srt.layers.quantization.marlin_utils_fp8 import prepare_fp8_layer_for_marlin
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import marlin_quantize

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.parametrize("group_size", [128, -1])
@pytest.mark.parametrize(
    "size_k,size_n", [(128, 64), (1024, 256), (3072, 5120), (5120, 8192)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_marlin_w4_dequant_matches_reference(size_k, size_n, group_size, dtype):
    w = torch.randn((size_k, size_n), dtype=dtype, device="cuda") * 0.02
    w_ref, q_w, scales, _, _, _ = marlin_quantize(
        w, scalar_types.uint4b8, group_size, False
    )
    out = marlin_w4_dequant(q_w, scales, size_k, size_n, group_size)
    assert out.shape == (size_n, size_k) and out.dtype == dtype
    torch.testing.assert_close(out, w_ref.t().to(dtype), rtol=0, atol=0)


def _random_fp8(shape) -> torch.Tensor:
    # Every e4m3 code (subnormals included) except the two NaNs.
    codes = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    codes[(codes & 0x7F) == 0x7F] = 0
    return codes.view(torch.float8_e4m3fn)


@pytest.mark.parametrize("block", [True, False])
@pytest.mark.parametrize(
    "size_k,size_n", [(128, 64), (1024, 256), (3072, 5120), (5120, 8192)]
)
def test_marlin_fp8_dequant_matches_reference(size_k, size_n, block):
    w = _random_fp8((size_n, size_k))
    if block:
        scale = torch.rand((-(-size_n // 128), size_k // 128), device="cuda") * 1e-2
        scale_full = scale.repeat_interleave(128, 0)[:size_n].repeat_interleave(128, 1)
        layer_weight, scale_name, block_size = w, "weight_scale_inv", [128, 128]
    else:
        scale = torch.rand((size_n,), device="cuda") * 1e-2
        scale_full = scale[:, None].expand(size_n, size_k)
        layer_weight, scale_name, block_size = w.t().contiguous(), "weight_scale", None
    scale_bf16 = scale_full.to(torch.bfloat16).float()
    ref = (w.float() * scale_bf16).to(torch.bfloat16)

    layer = SimpleNamespace(
        weight=layer_weight,
        output_size_per_partition=size_n,
        input_size_per_partition=size_k,
        weight_block_size=block_size,
        orig_dtype=torch.bfloat16,
        bias=None,
        **{scale_name: scale},
    )
    prepare_fp8_layer_for_marlin(layer, size_k_first=not block)
    out = marlin_fp8_dequant(
        layer.weight, layer.weight_scale, size_k, size_n, 128 if block else -1
    )
    assert out.shape == (size_n, size_k) and out.dtype == torch.bfloat16
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
