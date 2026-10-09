import sys

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.quantization.marlin_dequant import marlin_w4_dequant
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
