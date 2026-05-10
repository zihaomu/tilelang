import sys
from pathlib import Path

import torch
import tilelang.testing
from tilelang.utils import determine_torch_fp8_type


EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v4"
sys.path.insert(0, str(EXAMPLE_DIR))

import act_quant  # noqa: E402


@tilelang.testing.requires_rocm
def test_deepseek_v4_fp8_act_quant_uses_rocm_dtype_range():
    fp8_dtype = determine_torch_fp8_type("e4m3")
    assert fp8_dtype is torch.float8_e4m3fnuz
    assert torch.finfo(fp8_dtype).max == 240.0

    act_quant.test_fp8_act_quant(M=64, N=256, block_size=128)


if __name__ == "__main__":
    tilelang.testing.main()
