import pytest
import torch

from tilelang.contrib.dlpack import convert_func


def _converted_dtype_name(torch_dtype: torch.dtype) -> str:
    seen = []
    wrapped = convert_func(lambda arg: seen.append(str(arg.dtype)), torch.Tensor, torch.utils.dlpack.to_dlpack)
    wrapped(torch.empty((4,), dtype=torch_dtype))
    return seen[0]


@pytest.mark.skipif(
    not all(hasattr(torch, name) for name in ("float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2", "float8_e5m2fnuz")),
    reason="PyTorch float8 dtypes are unavailable",
)
def test_dlpack_preserves_float8_fnuz_dtype_names():
    assert _converted_dtype_name(torch.float8_e4m3fn) == "float8_e4m3"
    assert _converted_dtype_name(torch.float8_e4m3fnuz) == "float8_e4m3fnuz"
    assert _converted_dtype_name(torch.float8_e5m2) == "float8_e5m2"
    assert _converted_dtype_name(torch.float8_e5m2fnuz) == "float8_e5m2fnuz"
