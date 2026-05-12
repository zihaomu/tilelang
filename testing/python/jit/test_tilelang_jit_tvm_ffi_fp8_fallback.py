from dataclasses import dataclass

import pytest
import torch

from tilelang.jit.adapter import tvm_ffi as tvm_ffi_adapter


@dataclass
class _FakeParam:
    dtype: str
    torch_dtype_value: torch.dtype

    def is_float8(self):
        return self.dtype.startswith("float8")

    def torch_dtype(self):
        return self.torch_dtype_value


def _require_torch_fp8():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        pytest.skip("torch.float8_e4m3fn is unavailable")
    return dtype


def test_rocm_fp8_fallback_uses_native_dlpack_when_supported(monkeypatch):
    torch_dtype = _require_torch_fp8()

    monkeypatch.setattr(torch.version, "hip", "test-rocm", raising=False)
    monkeypatch.setenv("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "1")
    monkeypatch.setattr(tvm_ffi_adapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tvm_ffi_adapter, "_torch_dlpack_supports_dtype", lambda _dtype, _device: True)

    assert tvm_ffi_adapter._rocm_float8_storage_view_param_mask(
        [_FakeParam("float8_e4m3fn", torch_dtype)]
    ) is None


def test_rocm_fp8_fallback_uses_storage_view_when_native_dlpack_rejects_fp8(monkeypatch):
    torch_dtype = _require_torch_fp8()

    monkeypatch.setattr(torch.version, "hip", "test-rocm", raising=False)
    monkeypatch.setenv("TVM_FFI_SKIP_DLPACK_C_EXCHANGE_API", "1")
    monkeypatch.setattr(tvm_ffi_adapter.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tvm_ffi_adapter, "_torch_dlpack_supports_dtype", lambda _dtype, _device: False)

    assert tvm_ffi_adapter._rocm_float8_storage_view_param_mask(
        [
            _FakeParam("float8_e4m3fn", torch_dtype),
            _FakeParam("float32", torch.float32),
        ]
    ) == (True, False)


def test_float8_storage_view_preserves_logical_dtype_and_shape():
    torch_dtype = _require_torch_fp8()

    tensor = torch.empty((4,), dtype=torch_dtype)
    view = tvm_ffi_adapter._export_float8_as_tvm_view(tensor, "float8_e4m3fn")

    assert tuple(view.shape) == (4,)
    assert str(view.dtype) == "float8_e4m3fn"
