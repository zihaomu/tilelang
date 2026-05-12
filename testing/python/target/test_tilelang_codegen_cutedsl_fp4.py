import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T


def _is_cutedsl_available():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
        return True
    except Exception:
        return False


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def _fp4_pack_probe(x):
    M = T.dynamic("M")
    N = T.const("N")
    in_dtype = T.bfloat16
    out_dtype = T.float4_e2m1fn

    x: T.Tensor[(M, N), in_dtype]

    y = T.empty((M, N), out_dtype)
    blk_m = 32
    blk_n = 32

    with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, blk_n), threads=128) as (bx, by):
        x_shared = T.alloc_shared((blk_m, blk_n), in_dtype)
        x_local = T.alloc_fragment((blk_m, blk_n), in_dtype)
        y_local = T.alloc_fragment((blk_m, blk_n), out_dtype)
        y_shared = T.alloc_shared((blk_m, blk_n), out_dtype)

        T.copy(x[bx * blk_m, by * blk_n], x_shared, disable_tma=True)
        T.copy(x_shared, x_local)
        for i, j in T.Parallel(blk_m, blk_n):
            y_local[i, j] = T.clamp(T.cast(x_local[i, j], "float32"), -6.0, 6.0)
        T.copy(y_local, y_shared)
        T.copy(y_shared, y[bx * blk_m, by * blk_n], disable_tma=True)

    return y


@pytest.mark.skipif(not _is_cutedsl_available(), reason="CuTeDSL not installed")
@tilelang.testing.requires_cuda
def test_cutedsl_fp4_pack_uses_uint8_backing_store():
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)

    source = _fp4_pack_probe.get_kernel_source(x)

    assert "tl.pack_float32_to_fp4_e2m1fn_x2" in source
    assert "tl.make_rmem_tensor((4,), cutlass.Uint8)" in source
    assert "dtype=cutlass.Uint8" in source
    assert "tl.make_rmem_tensor((4,), cutlass.Float4E2M1FN" not in source


if __name__ == "__main__":
    tilelang.testing.main()
