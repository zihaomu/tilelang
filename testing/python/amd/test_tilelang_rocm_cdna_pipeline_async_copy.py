import pytest
import torch

import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.utils.target import determine_target, target_has_async_copy


def _matmul_kernel(num_stages, M=256, N=256, K=256, block_M=128, block_N=128, block_K=32):

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_N, block_K), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=4)
                T.copy(B[bx * block_N, ko * block_K], B_shared, coalesced_width=4)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_rocm
def test_rocm_cdna_num_stages_uses_async_copy():
    target = determine_target("auto", return_object=True)
    if not target_has_async_copy(target):
        pytest.skip("requires a ROCm CDNA target with async-copy support")

    kernel = tl.compile(_matmul_kernel(num_stages=1), out_idx=[2])
    src = kernel.get_kernel_source()
    assert "tl::cp_async_gs<" in src or "tl::cp_async_gs_conditional<" in src

    A = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    B = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    C = kernel(A, B)
    ref = A.float() @ B.float().T
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
