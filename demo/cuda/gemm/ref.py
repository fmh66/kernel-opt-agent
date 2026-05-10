import torch

atol = 1e-3
rtol = 1e-2


def reference(A, B, C, M, K, N, **kwargs):
    """PyTorch reference for GEMM: C = A @ B.

    Buffers are flat 1D tensors:
      A: M x K (row-major)
      B: K x N (row-major)
      C: M x N (row-major)
    """
    A_mat = A[:M * K].view(M, K)
    B_mat = B[:K * N].view(K, N)
    C[:M * N].copy_((A_mat @ B_mat).reshape(-1))
