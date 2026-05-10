import torch


def reference(**kwargs):
    A = kwargs["A"]
    B = kwargs["B"]
    C = kwargs["C"]
    M = int(kwargs["M"])
    K = int(kwargs["K"])
    N = int(kwargs["N"])
    result = A.reshape(M, K)[:M, :K] @ B.reshape(K, N)[:K, :N]
    C.reshape(M, N)[:] = result
