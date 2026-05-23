import math
import torch
import triton
import triton.language as tl


@triton.jit
def flash_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    B, H, N, d,
    stride_q_bs, stride_q_h, stride_q_seq, stride_q_d,
    stride_k_bs, stride_k_h, stride_k_seq, stride_k_d,
    stride_v_bs, stride_v_h, stride_v_seq, stride_v_d,
    stride_o_bs, stride_o_h, stride_o_seq, stride_o_d,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    m_start = tl.program_id(2) * BLOCK_M

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = m_start + tl.arange(0, BLOCK_M)
    m_mask = offs_m < N

    q_ptrs = (
        q_ptr
        + batch * stride_q_bs
        + head * stride_q_h
        + offs_m[:, None] * stride_q_seq
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k_ptrs = (
            k_ptr
            + batch * stride_k_bs
            + head * stride_k_h
            + offs_n[:, None] * stride_k_seq
            + offs_d[None, :] * stride_k_d
        )
        k = tl.load(k_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_new = alpha * l_i + tl.sum(p, axis=1)

        v_ptrs = (
            v_ptr
            + batch * stride_v_bs
            + head * stride_v_h
            + offs_n[:, None] * stride_v_seq
            + offs_d[None, :] * stride_v_d
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None] & (offs_d[None, :] < d), other=0.0).to(tl.float32)

        acc = acc * alpha[:, None] + tl.dot(p, v)

        m_i = m_new
        l_i = l_new

    acc = acc / l_i[:, None]

    out_ptrs = (
        out_ptr
        + batch * stride_o_bs
        + head * stride_o_h
        + offs_m[:, None] * stride_o_seq
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, acc, mask=m_mask[:, None] & (offs_d[None, :] < d))


def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float = None,
):
    B, H, N, d = Q.shape

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_D = min(64, triton.next_power_of_2(d))

    grid = (B, H, triton.cdiv(N, BLOCK_M))

    flash_attention_kernel[grid](
        Q, K, V, output,
        B, H, N, d,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        softmax_scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return output


def setup(B=4, H=12, N=2048, d=64, seed=42, dtype=torch.float16, **kwargs):
    torch.manual_seed(seed)
    Q = torch.randn((B, H, N, d), device="cuda", dtype=dtype)
    K = torch.randn((B, H, N, d), device="cuda", dtype=dtype)
    V = torch.randn((B, H, N, d), device="cuda", dtype=dtype)
    output = torch.empty((B, H, N, d), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "Q": Q, "K": K, "V": V,
            "output": output,
            "softmax_scale": 1.0 / math.sqrt(d),
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(kwargs["Q"], kwargs["K"], kwargs["V"], kwargs["output"], kwargs["softmax_scale"])
