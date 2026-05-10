import math
import torch
import triton
import triton.language as tl


@triton.jit
def fused_mha_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    H,
    N,
    d_k,
    stride_q_h,
    stride_q_n,
    stride_q_d,
    stride_k_h,
    stride_k_n,
    stride_k_d,
    stride_v_h,
    stride_v_n,
    stride_v_d,
    stride_o_h,
    stride_o_n,
    stride_o_d,
    inv_sqrt_dk,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)
    i = tl.program_id(1)

    if h >= H or i >= N:
        return

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < d_k

    q_ptrs = q_ptr + h * stride_q_h + i * stride_q_n + offs_d * stride_q_d
    q_vec = tl.load(q_ptrs, mask=d_mask, other=0.0).to(tl.float32)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        k_ptrs = (
            k_ptr
            + h * stride_k_h
            + offs_n[:, None] * stride_k_n
            + offs_d[None, :] * stride_k_d
        )
        k_mask = n_mask[:, None] & d_mask[None, :]
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        scores = tl.sum(k_tile * q_vec[None, :], axis=1) * inv_sqrt_dk
        scores = tl.where(n_mask, scores, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new)
        l_new = alpha * l_i + tl.sum(p, axis=0)

        v_ptrs = (
            v_ptr
            + h * stride_v_h
            + offs_n[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v_tile = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        acc = alpha * acc + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new
        l_i = l_new

    out_val = acc / l_i
    out_loc = out_ptr + h * stride_o_h + i * stride_o_n + offs_d * stride_o_d
    tl.store(out_loc, out_val, mask=d_mask)


def solve(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    N: int,
    d_model: int,
    num_heads: int,
):
    d_k = d_model // num_heads

    q = Q.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()
    k = K.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()
    v = V.view(N, num_heads, d_k).permute(1, 0, 2).contiguous()

    context = torch.empty((num_heads, N, d_k), device=Q.device, dtype=Q.dtype)

    block_d = min(128, triton.next_power_of_2(d_k))
    block_n = 128

    grid = (num_heads, N)
    fused_mha_kernel[grid](
        q,
        k,
        v,
        context,
        num_heads,
        N,
        d_k,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        context.stride(0),
        context.stride(1),
        context.stride(2),
        1.0 / math.sqrt(d_k),
        BLOCK_N=block_n,
        BLOCK_D=block_d,
    )

    out = context.transpose(0, 1).contiguous().view(N, d_model)
    output.copy_(out)
    return output


def setup(N=1024, d_model=1024, num_heads=16, seed=42, dtype=torch.float32, **kwargs):
    if d_model % num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads")

    torch.manual_seed(seed)
    Q = torch.randn((N, d_model), device="cuda", dtype=dtype)
    K = torch.randn((N, d_model), device="cuda", dtype=dtype)
    V = torch.randn((N, d_model), device="cuda", dtype=dtype)
    output = torch.empty((N, d_model), device="cuda", dtype=dtype)

    return {
        "inputs": {
            "Q": Q,
            "K": K,
            "V": V,
            "output": output,
            "N": int(N),
            "d_model": int(d_model),
            "num_heads": int(num_heads),
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["Q"],
        kwargs["K"],
        kwargs["V"],
        kwargs["output"],
        int(kwargs["N"]),
        int(kwargs["d_model"]),
        int(kwargs["num_heads"]),
    )
