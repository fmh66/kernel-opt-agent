import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    input_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    N,
    D: tl.constexpr,
    stride_input_row,
    stride_gamma,
    stride_beta,
    stride_output_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= N:
        return

    row_input_ptr = input_ptr + row * stride_input_row
    row_output_ptr = output_ptr + row * stride_output_row

    col_offsets = tl.arange(0, BLOCK_SIZE)

    x = tl.load(row_input_ptr + col_offsets).to(tl.float32)
    gamma = tl.load(gamma_ptr + col_offsets).to(tl.float32)
    beta = tl.load(beta_ptr + col_offsets).to(tl.float32)

    mean = tl.sum(x, axis=0) / D
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / D

    inv_std = tl.math.rsqrt(var + eps)
    out = diff * inv_std * gamma + beta

    tl.store(row_output_ptr + col_offsets, out)


def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    N: int,
    D: int,
    eps: float,
):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
    layernorm_kernel[grid](
        input, gamma, beta, output,
        N,
        D=D,
        stride_input_row=input.stride(0),
        stride_gamma=gamma.stride(0),
        stride_beta=beta.stride(0),
        stride_output_row=output.stride(0),
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def setup(N=1024, D=1024, eps=1e-5, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    input_tensor = torch.randn((N, D), device="cuda", dtype=dtype)
    gamma = torch.randn((D,), device="cuda", dtype=dtype)
    beta = torch.randn((D,), device="cuda", dtype=dtype)
    output = torch.empty((N, D), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "input": input_tensor,
            "gamma": gamma,
            "beta": beta,
            "output": output,
            "N": int(N),
            "D": int(D),
            "eps": eps,
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["input"], kwargs["gamma"], kwargs["beta"],
        kwargs["output"],
        int(kwargs["N"]), int(kwargs["D"]), float(kwargs["eps"]),
    )
