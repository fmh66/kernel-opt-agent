import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    input_ptr,
    gamma_ptr,
    output_ptr,
    N,
    D,
    stride_input_row,
    stride_gamma,
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
    mask = col_offsets < D

    x = tl.load(tl.max_contiguous(tl.multiple_of(row_input_ptr + col_offsets, BLOCK_SIZE), BLOCK_SIZE),
                mask=mask, other=0.0).to(tl.float32)
    gamma = tl.load(tl.max_contiguous(tl.multiple_of(gamma_ptr + col_offsets, BLOCK_SIZE), BLOCK_SIZE),
                    mask=mask, other=0.0).to(tl.float32)

    rms = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
    out = x * rms * gamma

    tl.store(tl.max_contiguous(tl.multiple_of(row_output_ptr + col_offsets, BLOCK_SIZE), BLOCK_SIZE),
             out, mask=mask)


def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    output: torch.Tensor,
    N: int,
    D: int,
    eps: float,
):
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)
    rmsnorm_kernel[grid](
        input, gamma, output, N, D,
        input.stride(0), gamma.stride(0), output.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def setup(N=1024, D=1024, eps=1e-5, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    input_tensor = torch.randn((N, D), device="cuda", dtype=dtype)
    gamma = torch.randn((D,), device="cuda", dtype=dtype)
    output = torch.empty((N, D), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "input": input_tensor,
            "gamma": gamma,
            "output": output,
            "N": int(N),
            "D": int(D),
            "eps": eps,
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["input"], kwargs["gamma"], kwargs["output"],
        int(kwargs["N"]), int(kwargs["D"]), float(kwargs["eps"]),
    )
