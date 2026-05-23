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
    row_lock,
):
    pid = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    row = tl.atomic_add(row_lock, 1)
    while row < N:
        row_input_ptr = input_ptr + row * stride_input_row
        row_output_ptr = output_ptr + row * stride_output_row

        x = tl.load(row_input_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

        rms = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + eps)
        out = x * rms * gamma

        tl.store(row_output_ptr + col_offsets, out, mask=mask)

        row = tl.atomic_add(row_lock, 1)


def solve(
    input: torch.Tensor,
    gamma: torch.Tensor,
    output: torch.Tensor,
    N: int,
    D: int,
    eps: float,
):
    BLOCK_SIZE = triton.next_power_of_2(D)
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    grid = (num_sms * 2,)
    row_lock = torch.zeros((1,), dtype=torch.int32, device="cuda")
    row_lock.zero_()
    rmsnorm_kernel[grid](
        input, gamma, output, N, D,
        input.stride(0), gamma.stride(0), output.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        row_lock=row_lock,
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
