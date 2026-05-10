import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel_v1(
    input_ptr,
    output_ptr,
    N,
    D,
    stride_input_row,
    stride_output_row,
    ROWS_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    if row_start >= N:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < D

    for r in range(ROWS_PER_BLOCK):
        row = row_start + r
        row_in_bounds = row < N

        row_input_ptr = input_ptr + row * stride_input_row
        row_output_ptr = output_ptr + row * stride_output_row

        load_mask = mask & row_in_bounds
        row_data = tl.load(row_input_ptr + col_offsets, mask=load_mask, other=-float('inf'))

        max_val = tl.max(row_data, axis=0)
        numerator = tl.exp(row_data - max_val)
        sum_val = tl.sum(numerator, axis=0)
        softmax_out = numerator / sum_val

        tl.store(row_output_ptr + col_offsets, softmax_out, mask=load_mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int, D: int):
    ROWS_PER_BLOCK = 2
    BLOCK_SIZE = triton.next_power_of_2(D)

    grid = ((N + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK,)

    softmax_kernel_v1[grid](
        input,
        output,
        N,
        D,
        input.stride(0),
        output.stride(0),
        ROWS_PER_BLOCK=ROWS_PER_BLOCK,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )

    return output


def setup(N=1024, D=1024, seed=42, dtype=torch.float32, **kwargs):
    torch.manual_seed(seed)
    input_tensor = torch.randn((N, D), device="cuda", dtype=dtype)
    output_tensor = torch.empty((N, D), device="cuda", dtype=dtype)
    return {
        "inputs": {
            "input": input_tensor,
            "output": output_tensor,
            "N": int(N),
            "D": int(D),
        },
        "outputs": ["output"],
    }


def run_kernel(**kwargs):
    solve(
        kwargs["input"],
        kwargs["output"],
        int(kwargs["N"]),
        int(kwargs["D"]),
    )
