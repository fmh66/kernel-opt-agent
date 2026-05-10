import torch


def reference(input: torch.Tensor, output: torch.Tensor, N: int, D: int, **kwargs):
    """PyTorch reference for softmax along dim=1 (row-wise)."""
    inp = input.reshape(N, D)
    out = torch.nn.functional.softmax(inp, dim=1)
    output.copy_(out.reshape(-1))
    return output
