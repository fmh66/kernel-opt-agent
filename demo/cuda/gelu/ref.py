import torch

def reference(input, output, N):
    """Reference PyTorch GELU implementation (in-place write)."""
    output.copy_(torch.nn.functional.gelu(input, approximate='tanh'))
