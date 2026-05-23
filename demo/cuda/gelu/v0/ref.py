import torch

def reference(input_tensor):
    """Reference PyTorch GELU implementation."""
    return torch.nn.functional.gelu(input_tensor)
