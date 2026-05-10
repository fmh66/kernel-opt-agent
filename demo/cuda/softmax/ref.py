import torch

atol = 1e-3
rtol = 1e-2


def reference(input, output, N, D):
    inp = input[:N * D].view(N, D)
    out = torch.softmax(inp, dim=-1)
    output[:N * D].copy_(out.reshape(-1))
