import torch

atol = 1e-4
rtol = 1e-3


def reference(**kwargs):
    input_tensor = kwargs["input"]
    gamma = kwargs["gamma"]
    output = kwargs["output"]
    N = kwargs["N"]
    D = kwargs["D"]

    x = input_tensor.view(N, D)
    g = gamma[:D]
    rms = torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-5)
    out = x * rms * g.unsqueeze(0)
    output.copy_(out.reshape(-1))
    return output
