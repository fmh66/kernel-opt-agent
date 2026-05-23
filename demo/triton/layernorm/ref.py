import torch
import torch.nn.functional as F


def reference(**kwargs):
    input_tensor = kwargs["input"]
    gamma = kwargs["gamma"]
    beta = kwargs["beta"]
    output = kwargs["output"]
    eps = float(kwargs["eps"])

    N, D = input_tensor.shape
    with torch.no_grad():
        result = F.layer_norm(input_tensor, (D,), gamma, beta, eps)
        output_view = output.reshape(N, D)
        output_view.copy_(result)
