import torch

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
    eps = float(kwargs.get("eps", 1e-5))
    solve(
        kwargs["input"], kwargs["gamma"], kwargs["beta"],
        kwargs["output"],
        int(kwargs["N"]), int(kwargs["D"]), eps,
    )


def solve(input, gamma, beta, output, N, D, eps):
    import torch.nn.functional as F
    in_2d = input.view(N, D)
    gamma_d = gamma[:D]
    beta_d = beta[:D]
    out = F.layer_norm(in_2d, (D,), weight=gamma_d, bias=beta_d, eps=eps)
    output.copy_(out.view(-1))


def reference(**kwargs):
    eps = float(kwargs.get("eps", 1e-5))
    solve(
        kwargs["input"], kwargs["gamma"], kwargs["beta"],
        kwargs["output"],
        int(kwargs["N"]), int(kwargs["D"]), eps,
    )
