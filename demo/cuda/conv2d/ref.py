import torch
import torch.nn.functional as F

atol = 5e-2
rtol = 1e-2

def reference(input, weight, output, N, C_in, H, W, C_out, K, stride, pad):
    inp_elems = N * C_in * H * W
    w_elems = C_out * C_in * K * K
    inp = input[:inp_elems].view(N, C_in, H, W)
    w = weight[:w_elems].view(C_out, C_in, K, K)
    out_H = (H + 2 * pad - K) // stride + 1
    out_W = (W + 2 * pad - K) // stride + 1
    out = F.conv2d(inp, w, stride=stride, padding=pad)
    out_elems = N * C_out * out_H * out_W
    output[:out_elems].copy_(out.reshape(-1))
