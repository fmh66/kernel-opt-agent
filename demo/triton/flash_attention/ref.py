import math
import torch


def reference(Q, K, V, output, softmax_scale):
    B, H, N, d = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, V)
    output.reshape(B, H, N, d).copy_(out)
