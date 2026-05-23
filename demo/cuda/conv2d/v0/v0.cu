#include <cuda_runtime.h>

__global__ void naive_conv2d(
    const float* __restrict__ input,
    const float* __restrict__ weight,
          float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    int n  = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;

    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;
    if (oh >= out_H || ow >= out_W) return;

    float acc = 0.0f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            for (int kw = 0; kw < K; kw++) {
                int ih = oh * stride - pad + kh;
                int iw = ow * stride - pad + kw;

                float val = 0.0f;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    val = input[n * (C_in * H * W) + ic * (H * W) + ih * W + iw];
                }

                acc += val * weight[oc * (C_in * K * K) + ic * (K * K) + kh * K + kw];
            }
        }
    }

    int out_idx = n * (C_out * out_H * out_W) + oc * (out_H * out_W) + oh * out_W + ow;
    output[out_idx] = acc;
}

extern "C" void solve(
    float* input, float* weight, float* output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;

    dim3 block(16, 16);
    dim3 grid(
        (out_W + block.x - 1) / block.x,
        (out_H + block.y - 1) / block.y,
        N * C_out
    );

    naive_conv2d<<<grid, block>>>(input, weight, output,
        N, C_in, H, W, C_out, K, stride, pad);
    cudaDeviceSynchronize();
}
