#include <cuda_runtime.h>

__global__ void conv2d_v3(
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
    int base_in = n * (C_in * H * W);
    int base_wt = oc * (C_in * K * K);

    for (int ic = 0; ic < C_in; ic++) {
        int ic_HW = ic * (H * W);
        int ic_K2 = ic * (K * K);
        const float* wt_ptr = weight + base_wt + ic_K2;

        // Prefetch 9 weight values via __ldg (read-only cache)
        float w00 = __ldg(wt_ptr);
        float w01 = __ldg(wt_ptr + 1);
        float w02 = __ldg(wt_ptr + 2);
        float w10 = __ldg(wt_ptr + 3);
        float w11 = __ldg(wt_ptr + 4);
        float w12 = __ldg(wt_ptr + 5);
        float w20 = __ldg(wt_ptr + 6);
        float w21 = __ldg(wt_ptr + 7);
        float w22 = __ldg(wt_ptr + 8);

        // kh=0
        {
            int ih = oh * stride - pad;
            bool ih_valid = (ih >= 0 && ih < H);
            int in_row = base_in + ic_HW + ih * W;

            int iw0 = ow * stride - pad;
            int iw1 = iw0 + 1;
            int iw2 = iw0 + 2;

            float v0 = (ih_valid && iw0 >= 0 && iw0 < W) ? __ldg(input + in_row + iw0) : 0.0f;
            float v1 = (ih_valid && iw1 >= 0 && iw1 < W) ? __ldg(input + in_row + iw1) : 0.0f;
            float v2 = (ih_valid && iw2 >= 0 && iw2 < W) ? __ldg(input + in_row + iw2) : 0.0f;

            acc = __fmaf_rn(v0, w00, acc);
            acc = __fmaf_rn(v1, w01, acc);
            acc = __fmaf_rn(v2, w02, acc);
        }

        // kh=1
        {
            int ih = oh * stride - pad + 1;
            bool ih_valid = (ih >= 0 && ih < H);
            int in_row = base_in + ic_HW + ih * W;

            int iw0 = ow * stride - pad;
            int iw1 = iw0 + 1;
            int iw2 = iw0 + 2;

            float v0 = (ih_valid && iw0 >= 0 && iw0 < W) ? __ldg(input + in_row + iw0) : 0.0f;
            float v1 = (ih_valid && iw1 >= 0 && iw1 < W) ? __ldg(input + in_row + iw1) : 0.0f;
            float v2 = (ih_valid && iw2 >= 0 && iw2 < W) ? __ldg(input + in_row + iw2) : 0.0f;

            acc = __fmaf_rn(v0, w10, acc);
            acc = __fmaf_rn(v1, w11, acc);
            acc = __fmaf_rn(v2, w12, acc);
        }

        // kh=2
        {
            int ih = oh * stride - pad + 2;
            bool ih_valid = (ih >= 0 && ih < H);
            int in_row = base_in + ic_HW + ih * W;

            int iw0 = ow * stride - pad;
            int iw1 = iw0 + 1;
            int iw2 = iw0 + 2;

            float v0 = (ih_valid && iw0 >= 0 && iw0 < W) ? __ldg(input + in_row + iw0) : 0.0f;
            float v1 = (ih_valid && iw1 >= 0 && iw1 < W) ? __ldg(input + in_row + iw1) : 0.0f;
            float v2 = (ih_valid && iw2 >= 0 && iw2 < W) ? __ldg(input + in_row + iw2) : 0.0f;

            acc = __fmaf_rn(v0, w20, acc);
            acc = __fmaf_rn(v1, w21, acc);
            acc = __fmaf_rn(v2, w22, acc);
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

    conv2d_v3<<<grid, block>>>(input, weight, output,
        N, C_in, H, W, C_out, K, stride, pad);
    cudaDeviceSynchronize();
}
