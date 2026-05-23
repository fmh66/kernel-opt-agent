#include <cuda_runtime.h>

__global__ void conv2d_v4(
    const float* __restrict__ input,
    const float* __restrict__ weight,
          float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    int n  = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow0 = blockIdx.x * (blockDim.x * 2) + tx;
    int ow1 = ow0 + blockDim.x;
    int oh  = blockIdx.y * blockDim.y + ty;

    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;
    bool valid0 = (oh < out_H && ow0 < out_W);
    bool valid1 = (oh < out_H && ow1 < out_W);

    if (!valid0 && !valid1) return;

    float acc0 = 0.0f, acc1 = 0.0f;
    int base_in = n * (C_in * H * W);
    int base_wt = oc * (C_in * K * K);

    for (int ic = 0; ic < C_in; ic++) {
        int ic_HW = ic * (H * W);
        int ic_K2 = ic * (K * K);
        const float* wt_ptr = weight + base_wt + ic_K2;

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
            int in_row = ic_HW + ih * W;

            int iw00 = ow0 * stride - pad, iw01 = iw00 + 1, iw02 = iw00 + 2;
            int iw10 = ow1 * stride - pad, iw11 = iw10 + 1, iw12 = iw10 + 2;

            if (valid0) {
                float v0 = (ih_valid && iw00 >= 0 && iw00 < W) ? __ldg(input + base_in + in_row + iw00) : 0.0f;
                float v1 = (ih_valid && iw01 >= 0 && iw01 < W) ? __ldg(input + base_in + in_row + iw01) : 0.0f;
                float v2 = (ih_valid && iw02 >= 0 && iw02 < W) ? __ldg(input + base_in + in_row + iw02) : 0.0f;
                acc0 = __fmaf_rn(v0, w00, acc0);
                acc0 = __fmaf_rn(v1, w01, acc0);
                acc0 = __fmaf_rn(v2, w02, acc0);
            }
            if (valid1) {
                float v0 = (ih_valid && iw10 >= 0 && iw10 < W) ? __ldg(input + base_in + in_row + iw10) : 0.0f;
                float v1 = (ih_valid && iw11 >= 0 && iw11 < W) ? __ldg(input + base_in + in_row + iw11) : 0.0f;
                float v2 = (ih_valid && iw12 >= 0 && iw12 < W) ? __ldg(input + base_in + in_row + iw12) : 0.0f;
                acc1 = __fmaf_rn(v0, w00, acc1);
                acc1 = __fmaf_rn(v1, w01, acc1);
                acc1 = __fmaf_rn(v2, w02, acc1);
            }
        }

        // kh=1
        {
            int ih = oh * stride - pad + 1;
            bool ih_valid = (ih >= 0 && ih < H);
            int in_row = ic_HW + ih * W;

            int iw00 = ow0 * stride - pad, iw01 = iw00 + 1, iw02 = iw00 + 2;
            int iw10 = ow1 * stride - pad, iw11 = iw10 + 1, iw12 = iw10 + 2;

            if (valid0) {
                float v0 = (ih_valid && iw00 >= 0 && iw00 < W) ? __ldg(input + base_in + in_row + iw00) : 0.0f;
                float v1 = (ih_valid && iw01 >= 0 && iw01 < W) ? __ldg(input + base_in + in_row + iw01) : 0.0f;
                float v2 = (ih_valid && iw02 >= 0 && iw02 < W) ? __ldg(input + base_in + in_row + iw02) : 0.0f;
                acc0 = __fmaf_rn(v0, w10, acc0);
                acc0 = __fmaf_rn(v1, w11, acc0);
                acc0 = __fmaf_rn(v2, w12, acc0);
            }
            if (valid1) {
                float v0 = (ih_valid && iw10 >= 0 && iw10 < W) ? __ldg(input + base_in + in_row + iw10) : 0.0f;
                float v1 = (ih_valid && iw11 >= 0 && iw11 < W) ? __ldg(input + base_in + in_row + iw11) : 0.0f;
                float v2 = (ih_valid && iw12 >= 0 && iw12 < W) ? __ldg(input + base_in + in_row + iw12) : 0.0f;
                acc1 = __fmaf_rn(v0, w10, acc1);
                acc1 = __fmaf_rn(v1, w11, acc1);
                acc1 = __fmaf_rn(v2, w12, acc1);
            }
        }

        // kh=2
        {
            int ih = oh * stride - pad + 2;
            bool ih_valid = (ih >= 0 && ih < H);
            int in_row = ic_HW + ih * W;

            int iw00 = ow0 * stride - pad, iw01 = iw00 + 1, iw02 = iw00 + 2;
            int iw10 = ow1 * stride - pad, iw11 = iw10 + 1, iw12 = iw10 + 2;

            if (valid0) {
                float v0 = (ih_valid && iw00 >= 0 && iw00 < W) ? __ldg(input + base_in + in_row + iw00) : 0.0f;
                float v1 = (ih_valid && iw01 >= 0 && iw01 < W) ? __ldg(input + base_in + in_row + iw01) : 0.0f;
                float v2 = (ih_valid && iw02 >= 0 && iw02 < W) ? __ldg(input + base_in + in_row + iw02) : 0.0f;
                acc0 = __fmaf_rn(v0, w20, acc0);
                acc0 = __fmaf_rn(v1, w21, acc0);
                acc0 = __fmaf_rn(v2, w22, acc0);
            }
            if (valid1) {
                float v0 = (ih_valid && iw10 >= 0 && iw10 < W) ? __ldg(input + base_in + in_row + iw10) : 0.0f;
                float v1 = (ih_valid && iw11 >= 0 && iw11 < W) ? __ldg(input + base_in + in_row + iw11) : 0.0f;
                float v2 = (ih_valid && iw12 >= 0 && iw12 < W) ? __ldg(input + base_in + in_row + iw12) : 0.0f;
                acc1 = __fmaf_rn(v0, w20, acc1);
                acc1 = __fmaf_rn(v1, w21, acc1);
                acc1 = __fmaf_rn(v2, w22, acc1);
            }
        }
    }

    int out_idx_base = n * (C_out * out_H * out_W) + oc * (out_H * out_W) + oh * out_W;
    if (valid0) output[out_idx_base + ow0] = acc0;
    if (valid1) output[out_idx_base + ow1] = acc1;
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
        (out_W + block.x * 2 - 1) / (block.x * 2),
        (out_H + block.y - 1) / block.y,
        N * C_out
    );

    conv2d_v4<<<grid, block>>>(input, weight, output,
        N, C_in, H, W, C_out, K, stride, pad);
    cudaDeviceSynchronize();
}
