#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16

__global__ void conv2d_v1(
    const float* __restrict__ input,
    const float* __restrict__ weight,
          float* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    extern __shared__ float smem_in[];

    int n  = blockIdx.z / C_out;
    int oc = blockIdx.z % C_out;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * TILE_W + tx;
    int oh = blockIdx.y * TILE_H + ty;

    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;
    bool valid = (oh < out_H && ow < out_W);

    int in_H = TILE_H + K - 1;
    int in_W = TILE_W + K - 1;
    int smem_size = in_H * in_W;

    int in_h_start = blockIdx.y * TILE_H * stride - pad;
    int in_w_start = blockIdx.x * TILE_W * stride - pad;

    float acc = 0.0f;
    int tid = ty * TILE_W + tx;

    for (int ic = 0; ic < C_in; ic++) {
        // Cooperative load of input tile into shared memory
        for (int idx = tid; idx < smem_size; idx += TILE_H * TILE_W) {
            int si = idx / in_W;
            int sj = idx % in_W;
            int ih = in_h_start + si;
            int iw = in_w_start + sj;
            smem_in[idx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[n * (C_in * H * W) + ic * (H * W) + ih * W + iw]
                : 0.0f;
        }
        __syncthreads();

        // Compute partial convolution using shared memory
        if (valid) {
            #pragma unroll
            for (int kh = 0; kh < K; kh++) {
                #pragma unroll
                for (int kw = 0; kw < K; kw++) {
                    acc += smem_in[(ty * stride + kh) * in_W + (tx * stride + kw)]
                         * weight[oc * (C_in * K * K) + ic * (K * K) + kh * K + kw];
                }
            }
        }
        __syncthreads();
    }

    if (valid) {
        int out_idx = n * (C_out * out_H * out_W) + oc * (out_H * out_W) + oh * out_W + ow;
        output[out_idx] = acc;
    }
}

extern "C" void solve(
    float* input, float* weight, float* output,
    int N, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    int out_H = (H + 2 * pad - K) / stride + 1;
    int out_W = (W + 2 * pad - K) / stride + 1;

    int in_H = TILE_H + K - 1;
    int in_W = TILE_W + K - 1;
    int smem_bytes = in_H * in_W * sizeof(float);

    dim3 block(TILE_W, TILE_H);
    dim3 grid(
        (out_W + TILE_W - 1) / TILE_W,
        (out_H + TILE_H - 1) / TILE_H,
        N * C_out
    );

    conv2d_v1<<<grid, block, smem_bytes>>>(input, weight, output,
        N, C_in, H, W, C_out, K, stride, pad);
    cudaDeviceSynchronize();
}
