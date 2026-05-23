#include <cuda_runtime.h>

#define TILE_H 16
#define TILE_W 16

__global__ void conv2d_v2(
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
    int in_h_start = blockIdx.y * TILE_H - pad;
    int in_w_start = blockIdx.x * TILE_W - pad;

    float acc = 0.0f;

    int base_in = n * (C_in * H * W);
    int base_wt = oc * (C_in * K * K);

    for (int ic = 0; ic < C_in; ic++) {
        // Region 1: main tile (rows 0..15, cols 0..15) — all threads
        // coalesced: consecutive tx → consecutive global memory addresses
        {
            int ih = in_h_start + ty;
            int iw = in_w_start + tx;
            smem_in[ty * in_W + tx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[base_in + ic * (H * W) + ih * W + iw]
                : 0.0f;
        }

        // Region 2: right halo (rows 0..15, cols 16,17) — threads tx=0,1
        if (tx < (in_W - TILE_W)) {
            int ih = in_h_start + ty;
            int iw = in_w_start + TILE_W + tx;
            smem_in[ty * in_W + TILE_W + tx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[base_in + ic * (H * W) + ih * W + iw]
                : 0.0f;
        }

        // Region 3: bottom halo (rows 16,17, cols 0..15) — threads ty=0,1
        if (ty < (in_H - TILE_H)) {
            int ih = in_h_start + TILE_H + ty;
            int iw = in_w_start + tx;
            smem_in[(TILE_H + ty) * in_W + tx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[base_in + ic * (H * W) + ih * W + iw]
                : 0.0f;
        }

        // Region 4: bottom-right corner (rows 16,17, cols 16,17) — threads tx=0,1 && ty=0,1
        if (tx < (in_W - TILE_W) && ty < (in_H - TILE_H)) {
            int ih = in_h_start + TILE_H + ty;
            int iw = in_w_start + TILE_W + tx;
            smem_in[(TILE_H + ty) * in_W + TILE_W + tx] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                ? input[base_in + ic * (H * W) + ih * W + iw]
                : 0.0f;
        }

        __syncthreads();

        if (valid) {
            int ic_K2 = ic * (K * K);
            #pragma unroll
            for (int kh = 0; kh < K; kh++) {
                int smem_row = (ty + kh) * in_W + tx;
                int wt_row = ic_K2 + kh * K;
                #pragma unroll
                for (int kw = 0; kw < K; kw++) {
                    acc += smem_in[smem_row + kw]
                         * weight[base_wt + wt_row + kw];
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

    conv2d_v2<<<grid, block, smem_bytes>>>(input, weight, output,
        N, C_in, H, W, C_out, K, stride, pad);
    cudaDeviceSynchronize();
}
