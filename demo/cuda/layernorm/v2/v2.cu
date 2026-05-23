#include <cuda_runtime.h>

__global__ void layernorm_v2(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
          float* __restrict__ output,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;

    const float* in_row  = input  + row * D;
          float* out_row = output + row * D;

    extern __shared__ float smem[];
    int num_warps = blockDim.x >> 5;
    float* warp_sum_x  = smem;
    float* warp_sum_x2 = smem + num_warps;
    float* block_scalars = smem + 2 * num_warps;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // --- single pass: accumulate sum_x and sum_x2 ---
    float sum_x = 0.0f, sum_x2 = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float x = in_row[i];
        sum_x  += x;
        sum_x2 += x * x;
    }

    // reduce sum_x within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x += __shfl_down_sync(0xffffffff, sum_x, offset);
    }
    // reduce sum_x2 within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, offset);
    }

    if (lane_id == 0) {
        warp_sum_x [warp_id] = sum_x;
        warp_sum_x2[warp_id] = sum_x2;
    }
    __syncthreads();

    // warp 0 reduces across warps
    if (warp_id == 0) {
        float wsx  = (lane_id < num_warps) ? warp_sum_x [lane_id] : 0.0f;
        float wsx2 = (lane_id < num_warps) ? warp_sum_x2[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            wsx  += __shfl_down_sync(0xffffffff, wsx,  offset);
            wsx2 += __shfl_down_sync(0xffffffff, wsx2, offset);
        }
        if (lane_id == 0) {
            float mean = wsx / D;
            float var = wsx2 / D - mean * mean;
            block_scalars[0] = mean;
            block_scalars[1] = rsqrtf(var + eps);
        }
    }
    __syncthreads();

    float mean    = block_scalars[0];
    float inv_std = block_scalars[1];

    // --- normalize ---
    for (int i = tid; i < D; i += blockDim.x) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

extern "C" void solve(
    float* input, float* gamma, float* beta, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    float eps = 1e-5f;
    int smem_size = ((threadsPerBlock >> 5) * 2 + 2) * sizeof(float);
    layernorm_v2<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, gamma, beta, output, N, D, eps);
    cudaDeviceSynchronize();
}
