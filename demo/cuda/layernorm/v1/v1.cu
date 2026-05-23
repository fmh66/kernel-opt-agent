#include <cuda_runtime.h>

__global__ void layernorm_v1(
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
    float* warp_sums = smem;
    float* warp_vars = smem + num_warps;
    float* block_scalars = smem + 2 * num_warps;

    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    // --- mean ---
    float sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        sum += in_row[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    float mean;
    if (warp_id == 0) {
        float wsum = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            wsum += __shfl_down_sync(0xffffffff, wsum, offset);
        }
        if (lane_id == 0) {
            block_scalars[0] = wsum / D;
        }
    }
    __syncthreads();
    mean = block_scalars[0];

    // --- var ---
    float var = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        float diff = in_row[i] - mean;
        var += diff * diff;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }

    if (lane_id == 0) {
        warp_vars[warp_id] = var;
    }
    __syncthreads();

    float inv_std;
    if (warp_id == 0) {
        float wvar = (lane_id < num_warps) ? warp_vars[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            wvar += __shfl_down_sync(0xffffffff, wvar, offset);
        }
        if (lane_id == 0) {
            block_scalars[1] = rsqrtf(wvar / D + eps);
        }
    }
    __syncthreads();
    inv_std = block_scalars[1];

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
    layernorm_v1<<<blocksPerGrid, threadsPerBlock, smem_size>>>(input, gamma, beta, output, N, D, eps);
    cudaDeviceSynchronize();
}
