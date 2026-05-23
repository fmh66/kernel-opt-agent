#include <cuda_runtime.h>

__global__ void rmsnorm_v1(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
          float* __restrict__ output,
    int N, int D, float eps)
{
    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* in_row  = input  + row * D;
          float* out_row = output + row * D;

    float sum_sq = 0.0f;
    for (int i = tid; i < D; i += stride) {
        float val = in_row[i];
        sum_sq += val * val;
    }

    // Warp-level reduction (butterfly)
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_partial[32]; // max 32 warps per block (1024 threads)
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_partial[warp_id] = sum_sq;
    }
    __syncthreads();

    // First warp reduces partial sums from all warps
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        sum_sq = (lane_id < num_warps) ? s_partial[lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
        }
        if (lane_id == 0) {
            s_partial[0] = rsqrtf(sum_sq / D + eps);
        }
    }
    __syncthreads();
    float rms = s_partial[0];

    // Coalesced write
    for (int i = tid; i < D; i += stride) {
        out_row[i] = in_row[i] * rms * gamma[i];
    }
}

extern "C" void solve(
    float* input, float* gamma, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    float eps = 1e-5f;
    rmsnorm_v1<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, N, D, eps);
    cudaDeviceSynchronize();
}
