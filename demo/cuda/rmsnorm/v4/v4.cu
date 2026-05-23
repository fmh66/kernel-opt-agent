#include <cuda_runtime.h>

__global__ void rmsnorm_v4(
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

    // 2-way ILP: process 2 elements per iteration, loads through read-only cache
    float sum_sq = 0.0f;
    int i = tid;
    for (; i + stride < D; i += stride * 2) {
        float v0 = __ldg(in_row + i);
        float v1 = __ldg(in_row + i + stride);
        sum_sq += v0 * v0 + v1 * v1;
    }
    for (; i < D; i += stride) {
        float v0 = __ldg(in_row + i);
        sum_sq += v0 * v0;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    }

    __shared__ float s_partial[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_partial[warp_id] = sum_sq;
    }
    __syncthreads();

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

    // 2-way ILP output
    i = tid;
    for (; i + stride < D; i += stride * 2) {
        float v0 = __ldg(in_row + i);
        float v1 = __ldg(in_row + i + stride);
        float g0 = __ldg(gamma + i);
        float g1 = __ldg(gamma + i + stride);
        out_row[i]           = v0 * rms * g0;
        out_row[i + stride]  = v1 * rms * g1;
    }
    for (; i < D; i += stride) {
        out_row[i] = __ldg(in_row + i) * rms * __ldg(gamma + i);
    }
}

extern "C" void solve(
    float* input, float* gamma, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = N;
    float eps = 1e-5f;
    rmsnorm_v4<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, N, D, eps);
    cudaDeviceSynchronize();
}
