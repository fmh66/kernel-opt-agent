#include <cuda_runtime.h>

__global__ void rmsnorm_v5(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
          float* __restrict__ output,
    int N, int D, float eps)
{
    int row0 = blockIdx.x * 2;
    if (row0 >= N) return;
    int row1 = row0 + 1;
    bool has_row1 = (row1 < N);

    int tid = threadIdx.x;
    int stride = blockDim.x;

    const float* in_row0  = input  + row0 * D;
    const float* in_row1  = input  + row1 * D;
          float* out_row0 = output + row0 * D;
          float* out_row1 = output + row1 * D;

    float sum_sq0 = 0.0f;
    float sum_sq1 = 0.0f;

    for (int i = tid; i < D; i += stride) {
        float v0 = in_row0[i];
        sum_sq0 += v0 * v0;
        if (has_row1) {
            float v1 = in_row1[i];
            sum_sq1 += v1 * v1;
        }
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq0 += __shfl_xor_sync(0xffffffff, sum_sq0, offset);
        if (has_row1) {
            sum_sq1 += __shfl_xor_sync(0xffffffff, sum_sq1, offset);
        }
    }

    __shared__ float s_partial[2][32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    if (lane_id == 0) {
        s_partial[0][warp_id] = sum_sq0;
        if (has_row1) {
            s_partial[1][warp_id] = sum_sq1;
        }
    }
    __syncthreads();

    float rms0, rms1;
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;

        sum_sq0 = (lane_id < num_warps) ? s_partial[0][lane_id] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_sq0 += __shfl_xor_sync(0xffffffff, sum_sq0, offset);
        }
        rms0 = rsqrtf(sum_sq0 / D + eps);

        if (has_row1) {
            sum_sq1 = (lane_id < num_warps) ? s_partial[1][lane_id] : 0.0f;
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum_sq1 += __shfl_xor_sync(0xffffffff, sum_sq1, offset);
            }
            rms1 = rsqrtf(sum_sq1 / D + eps);
        }

        if (lane_id == 0) {
            s_partial[0][0] = rms0;
            if (has_row1) s_partial[1][0] = rms1;
        }
    }
    __syncthreads();
    rms0 = s_partial[0][0];
    rms1 = s_partial[1][0];

    for (int i = tid; i < D; i += stride) {
        out_row0[i] = in_row0[i] * rms0 * gamma[i];
        if (has_row1) {
            out_row1[i] = in_row1[i] * rms1 * gamma[i];
        }
    }
}

extern "C" void solve(
    float* input, float* gamma, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + 1) / 2;
    float eps = 1e-5f;
    rmsnorm_v5<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, N, D, eps);
    cudaDeviceSynchronize();
}
