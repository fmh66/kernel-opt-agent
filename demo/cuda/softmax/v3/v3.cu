#include <cuda_runtime.h>
#include <cfloat>

#define WARP_SIZE 32
#define BLOCK_DIM 256

__global__ void softmax_v3(float* input, float* output, int N, int D) {
    extern __shared__ float smem[];
    float* exp_shared = smem;

    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    float* in_row  = input  + row * D;
    float* out_row = output + row * D;

    // Step 1: float4 vectorized max reduction
    float max_val = -FLT_MAX;
    for (int i = tid * 4; i < D; i += BLOCK_DIM * 4) {
        float4 v = *reinterpret_cast<const float4*>(in_row + i);
        max_val = fmaxf(max_val, v.x);
        max_val = fmaxf(max_val, v.y);
        max_val = fmaxf(max_val, v.z);
        max_val = fmaxf(max_val, v.w);
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    __shared__ float shared_max[BLOCK_DIM / WARP_SIZE];
    if (lane == 0) {
        shared_max[tid / WARP_SIZE] = max_val;
    }
    __syncthreads();

    max_val = shared_max[0];
    for (int i = 1; i < BLOCK_DIM / WARP_SIZE; i++) {
        max_val = fmaxf(max_val, shared_max[i]);
    }

    // Step 2: float4 vectorized exp + sum, store to shared memory
    float sum_val = 0.0f;
    for (int i = tid * 4; i < D; i += BLOCK_DIM * 4) {
        float4 v = *reinterpret_cast<const float4*>(in_row + i);
        float e0 = expf(v.x - max_val);
        float e1 = expf(v.y - max_val);
        float e2 = expf(v.z - max_val);
        float e3 = expf(v.w - max_val);
        exp_shared[i]     = e0;
        exp_shared[i + 1] = e1;
        exp_shared[i + 2] = e2;
        exp_shared[i + 3] = e3;
        sum_val += e0 + e1 + e2 + e3;
    }
    __syncthreads();

    // Warp reduction for sum
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_xor_sync(0xffffffff, sum_val, offset);
    }

    __shared__ float shared_sum[BLOCK_DIM / WARP_SIZE];
    if (lane == 0) {
        shared_sum[tid / WARP_SIZE] = sum_val;
    }
    __syncthreads();

    sum_val = shared_sum[0];
    for (int i = 1; i < BLOCK_DIM / WARP_SIZE; i++) {
        sum_val += shared_sum[i];
    }

    // Step 3: float4 vectorized normalize + write to global memory
    float inv_sum = 1.0f / sum_val;
    for (int i = tid * 4; i < D; i += BLOCK_DIM * 4) {
        float4 result;
        result.x = exp_shared[i]     * inv_sum;
        result.y = exp_shared[i + 1] * inv_sum;
        result.z = exp_shared[i + 2] * inv_sum;
        result.w = exp_shared[i + 3] * inv_sum;
        *reinterpret_cast<float4*>(out_row + i) = result;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    size_t smem_bytes = D * sizeof(float);
    softmax_v3<<<N, BLOCK_DIM, smem_bytes>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
