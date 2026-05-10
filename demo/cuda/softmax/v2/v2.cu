#include <cuda_runtime.h>
#include <cfloat>

#define WARP_SIZE 32
#define BLOCK_DIM 256

__global__ void softmax_v2(float* input, float* output, int N, int D) {
    extern __shared__ float smem[];
    float* exp_shared = smem; // D floats for exp values

    int row = blockIdx.x;
    if (row >= N) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;

    float* in_row  = input  + row * D;
    float* out_row = output + row * D;

    // Step 1: Find row-wise max via warp-level reduction (read from global)
    float max_val = -FLT_MAX;
    for (int i = tid; i < D; i += BLOCK_DIM) {
        max_val = fmaxf(max_val, in_row[i]);
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

    // Step 2: Compute exp, store to shared memory, compute sum (single pass)
    float sum_val = 0.0f;
    for (int i = tid; i < D; i += BLOCK_DIM) {
        float val = expf(in_row[i] - max_val);
        exp_shared[i] = val;
        sum_val += val;
    }
    __syncthreads(); // ensure all exp values are written to shared memory

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

    // Step 3: Normalize from shared memory, write to output (single global write pass)
    float inv_sum = 1.0f / sum_val;
    for (int i = tid; i < D; i += BLOCK_DIM) {
        out_row[i] = exp_shared[i] * inv_sum;
    }
}

extern "C" void solve(float* input, float* output, int N, int D) {
    size_t smem_bytes = D * sizeof(float);
    softmax_v2<<<N, BLOCK_DIM, smem_bytes>>>(input, output, N, D);
    cudaDeviceSynchronize();
}
