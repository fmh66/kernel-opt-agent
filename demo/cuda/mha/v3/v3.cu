#include <cuda_runtime.h>
#include <math.h>

__global__ void multi_head_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int N, int d_model, int h, int d_k)
{
    int head = blockIdx.x;
    int i    = blockIdx.y;
    int tid  = threadIdx.x;

    if (tid >= d_k) return;

    extern __shared__ float scores[];

    float scale = rsqrtf((float)d_k);

    const float4* q_ptr4 = reinterpret_cast<const float4*>(Q + i * d_model + head * d_k);

    // Each thread computes scores for strided key positions
    float local_max = -INFINITY;
    for (int j = tid; j < N; j += d_k) {
        const float4* k_ptr4 = reinterpret_cast<const float4*>(K + j * d_model + head * d_k);
        float dot = 0.0f;
        for (int d4 = 0; d4 < d_k / 4; d4++) {
            float4 qv = q_ptr4[d4];
            float4 kv = k_ptr4[d4];
            dot += qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        }
        scores[j] = dot * scale;
        local_max = fmaxf(local_max, scores[j]);
    }

    // Warp-level max reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
    }

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    int num_warps = (d_k + warpSize - 1) / warpSize;

    // Cross-warp max reduction via shared memory
    if (lane_id == 0) {
        scores[N + warp_id] = local_max;
    }
    __syncthreads();

    float global_max = scores[N];
    for (int w = 1; w < num_warps; w++) {
        global_max = fmaxf(global_max, scores[N + w]);
    }

    // Compute exp scores and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += d_k) {
        scores[j] = expf(scores[j] - global_max);
        local_sum += scores[j];
    }

    // Warp-level sum reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }

    // Cross-warp sum reduction via shared memory
    if (lane_id == 0) {
        scores[N + warp_id] = local_sum;
    }
    __syncthreads();

    float global_sum = 0.0f;
    for (int w = 0; w < num_warps; w++) {
        global_sum += scores[N + w];
    }

    // Normalize scores
    float inv_sum = 1.0f / global_sum;
    for (int j = tid; j < N; j += d_k) {
        scores[j] *= inv_sum;
    }

    __syncthreads();

    // Compute output: weighted sum of V vectors
    float val = 0.0f;
    for (int j = 0; j < N; j++) {
        val += scores[j] * V[j * d_model + head * d_k + tid];
    }

    output[i * d_model + head * d_k + tid] = val;
}

extern "C" void solve(const float* Q, const float* K, const float* V,
                      float* output, int N, int d_model, int num_heads)
{
    int d_k = d_model / num_heads;
    int num_warps = (d_k + 31) / 32;

    dim3 grid(num_heads, N);
    dim3 block(d_k);
    size_t shared_mem = (N + num_warps) * sizeof(float);

    multi_head_attention_kernel<<<grid, block, shared_mem>>>(
        Q, K, V, output, N, d_model, num_heads, d_k);

    cudaDeviceSynchronize();
}
