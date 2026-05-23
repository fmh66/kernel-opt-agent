#include <cuda_runtime.h>

__global__ void rmsnorm_v2(
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

    // float4 vectorized accumulation
    int vec_elems = D & ~3; // D rounded down to multiple of 4
    float sum_sq = 0.0f;

    for (int i = tid * 4; i < vec_elems; i += stride * 4) {
        const float4* in_vec = reinterpret_cast<const float4*>(in_row + i);
        float4 vals = *in_vec;
        sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }

    // Scalar tail for non-multiple-of-4 remainder
    for (int i = vec_elems + tid; i < D; i += stride) {
        float val = in_row[i];
        sum_sq += val * val;
    }

    // Warp-level reduction (butterfly)
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

    // float4 vectorized output write
    for (int i = tid * 4; i < vec_elems; i += stride * 4) {
        const float4* in_vec  = reinterpret_cast<const float4*>(in_row + i);
        const float4* g_vec   = reinterpret_cast<const float4*>(gamma + i);
        float4 out_vals;
        float4 in_vals  = *in_vec;
        float4 g_vals   = *g_vec;
        out_vals.x = in_vals.x * rms * g_vals.x;
        out_vals.y = in_vals.y * rms * g_vals.y;
        out_vals.z = in_vals.z * rms * g_vals.z;
        out_vals.w = in_vals.w * rms * g_vals.w;
        *reinterpret_cast<float4*>(out_row + i) = out_vals;
    }

    // Scalar tail
    for (int i = vec_elems + tid; i < D; i += stride) {
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
    rmsnorm_v2<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, N, D, eps);
    cudaDeviceSynchronize();
}
