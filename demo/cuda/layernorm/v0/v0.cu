#include <cuda_runtime.h>

__global__ void naive_layernorm(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
          float* __restrict__ output,
    int N, int D, float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* in_row  = input  + row * D;
          float* out_row = output + row * D;

    float mean = 0.0f;
    for (int i = 0; i < D; i++) {
        mean += in_row[i];
    }
    mean /= D;

    float var = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = in_row[i] - mean;
        var += diff * diff;
    }
    var /= D;

    float inv_std = rsqrtf(var + eps);

    for (int i = 0; i < D; i++) {
        out_row[i] = (in_row[i] - mean) * inv_std * gamma[i] + beta[i];
    }
}

extern "C" void solve(
    float* input, float* gamma, float* beta, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float eps = 1e-5f;
    naive_layernorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, beta, output, N, D, eps);
    cudaDeviceSynchronize();
}
