#include <cuda_runtime.h>

__global__ void naive_rmsnorm(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
          float* __restrict__ output,
    int N, int D, float eps)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const float* in_row  = input  + row * D;
          float* out_row = output + row * D;

    float rms = 0.0f;
    for (int i = 0; i < D; i++) {
        rms += in_row[i] * in_row[i];
    }
    rms = rsqrtf(rms / D + eps);

    for (int i = 0; i < D; i++) {
        out_row[i] = in_row[i] * rms * gamma[i];
    }
}

extern "C" void solve(
    float* input, float* gamma, float* output,
    int N, int D)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    float eps = 1e-5f;
    naive_rmsnorm<<<blocksPerGrid, threadsPerBlock>>>(input, gamma, output, N, D, eps);
    cudaDeviceSynchronize();
}
