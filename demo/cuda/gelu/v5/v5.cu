#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int idx = tid; idx < N; idx += stride) {
        float x = input[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        output[idx] = x * cdf;
    }
}

extern "C" void solve(float* input, float* output, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = 168;
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
