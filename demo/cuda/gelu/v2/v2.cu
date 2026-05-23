#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(
    const float* __restrict__ input,
          float* __restrict__ output,
    int N)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= N) return;

    float4 in = reinterpret_cast<const float4*>(input)[blockIdx.x * blockDim.x + threadIdx.x];
    float4 out;

    int remaining = min(N - idx, 4);
    if (remaining == 4) {
        float cdf0 = 0.5f * (1.0f + tanhf(0.7978845608f * (in.x + 0.044715f * in.x * in.x * in.x)));
        float cdf1 = 0.5f * (1.0f + tanhf(0.7978845608f * (in.y + 0.044715f * in.y * in.y * in.y)));
        float cdf2 = 0.5f * (1.0f + tanhf(0.7978845608f * (in.z + 0.044715f * in.z * in.z * in.z)));
        float cdf3 = 0.5f * (1.0f + tanhf(0.7978845608f * (in.w + 0.044715f * in.w * in.w * in.w)));
        out.x = in.x * cdf0;
        out.y = in.y * cdf1;
        out.z = in.z * cdf2;
        out.w = in.w * cdf3;
        reinterpret_cast<float4*>(output)[blockIdx.x * blockDim.x + threadIdx.x] = out;
    } else {
        float elements[4] = {in.x, in.y, in.z, in.w};
        for (int i = 0; i < remaining; i++) {
            float x = elements[i];
            float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
            output[idx + i] = x * cdf;
        }
    }
}

extern "C" void solve(float* input, float* output, int N)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    gelu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
