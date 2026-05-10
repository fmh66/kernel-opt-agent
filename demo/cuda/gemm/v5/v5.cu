#include <cuda_runtime.h>

#define TILE 16
#define COARSE 2

__global__ void tiled_gemm_v5(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x * COARSE;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < K; t += TILE) {
        // Load A tile: each thread loads 2 elements (tx and tx+8)
        if (row < M && (t + tx) < K)
            As[ty][tx] = A[row * K + (t + tx)];
        else
            As[ty][tx] = 0.0f;

        if (row < M && (t + tx + 8) < K)
            As[ty][tx + 8] = A[row * K + (t + tx + 8)];
        else
            As[ty][tx + 8] = 0.0f;

        // Load B tile: each thread loads 2 elements (for cols tx*2 and tx*2+1)
        if ((t + ty) < K && (col) < N)
            Bs[ty][tx * 2] = B[(t + ty) * N + col];
        else
            Bs[ty][tx * 2] = 0.0f;

        if ((t + ty) < K && (col + 1) < N)
            Bs[ty][tx * 2 + 1] = B[(t + ty) * N + (col + 1)];
        else
            Bs[ty][tx * 2 + 1] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            float a_val = As[ty][k];
            acc0 += a_val * Bs[k][tx * 2];
            acc1 += a_val * Bs[k][tx * 2 + 1];
        }

        __syncthreads();
    }

    if (row < M) {
        if (col < N)
            C[row * N + col] = acc0;
        if ((col + 1) < N)
            C[row * N + col + 1] = acc1;
    }
}

extern "C" void solve(
    float* A, float* B, float* C,
    int M, int K, int N)
{
    dim3 threadsPerBlock(TILE / COARSE, TILE);

    dim3 blocksPerGrid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    tiled_gemm_v5<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
