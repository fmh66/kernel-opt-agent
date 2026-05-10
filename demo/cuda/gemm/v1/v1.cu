#include <cuda_runtime.h>

#define TILE 16

__global__ void tiled_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float acc = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < K; t += TILE) {
        // Cooperative load of A tile (row,col) = (row, t+tx)
        if (row < M && (t + tx) < K)
            As[ty][tx] = A[row * K + (t + tx)];
        else
            As[ty][tx] = 0.0f;

        // Cooperative load of B tile (row,col) = (t+ty, col)
        if ((t + ty) < K && col < N)
            Bs[ty][tx] = B[(t + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product from shared memory
        for (int k = 0; k < TILE; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

extern "C" void solve(
    float* A, float* B, float* C,
    int M, int K, int N)
{
    dim3 threadsPerBlock(TILE, TILE);

    dim3 blocksPerGrid(
        (N + TILE - 1) / TILE,
        (M + TILE - 1) / TILE
    );

    tiled_gemm<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
