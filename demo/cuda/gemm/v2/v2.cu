#include <cuda_runtime.h>

#define TILE 16

__global__ void tiled_gemm_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As0[TILE][TILE];
    __shared__ float Bs0[TILE][TILE];
    __shared__ float As1[TILE][TILE];
    __shared__ float Bs1[TILE][TILE];

    float acc = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < K; t += 2 * TILE) {
        // Load tile 0: A[row][t+tx], B[t+ty][col]
        if (row < M && (t + tx) < K)
            As0[ty][tx] = A[row * K + (t + tx)];
        else
            As0[ty][tx] = 0.0f;

        if ((t + ty) < K && col < N)
            Bs0[ty][tx] = B[(t + ty) * N + col];
        else
            Bs0[ty][tx] = 0.0f;

        // Load tile 1: A[row][t+TILE+tx], B[t+TILE+ty][col]
        if (row < M && (t + TILE + tx) < K)
            As1[ty][tx] = A[row * K + (t + TILE + tx)];
        else
            As1[ty][tx] = 0.0f;

        if ((t + TILE + ty) < K && col < N)
            Bs1[ty][tx] = B[(t + TILE + ty) * N + col];
        else
            Bs1[ty][tx] = 0.0f;

        __syncthreads();

        // Compute both tiles from shared memory
        for (int k = 0; k < TILE; k++) {
            acc += As0[ty][k] * Bs0[k][tx];
            acc += As1[ty][k] * Bs1[k][tx];
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

    tiled_gemm_v2<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
