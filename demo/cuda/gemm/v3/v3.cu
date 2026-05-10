#include <cuda_runtime.h>

#define TILE 16

__global__ void tiled_gemm_v3(
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
    __shared__ float As2[TILE][TILE];
    __shared__ float Bs2[TILE][TILE];
    __shared__ float As3[TILE][TILE];
    __shared__ float Bs3[TILE][TILE];

    float acc = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < K; t += 4 * TILE) {
        // Load tile 0
        if (row < M && (t + tx) < K)
            As0[ty][tx] = A[row * K + (t + tx)];
        else
            As0[ty][tx] = 0.0f;
        if ((t + ty) < K && col < N)
            Bs0[ty][tx] = B[(t + ty) * N + col];
        else
            Bs0[ty][tx] = 0.0f;

        // Load tile 1
        int t1 = t + TILE;
        if (row < M && (t1 + tx) < K)
            As1[ty][tx] = A[row * K + (t1 + tx)];
        else
            As1[ty][tx] = 0.0f;
        if ((t1 + ty) < K && col < N)
            Bs1[ty][tx] = B[(t1 + ty) * N + col];
        else
            Bs1[ty][tx] = 0.0f;

        // Load tile 2
        int t2 = t + 2 * TILE;
        if (row < M && (t2 + tx) < K)
            As2[ty][tx] = A[row * K + (t2 + tx)];
        else
            As2[ty][tx] = 0.0f;
        if ((t2 + ty) < K && col < N)
            Bs2[ty][tx] = B[(t2 + ty) * N + col];
        else
            Bs2[ty][tx] = 0.0f;

        // Load tile 3
        int t3 = t + 3 * TILE;
        if (row < M && (t3 + tx) < K)
            As3[ty][tx] = A[row * K + (t3 + tx)];
        else
            As3[ty][tx] = 0.0f;
        if ((t3 + ty) < K && col < N)
            Bs3[ty][tx] = B[(t3 + ty) * N + col];
        else
            Bs3[ty][tx] = 0.0f;

        __syncthreads();

        // Compute all 4 tiles from shared memory
        for (int k = 0; k < TILE; k++) {
            acc += As0[ty][k] * Bs0[k][tx];
            acc += As1[ty][k] * Bs1[k][tx];
            acc += As2[ty][k] * Bs2[k][tx];
            acc += As3[ty][k] * Bs3[k][tx];
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

    tiled_gemm_v3<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
