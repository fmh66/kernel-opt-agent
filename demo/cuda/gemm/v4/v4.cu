#include <cuda_runtime.h>

#define TILE_M 16
#define TILE_N 16
#define TILE_K 64

__global__ void tiled_gemm_v4(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int M, int K, int N)
{
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    float acc = 0.0f;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    for (int t = 0; t < K; t += TILE_K) {
        // Cooperative load of A tile (M x K): 4 elements per thread
        {
            int k0 = t + tx;
            int k1 = t + 16 + tx;
            int k2 = t + 32 + tx;
            int k3 = t + 48 + tx;
            if (row < M) {
                As[ty][tx]      = (k0 < K) ? A[row * K + k0] : 0.0f;
                As[ty][tx + 16] = (k1 < K) ? A[row * K + k1] : 0.0f;
                As[ty][tx + 32] = (k2 < K) ? A[row * K + k2] : 0.0f;
                As[ty][tx + 48] = (k3 < K) ? A[row * K + k3] : 0.0f;
            } else {
                As[ty][tx]      = 0.0f;
                As[ty][tx + 16] = 0.0f;
                As[ty][tx + 32] = 0.0f;
                As[ty][tx + 48] = 0.0f;
            }
        }

        // Cooperative load of B tile (K x N): 4 elements per thread
        {
            int k0 = t + ty;
            int k1 = t + 16 + ty;
            int k2 = t + 32 + ty;
            int k3 = t + 48 + ty;
            if (col < N) {
                Bs[ty][tx]      = (k0 < K) ? B[k0 * N + col] : 0.0f;
                Bs[ty + 16][tx] = (k1 < K) ? B[k1 * N + col] : 0.0f;
                Bs[ty + 32][tx] = (k2 < K) ? B[k2 * N + col] : 0.0f;
                Bs[ty + 48][tx] = (k3 < K) ? B[k3 * N + col] : 0.0f;
            } else {
                Bs[ty][tx]      = 0.0f;
                Bs[ty + 16][tx] = 0.0f;
                Bs[ty + 32][tx] = 0.0f;
                Bs[ty + 48][tx] = 0.0f;
            }
        }

        __syncthreads();

        // Compute: 64 FMAs from shared memory
        for (int k = 0; k < TILE_K; k++) {
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
    dim3 threadsPerBlock(TILE_N, TILE_M);

    dim3 blocksPerGrid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M
    );

    tiled_gemm_v4<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, K, N);
    cudaDeviceSynchronize();
}
