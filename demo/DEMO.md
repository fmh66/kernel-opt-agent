# CUDA / Triton Kernel Optimization Demo

This page summarizes the current demo artifacts under `demo/cuda/*` and `demo/triton/*`.

**Primary environment**: NVIDIA RTX A6000 (CC 8.6 / sm_86), CUDA 12.6, Nsight Compute 2024.3.2.0, nsight-python 0.9.6. Most updated demos use Triton 3.6.0 and PyTorch 2.11.0+cu126; the Triton GEMM report was captured with Triton 3.4.0 and PyTorch 2.6.0+cu124.

## Overview

| Backend | Case | Shape | Best | Iteration Speedup | Benchmark vs PyTorch Eager | Correctness |
| --- | --- | --- | --- | ---: | ---: | --- |
| CUDA | Softmax | N=4096, D=4096 | v2 | 11.72x | 2.73x faster | PASS |
| CUDA | GEMM | M=K=N=1024 | v5 | 1.80x | 0.37x (slower) | PASS |
| CUDA | MHA | N=512, d_model=1024, heads=16 | v5 | 8.90x | 0.47x (slower) | PASS |
| Triton | GEMM | M=K=N=1024 | v5 | 1.02x | 1.27x faster | PASS |
| Triton | MHA | N=1024, d_model=1024, heads=16 | v5 | 731x | 4.76x faster | PASS |
| Triton | Softmax | N=1024, D=1024 | v0 | 1.00x | 1.88x faster | PASS |

---

## CUDA

### CUDA Softmax

Artifacts: `demo/cuda/softmax/`

**Shape**: N=4096, D=4096

| Version | Time (ms) | Speedup | Mem Throughput (%) | SM Throughput (%) | Occupancy (%) | Load Eff (%) | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 3.4769 | 1.00x | 13.35 | 1.21 | 16.66 | 12.50 | Memory | Naive row mapping, poor coalescing |
| v1 | 0.4516 | 7.70x | 92.38 | 9.08 | 95.75 | 100.00 | Memory | One block per row + warp reductions |
| v2 | **0.2967** | **11.72x** | 93.72 | 14.38 | 81.07 | 100.00 | Memory | Shared memory cache for exp values |
| v3 | 0.3018 | 11.52x | 93.37 | 6.86 | 80.53 | 100.00 | Memory | `float4` vectorization attempt |
| v4 | 0.2961 | 11.75x | 93.89 | 14.32 | 81.65 | 100.00 | Memory | `__expf` fast math |
| v5 | 0.2960 | 11.75x | N/A | N/A | N/A | N/A | Memory | `__launch_bounds__` |

**Benchmark**: v2 vs PyTorch

| Metric | v2 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | **0.2971** | 0.8112 | 0.7150 |
| Speedup | **2.73x** | - | 2.41x |
| Mem Throughput (%) | 93.60 | 93.58 | 93.62 |
| DRAM Bandwidth (GB/s) | 682 | 682 | 682 |
| Occupancy (%) | 80.43 | 80.52 | 80.51 |

Main result: v2 reaches the DRAM bandwidth roofline. Coalesced row blocks and shared-memory caching provide the practical gains; later instruction-level changes are neutral.

### CUDA GEMM

Artifacts: `demo/cuda/gemm/`

**Shape**: M=K=N=1024

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | Occupancy (%) | FMA Util (%) | Barrier Stall | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 0.9445 | 1.00x | 97.33 | 2.19 | 95.04 | 19.17 | 0.00 | Memory | Baseline GEMM |
| v1 | 0.7025 | 1.34x | 94.53 | 2.84 | 95.21 | 9.97 | 5.96 | Barrier | Shared-memory tiling |
| v2 | 0.6912 | 1.37x | 95.69 | 2.88 | 95.03 | 9.59 | 6.30 | Barrier | Fuse two K tiles per barrier |
| v3 | 0.6842 | 1.38x | 96.36 | 2.92 | 94.93 | 9.99 | 5.76 | Barrier | Fuse four K tiles per barrier |
| v4 | 0.7200 | 1.31x | 96.59 | 2.76 | 95.07 | 9.43 | 8.51 | Barrier | Larger TILE_K=64 attempt |
| v5 | **0.5232** | **1.80x** | 76.70 | 3.89 | 90.21 | 12.46 | 8.44 | Compute+Barrier | Thread coarsening |

**Benchmark**: v5 vs PyTorch

| Metric | v5 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | 0.5306 | **0.1977** | 0.2446 |
| Relative speed | 0.37x | - | 0.46x |
| SM Throughput (%) | 76.61 | 76.68 | 76.77 |
| Mem Throughput (%) | 3.88 | 3.88 | 3.89 |
| Occupancy (%) | 90.21 | 90.10 | 90.25 |

Main result: shared-memory tiling and thread coarsening improve the custom CUDA kernel, but it remains slower than PyTorch/cuBLAS because it does not use Tensor Cores or cuBLAS-level scheduling.

### CUDA MHA

Artifacts: `demo/cuda/mha/`

**Shape**: N=512, d_model=1024, num_heads=16

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | Occupancy (%) | Load Eff (%) | Long SB | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 7.7245 | 1.00x | 87.52 | 0.17 | 62.73 | 17.65 | 4.97 | Compute | Serial/under-parallelized baseline |
| v1 | 1.8770 | 4.12x | 19.11 | 0.72 | 64.41 | 21.96 | 30.37 | Latency | Thread-level QK parallelism |
| v2 | 1.9298 | 4.00x | 18.80 | 0.70 | 64.39 | 21.96 | 32.31 | Latency | Contiguous mapping attempt |
| v3 | 1.0956 | 7.05x | 19.84 | 1.32 | 64.25 | 66.33 | 35.02 | Latency | `float4` vectorized loads |
| v4 | 0.8696 | 8.88x | 24.24 | 1.63 | 65.31 | 66.33 | 44.67 | Latency | Loop unrolling / ILP |
| v5 | **0.8680** | **8.90x** | 24.34 | 1.68 | 65.05 | 66.33 | 44.51 | Latency | `const __restrict__` |

**Benchmark**: v5 vs PyTorch

| Metric | v5 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | 0.8944 | 0.4241 | **0.3422** |
| Relative speed | 0.47x | - | 0.38x |
| SM Throughput (%) | 24.20 | 24.07 | 24.20 |
| Mem Throughput (%) | 1.63 | 1.61 | 1.59 |
| Occupancy (%) | 65.09 | 65.09 | 65.09 |

Main result: the custom CUDA MHA gains 8.9x through parallel QK work, vectorized loads, and ILP, but still trails PyTorch because the remaining path is latency-bound and lacks FlashAttention/Tensor Core-level algorithmic structure.

---

## Triton

### Triton GEMM

Artifacts: `demo/triton/gemm/`

**Shape**: M=K=N=1024

| Version | Time (ms) | Speedup | SM Throughput (%) | Mem Throughput (%) | FMA Util (%) | Occupancy (%) | Regs/Thread | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 0.1613 | 1.00x | 51.71 | 13.94 | 59.53 | 22.98 | 96 | Balanced | Baseline 64x64x32 tiled FP32 GEMM |
| v1 | 0.1645 | 0.98x | 51.76 | 13.60 | 59.25 | 23.02 | 96 | Balanced | `make_block_ptr` attempt |
| v2 | 0.1819 | 0.89x | 47.47 | 12.52 | 49.11 | 42.64 | 80 | Shared-memory contention | `num_warps=8` |
| v3 | 0.1916 | 0.84x | 80.88 | 12.64 | 37.80 | 37.13 | 56 | Overhead | Smaller tiles / larger grid |
| v4 | 0.1699 | 0.95x | 46.77 | 20.03 | 51.92 | 8.32 | 128 | Low occupancy | Larger BLOCK_K=64 |
| v5 | **0.1587** | **1.02x** | 53.42 | 13.31 | 57.32 | 23.48 | 96 | Balanced | BLOCK_K=16 + 4-stage pipelining |

**Benchmark**: v5 vs PyTorch

| Metric | v5 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | **0.1749** | 0.2214 | 0.2719 |
| Speedup | **1.27x** | - | 1.55x |

Main result: v0 was already strong for this small FP32 GEMM. v5 gains 1.6% by reducing per-stage shared memory and improving software pipelining; TF32/Tensor Core mode is the major remaining opportunity if looser tolerance is allowed.

### Triton MHA

Artifacts: `demo/triton/mha/`

**Shape**: N=1024, d_model=1024, num_heads=16

| Version | Time (ms) | Speedup | Mem Throughput (%) | SM Throughput (%) | TC Util (%) | Occupancy (%) | Regs/Thread | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | 141.23 | 1.00x | 77.66 | 37.80 | 0.00 | 102.05 | 80 | Memory | Grid over `(H,N,d_k)`, redundant work |
| v1 | 3.891 | 36.3x | 90.46 | 26.75 | 0.00 | 101.74 | 80 | Memory | Fuse d dimension into one program |
| v2 | 0.284 | 497x | 78.08 | 35.26 | 45.25 | 100.11 | 140 | Memory | Query tiling + `tl.dot` Tensor Cores |
| v3 | 0.264 | 535x | 78.45 | 41.97 | 50.73 | 100.29 | 255 | Memory | Larger K/V tile, BLOCK_N=128 |
| v4 | 0.407 | 347x | 79.65 | 19.02 | 21.96 | 103.47 | 206 | Memory | Reduced query tile, BLOCK_I=32 |
| v5 | **0.193** | **731x** | 80.01 | 42.76 | 51.52 | 77.91 | 255 | Memory | Pre-scale Q to fuse inner-loop multiply |

**Benchmark**: v5 vs PyTorch

| Metric | v5 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | **0.1939** | 0.9232 | 0.8104 |
| Speedup | **4.76x** | - | 4.18x |
| Correctness | PASS | - | - |

Main result: v5 eliminates 64x redundant d-dimension work, adds query-side tiling with Tensor Cores, improves K/V tiling, and pre-scales Q to remove an inner-loop multiply. The updated benchmark is correctness PASS and 4.76x faster than PyTorch eager.

### Triton Softmax

Artifacts: `demo/triton/softmax/`

**Shape**: N=1024, D=1024

| Version | Time (ms) | Speedup | Mem Throughput (%) | SM Throughput (%) | Occupancy (%) | Regs/Thread | Bottleneck | Key Optimization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| v0 | **0.0399** | **1.00x** | 85.57 | 12.02 | 78.81 | 23 | Memory | Baseline already optimal |
| v1 | 0.0415 | 0.96x | 77.69 | 21.24 | 98.34 | 28 | Memory | 2 rows/block + 8 warps |
| v2 | 0.0438 | 0.91x | 79.89 | 10.22 | 52.78 | 26 | Memory | 2 rows/block + 4 warps |
| v3 | 0.0406 | 0.98x | 85.82 | 12.43 | 77.50 | 23 | Memory | `tl.softmax` builtin |
| v4 | 0.0646 | 0.62x | 83.12 | 22.85 | 93.44 | 20 | Memory | Single row + 8 warps |
| v5 | 0.0437 | 0.91x | 85.68 | 7.83 | 40.24 | 33 | Memory | Single row + 2 warps |

**Benchmark**: v0 vs PyTorch

| Metric | v0 | PyTorch Eager | PyTorch Compile |
| --- | ---: | ---: | ---: |
| Time (ms) | **0.0388** | 0.0731 | 0.1498 |
| Speedup | **1.88x** | - | 3.86x |
| Mem Throughput (%) | 74.06 | 73.08 | 73.75 |
| DRAM Bandwidth (GB/s) | 537 | 531 | 537 |
| Occupancy (%) | 19.75 | 19.98 | 20.03 |

Main result: the starting Triton kernel was already the best version. Warp count tuning, multi-row batching, and builtin replacement did not reduce memory traffic and therefore regressed or stayed neutral.
