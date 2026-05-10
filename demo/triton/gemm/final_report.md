# CUDA Optimization Final Report — `gemm.py` (`2026-05-10`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | N/A (Triton backend) |
| ncu | NVIDIA Nsight Compute |
| nsight-python | N/A |
| Triton | 3.4.0 |
| PyTorch | 2.6.0+cu124 |
| Kernel file | /home/kernel-opt-skill/test/gemm.py |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 (best) |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 0.1613 | 0.1645 | 0.1819 | 0.1916 | 0.1699 | **0.1587** |
| Speedup (x) | 1.00 | 0.98 | 0.89 | 0.84 | 0.95 | **1.02** |
| SM Throughput (% peak) | 51.71 | 51.76 | 47.47 | 80.88 | 46.77 | 53.42 |
| Memory Throughput (% peak) | 13.94 | 13.60 | 12.52 | 12.64 | 20.03 | 13.31 |
| FMA Pipe Utilization (% peak) | 59.53 | 59.25 | 49.11 | 37.80 | 51.92 | 57.32 |
| Tensor Core Utilization (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Achieved Occupancy (%) | 22.98 | 23.02 | 42.64 | 37.13 | 8.32 | 23.48 |
| Theoretical Occupancy (%) | 25.00 | 25.00 | 50.00 | 41.67 | 8.33 | 33.33 |
| Block Size (threads) | 128 | 128 | 256 | 128 | 128 | 128 |
| Grid Size | 256 | 256 | 256 | 1024 | 256 | 256 |
| Registers / Thread | 96 | 96 | 80 | 56 | 128 | 96 |
| Dynamic Shared Memory (bytes) | 32768 | 32768 | 32768 | 16384 | 65536 | 24576 |
| Stall: Short Scoreboard | 0.6988 | 0.7486 | 2.6402 | 1.1644 | 0.2383 | 0.5725 |
| Stall: Long Scoreboard | 0.1105 | 0.1158 | 0.2173 | 0.1853 | 0.0939 | 0.0815 |
| Stall: Barrier | 0.2222 | 0.1860 | 0.6421 | 1.0349 | 0.0410 | 0.3355 |
| Stall: Not Selected | 1.1350 | 1.1114 | 1.5744 | 1.2436 | 0.0000 | 1.1612 |
| L1 Bank Conflicts | 24355 | 21923 | 22255 | 46099 | 0 | 20455 |
| Shared Memory Bandwidth (bytes/s) | 2.96e12 | 2.91e12 | 4.61e12 | 6.04e12 | 2.67e12 | 3.86e12 |
| L2 Hit Rate (%) | 93.31 | 93.23 | 93.05 | 96.56 | 89.41 | 93.82 |
| Global Load Efficiency (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Branch Divergence (total) | 0 | 0 | 0 | 0 | 0 | 0 |
| Waves / SM | 1.02 | 1.02 | 1.02 | 2.44 | 3.05 | 0.76 |
| IPC | 0.7074 | 0.7120 | 0.6247 | 0.5727 | 0.6067 | 0.6980 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Coalesced memory access (make_block_ptr) | x | - | - | - | - |
| Higher occupancy (num_warps) | - | x | - | - | - |
| Finer grid parallelism (smaller tiles) | - | - | x | - | - |
| Larger K tile (reduce iterations) | - | - | - | x | - |
| Deeper SW pipelining (num_stages=4) | - | - | - | - | x |
| Tensor Core (TF32) | - | - | - | - | - |

**Decision rationale per version:**

- **v1 (make_block_ptr):** v0 Global Load Efficiency = 0% suggested the compiler was not generating coalesced vectorized loads. `tl.make_block_ptr` with `order=(1,0)` tells the compiler the contiguous dimension explicitly. However, the Triton compiler already inferred the same access pattern from manual pointer arithmetic — zero effect on all metrics.

- **v2 (num_warps=8):** v0 Achieved Occupancy = 23% (Theoretical = 25%) limited by shared memory (32KB/block -> 4 blocks/SM max). Doubling num_warps to 8 doubled threads per block but kept shared memory per block fixed, theoretically doubling occupancy to 50%. Regression: Short Scoreboard stall tripled (0.70 -> 2.64) due to more warps contending for shared memory banks. Classic "occupancy above all else" anti-pattern.

- **v3 (smaller tiles 64->32):** v0 grid size = 256 blocks / 84 SMs = ~3 blocks/SM, providing only 1 wave/SM. Reducing BLOCK_M/N from 64 to 32 increased grid to 1024 blocks for 2.4 waves/SM. SM utilization rose to 80.88% but FMA pipe dropped from 59% to 38%. More blocks meant 4x more total barriers and launch overhead per unit of actual compute.

- **v4 (larger BLOCK_K 32->64):** Halved K-loop iterations (32->16) and eliminated L1 bank conflicts (24K -> 0). Barrier stall dropped 82% (0.22->0.04), Short Scoreboard dropped 65%. But shared memory doubled to 64KB, collapsing occupancy from 23% to 8.3%. The stall reduction nearly compensated for the occupancy loss (-5.3% latency), but not enough to win.

- **v5 (BLOCK_K=16 + num_stages=4):** The winning variant. Halved BLOCK_K to 16, reducing per-stage shared memory from 16KB to 8KB. At 8KB/stage, 4-stage pipelining fit in 24KB SMEM (less than v0's 32KB for 2 stages). Deeper pipelining reduced Long Scoreboard by 26% and Short Scoreboard by 18% via better load/compute overlap. The narrower dot products (64x64x16 vs 64x64x32) reduced per-iteration shared memory contention. Barrier stall increased 51% due to 64 vs 32 iterations, partially offsetting gains.

---

## Best Version Conclusion

**Best version:** `v5` — execution time reduced from 0.1613 ms to 0.1587 ms, speedup **1.02x** (1.6% improvement).

Key gains:
- Deeper software pipelining (num_stages=4) better hid global load latency
- Narrower per-iteration dot products (BLOCK_K=16) reduced shared memory bank contention
- 25% less shared memory per block (24KB vs 32KB), raising theoretical occupancy from 25% to 33%

Stopping reason: Max iterations (5) reached. Modest improvement achieved; the kernel is near-optimal for this small 1024x1024 problem size given the IEEE FP32 precision constraint.

**Benchmark (v5 vs PyTorch):**
- v5 Triton: **0.1749 ms** (benchmark conditions differ slightly from NCU timing)
- PyTorch eager: 0.2214 ms — v5 is **21% faster**
- PyTorch compile: 0.2719 ms — v5 is **36% faster**

**Remaining optimization opportunities:**
- Problem size scaling: the 1024x1024 GEMM is small (all data fits in L2). Larger matrices (4096+) would benefit from swizzle for L2 hit rate and larger tiles for higher arithmetic intensity.
- TF32 Tensor Cores: switching to `input_precision="tf32"` would enable ~8x faster MMA instructions but failed correctness (mean rel error 0.00619 > rtol 0.001). Relaxing tolerance or using mixed precision would unlock significant speedup.
- autotuning: using `@triton.autotune` with a grid of BLOCK sizes across multiple shape ranges would automatically select optimal configs per shape.
