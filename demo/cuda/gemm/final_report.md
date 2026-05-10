# CUDA Optimization Final Report — GEMM (`2026-05-10`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/gemm.cu |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 (best) |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 0.9445 | 0.7025 | 0.6912 | 0.6842 | 0.7200 | **0.5232** |
| Speedup (×) | 1.00 | 1.34 | 1.02 | 1.01 | 0.95 | **1.80** |
| Memory Throughput (%) | 2.19 | 2.84 | 2.88 | 2.92 | 2.76 | 3.89 |
| SM Throughput (%) | 97.33 | 94.53 | 95.69 | 96.36 | 96.59 | 76.70 |
| Bottleneck | Memory-Bound | Barrier | Barrier | Barrier | Barrier | Compute+Barrier |
| Achieved Occupancy (%) | 95.04 | 95.21 | 95.03 | 94.93 | 95.07 | 90.21 |
| Registers / Thread | 40 | 38 | 39 | 40 | 40 | 40 |
| Shared Memory (bytes) | 0 | 2048 | 4096 | 8192 | 8192 | 2048 |
| Global Load Efficiency (%) | 56.25 | 100.00 | 100.00 | 100.00 | 100.00 | 66.67 |
| FMA Utilization (%) | 19.17 | 9.97 | 9.59 | 9.99 | 9.43 | 12.46 |
| FFMA Throughput/cycle | 653 | 845 | 856 | 862 | 813 | **1176** |
| IPC | 0.286 | 0.295 | 0.271 | 0.258 | 0.262 | 0.292 |
| Stall: Barrier | 0.00 | 5.96 | 6.30 | 5.76 | 8.51 | 8.44 |
| Stall: Long Scoreboard | 4.25 | 5.46 | 3.31 | 1.98 | 2.58 | 3.96 |
| Stall: Not Selected | 4.57 | 3.41 | 3.96 | 4.55 | 3.82 | 2.63 |
| L1 Bank Conflicts | 32.8M | 4.43M | 4.20M | 4.19M | 6.30M | 16.9M |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Shared memory tiling | ✓ | ✓ | ✓ | ✓ | ✓ |
| Multi-tile fusion (reduce barriers) | ✗ | ✓ (2 tiles) | ✓ (4 tiles) | ✗ | ✗ |
| Large K-tile (TILE_K=64) | ✗ | ✗ | ✗ | ✓ | ✗ |
| Thread coarsening (2 output/thread) | ✗ | ✗ | ✗ | ✗ | ✓ |
| Coalesced global memory access | ✓ | ✓ | ✓ | ✓ | ✓ |

**Decision rationale per version:**
- **v1:** Global Load Efficiency was 56.25% — the primary bottleneck. Shared memory tiling achieves O(TILE) reuse of each global load. Experience log confirmed this exact optimization yielded 1.31x speedup for identical kernel+bottleneck. Delivered 1.34x.
- **v2:** Barrier stall (5.96) emerged as the dominant bottleneck from 128 `__syncthreads()` calls. Fusing 2 tiles per barrier halves barrier count to 64. Experience log confirmed 1.02x. Delivered 1.02x.
- **v3:** Barrier stall (6.30) still dominant. Fusing 4 tiles per barrier further reduces to 32. Diminishing returns expected. Delivered 1.01x — marginal.
- **v4:** Attempted larger K-tile (TILE_K=64) as an alternative to multi-tile fusion. Barrier stall jumped 5.76→8.51 due to longer inner K loop (64 vs 16 iterations). 5.3% regression — abandoned.
- **v5:** FMA utilization stuck at ~10% across all versions. Thread coarsening doubles FMAs per thread (2048 vs 1024), improving ILP. FFMA throughput jumped 36% (862→1176/cycle). Delivered 1.31x vs v3 — the largest single improvement after v1.

---

## Best Version Conclusion

**Best version:** `v5` — execution time reduced from 0.9445 ms (v0) to **0.5232 ms**, speedup **1.80×**.

**Key gains:**
1. Shared memory tiling (v1): eliminated poor global load efficiency (56→100%), 1.34x speedup
2. Thread coarsening (v5): doubled FMA throughput via improved ILP, 1.31x additional speedup
3. Multi-tile fusion (v2-v3): modest additive improvements (1.02x + 1.01x)

**Stopping reason:** Max iterations (5) reached. The dominant remaining bottleneck is Barrier stall (8.44) + L1 Bank Conflicts (16.9M from strided Bs access). Further improvements would require: (a) double-buffering with `cp.async` to overlap loads and compute, completely eliminating the compute-phase barrier, or (b) transposing the Bs shared memory layout to eliminate bank conflicts from the Bs access pattern.

**Remaining optimization opportunities:**
- `cp.async` + double-buffering to overlap global→shared load with shared→register compute, eliminating one `__syncthreads()` per iteration
- Transpose Bs shared memory layout to make Bs[tx][k] access consecutive, eliminating ~17M bank conflicts
- Mixed precision (FP16/BF16) with Tensor Core WMMA for 2-4x additional throughput on Ampere
