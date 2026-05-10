# CUDA Optimization Final Report — MHA (`2026-05-10`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/mha.cu |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 (best) |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 7.7245 | 1.8770 | 1.9298 | 1.0956 | 0.8696 | 0.8680 |
| Speedup (x) | 1.00 | 4.12 | 4.00 | 7.05 | 8.88 | 8.90 |
| SM Throughput (% of peak) | 87.52 | 19.11 | 18.80 | 19.84 | 24.24 | 24.34 |
| Memory Throughput (% of peak) | 0.17 | 0.72 | 0.70 | 1.32 | 1.63 | 1.68 |
| Global Load Efficiency (%) | 17.65 | 21.96 | 21.96 | 66.33 | 66.33 | 66.33 |
| L1 Hit Rate (%) | 89.08 | 75.19 | 67.18 | 19.92 | 44.16 | 43.99 |
| L2 Hit Rate (%) | 99.61 | 99.73 | 99.85 | 99.89 | 99.45 | 100.52 |
| Achieved Occupancy (%) | 62.73 | 64.41 | 64.39 | 64.25 | 65.31 | 65.05 |
| Theoretical Occupancy (%) | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 |
| Registers / Thread | 48 | 40 | 40 | 56 | 44 | 44 |
| IPC | 0.259 | 0.079 | 0.078 | 0.099 | 0.141 | 0.141 |
| Issue Slot Utilization (%) | 25.89 | 7.92 | 7.77 | 9.92 | 14.13 | 14.13 |
| Warp Execution Efficiency (%) | 1.73 | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| Stall: Barrier (%) | 14.41 | 1.36 | 1.50 | 2.09 | 0.24 | 0.25 |
| Stall: Long Scoreboard (%) | 4.97 | 30.37 | 32.31 | 35.02 | 44.67 | 44.51 |
| L1 Bank Conflicts (total) | 13,612 | 227M | 217M | 36.8M | 43.8M | 43.7M |
| FFMA Throughput (per cycle) | 39.3 | 157.8 | 155.3 | 259.9 | 317.5 | 318.9 |
| Bottleneck | Compute | Latency | Latency | Latency | Latency | Latency |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Thread-level parallelization (warp reduction) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Contiguous thread-to-data mapping | ✗ | ✓ | ✗ | ✗ | ✗ |
| Vectorized loads (`float4`) | ✗ | ✗ | ✓ | ✓ | ✓ |
| ILP (loop unrolling) | ✗ | ✗ | ✗ | ✓ | ✓ |
| `const __restrict__` / LDG path | ✗ | ✗ | ✗ | ✗ | ✓ |

**Decision rationale per version:**
- **v1:** Distributed QK score computation from thread-0-only to all 64 threads using strided key-position assignment with warp-shuffle max/sum reductions. Warp Execution Efficiency 1.73% → 32.00%, Stall Barrier 14.41% → 1.36%. Latency ↓ 75.7%.
- **v2:** Attempted contiguous thread-to-key mapping to reduce shared memory bank conflicts. 2.8% regression — L1 bank conflicts unchanged (227M → 217M) because the metric tracks L1 data cache (global memory), not shared memory. The contiguous mapping does not change global memory access patterns.
- **v3:** Added float4 vectorized loads in the QK dot-product loop, reducing 4 scalar loads to one 16-byte load instruction. Global Load Efficiency 22% → 66%, L1 Bank Conflicts 227M → 37M (-84%). Latency ↓ 41.6% vs v1.
- **v4:** Added `#pragma unroll` on the output weighted-sum loop and reduction loops. IPC 0.099 → 0.141 (+42%), L1 Hit Rate 20% → 44%, FFMA throughput 260 → 318. Latency ↓ 20.6% vs v3.
- **v5:** Added `const __restrict__` to Q, K, V pointers to enable LDG read-only cache path. Neutral (0.2% within noise) — the compiler already routes const-qualified pointers through the read-only cache on sm_86.

---

## Best Version Conclusion

**Best version:** `v5` — execution time reduced from 7.72 ms to 0.87 ms, speedup **8.90x**.
Key gains: thread-level parallelization (4.1x) + float4 vectorization (1.7x) + loop unrolling (1.3x) + __restrict__ (neutral).
Stopping reason: max iterations (5) reached. Kernel now memory-latency-bound (Long Scoreboard = 44.5%). Further gains require algorithmic changes (online softmax fusion, shared-memory K/V tiling) or warp specialization.

**Remaining optimization opportunities:**
- **Online softmax**: Fuse exp/normalize with the output weighted-sum to eliminate one shared memory round-trip and one `__syncthreads()`
- **Shared memory tiling of K**: Prefetch K tiles into shared memory for reuse across query positions within the same head (512 query positions share the same head's K)
- **Warp specialization**: Dedicate some warps to K/V prefetch while others compute
- **Tensor Core**: Use WMMA for the QK dot-product matrix multiply (requires data layout restructuring)
