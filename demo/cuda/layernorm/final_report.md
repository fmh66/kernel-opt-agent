# CUDA Optimization Final Report — LayerNorm (`2026-05-18`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/layernorm/layernorm.cu |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 | best (v2) |
|---|---|---|---|---|---|---|---|---|
| Execution Time (ms) | 8.5155 | 2.4509 | 1.8653 | 1.8648 | 1.8608 | 1.8617 | **1.8653** |
| Speedup (×) | 1.00 | 3.47 | 4.56 | 4.57 | 4.58 | 4.57 | **4.56** |
| Memory Throughput (%) | 27.43 | 94.76 | 93.93 | 94.01 | 94.01 | 94.09 | 93.93 |
| SM Throughput (%) | 2.91 | 10.73 | 11.91 | 11.98 | 11.94 | 12.73 | 11.91 |
| Bottleneck | Latency | Memory | Memory | Memory | Memory | Memory | Memory |
| Achieved Occupancy (%) | 16.62 | 98.12 | 97.86 | 97.67 | 95.11 | 90.63 | 97.86 |
| Waves / SM | 0.08 | 20.32 | 20.32 | 20.32 | 20.32 | 40.63 | 20.32 |
| Registers / Thread | 40 | 23 | 22 | 20 | 40 | 22 | 22 |
| Global Load Efficiency (%) | 12.50 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| Stall: Long Scoreboard (%) | 47.11 | 126.33 | 122.84 | 122.74 | 133.49 | 101.62 | 122.84 |
| Stall: Barrier (%) | 0.00 | 2.17 | 1.78 | 1.67 | 5.15 | 9.24 | 1.78 |
| Branch Divergence (%) | 0.00 | 0.20 | 0.15 | 0.15 | 0.51 | 0.15 | 0.15 |
| L1 Hit Rate (%) | 89.50 | 25.42 | 32.37 | 32.08 | 26.23 | 26.99 | 32.37 |
| L2 Hit Rate (%) | 72.97 | 33.12 | 40.84 | 40.49 | 45.62 | 45.09 | 40.84 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Coalesced global memory access | ✓ | ✓ | ✓ | ✓ | ✓ |
| Warp shuffle reduction | ✓ | ✓ | ✓ | ✓ | ✓ |
| Single-pass mean+var (fused) | ✗ | ✓ | ✓ | ✗ | ✓ |
| `__ldg` read-only cache | ✗ | ✗ | ✓ | ✗ | ✗ |
| Thread coarsening (×4) | ✗ | ✗ | ✗ | ✓ | ✗ |
| Larger block size (512) | ✗ | ✗ | ✗ | ✗ | ✓ |

**Decision rationale per version:**
- **v1:** Changed from one-thread-per-row to one-block-per-row with warp-level reductions. Grid expanded from 40→10240 blocks. Coalescing 12.5%→100%, occupancy 16.6%→98.1%. Past experience #1 had identical symptoms with 7.7x success.
- **v2:** Fused mean and variance computation into a single pass using sum_x + sum_x² accumulation (var = sum_x²/D − mean²). Reduced input reads from 3× to 2×, cutting DRAM traffic ~33%. Barrier stall ↓18%.
- **v3:** `__ldg()` for gamma/beta to route through read-only cache. Neutral — L2 already effectively caching gamma/beta at 41% hit rate.
- **v4:** Thread coarsening (×4 elements per loop iteration). Neutral — register spike 22→40 hurt occupancy, barrier stall increased 1.78→5.15%.
- **v5:** Block size 256→512. Neutral — barrier stall 1.78→9.24% cancelled Long Scoreboard improvement.

---

## Best Version Conclusion

**Best version:** `v2` — execution time reduced from 8.5155 ms to 1.8653 ms, speedup **4.56×**.

**Key gains:**
1. Block-level parallelism (grid 40→10240): 3.47×
2. Fused mean+var pass (input reads 3→2): 1.31×

**Stopping reason:** Maximum iterations reached (N=5). Kernel is saturated at 94% Memory SOL — approaching theoretical bandwidth limit.

**Remaining optimization opportunities:**
- Shared memory caching of gamma/beta could reduce remaining redundant DRAM reads, but A6000's 100KB shared memory per SM limits tile size.
- Mixed precision (FP16/BF16) could halve memory traffic but requires precision validation.
- CUDA Graphs could reduce launch overhead for repeated inference scenarios.

---

## Benchmark vs PyTorch

| Implementation | Latency (mean) |
|---|---|
| v0 (naive) | 8.447 ms |
| **v2 (optimized)** | **1.865 ms** |
| PyTorch eager | 3.069 ms |
| PyTorch compile | 1.827 ms |

Our optimized kernel is **1.65× faster** than PyTorch eager and within **2% of torch.compile**.
