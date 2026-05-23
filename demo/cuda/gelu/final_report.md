# CUDA Optimization Final Report — GELU (2026-05-18)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / nvcc 12.6.85 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/gelu/gelu.cu → demo/cuda/gelu/v0/v0.cu |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 0.0188 | 0.0191 | 0.0192 | 0.0190 | 0.0190 | 0.0193 |
| Speedup (vs v0) | 1.00× | 0.98× | 0.98× | 0.99× | 0.99× | 0.97× |
| Memory SOL (%) | 23.41 | 23.31 | 21.82 | 23.64 | 23.72 | 18.64 |
| SM SOL (%) | 9.03 | 8.64 | 5.83 | 8.98 | 8.91 | 8.22 |
| Bottleneck | Latency | Latency | Latency | Latency | Latency | Latency |
| Achieved Occupancy (%) | 60.67 | 56.10 | 18.55 | 60.88 | 64.20 | 29.25 |
| Waves / SM | 0.79 | 0.79 | 0.20 | 0.79 | 0.79 | 0.33 |
| Registers / Thread | 16 | 16 | 19 | 16 | 16 | 24 |
| Warp Stall — Long SB (%) | 20.82 | 21.92 | 7.64 | 20.85 | 20.84 | 11.38 |
| Warp Stall — Short SB (%) | 1.35 | 1.21 | 1.07 | 1.35 | 1.35 | 1.04 |
| Branch Divergence (targets) | 3200/6400 | 3200/6400 | 3200/7200 | 3200/6400 | 3200/6400 | 3200/10944 |
| L2 Hit Rate (%) | 59.57 | 55.18 | 55.98 | 54.93 | 55.37 | 55.55 |
| Warp Exec Efficiency (%) | 25.01 | 25.01 | 22.82 | 25.01 | 25.01 | 26.33 |
| IPC | 0.164 | 0.153 | 0.110 | 0.166 | 0.166 | 0.145 |
| FMA Pipe Util (%) | 7.03 | 6.63 | 5.50 | 7.17 | 7.17 | 6.69 |
| LSU Pipe Util (%) | 2.24 | 2.09 | 0.49 | 2.26 | 2.26 | 1.29 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Tuned block size / `__launch_bounds__` | ✓ | ✗ | ✗ | ✗ | ✗ |
| Vectorized loads (`float4`) | ✗ | ✓ | ✗ | ✗ | ✗ |
| `__ldg` read-only cache | ✗ | ✗ | ✓ | ✗ | ✗ |
| Fast-math compilation (`--use_fast_math`) | ✗ | ✗ | ✗ | ✓ | ✗ |
| Grid-stride loop / persistent kernel | ✗ | ✗ | ✗ | ✗ | ✓ |

**Decision rationale per version:**

- **v1 (Block 256→128):** Latency-bound kernel with 0.79 waves/SM. Smaller blocks → 2× more blocks to spread across SMs. Failed because total warp count (3200) remained identical — no actual increase in parallelism. 0.98× speedup.

- **v2 (float4 vectorization):** 20.8% Long Scoreboard stall from global memory. float4 reduces load/store instruction count 4×. Long Scoreboard improved (20.8→7.6%) but grid shrank 400→100, causing Waves/SM to crash 0.79→0.20 and Occupancy 60.7→18.6%. Parallelism loss dominated the memory benefit. 0.98× speedup.

- **v3 (`__ldg()`):** L1 Hit Rate 0% — attempted to bypass via read-only cache path. All metrics unchanged — `__ldg()` provides no benefit for already-coalesced (100% efficiency) single-element loads on simple element-wise kernels. 0.99× speedup.

- **v4 (`--use_fast_math`):** Attempted to accelerate math-heavy GELU computation (tanh + 5 FMUL + FADD per element) via faster intrinsics. All metrics nearly identical to v0 — math is too small a fraction (~7% FMA utilization) of total execution time for fast-math to have measurable impact. Memory latency dominates. 0.99× speedup.

- **v5 (Grid-stride loop):** Reduced grid from 400→168 blocks to improve L2 cache locality via temporal reuse. Occupancy cratered (60.7→29.3%), Waves/SM 0.79→0.33, registers increased 16→24 from loop induction variable. Long Scoreboard improved (20.8→11.4%) but occupancy loss was fatal. 0.97× speedup.

---

## Best Version Conclusion

**Best version: v0** — the original unoptimized kernel at 0.0188 ms. All five optimization attempts produced regressions or neutral results.

The kernel is fundamentally limited by its tiny problem size (102,400 elements, 0.4 MB per buffer). At 18.8 μs execution time with 23% memory utilization and 9% SM utilization, the GPU is severely underutilized regardless of kernel design. Every optimization that reduced grid parallelism (float4, grid-stride) hurt performance; those that preserved it (`__ldg__`, fast-math, block tuning) were neutral.

**Benchmark context:** PyTorch eager achieves 0.0079 ms (2.4× faster) and `torch.compile` achieves 0.0052 ms (3.6× faster) on the same workload — confirming that the hand-written CUDA kernel carries measurable overhead vs. PyTorch's highly-optimized fused GELU implementation.

**Remaining optimization opportunities:**
- **Kernel fusion:** Integrate GELU into a larger upstream/downstream kernel (e.g., linear → GELU) to amortize memory round-trips
- **CUDA Graphs:** Reduce launch overhead for such tiny kernels — likely the largest single remaining gain
- **Larger problem size:** With N ≥ 1M+ elements, the parallelism-exploiting optimizations (float4, grid-stride) might become beneficial
- **Ampere-specific:** `cp.async` prefetch is irrelevant for this single-pass element-wise kernel with no shared memory stage
