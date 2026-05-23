# CUDA Optimization Final Report — `conv2d` (`2026-05-17`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / release 12.6, V12.6.85 |
| ncu | 2024.3.2.0 (build 34861637) |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/conv2d/conv2d.cu |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 | **v3 (best)** |
|---|---|---|---|---|---|---|---|
| Execution Time (ms) | 2.5415 | 3.1676 | 2.9033 | 2.0619 | 2.0364 | 2.0652 | **2.0619** |
| Speedup (×) | 1.00 | 0.80 | 0.88 | **1.23** | 1.25 | 1.23 | **1.23** |
| SM Throughput (%) | 76.80 | 72.65 | 78.71 | 98.07 | 80.63 | 97.85 | 98.07 |
| Memory Throughput (%) | 1.17 | 0.90 | 0.97 | 1.48 | 1.44 | 1.47 | 1.48 |
| FMA Pipe Utilization (%) | 31.09 | 29.56 | 22.97 | 13.58 | 13.02 | 13.58 | 13.58 |
| FFMA Throughput (per cycle) | 450.3 | 343.6 | 368.3 | 575.0 | 555.2 | 575.0 | 575.0 |
| IPC | 0.751 | 0.731 | 0.630 | 0.692 | 0.715 | 0.692 | 0.692 |
| Achieved Occupancy (%) | 91.12 | 98.81 | 98.79 | 76.39 | 59.38 | 76.21 | 76.39 |
| Theoretical Occupancy (%) | 100.0 | 100.0 | 100.0 | 83.33 | 66.67 | 83.33 | 83.33 |
| Registers / Thread | 37 | 40 | 38 | 48 | 60 | 48 | 48 |
| Global Load Efficiency (%) | 69.56 | 31.52 | 30.33 | 68.05 | 73.99 | 68.05 | 68.05 |
| L1 Hit Rate (%) | 86.96 | 60.38 | 63.95 | 85.28 | 89.88 | 85.28 | 85.28 |
| Warp Exec Efficiency (%) | 28.04 | 29.07 | 27.63 | 28.05 | 28.87 | 28.05 | 28.05 |
| Stall: Barrier (%) | 0.00 | 3.84 | 4.94 | 0.00 | 0.00 | 0.00 | 0.00 |
| Stall: Long Scoreboard (%) | 7.80 | 3.65 | 5.50 | 1.58 | 3.67 | 1.58 | 1.58 |
| Stall: Math Pipe Throttle (%) | 0.78 | 0.83 | 0.38 | 2.38 | 1.15 | 2.38 | 2.38 |
| Stall: Not Selected (%) | 1.81 | 2.10 | 1.75 | 3.55 | 1.51 | 3.55 | 3.55 |
| L1 Bank Conflicts (total) | 30.5M | 56.0M | 66.2M | 38.1M | 16.9M | 38.1M | 38.1M |
| Branch Targets (total) | 118M | 161M | 149M | 7.45M | 25.8M | 7.45M | 7.45M |
| Bottleneck | Compute | Barrier | Barrier | MathPipe | RegPressure | MathPipe | MathPipe |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Shared memory tiling | ✓ | ✓ | ✗ | ✗ | ✗ |
| Coalesced global memory access | ✓ | ✓ | ✓ | ✓ | ✓ |
| `__ldg` / read-only cache | ✗ | ✗ | ✓ | ✓ | ✓ |
| Explicit FMA intrinsic (`__fmaf_rn`) | ✗ | ✗ | ✓ | ✓ | ✓ |
| ILP (manual loop unrolling) | ✗ | ✗ | ✓ | ✓ | ✓ |
| Thread coarsening | ✗ | ✗ | ✗ | ✓ | ✗ |
| `--use_fast_math` | ✗ | ✗ | ✗ | ✗ | ✓ |

**Decision rationale per version:**

- **v1 (Shared memory tiling, per-channel):** Hypothesis that loading input tiles into shared memory once per channel would reduce redundant global loads (L1 load bandwidth was 608× DRAM). **Failed (0.80×)** — 128 `__syncthreads()` barriers dominated (Barrier stall 0→3.85%), and uncoalesced SMEM loads dropped Global Load Efficiency from 69.6% to 31.5%.

- **v2 (Coalesced SMEM loading):** Fixed the tile loading pattern to use 2D thread-to-input mapping for coalesced accesses. **Still regression (0.88×)** — barrier overhead from per-channel syncing remained (4.94% stall), proving the SMEM approach fundamentally doesn't work for C_in=64 with per-channel barriers.

- **v3 (Manual unrolling + __ldg + __fmaf_rn):** Abandoned shared memory. Fully unrolled the K=3 loops, prefetched all 9 weights per (oc,ic) via `__ldg()`, and used explicit `__fmaf_rn()` for FMAs. **Success (1.23×)** — SM throughput jumped 76.8→98.1%, Long Scoreboard dropped 7.80→1.58%, Branch targets collapsed 118M→7.45M. 48 registers limit theoretical occupancy to 83%.

- **v4 (Thread coarsening 2×):** Each thread computes 2 output pixels to increase ILP. **Neutral (1.25×)** — marginal gain within noise. Duplicated inner loop ballooned register count to 60, dropping occupancy to 59%, negating the ILP benefit.

- **v5 (--use_fast_math):** Compiler flag on v3 code. **Neutral (1.00× vs v3)** — no change since the kernel already uses explicit `__fmaf_rn()` and `__ldg()`.

---

## Best Version Conclusion

**Best version:** `v3` — execution time reduced from **2.5415 ms** (v0) to **2.0619 ms**, **1.23× speedup**.

Key gains:
- Manual loop unrolling eliminated 110M branch instructions (118M→7.45M)
- `__ldg()` reduced global memory load latency (Long Scoreboard stall 7.80→1.58%)
- `__fmaf_rn()` ensured fused multiply-add without relying on compiler heuristics
- FFMA throughput increased 28% (450→575 per cycle)

Stopping reason: Maximum iterations reached (N=5). SM throughput at 98% — near hardware peak for this workload pattern.

**Remaining optimization opportunities:**
- Tensor Core WMMA for the convolution (would require im2col transform or implicit GEMM formulation) — potentially 2-3× speedup by leveraging dedicated matrix hardware
- Warp execution efficiency (28%) remains low due to predicated edge-pixel boundary checks; splitting interior/edge code paths could improve this
- `cp.async` prefetching for input data could reduce remaining Long Scoreboard stall, but benefit is limited given SM already at 98%

---

## Benchmark (do_bench: steady-state latency, 20ms warmup / 100ms rep)

| Metric | v0 (baseline) | **v3 (best)** | PyTorch Eager | PyTorch Compile |
|---|---|---|---|---|
| Mean (ms) | 2.4118 | **1.9392** | 0.1451 | 0.1736 |
| Median (ms) | 2.4387 | **1.9663** | 0.1444 | 0.1739 |
| Min (ms) | 2.2252 | **1.8136** | 0.1413 | 0.1710 |
| Max (ms) | 2.4750 | **2.0265** | 0.1536 | 0.1761 |
| Std dev (ms) | 0.0793 | **0.0561** | 0.0033 | 0.0009 |
| Speedup vs v0 | 1.00× | **1.24×** | - | - |

**Correctness:** PASS (atol=5e-2, rtol=1e-2 for direct-conv vs cuDNN accumulation differences)

Note: PyTorch (cuDNN) is ~13× faster because it uses highly optimized implicit GEMM / Winograd algorithms, not naive direct convolution. Our 1.24× improvement is measured within the direct convolution algorithmic space.
