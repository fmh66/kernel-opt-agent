# CUDA Optimization Final Report — `softmax` (`2026-05-10`)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6, sm_86) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| ncu | 2024.3.2.0 (build 34861637) |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | test/softmax.cu → demo/cuda/softmax |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 | best (v2) |
|---|---|---|---|---|---|---|---|
| Execution Time (ms) | 3.4769 | 0.4516 | 0.2967 | 0.3018 | 0.2961 | 0.2960 | 0.2967 |
| Speedup (× vs v0) | 1.00 | 7.70× | 11.72× | 11.52× | 11.75× | 11.75× | 11.72× |
| Memory Throughput (% peak) | 13.35 | 92.38 | 93.72 | 93.37 | 93.89 | N/A | 93.72 |
| SM Throughput (% peak) | 1.21 | 9.08 | 14.38 | 6.86 | 14.32 | N/A | 14.38 |
| Bottleneck | Memory-Bound | Memory-Bound | Memory-Bound | Memory-Bound | Memory-Bound | Memory-Bound | Memory-Bound |
| Achieved Occupancy (%) | 16.66 | 95.75 | 81.07 | 80.53 | 81.65 | N/A | 81.07 |
| Theoretical Occupancy (%) | 100.00 | 100.00 | 83.33 | 83.33 | 83.33 | N/A | 83.33 |
| Registers / Thread | 38 | 40 | 40 | 40 | 40 | N/A | 40 |
| Global Load Efficiency (%) | 12.50 | 100.00 | 100.00 | 100.00 | 100.00 | N/A | 100.00 |
| Global Store Efficiency (%) | 12.50 | 100.00 | 100.00 | 100.00 | 100.00 | N/A | 100.00 |
| Long Scoreboard Stall (%) | 46.33 | 140.82 | 63.84 | 66.17 | 72.12 | N/A | 63.84 |
| Barrier Stall (%) | 0.00 | 27.11 | 10.65 | 21.35 | 12.28 | N/A | 10.65 |
| Shared Memory / Block (bytes) | 0 | 64 | 16448 | 16448 | 16448 | N/A | 16448 |
| L1 Bank Conflicts | 6.89e+07 | 103900 | 116332 | 110853 | 114688 | N/A | 116332 |
| IPC | 0.039 | 0.058 | 0.089 | 0.071 | 0.080 | N/A | 0.089 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Coalesced global memory access (one-block-per-row) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Shared memory caching (exp values) | ✗ | ✓ | ✓ | ✓ | ✓ |
| Warp-level reduction (shuffle) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Vectorized loads (`float4`) | ✗ | ✗ | ✓ | ✗ | ✗ |
| Fast-math intrinsic (`__expf`) | ✗ | ✗ | ✗ | ✓ | ✗ |
| `__launch_bounds__` | ✗ | ✗ | ✗ | ✗ | ✓ |

**Decision rationale per version:**

- **v0→v1:** Parallelized across the D dimension (one block per row) with warp-level reductions. v0 had one thread per row with stride-D access → 12.5% global load efficiency and only 16 blocks across 84 SMs. Changing to one-block-per-row made threads within a warp access consecutive elements → 100% coalescing, and launched N=4096 blocks → 95.8% occupancy. **Result: 7.7× speedup.**

- **v1→v2:** Used shared memory (16KB) to cache exp values, eliminating one global read pass and one global write pass. v1 read input ×3 and wrote output ×2 per row (5 passes). v2 reads input ×2 and writes output ×1 (3 passes), using shared memory as the intermediate buffer. Memory traffic reduced ~40%. **Result: 1.52× speedup.**

- **v2→v3:** Vectorized global loads/stores with float4. Since memory was already at 93.7% bandwidth saturation, reducing instruction count didn't improve throughput. Float4 access to shared memory increased short scoreboard stalls (5.6%→17.6%). **Result: slight regression (0.98×). Reverted in v4.**

- **v3→v4:** Replaced `expf()` with `__expf()` fast-math intrinsic. Kernel is memory-bandwidth saturated at 93.9% — compute-side optimizations have no measurable effect. **Result: neutral (1.00×).**

- **v4→v5:** Added `__launch_bounds__(256, 6)` to hint compiler about register allocation. Registers stayed at 40, occupancy unchanged. **Result: neutral (1.00×).**

---

## Best Version Conclusion

**Best version:** `v2` — execution time reduced from **3.4769 ms** (v0) to **0.2967 ms** (v2), an **11.72× speedup**.

Compared to PyTorch reference:
- 2.73× faster than PyTorch eager (0.8112 ms)
- 2.41× faster than PyTorch compile (0.7150 ms)

**Key gains:**
1. Coalesced memory access (one block per row, 12.5%→100% global load efficiency) — dominant contributor
2. Shared memory caching of exp values (3 passes→2 global reads + 1 write) — secondary contributor
3. Warp-level reductions via shuffle instructions — eliminated shared memory for intra-warp communication

**Stopping reason:** Max iterations (5) reached. Kernel is at the DRAM bandwidth roofline (93.7% of peak). Further compute-side or instruction-level optimizations (float4, __expf, __launch_bounds__) yielded no improvement. The remaining opportunity is to reduce global memory traffic below the theoretical minimum of 2 reads + 1 write per element, which is not feasible for standard softmax.

**Remaining optimization opportunities:**
- Mixed precision (FP16/BF16) would reduce memory traffic by 2× if acceptable for the use case
- Kernel fusion: integrate softmax with upstream/downstream ops to eliminate global memory round-trips
- For batched small-D use cases, consider warp-level softmax instead of block-level
