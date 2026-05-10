# Triton Softmax Optimization Final Report — `softmax` (2026-05-10)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6, sm_86) |
| CUDA / nvcc | 12.6 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | `/home/kernel-opt-skill/test/softmax.py` |
| Problem size | N=1024, D=1024 (fp32) |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| Execution Time (ms) | **0.0399** | 0.0415 | 0.0438 | 0.0406 | 0.0646 | 0.0437 |
| Speedup (vs v0) | 1.00× | 0.96× | 0.91× | 0.98× | 0.62× | 0.91× |
| Memory Throughput (% peak) | 85.57 | 77.69 | 79.89 | 85.82 | 83.12 | 85.68 |
| SM Throughput (% peak) | 12.02 | 21.24 | 10.22 | 12.43 | 22.85 | 7.83 |
| Bottleneck | Memory | Memory | Memory | Memory | Memory | Memory |
| Achieved Occupancy (%) | 78.81 | 98.34 | 52.78 | 77.50 | 93.44 | 40.24 |
| Theoretical Occupancy (%) | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 66.67 |
| Waves / SM | 1.02 | 1.02 | 0.51 | 1.02 | 2.03 | 0.76 |
| Registers / Thread | 23 | 28 | 26 | 23 | 20 | 33 |
| Block Size | 128 | 256 | 128 | 128 | 256 | 64 |
| Grid Size | 1024 | 512 | 512 | 1024 | 1024 | 1024 |
| Long Scoreboard Stall (%) | 37.23 | 26.95 | 21.99 | 38.42 | 26.13 | 26.07 |
| Short Scoreboard Stall (%) | 3.72 | 4.37 | 2.57 | 3.63 | 4.84 | 2.48 |
| L1 Bank Conflicts | 2839 | 4192 | 3703 | 2981 | 4452 | 2753 |
| IPC | 0.1525 | 0.2136 | 0.1491 | 0.1511 | 0.2153 | 0.1218 |
| L1 Hit Rate (%) | 0 | 0 | 0 | 0 | 0 | 0 |
| L2 Hit Rate (%) | 50.78 | 51.74 | 51.32 | 51.33 | 51.31 | 50.87 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Warp count tuning (num_warps) | 8↑ | 4 (default) | 4 (default) | 8↑ | 2↓ |
| Multi-row per block (2 rows) | ✓ | ✓ | — | — | — |
| tl.softmax builtin | — | — | ✓ | — | — |
| Single-row baseline | — | — | ✓ | ✓ | ✓ |

**Decision rationale per version:**
- **v1:** Increase num_warps 4→8 and process 2 rows per block to hide memory latency (Long Scoreboard=37.23%). Expected better occupancy and memory stall hiding.
- **v2:** Revert num_warps to default (4) while keeping 2 rows per block, aiming to reduce bank conflicts while preserving row-batching benefit.
- **v3:** Replace manual softmax with `tl.softmax` builtin, expecting Triton's optimized implementation to handle reductions more efficiently.
- **v4:** Isolate num_warps=8 on single row per block (remove the 2-row loop from v1), testing whether warps alone help latency hiding.
- **v5:** Reduce num_warps to 2 (64 threads/block) to minimize shared memory bank conflicts in the reduction tree.

---

## Best Version Conclusion

**Best version:** `v0` (original, unoptimized) — execution time **0.0399 ms**.

All optimization attempts (v1–v5) resulted in regressions or neutral outcomes. The v0 kernel is already optimally configured for this problem size and hardware:

- **1.88× faster** than PyTorch eager (0.0731 ms)
- **3.86× faster** than PyTorch compiled (0.1498 ms)

**Why optimizations failed:** The kernel is fundamentally **memory-bandwidth-bound** at 85.57% of peak DRAM bandwidth. All attempts to change thread/warp configuration introduced one or more of:
1. **Increased shared memory bank conflicts** (v1: +48%, v4: +57%) — more warps = more cross-warp reduction contention
2. **Increased register pressure** (v5: 23→33 regs/thread) — fewer threads = more elements/thread = higher register demand
3. **Reduced grid parallelism** (v1/v2: grid 1024→512) — multi-row batching halved the number of blocks, dropping waves/SM
4. **No reduction in memory traffic** — all versions read+write the same 8 MB; the memory pipeline is the bottleneck

**Stopping reason:** Max iterations (N=5) reached. All optimization directions exhausted: warp tuning (up, default, down), multi-row batching, and builtin replacement all failed to beat the baseline.

**Remaining optimization opportunities:**
- **FP16 precision** — halving data types would reduce memory traffic ~50%, potentially yielding ~1.5–2× speedup, but requires API change
- **Fusion with upstream/downstream kernels** — combining softmax with adjacent operations (e.g., in attention) would eliminate intermediate global memory round-trips
- **Different hardware** — on Hopper (sm_90) with TMA and larger shared memory, different optimization strategies may apply
- **Larger problem sizes** — for D ≫ 1024 where online softmax tiling becomes necessary, alternative tiling strategies would need evaluation
