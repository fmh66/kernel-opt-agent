# Triton MHA Optimization Final Report — `mha.py` (2026-05-10)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 |
| ncu | 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | /home/kernel-opt-skill/test/mha.py |

---

## Version Iteration Comparison

| Metric | v0 | v1 | v2 | v3 | v4 | v5 (best) |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 141.23 | 3.891 | 0.284 | 0.264 | 0.407 | **0.193** |
| Speedup (×) | 1.00 | 36.3 | 497 | 535 | 347 | **731** |
| Memory Throughput (%) | 77.66 | 90.46 | 78.08 | 78.45 | 79.65 | 80.01 |
| SM Throughput (%) | 37.80 | 26.75 | 35.26 | 41.97 | 19.02 | 42.76 |
| Tensor Core Util (%) | 0.00 | 0.00 | 45.25 | 50.73 | 21.96 | 51.52 |
| Bottleneck | Memory | Memory | Memory | Memory | Memory | Memory |
| Achieved Occupancy (%) | 102.05 | 101.74 | 100.11 | 100.29 | 103.47 | 77.91 |
| Registers / Thread | 80 | 80 | 140 | 255 | 206 | 255 |
| Dynamic SMEM (KB) | 0.5 | 1.0 | 80.0 | 48.0 | 56.0 | 48.0 |
| Grid Size | 1,048,576 | 16,384 | 4,096 | 4,096 | 4,096 | 4,096 |
| L2 Hit Rate (%) | 94.44 | 74.34 | 78.72 | 83.90 | 79.45 | 84.00 |
| Warp Stall — Long SB (%) | 30.18 | 24.17 | 24.19 | 24.40 | 24.30 | 40.07 |
| Warp Execution Eff (%) | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 |
| L1 Bank Conflicts | 731M | 30.5M | 8,167 | 78,081 | 524,288 | 80,790 |
| Branch Divergence | 0 | 0 | 0 | 0 | 65,536 | 0 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Fused d-dimension (grid collapse) | ✓ | — | — | — | — |
| Query-side tiling (BLOCK_I) | — | ✓ | — | — | — |
| Tensor Core (tl.dot) | — | ✓ | — | — | — |
| Larger K/V tile (BLOCK_N=128) | — | — | ✓ | — | — |
| Reduced query tile (BLOCK_I=32) | — | — | — | ✓ | — |
| Pre-scale Q (fuse multiply) | — | — | — | — | ✓ |

**Decision rationale per version:**
- **v1:** Fuse d dimension — grid from (H,N,d_k) to (H,N). Each Q[h,i,:] was loaded 64x (once per output d). Eliminates 64× redundant Q/K loads. Expected: >30x speedup.
- **v2:** Query-side tiling BLOCK_I=64 with tl.dot. Each program processes 64 queries, reuses K/V 64×. Enables Tensor Core for QK^T and PV matmuls. Expected: >10x over v1.
- **v3:** BLOCK_N=128 (num_stages=1 forced). Past experience shows online softmax prevents effective cross-iteration pipelining, so losing num_stages shouldn't hurt. Halves loop iterations (16→8). Expected: small improvement.
- **v4:** BLOCK_I=32 with BLOCK_N=128. Attempt to reduce register pressure from v3's 255 reg spill. Expected: lower registers, similar latency.
- **v5:** Pre-scale Q by inv_sqrt_dk. Removes one multiply from the inner K/V loop. Expected: small improvement.

---

## Best Version Conclusion

**Best version:** `v5` — execution time reduced from 141.23 ms to 0.193 ms, speedup **731×**.

Key gains:
1. **64× redundant load elimination** (v1): grid (H,N,d_k) → (H,N), each program outputs full d_k vector
2. **Query-side tiling + Tensor Core** (v2): BLOCK_I=64, tl.dot for QK^T and PV, 45→51% Tensor Core utilization
3. **Larger K/V tiles** (v3): BLOCK_N=128, better L2 reuse (83.9%), fewer loop iterations
4. **Pre-scaled Q** (v5): eliminated inner-loop multiply, improved MMA pipeline dataflow

Stopping reason: Max iterations reached (N=5). All 5 iterations completed with measurable improvements or learning.

**Remaining optimization opportunities:**
- FP16 data path could halve memory traffic (current fp32 limits memory bandwidth at 80%)
- FlashAttention-2 style causal masking for decoder-only workloads
- Split-K or persistent kernel patterns for very large sequence lengths (>4096)
- The 32% Warp Execution Efficiency across all versions suggests BLOCK_D=64 may not perfectly fill MMA tiles — exploring BLOCK_D=128 with padding could help
