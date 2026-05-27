# CUDA Optimization Final Report - `gqa_paged_decode_h16_kv2_d128_ps1` (2026-05-27)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6) |
| CUDA / nvcc | 12.6 / V12.6.85 |
| Triton | 3.7.0 |
| PyTorch | 2.12.0+cu126 |
| Kernel file | v3/kernel.py (best) |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 | best |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Correctness | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | — |
| Execution Time (ms) | 0.0524 | 0.0734 | 0.0530 | **0.0497** | 0.0510 | 0.0519 | 0.0499 | 0.0572 | 0.0520 | 0.0509 | 0.0516 | 0.0497 |
| Speedup (x) | 1.00 | 0.71 | 0.99 | **1.05** | 1.03 | 1.01 | 1.05 | 0.92 | 1.01 | 1.03 | 1.02 | 1.05 |
| Memory Throughput (%) | 3.58 | 0.99 | 3.42 | 3.24 | 2.30 | 3.28 | 3.41 | 3.42 | 3.46 | 3.66 | 3.42 | 3.24 |
| Compute Throughput (%) | 6.80 | 4.15 | 13.11 | 10.03 | 7.77 | 8.73 | 10.05 | 10.21 | 5.28 | 5.37 | 10.03 | 10.03 |
| Bottleneck | Latency | Latency (starve) | Latency | Latency | Latency (starve) | Latency | Latency | Latency | Latency | Latency | Latency | Latency |
| Achieved Occupancy (%) | 24.76 | 8.34 | 50.26 | 25.38 | 16.63 | 25.34 | 25.27 | 25.48 | 12.61 | 12.70 | 25.40 | 25.38 |
| Grid Size | 256 | 32 | 256 | 128 | 64 | 128 | 128 | 128 | 128 | 128 | 128 | 128 |
| Block Size / Threads | 128 | 128 | 256 | 256 | 256 | 256 | 256 | 256 | 128 | 128 | 256 | 256 |
| Registers / Thread | 32 | 38 | 32 | 39 | 40 | 40 | 38 | 38 | 38 | 37 | 38 | 39 |
| Stall: Long Scoreboard (%) | 11.15 | 2.98 | 10.30 | 4.74 | 3.15 | 5.57 | 4.40 | 4.40 | 6.12 | 5.48 | 4.45 | 4.74 |
| Stall: Short Scoreboard (%) | 3.11 | 3.33 | 3.59 | 2.86 | 2.84 | 3.82 | 3.11 | 3.13 | 2.17 | 2.61 | 3.16 | 2.86 |
| L1 Hit Rate (%) | 21.16 | 86.83 | 60.09 | 59.52 | 58.74 | 59.56 | 51.41 | 7.68 | 21.05 | 6.15 | 51.44 | 59.52 |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8 | v9 | v10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Q-head fusion (KV data reuse) | yes (8Q) | no | yes (2Q) | yes (4Q) | no | no | no | no | no | no |
| num_warps=8 (256 threads) | no | yes | yes | yes | yes | yes | yes | no | no | yes |
| num_warps=4 (128 threads) | yes | no | no | no | no | no | no | yes | yes | no |
| Fast math exp2 (tl.math.exp2) | no | no | no | no | yes | no | no | no | no | no |
| Precomputed page offsets | no | no | no | no | no | yes | yes | no | yes | yes |
| Cache modifier (.cg on K/V) | no | no | no | no | no | no | yes | no | no | no |

---

## Hypothesis Outcomes

| Transition | NCU Symptom | KBS Pattern | Hypothesis | Result | Evidence |
|---|---|---|---|---|---|
| v0 -> v1 | Low occupancy + repeated KV loads | technique-ampere-optimization | Fuse 8 Q-heads per block via (B, H_kv) grid | REGRESSION (0.0524→0.0734) | Grid starvation at 32 blocks |
| v1 -> v2 | Grid starvation (32 blocks) | technique-ampere-optimization | Revert to v0 grid + num_warps=8 | NEUTRAL (0.0734→0.0530) | Doubled occupancy but no speedup — no KV reuse |
| v2 -> v3 | Zero KV data reuse limits any gains | technique-ampere-optimization | 2-head fusion with page-outer loop (grid=128) | IMPROVED (0.0530→0.0497) | 5.2% over v0, 2x KV reuse at sustainable grid |
| v3 -> v4 | 2x KV reuse = 5% gain, try 4x | technique-ampere-optimization | 4-head fusion (grid=64) | REGRESSION (0.0497→0.0510) | Grid starvation outweighs extra reuse |
| v4 -> v5 | Grid starvation; revert to v3 + compute opt | technique-ampere-optimization | Fast exp2 replacing tl.exp | REGRESSION (0.0510→0.0519) | Math opt ineffective on memory-latency-bound kernel |
| v5 -> v6 | Redundant address calc per iteration | technique-ampere-optimization | Precompute base page offsets | NEUTRAL (0.0519→0.0499) | Tied with v3's best; instruction savings don't move runtime |
| v6 -> v7 | KV streaming wastes L1 | technique-ampere-optimization | Cache modifier .cg on K/V loads | REGRESSION (0.0499→0.0572) | .cg L1 bypass hurts intra-block 2-head KV reuse |
| v7 -> v8 | .cg bypass hurts; need fresh direction | technique-ampere-optimization | Revert to v3 + num_warps=4 | NEUTRAL (0.0572→0.0520) | num_warps=4 halves occupancy without benefit |
| v8 -> v9 | num_warps=4 neutral; combine with precompute | technique-ampere-optimization | v6 precompute offsets + num_warps=4 | NEUTRAL (0.0520→0.0509) | Near v3 but slightly slower |
| v9 -> v10 | Optimization space exhausted | technique-ampere-optimization | Combine v3 structure + v6 precompute offsets | NEUTRAL (0.0509→0.0516) | Commutative effect within measurement noise |

---

## KBS Evidence

| Version | Query | Doc ID | Canonical path | Confidence | Used for / Applicability |
|---|---|---|---|---|---|
| v0 | `triton flash attention decode paged kv GQA sm86` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Guiding occupancy tuning and data reuse strategies |
| v1 | `triton grid parallelism warmup CUDA block` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Grid starvation diagnosis |
| v2 | `triton Q head fusion KV reuse GQA` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | KV data reuse through Q-head fusion |
| v3 | `triton flash attention decode v1 split KV` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Grid vs reuse balance validation |
| v4 | `triton grid occupancy block size tuning` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | 64-block grid starvation confirmed |
| v5 | `triton fast math exp exp2 Ampere` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Math ops irrelevant for memory-bound kernel |
| v6 | `triton address compute precompute base offset` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Instruction-level savings neutral for latency-bound |
| v7 | `triton cache modifier L1 bypass streaming` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | .cg counterproductive for data shared within block |
| v8 | `triton num_warps occupancy threads block size` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Warp count tuning saturated |
| v9 | `triton flash attention decode paged kv cache sm86` | technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Final combination validation |

No rejected KBS results — all queries returned consistent guidance.

---

## NCU + KBS Synthesis

| Version | NCU fact set | KBS evidence | Decision |
|---|---|---|---|
| v1 | 0.0734 ms, grid starvation, 8.34% occupancy | technique-ampere-optimization | Too few blocks (32) cannot fill 84 SMs; need more grid parallelism |
| v2 | 0.0530 ms, 50% occupancy but no gain | technique-ampere-optimization | More warps help occupancy but wasted when no KV reuse exists |
| v3 | 0.0497 ms, 25% occupancy with 2x reuse | technique-ampere-optimization | Optimal balance: 128 blocks fills SMs adequately while 2x reuse cuts memory traffic |
| v4 | 0.0510 ms, 64-block grid starvation | technique-ampere-optimization | Grid must stay >= 128 blocks on 84-SM GPU; 4x reuse loses more to parallelism loss |
| v5 | 0.0519 ms, fast math ineffective | technique-ampere-optimization | Memory-latency-bound kernel ignores compute pipeline improvements |
| v6 | 0.0499 ms, precomputed offsets neutral | technique-ampere-optimization | Address ALU savings imperceptible vs DRAM latency |
| v7 | 0.0572 ms, .cg L1 bypass harmful | technique-ampere-optimization | Intra-block K/V reuse depends on L1; cache hints counterproductive |
| v8 | 0.0520 ms, num_warps=4 halves occupancy | technique-ampere-optimization | 4 warps/block too few for latency hiding when grid is already sufficient |
| v9 | 0.0509 ms, num_warps=4 + precompute neutral | technique-ampere-optimization | Combination within noise of v3, but lower occupancy is a net negative |
| v10 | 0.0516 ms, v3+v6 combination neutral | technique-ampere-optimization | Final combination confirms optimization space is saturated |

---

## Final Benchmark

| Item | Value |
|---|---|
| Best kernel | v3/kernel.py |
| Baselines | FlashInfer, Torch Compile |
| Best execution time (CUDA event) | 0.0164 ms |
| FlashInfer execution time | 0.2365 ms |
| Torch Compile execution time | 6.2403 ms |
| Benchmark speedup vs FlashInfer | 14.4x |
| Benchmark speedup vs Torch Compile | 378.7x |
| Benchmark artifact | benchmark.md |

---

## Best Version Conclusion

**Best version:** `v3` — execution time reduced from 0.0524 ms (v0 baseline) to 0.0497 ms, speedup 1.05x via NCU profiling. CUDA event benchmark shows 0.0164 ms, 14.4x faster than FlashInfer and 378.7x faster than torch.compile.

**Key gains:** Q-head fusion with page-outer loop ordering provides 2x KV data reuse per block (each K/V page loaded once, used by 2 Q-heads). The grid of 128 blocks on 84-SM RTX A6000 provides sufficient parallelism without starvation. num_warps=8 gives adequate latency hiding for the memory-bound access pattern.

**Optimization dimensions explored:**
1. KV data reuse (Q-head fusion): 2-head fusion at grid=128 is optimal (v3, v6). More fusion (v4, 4-head) causes grid starvation. Less (v0, 1-head) wastes memory bandwidth.
2. num_warps: 8 warps (256 threads) is optimal. 4 warps (128 threads) significantly reduces occupancy without benefit.
3. Compute-side: fast exp2 (v5), address precomputation (v6), cache hints (v7) — all neutral or harmful on this memory-latency-bound kernel.
4. The fundamental bottleneck is scattered KV page access from DRAM (~2.4e10 bytes/s DRAM read), which no compute-side optimization can bypass.

**Stopping reason:** Maximum iterations reached (10) and optimization space saturated for this kernel configuration. The 5% improvement from v0 to v3 represents the achievable headroom given the memory-latency bottleneck.

**Remaining optimization opportunities:**
- Shared memory pooling: prefetch consecutive KV pages into shared memory before computing dot products (limited benefit with page_size=1 and unpredictable page indices)
- Persistent kernel pattern: use cooperative groups with CTA-level persistent scheduling to keep KV pages hot across iterations (requires Triton 3.2+ persistent thread block support, significant complexity)
- Mixed precision: FP8 KV cache with scaled dot-product attention (requires hardware validation and broader system changes)
