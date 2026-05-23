# Triton Flash Attention Optimization Final Report — `flash_attention` (2026-05-17)

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (CC 8.6, sm_86) |
| CUDA / nvcc | CUDA 12.6 / nvcc 12.6.85 |
| ncu | NVIDIA Nsight Compute 2024.3.2.0 |
| nsight-python | 0.9.6 |
| Triton | 3.6.0 |
| PyTorch | 2.11.0+cu126 |
| Kernel file | flash_attention.py |
| Problem dims | B=4, H=12, N=4096, d=64 (fp16) |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| Execution Time (ms) | 4.1847 | 4.8214 | 5.0377 | 8.7068 | 5.2533 | 6.6973 |
| Speedup (×) | 1.00 | 0.87 | 0.83 | 0.48 | 0.80 | 0.62 |
| Memory Throughput (% SOL) | 89.66 | 60.21 | 90.02 | 34.05 | 27.27 | 94.70 |
| SM Throughput (% SOL) | 63.49 | 63.46 | 56.91 | 66.61 | 50.56 | 41.98 |
| Bottleneck | Memory | Compute | Memory | Compute | Latency | Memory |
| Achieved Occupancy (%) | 16.54 | 15.21 | 8.33 | 16.67 | 8.33 | 21.55 |
| Theoretical Occupancy (%) | 16.67 | 16.67 | 8.33 | 16.67 | 8.33 | 16.67 |
| Registers / Thread | 204 | 255 | 255 | 148 | 255 | 108 |
| Block Size | 128 | 128 | 128 | 256 | 128 | 128 |
| Grid Size | 3072 | 384 | 3072 | 3072 | 1536 | 6144 |
| Shared Memory (bytes) | 49152 | 49152 | 81920 | 32768 | 65536 | 49664 |
| Long Scoreboard Stall | 1.406 | 0.130 | 0.224 | 0.777 | 0.053 | 0.943 |
| Short Scoreboard Stall | 0.544 | 0.607 | 0.367 | 0.223 | 0.712 | 0.684 |
| Math Pipe Throttle | 1.456 | 1.623 | 0.764 | 2.224 | 1.059 | 0.608 |
| L1 Hit Rate (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| L2 Hit Rate (%) | 17.52 | 47.04 | 9.88 | 42.32 | 42.45 | 44.16 |
| L1 Bank Conflicts | 500525 | 464163 | 452175 | 447842 | 406640 | 6820000 |
| Warp Execution Efficiency (%) | 32.00 | 32.00 | 32.00 | 32.00 | 32.00 | 31.59 |
| Tensor Core Utilization (%) | 64.42 | 69.64 | 57.60 | 67.47 | 52.55 | 51.41 |
| Correctness | PASS | PASS | PASS | PASS | PASS | PASS |

---

## Optimization Strategies per Version

| Strategy | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| Tensor Core (tl.dot) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Mixed precision (fp16 load, fp32 compute) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Query-side tiling (multi-tile per program) | ✓ | ✗ | ✗ | ✗ | ✗ |
| Tile size tuning | ✗ | ✓ (N↑) | ✗ | ✓ (M↑) | ✓ (M↓) |
| Software pipelining (num_stages) | ✗ | ✗ | ✓ | ✗ | ✗ |
| Warp count tuning | ✗ | ✗ | ✓ (8w) | ✗ | ✗ |
| Coalesced global access | ✓ | ✓ | ✓ | ✓ | ✓ |

**Decision rationale per version:**
- **v1:** Query-side tiling with BLOCK_I=8 — attempted to reuse K/V loads across 8 Q tiles via L2 cache. L2 hit rate improved (17.5%→47%) but grid dropped from 3072→384 programs, registers increased 204→255. Regression.
- **v2:** BLOCK_N=128 — halved inner loop iterations (64→32) to reduce load overhead. Registers increased 204→255, L2 hit rate dropped (17.5%→9.9%), occupancy halved. Regression.
- **v3:** num_stages=2 + num_warps=8 — software pipelining to hide memory latency. Registers improved (204→148) but instructions doubled, creating math pipe contention (2.224 stall). Worst regression.
- **v4:** BLOCK_M=128 — larger Q tiles to reduce total K/V traffic by halving grid. Registers increased (204→255), occupancy halved (16.5%→8.3%). Regression.
- **v5:** BLOCK_M=32 — smaller Q tiles to reduce registers and increase occupancy. Registers dropped to 108 and occupancy improved (16.5%→21.5%), but L1 bank conflicts exploded 13x (500K→6.8M), branch divergence appeared (67%), memory SOL hit 94.7%. Worst regression for tile-only changes.

---

## Best Version Conclusion

**Best version:** `v0` (original unoptimized kernel) — 4.1847 ms

All 5 optimization attempts resulted in regressions (0.48× to 0.87× speedup). The original kernel is at a local optimum for the given problem parameters (B=4, H=12, N=4096, d=64) on RTX A6000 (sm_86).

**Key finding:** The kernel is strongly memory-bandwidth bound (89.7% SOL) due to the fundamental flash attention algorithm structure — each program independently loads the entire K and V tensors from DRAM. Parameter tuning (tile sizes, num_warps, num_stages) cannot overcome this bottleneck because:
1. Larger tiles increase register pressure (204→255), reducing occupancy
2. Smaller tiles increase total K/V traffic, worsening the memory bottleneck (89.7%→94.7%)
3. Software pipelining doubles instruction count without sufficient memory/compute overlap
4. Warp count changes alter shared memory allocation, creating new bottlenecks

---

## Benchmark vs PyTorch Reference

| Metric | Solution (v0) | PyTorch Eager | PyTorch Compile |
|---|---|---|---|
| Mean Time (ms) | 4.0838 | 17.1028 | 2.0392 |
| Median Time (ms) | 4.0704 | 17.1090 | 2.0439 |
| Speedup vs Eager | **4.19×** | 1.00× | 8.39× |
| Speedup vs Compile | 0.50× | 0.12× | 1.00× |

The Triton kernel is 4.2× faster than PyTorch eager but 2× slower than `torch.compile` (which uses FlashAttention-2 internally). This confirms the NCU finding — the kernel is memory-bandwidth bound and would benefit from FlashAttention-2 style algorithmic improvements.

---

**Remaining optimization opportunities:**
- **Intra-block K/V sharing via shared memory**: Restructure to load K/V tiles into shared memory once per block and share across warps (FlashAttention-2 approach). This requires significant algorithm restructuring but could reduce per-program memory traffic.
- **Autotuning**: Use `@triton.autotune` to search over num_warps × num_stages × BLOCK sizes, which may find non-obvious configurations.
- **Different hardware**: The RTX A6000 (sm_86) has lower memory bandwidth than datacenter GPUs (A100/H100). The same kernel would perform proportionally better on higher-bandwidth hardware.
- **Larger sequence lengths**: For N >> 4096, query-side tiling (v1 approach) would have enough grid parallelism to be effective.
