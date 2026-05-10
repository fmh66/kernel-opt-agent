# CUDA Community Optimization Techniques

Advanced optimization techniques from community practice, research, and production deployments that go beyond the standard optimization guides.

---

## Warp Specialization

Warp specialization assigns different roles to different warps within a CTA, overlapping data movement with computation at warp granularity rather than block granularity.

### Pattern
```
Warp 0: TMA Loader    — loads A/B tiles from HBM → SMEM via cp.async.bulk.tensor
Warp 1: MMA Producer  — issues tcgen05.mma / wgmma instructions
Warp 2: Epilogue      — reads from accumulator → SMEM → GMEM
```

### Key Mechanism
- **Hopper `setmaxnreg`**: dynamically allocate register counts per warp — producer warps get fewer regs (they just issue TMA), consumer warps get more (they hold MMA accumulators)
- **`mbarrier` (asynchronous barriers)**: synchronize producer/consumer without stalling the entire block
- Warps become independent execution streams with explicit commit/wait points

### When to Use
- MMA-heavy kernels where data movement and compute can be cleanly separated
- Particularly effective on Hopper+ with TMA and `setmaxnreg`
- Less benefit on Ampere where `cp.async` already provides good overlap with simpler code

> Source: [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) · [FlashAttention-4](https://arxiv.org/abs/2603.05451)

---

## SMEM Register Spilling (CUDA 13.0+)

New in CUDA Toolkit 13.0: the compiler can spill high-pressure registers into shared memory instead of local (GMEM-backed) memory.

### How to Enable
```cpp
asm volatile(".pragma \"enable_smem_spilling\";");
```

### Benefits
- Spills go to SMEM (~7× faster than local memory which goes to L2→HBM)
- Reduces L2 cache pressure by 5-10% (no local memory traffic from spills)
- Allows higher occupancy without the performance cliff of local memory spills

### Caveats
- Reduces available shared memory for explicit tiling — trade off carefully
- Only available in CUDA 13.0+ toolchain
- Verify improvement with NCU: check local memory traffic before/after

> Source: [NVIDIA Developer Blog - SMEM Register Spilling](https://developer.nvidia.cn/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling/)

---

## Persistent Kernels

Instead of launching many short-lived blocks, launch as many CTAs as there are SMs, and have each CTA loop over tiles internally.

### Benefits
- Eliminates block launch overhead between waves (CUDA kernel launch is not free)
- Reuses constants, preloaded data, and TMA descriptors across tiles
- Reduces scheduling jitter on heavily loaded GPUs

### Implementation Pattern
```cpp
// Launch grid_size = SM_count (e.g., 132 for H100)
// Each block loops:
while (true) {
    int tile_idx = atomicAdd(&tile_counter, 1);  // or use CLC on Blackwell
    if (tile_idx >= total_tiles) break;
    // process tile[tile_idx]
}
```

### When to Use
- Large problems where grid count >> SM count would cause many waves
- Kernels with expensive setup per block (TMA descriptor construction, preloaded constants)
- Particularly effective combined with warp specialization on Hopper+

> Source: [PyTorch Blog - MoE Persistent Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) · [Hazy Research - One Kernel for All GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl)

---

## PTX-Level Causality Reasoning

Understanding PTX memory causality can eliminate unnecessary fences that waste throughput.

### Causality Chain (Blackwell)
```
TMA load (cp.async.bulk.tensor)
  → mbarrier complete-tx (release)
    → mbarrier try_wait (acquire)
      → tcgen05.mma (read)
```
This chain is already causally ordered — **no `fence.proxy.async` needed**.

### `elect.sync` vs `laneid() == 0`
When a single thread issues TMA loads, use `elect.sync` (PTX) or `warp::elect_leader()` instead of `laneid() == 0`. The assembler recognizes `elect` and avoids inserting serialization loops over all 32 lanes — saves instruction overhead per TMA issue.

### Key Redundant Fences to Audit
- `fence.proxy.async` between TMA commit and mbarrier wait (redundant when mbarrier is used)
- `tcgen05.fence` between cp and mma on Blackwell (implicitly pipelined)
- Multiple `__syncthreads()` that could be `__syncwarp()` or mbarrier wait

> Source: [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2)

---

## cuBLASDx (Device-side BLAS)

NVIDIA's device-side BLAS library auto-generates highly optimized GEMM kernels without hand-tuning.

### Key Features
- `suggest_layout_smem_*()` for optimal shared memory layout
- Automatic MMA instruction selection (WMMA/WGMMA based on target arch)
- Automatic TMA usage on Hopper+
- Template-based API — tile sizes, data types, pipeline depth as template params

### When to Use
- As a performance baseline before writing custom kernels
- For standard GEMM/convolution where custom tuning is not the bottleneck
- To study generated PTX/SASS for architecture-optimal patterns

> Source: [NVIDIA cuBLASDx Performance Guide](https://docs.nvda.net.cn/cuda/cublasdx/performance.html)

---

## Community Tools

| Tool | Purpose | Source |
|---|---|---|
| **ThunderKittens** | CUDA-embedded DSL for SOTA GEMM on Blackwell/Hopper | [Hazy Research](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) |
| **CUDA Optimizer** | Automated grid search over block sizes, strides, occupancy | [GitHub](https://github.com/GaryBoone/cuda_optimizer) |
| **FlashAttention 1-4** | Reference implementation for memory-efficient attention | [arXiv](https://arxiv.org/abs/2603.05451) |
| **Nsight Compute** | Kernel-level profiling with bottleneck analysis | NVIDIA (bundled with CUDA Toolkit) |

---

## Energy-Aware Optimization

A 2025 IEEE study on 6-level hierarchical optimization found:

1. Occupancy optimization delivers the best performance-per-watt — large throughput gains without proportional energy penalty
2. The optimization progression: naive → coalescing → tiling → occupancy → register blocking → multi-technique
3. Occupancy optimization alone: +111% perf on V100, +58% on T4
4. Register blocking provides incremental gains but at higher energy cost per FLOP

> Source: IEEE 2025 — Energy-Aware CUDA Optimization Taxonomy (search: "6-level hierarchical optimization taxonomy CUDA energy")

---

## Blackbird-Specific TMA Causality (Errata)

When using TMA on Blackwell:
- **Don't put `fence.proxy.async` between TMA commit and mbarrier wait** — the mbarrier already provides ordering
- **Don't put `fence.proxy.async` between TMA cluster fence and `fence.proxy.tensormap`** — these are independent operations
- **Do verify that your mbarrier uses `complete-tx` semantics** (not just `arrive`) to ensure TMA writes are visible before MMA reads

> Source: [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) — PTX Causality section
