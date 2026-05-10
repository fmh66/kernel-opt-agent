---
name: cuda
description: CUDA optimization strategies by bottleneck type. Assumes bottleneck has been classified by profiling/PROFILING.md.
---

# cuda

## Directory Structure

```
cuda/
├── CUDA.md
└── reference/
    ├── compute-opt.md        Compute-bound optimization
    ├── latency-opt.md        Latency-bound optimization
    ├── memory-opt.md         Memory-bound optimization
    ├── architecture-opt.md   Architecture-specific (Ampere/Hopper/Blackwell)
    └── community-opt.md      Community advanced techniques
```

## Memory-Bound

**Optimization priority:**
1. Kernel Fusion — eliminate Global Memory round-trips; keep intermediates in registers
2. Coalesced access + SoA layout + vectorization (`float4/int4`)
3. Shared Memory Tiling + Bank Conflict elimination (padding / swizzle)
4. `cp.async` + double-buffering / multi-stage pipeline
5. `__ldg()` / `const __restrict__` / L2 Persistence (CC 8.0+)
6. Pinned Memory + CUDA Stream pipeline

> Detailed entries → `reference/memory-opt.md` · Architecture-specific → `reference/architecture-opt.md`

---

## Compute-Bound

**Optimization priority:**
1. Tensor Core / WMMA / MMA PTX — first choice for matrix kernels
2. FMA (`__fmaf_rn()`) + strength reduction (`rsqrtf` / shifts) + `--use_fast_math`
3. Eliminate branch divergence: predication / select instructions / rearrange data by warp / `__all_sync()` early exit
4. `#pragma unroll` + loop transformations (split / merge / interchange) + software pipelining

> Detailed entries → `reference/compute-opt.md` · Advanced community patterns → `reference/community-opt.md`

---

## Latency-Bound

**Optimization priority:**
1. Tune block size (128 / 256 / 512 empirical testing) + `__launch_bounds__`
2. Warp Shuffle instead of Shared Memory three-step sync (write → sync → read)
3. `__syncwarp()` instead of `__syncthreads()` / Cooperative Groups minimum sync group
4. `cp.async` prefetch + increase per-thread independent work (ILP)
5. `--ptxas-options=-v` to check register spilling → reduce active variables / split kernel
6. CUDA Graphs — for dense small-kernel scenarios to reduce CPU launch overhead

> Detailed entries → `reference/latency-opt.md` · Persistent kernels & warp specialization → `reference/community-opt.md`

---

## Anti-Patterns (from experience)

These patterns have repeatedly led to regressions or neutral outcomes across real optimization sessions. Check for them before committing to a strategy.

- **Fixing bank conflicts at the cost of compute efficiency**: Shared memory padding (e.g., +1 column) can eliminate bank conflicts but may increase shared memory usage or alter the load-to-FMA ratio unfavorably. When fixing bank conflicts, verify the instructions-per-FMA ratio doesn't degrade.
- **Compute optimizations on memory-bound kernels**: When Memory SOL is already above ~90%, further compute-side changes (fast-math, FMA intrinsics, unrolling) yield negligible gains — the kernel is bandwidth-saturated. Focus on reducing memory traffic instead.
- **Unrolling without checking IPC**: `#pragma unroll` reduces branch instruction count but can lower IPC due to increased register pressure. Always compare IPC before/after unrolling.
- **Occupancy for occupancy's sake**: Higher occupancy does not guarantee better performance. For compute-bound kernels, lower occupancy with more registers per thread often wins. Use latency measurements, not occupancy numbers, as the success criterion.
- **Multiple changes in one iteration**: Violating the one-variable rule makes it impossible to attribute performance changes. If two changes interact, test them separately in successive iterations.

## General Principles

- **Occupancy is not always better when higher**: for Compute-Bound kernels, lower occupancy for more registers often yields faster kernels; use measured latency as the final criterion
- **Correctness before optimization**: pass correctness check each iteration before measuring performance
- **`--use_fast_math` requires caution**: may introduce precision issues; must re-validate numerics after enabling
- **Metrics-driven, not intuition-driven**: each optimization must include NCU evidence (summary + details)
- **Prioritize end-to-end gains**: single-kernel local improvements need validation with full benchmark
