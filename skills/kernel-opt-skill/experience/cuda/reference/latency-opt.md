# CUDA Kernel Latency Optimization (Latency-Bound)

---

## Occupancy & Launch Configuration

### Launch Configuration Tuning

Block size directly affects occupancy and hardware utilization. Common choices are 128/256/512, but the optimal value depends on the kernel's resource consumption. The CUDA Occupancy Calculator and `cudaOccupancyMaxPotentialBlockSize` API can assist in making the decision.

### Control Register Usage

The more registers each thread uses, the fewer warps can reside simultaneously (lower occupancy), and the weaker the scheduler's ability to hide latency. Use `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` to hint the compiler to control register allocation.

### Register Spilling

When registers are insufficient, the compiler spills variables to local memory (actually global memory, cached in L1/L2). Heavy spilling causes severe performance cliffs. Mitigate by reducing active variable count, shrinking loop unroll factor, or splitting the kernel. Use `--ptxas-options=-v` to inspect register and spill statistics.

### Occupancy Is Not Always Better When Higher

Higher occupancy means more warps available to hide latency, but also means fewer registers and shared memory available per thread. For compute-intensive kernels, a moderate reduction in occupancy in exchange for more registers (fewer spills, higher ILP) often yields better performance. Find the optimal balance through empirical testing.

---

## ILP (Instruction-Level Parallelism) Improvement

### Increase Per-Thread Independent Work

Have each thread process multiple data elements (thread coarsening), completing more computation in registers before writing back. This gives the scheduler more independent instructions to issue while a single warp is waiting, improving ILP.

### Loop Unrolling

Use `#pragma unroll` or `#pragma unroll N` to unroll loops. Unrolling reduces loop control instructions (comparisons, jumps) while exposing more independent instructions to the scheduler, improving ILP.

### Software Pipelining

Overlap the computation of the current iteration with the data prefetch for the next iteration within a loop body, maximizing functional unit utilization.

---

## Synchronization Optimization

### Reduce `__syncthreads()` Count

The most direct approach. If the data access pattern within a warp naturally has no cross-warp dependency, the sync is redundant. Audit every `__syncthreads()` call to confirm its necessity.

### Warp-level Sync Instead of Block-level Sync

The 32 threads within a warp naturally execute in lockstep (after Volta with independent thread scheduling, warp-level primitives are still valid). Using `__syncwarp()` instead of `__syncthreads()` reduces the sync granularity from the entire block to a single warp, dramatically lowering overhead.

- **Note (semantic boundary)**: Under Volta+ independent thread scheduling, "natural lockstep" should not be assumed as an implicit sync guarantee; intra-warp cooperation should still use the correct mask and explicit sync points to ensure visibility and convergence.

### Warp Shuffle

Data exchange between threads within a warp requires no shared memory, has no bank conflict issues, and has extremely low latency (~1 cycle). Suitable for reduction, prefix sum, and broadcast patterns. This directly eliminates the "write shared → sync → read shared" three-step overhead.

**Key primitives (CUDA 9+ `_sync` versions):**

| Instruction | Pattern | Use Case |
|---|---|---|
| `__shfl_down_sync(mask, val, delta)` | Shift-down (gather) | Parallel reduction: `val += __shfl_down_sync(mask, val, offset)` |
| `__shfl_up_sync(mask, val, delta)` | Shift-up (scatter) | Prefix sum backward fill |
| `__shfl_xor_sync(mask, val, laneMask)` | Butterfly (XOR swap) | Warp-level reduction without warp unrolling; all-reduce |
| `__shfl_sync(mask, val, srcLane)` | Broadcast from specific lane | Scalar→vector distribution |

**Warp-level reduction pattern** (standard, no shared memory needed):
```cpp
for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
// Lane 0 now holds the warp sum
```

On SM 8.0+ (Ampere), `__reduce_add_sync(mask, val)` can replace the entire loop with a single instruction.

**CUDA 9+ critical note**: always use `_sync` suffix and pass explicit mask (e.g., `0xffffffff` or `__activemask()`). Old `__shfl()` without mask is deprecated and can cause correctness bugs under Volta+ independent thread scheduling.

### Register Cache — Warp-level Virtual Cache

Instead of storing thread-local data in shared memory, distribute data across warp threads' registers in round-robin fashion:
- Thread i stores element i, thread i+1 stores element i+1, etc.
- When thread j needs element i, it reads from thread i's register via `__shfl_sync(mask, val, i)`
- Benefit: register bandwidth is higher than shared memory (~22 cycle latency vs ~26 for shared memory), no `__syncthreads()` needed, zero bank conflicts
- Pattern: "Remote Register Read" — effectively a warp-level L0 cache implemented via shuffle
- Best suited for: stencil computations, convolutions with small kernel windows, neighbor-based algorithms

> Source: [NVIDIA Developer Blog - Register Cache](https://developer.nvidia.com/blog/register-cache-warp-c-centric-cuda-programs/) · [CUDA Pro Tip: Kepler Shuffle](https://developer.nvidia.com/blog/cuda-pro-tip-kepler-shuffle/)

### Cooperative Groups

The cooperative group mechanism introduced in CUDA 9 allows defining thread groups of arbitrary granularity and syncing within that group — for example, syncing only a tile of 8 threads — avoiding unnecessary full-block waits.

**Core abstractions:**

| API | Scope | Use Case |
|---|---|---|
| `cg::this_thread_block()` | Entire block | Block-level sync (replaces `__syncthreads()`) |
| `cg::tiled_partition<32>(block)` | Warp-sized tile | Warp shuffle with auto-correct mask |
| `cg::tiled_partition<N>(block)` | Arbitrary tile (≤32) | Sub-warp cooperation (e.g., 8-thread tiles) |
| `cg::reduce(tile, data, op)` | Tile-level reduction | One-line warp/tile reduction with correct sync |
| `cg::multi_block_group` | Multi-block (SM 6.0+) | Cross-block cooperation within one kernel launch |
| `cg::grid_group` | Entire grid | Cooperative launch — all blocks syncable |

**Practical example:**
```cpp
namespace cg = cooperative_groups;
auto block = cg::this_thread_block();
auto warp_tile = cg::tiled_partition<32>(block);

// Warp-level reduction using CG — mask is auto-managed
int result = cg::reduce(warp_tile, data, cg::plus<int>());

// Sub-warp tile (8 threads) — finer granularity, less sync waste
auto tile8 = cg::tiled_partition<8>(block);
```

**Key advantage over raw PTX primitives**: CG manages sync masks automatically, eliminating the most common source of correctness bugs (wrong mask / stale mask) in warp-level programming. Use `cg::reduce()` instead of hand-written shuffle loops for reduction — it selects the optimal shuffle pattern and mask.

### Occupancy vs ILP Tradeoff (Volkov's Principle)

From the seminal work "Better Performance at Lower Occupancy" (Volkov, GTC 2010):

- If a kernel has sufficient ILP (Instruction-Level Parallelism), SM execution units stay saturated even at **25-50% occupancy**
- Higher occupancy ≠ better performance: more active warps mean fewer registers per thread, which limits ILP per warp
- **For compute-bound kernels**: lower occupancy + more registers/thread (higher ILP) often wins
- **For memory-bound kernels**: higher occupancy helps hide DRAM latency via warp switching
- Rule of thumb: use NCU's `Issue Slot Utilization` and `Eligible Warps Per Cycle` to judge — if issue slots are already saturated at 50% occupancy, increasing occupancy won't help (and may hurt)

---

### Reduction Optimization Stages (community-benchmarked)

Six-stage progressive optimization for parallel reduction, each with measurable gains:

| Stage | Technique | Approximate Gain |
|---|---|---|
| 1 | Naive global memory (interleaved addressing) | Baseline |
| 2 | Shared memory caching | 3-5× acceleration |
| 3 | Two-level hierarchical reduction (SMEM + GMEM) | +10-20% |
| 4 | Thread coarsening (multiple elements per thread) | +30-50% |
| 5 | **Warp shuffle** replaces shared memory | +10-30% |
| 6 | Bank conflict elimination + cooperative groups | +5-15% |

Final form: warp shuffle for intra-warp reduction → shared memory for inter-warp reduction within block → atomicAdd or second kernel for inter-block reduction.

> Source: [GitHub - MaxReduction-Cuda](https://github.com/mahmoudmaftah/MaxReduction-Cuda) (six-stage optimization with benchmarks) · [NVIDIA CUDA Samples - Reduction](https://deepwiki.com/NVIDIA/cuda-samples/7.1-egl-and-opengl-integration)

### `cuda::barrier` / `cuda::pipeline` (Ampere+)

In async copy and multi-stage pipelines, use explicit stage synchronization instead of "empirical synchronization". The core idea is to clearly specify producer-consumer commit/wait points, avoiding intermittent errors and performance jitter.

---

## Async Prefetch

### `cp.async` Prefetch

(CUDA 11+) Global → Shared Memory transfer is handled by hardware DMA, consuming no registers or compute units, and can overlap completely with computation. Combined with multi-stage buffers, this enables software pipelining that greatly hides global memory latency.

### `cudaMemPrefetchAsync`

In Unified Memory scenarios, proactively trigger page migration to avoid the high latency of on-demand page faults.

---

## Reduce Scheduling Overhead

### CUDA Graphs

When the kernel chain structure is stable and executed repeatedly, Graphs can reduce CPU submission and launch overhead, especially in dense small-kernel scenarios. Evaluate capture/update costs for dynamic graph scenarios.

---

## NCU Validation Checklist

Latency-Bound optimizations should include at least the following validations:

- **Sync wait reduction**: watch whether `Stall Barrier` and related wait stalls decrease.
- **Scheduling issuability**: watch whether `Eligible Warps Per Cycle` improves.
- **Occupancy change**: watch `Achieved Occupancy` combined with kernel latency to judge improvement.
- **Register spilling**: use `--ptxas-options=-v` + NCU to check if spills decrease.
- **Overall benefit**: final judgment by kernel latency (avg/median), not just individual sub-metrics.

Common misdiagnoses:

- Reducing `__syncthreads()` count while introducing data visibility errors.
- Only seeing occupancy rise without watching kernel latency — occupancy can go up while performance gets worse.
- Only seeing a single stall metric decrease without checking overall kernel latency and correctness.

Optimization entry point: `experience/cuda/CUDA.md` — optimization priorities organized by bottleneck type.
