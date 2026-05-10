# CUDA Kernel Memory Optimization

---

## Global Memory Access Optimization

### Coalesced Access

32 threads in a warp access consecutive, aligned addresses, and the hardware merges them into the minimum number of memory transactions. Ideally, a single 128B transaction serves the entire warp. Counter-examples are strided or random access, which cause an exponential increase in transaction count.

### Aligned Access

The starting address of an access is aligned to a 128B (or 32B sector) boundary. Misaligned access wastes bandwidth because the hardware reads in sector units — misalignment means extra sectors are fetched but unused.

### Vectorized Memory Access

Use wide types like `float2`, `float4`, `int4`, `double2` to read/write 128-bit in a single instruction. Benefits: fewer instructions, higher effective bytes per transaction. The address must be aligned to the vector width.

### Read-Only Data Path

- **`__ldg()` built-in**: explicitly routes through the read-only data cache (L1 texture cache), bypassing the regular L1 path, with better cache behavior for irregular access patterns.
  - **Note (architecture-specific)**: on newer architectures and compilers, `__ldg()` is no longer a universally beneficial optimization; gains depend on access patterns and cache behavior. Use NCU data to decide.
- **`const __restrict__` pointer annotation**: tells the compiler the pointed-to data is never written and has no aliases, allowing the compiler to automatically select the `__ldg()` path.

### L2 Cache Optimization (Compute Capability 8.0+)

- **L2 Persistence**: use the `cudaAccessPolicyWindow` API to "pin" hot data in the L2 cache, preventing eviction by cold data. Suitable for small but frequently accessed data (e.g., lookup tables, embeddings).
- **L2 access attributes**: `cudaAccessProperty::cudaAccessPropertyPersisting` vs. `Streaming` — fine-grained control over L2 residency policy for different data.

### Sector Granularity

From the Volta architecture onwards, the L1 cache operates at 32B sector granularity (not the traditional 128B cache line fetch). This means the penalty for non-contiguous access is lower than in the Kepler/Maxwell era, but contiguous access is still optimal.

### Coalescing Practical Rules (community-validated)

From GPU MODE Lecture 8 (CUDA Performance Checklist, Sep 2024) — benchmarks on RTX 4090:

- Contiguous thread→memory mapping yields **~2× speedup** (119µs vs 205µs for a reduction kernel)
- DRAM throughput stays similar (~93-94%), but L1 cache throughput rises from 9.5% → 17.5% — coalescing improves L1 hit rate
- Rule: threads 0..31 in a warp must access consecutive 4B/8B/16B elements starting at a 128B-aligned address
- Request alignment: the starting address of the 128B segment accessed by a warp must be aligned — misaligned access wastes bandwidth because hardware fetches extra sectors

### Thread Coarsening

Each thread processes 2+ elements instead of 1, reducing total memory transactions and increasing ILP:

- For memory-bound kernels (elementwise, reductions), thread coarsening is often the single largest win after coalescing
- A 2-element coarsening reduces global load instructions by ~50% and can improve throughput 30-50%
- Trade-off: increases register usage per thread, may reduce occupancy — verify with NCU
- General pattern: `for (int i = 2*idx; i < 2*idx+COARSE_FACTOR; i++)` processing multiple elements in registers before a single store

### Privatization

Keep local copies of data in registers to minimize repeated global memory traffic:

- Load data into a register, perform all intermediate computations, write once
- Sliding-window algorithms benefit most — each element is loaded once but used by multiple neighbors
- Combined with thread coarsening: privatization enables processing N elements from a single global load
- The "register caching" pattern (N4 template parameter) is a structured form of privatization for compile-time-known element counts

> Sources: [GPU MODE Lecture 8 - CUDA Performance Checklist](https://christianjmills.com/posts/cuda-mode-notes/lecture-008/index.html) · [NVIDIA Blog - Boosting CUDA Efficiency](https://developer.nvidia.com/blog/boosting-cuda-efficiency-with-essential-techniques-for-new-developers/)

---

## Shared Memory Optimization

### Tiling

Load data from global memory into tiles in Shared Memory for multiple on-chip reuses. The classic optimization for matrix multiplication — each tile is read from global memory once but reused O(N) times in Shared Memory.

### Bank Conflict Elimination

Shared Memory is divided into 32 banks, each 4B wide (or 8B in some modes). Multiple threads in the same warp accessing different addresses in the same bank are serialized. Solutions:

- **Padding**: add one element at the end of each row of a 2D array, e.g., `__shared__ float s[32][33]`, to offset bank mappings.
- **Swizzle/XOR indexing**: use XOR operations to remap indices — a more advanced and space-efficient approach.

### Async Copy

- **`cp.async` (CUDA 11+)**: Global → Shared Memory transfer handled by hardware DMA, consuming no registers or compute units; can overlap completely with computation.
- **`cuda::memcpy_async` (C++ API)**: a semantically clearer wrapper.
- **Multi-stage pipeline**: allocate multiple Shared Memory buffers — one loading data while another computes — implementing software pipelining that greatly hides global memory latency.

### Shared Memory Capacity Configuration

Use `cudaFuncSetAttribute` to adjust the L1/Shared Memory ratio toward Shared Memory (e.g., from 48KB up to 100KB+), suitable for kernels with high Shared Memory demands. Ampere architecture Shared Memory can reach up to 164KB.

### Dynamic Tile Sizing

Instead of hardcoding tile sizes, query device properties at runtime for portability:

```cpp
int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
int tileSize = sqrt(prop.sharedMemPerBlock / sizeof(float));  // dynamic sizing
```

This adapts to the GPU's actual shared memory capacity without manual per-architecture tuning. For production code, combine dynamic sizing with compile-time autotuning of tile shape (e.g., square vs rectangular tiles).

### CUTLASS Tiled Copy Insights

When implementing custom tiled copies (e.g., column-major 128×32 tile with 256 threads):
- The `TiledCopy` TV Layout controls thread-value mapping — threads within a warp must access **contiguous 128-byte aligned segments** for coalesced loads
- When data layout (row-major vs column-major) doesn't match the default tiled copy pattern, redesign the copy atom, not just the iterator
- `make_tiled_copy` with `raked_product` forces certain thread-value adjacency — may not be optimal for all data layouts
- Key insight: test with NCU's `Global Load Efficiency` metric — it should approach 100% for well-coalesced tiled copies

> Sources: [CUTLASS Issue #1545 - Tiled Copy Deep Dive](https://github.com/NVIDIA/cutlass/issues/1545) · [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/12.4.0/ampere-tuning-guide/contents.html)

---

## Constant Memory

### `__constant__` Memory

A total of 64KB with dedicated hardware cache. Most efficient when all threads in a warp read the same address (broadcast mechanism: a single read serves 32 threads). Suitable for storing kernel parameters, convolution weights, and other small read-only data. If threads in the same warp read different addresses, accesses are serialized and performance degrades.

---

## Texture Memory

### Texture / Surface Objects

- Hardware cache optimized for 2D spatial locality (Morton/Z-order layout), suitable for image processing, stencil computations, and other 2D access patterns.
- Automatic boundary handling (clamp, wrap modes), eliminating manual boundary checks.
- On modern GPUs, `__ldg()` can replace texture in most 1D scenarios; texture still has advantages for 2D spatial locality.

---

## Data Layout Optimization

### SoA vs AoS

- **AoS (Array of Structures)**: `struct { float x, y, z; } particles[N]` — fields of the same particle are stored contiguously. Warp access to the same field results in non-contiguous addresses; cannot be coalesced.
- **SoA (Structure of Arrays)**: `float x[N], y[N], z[N]` — all particles' same field stored contiguously. Naturally coalesced during warp access.
- **Almost always prefer SoA in CUDA** — the difference can be several times in performance.

### Padding and Alignment

Pad the row width of 2D arrays (`cudaMallocPitch` / `cudaMalloc3D`) to ensure the starting address of each row is aligned to the coalesced access boundary.

---

## Memory/Compute Overlap

### Double-buffering / Multi-stage Pipeline

Allocate two sets of buffers in Shared Memory or registers:

- Stage 1: buffer A computes; buffer B loads the next batch of data.
- Stage 2: buffer B computes; buffer A loads the next batch of data.

This is the core technique in all high-performance GEMM implementations (e.g., cuBLAS, CUTLASS).

### CUDA Streams Overlap

- Kernel execution, H2D copy, and D2H copy in different streams can run in parallel.
- Split large data into chunks and create a "copy-compute-copy-back" pipeline across multiple streams.

### Prefetch

- Use `__builtin_prefetch` or manually use extra threads to preload data for the next iteration from Global Memory.
- In Unified Memory scenarios, use `cudaMemPrefetchAsync` to explicitly trigger page migration and avoid the high latency of on-demand page faults.

---

## Reduce Unnecessary Memory Access

### Kernel Fusion

Merge adjacent producer-consumer kernels into one, keeping intermediate results in registers or Shared Memory and eliminating a complete Global Memory round-trip. This is often the single highest-gain optimization.

### Warp Shuffle

Threads within a warp exchange register values directly, without going through Shared Memory. Suitable for warp-level reduction, prefix sum, and broadcast. Latency is approximately 1 clock cycle — faster than Shared Memory with no bank conflicts.

### Cooperative Groups

Flexibly define sync groups smaller than a warp or spanning multiple blocks, with precise sync granularity control, avoiding unnecessary full-block waits from `__syncthreads()`.

### Register Caching

When the number of elements per thread can be determined at compile time, use template parameter `N4` to load all elements into a register array (`float4 regs[N4]`), complete all intermediate computation in registers (max, exp, sum), and only perform a single DRAM write at the end. Use `__launch_bounds__` to control register allocation; `N4` should not exceed 8.

---

## Pinned Memory & Unified Memory

### Pinned Memory (Page-locked)

Memory allocated with `cudaHostAlloc` or `cudaMallocHost` — H2D/D2H transfer bandwidth is much higher than pageable memory (up to 2x) and supports async transfers.

### Unified Memory Optimization

Use `cudaMemPrefetchAsync` to proactively migrate to the target device; use `cudaMemAdvise` (e.g., `cudaMemAdviseSetReadMostly`) to provide access pattern hints to the driver, avoiding the high latency of on-demand page faults.

---

## NCU Validation Checklist

Memory optimizations should include at least the following validations:

- **Bandwidth utilization**: watch `Memory SOL %` and `DRAM Throughput` to confirm they are approaching the expected upper bound.
- **Access quality**: watch `Global Load/Store Efficiency` and `Sectors/Request` to confirm coalescing/alignment has improved.
- **Cache behavior**: watch `L1 Hit Rate` and `L2 Hit Rate` to confirm optimization direction aligns with locality changes.
- **Shared path health**: watch `Shared Memory Efficiency` to confirm bank conflicts have decreased.
- **Overall benefit**: final judgment by kernel latency (avg/median), not just individual sub-metrics.

Optimization entry point: `experience/cuda/CUDA.md` — optimization priorities organized by bottleneck type.
