# Triton Community Experience & Advanced Patterns

Production-proven advanced patterns and community experience for pushing Triton kernel performance beyond the basics.

---

## Persistent Kernel Pattern

### Concept
Instead of launching `grid_size = ceil(total_work / tile_size)`, launch exactly `grid_size = SM_count` (e.g., 132 on H100) and have each program loop over multiple tiles internally.

### Why It Wins
- Eliminates block launch overhead between waves (CUDA kernel launch is ~5-10µs per wave)
- Reuses TMA descriptors, preloaded constants, and SMEM buffers across tiles
- Reduces scheduling jitter — no "long-tail" blocks delaying the next wave

### Implementation Sketch
```python
@triton.jit
def persistent_kernel(...):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # = SM_COUNT

    while True:
        tile_id = tl.atomic_add(tile_counter_ptr, 1)  # fetch next tile
        if tile_id >= total_tiles: break
        # Process tile[tile_id] normally
```

### Production Result (MoE Grouped GEMM)
- 2.62× speedup over naive PyTorch loop for DeepSeek-v3 training on H100
- Combined persistent kernel + grouped launch ordering + dynamic TMA descriptors

> Source: [PyTorch Blog - MoE Persistent Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/)

---

## Grouped Launch Ordering for L2 Cache

### The Problem
In large GEMM with linear `(pid_m, pid_n)` ordering, adjacent programs access different B columns simultaneously, thrashing the L2 cache.

### The Solution: Column-Major Group Traversal
Instead of linear row-major traversal of tiles, traverse in column-major order within a band of rows:
- Programs within a group share access to the same B tiles
- L2 cache lines for B are reused rather than evicted

### Production Result
- +60% L2 cache hit rate
- 1.33× speedup on H100 grouped GEMM
- Technique originated in Triton matmul tutorial's "swizzle" section

> Source: [PyTorch Blog - MoE Persistent Grouped GEMM](https://pytorch.org/blog/accelerating-moes-with-a-triton-persistent-cache-aware-grouped-gemm-kernel/) · [Triton Tutorial - Matrix Multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)

---

## TLX (Triton Low-Level Extensions)

TLX provides lower-level primitives beyond standard `tl.dot`, `tl.load`, `tl.store`, enabling patterns that map directly to Hopper/Blackwell hardware features.

### Key TLX Capabilities
- **Warp specialization**: `tlx.async_task` for producer/consumer warp groups
- **Pingpong scheduling**: two consumer groups + two SMEM buffers — while one computes, the other loads next data
- **TMA direct access**: `tlx.make_tensor_descriptor` for on-device TMA descriptor construction
- **Group tiling**: pack multiple logical tiles (e.g., GQA query heads) into one physical tile to reduce mask computation waste

### Production Result (2-Simplicial Attention)
- 588 BF16 TFLOPS on H100 (~60% Tensor Core utilization)
- Reduced GQA mask waste from 73% → 1.35% via group tiling
- Asymmetric sliding window: small W1 (persisted in SMEM) + large W2 (maximizes Tensor Core ratio)

> Source: [PyTorch Blog - Fast 2-Simplicial Attention](https://pytorch.org/blog/fast-2-simplicial-attention-hardware-efficient-kernels-in-tlx/)

---

## Optimizing Tensor Core Utilization

### Binary Decomposition
Ternary operations (e.g., trilinear products in attention variants) don't map well to WGMMA instructions. Decompose into binary ops:
```
Before: C = A * B * D  (ternary, no direct MMA mapping)
After:  T = tl.dot(A, B); C = T * D  (binary MMA + elementwise)
```

### Accumulation Precision
- Accumulator should stay fp32 even when inputs are fp16/bf16/fp8
- Follow the dtype passed by the upstream caller — kernel authors should not arbitrarily reduce precision
- Whether fp16/bf16/fp8 can be used is a framework/model-level decision

### MMA Shape Constraints
- Minimum granularity: typically 16×16×16 for MMA instructions
- Tile sizes that don't align to MMA shapes create "cleanup code" outside Tensor Core path
- Use powers of 2 tile sizes that are multiples of MMA instruction dimensions

---

## Split-K Parallelization

### When to Use
- K dimension is large but M×N is small — normal grid has few CTAs, unable to fill all SMs
- Example: certain attention projections, embedding backpropagation

### Pattern
```python
# Split K into S parts
for s in range(SPLIT_K):
    partial = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    # Process K chunk [s * K_chunk : (s+1) * K_chunk]
    for k in range(k_start, k_end):
        partial += tl.dot(a, b)
    tl.atomic_add(C, partial)  # atomic merge with other splits
```

### Trade-off
- Cost: multiple atomic adds (with contention) or secondary reduction kernel
- Benefit: fills SMs that would otherwise be idle
- Production result: 1.94× speedup on Llama3-70B FP8 inference

> Source: [PyTorch Blog - Accelerating Llama3 FP8 Inference with Triton](https://pytorch.com.tw/blog/accelerating-llama3/)

---

## CUDA Graphs with Triton Kernels

### When to Use
- Production inference with stable computation graph
- Dense small-kernel chains where launch overhead dominates

### Pattern
```python
# Warmup: run once to trigger autotuning and compilation
model(data)

# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(data)

# Replay — no JIT, no autotune, no launch overhead
g.replay()
```

### Production Result
- 6.4× end-to-end speedup in attention layers (SGLang case)
- Eliminates both Triton JIT overhead and CUDA kernel launch overhead

---

## Avoid the "Interpret Mode" Trap

Setting `os.environ["TRITON_INTERPRET"] = "1"` disables JIT compilation entirely:
- Kernels become **200×–1500× slower** than PyTorch native
- Only useful for debugging kernel logic (print statements work)
- Never use in production or benchmarking

---

## Roofline-Guided Optimization

The Roofline model is the single most important tool for determining optimization direction:

### Compute Ridge Point (H100 approximate)
- `Arithmetic Intensity = Peak FLOPs / Peak Bandwidth`
- Below ridge point → **memory-bound**: optimize coalescing, tiling, fusion, reduce redundant transfers
- Above ridge point → **compute-bound**: optimize Tensor Core utilization, instruction selection, pipeline depth

### Quick Estimation
```
AI ≈ (2 * M * N * K) / (2 * (M*K + K*N + M*N) * bytes_per_element)
```
For fp16 GEMM with tiling:
```
AI ≈ (2 * BLOCK_M * BLOCK_N * BLOCK_K) / (2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * 2)
```

### Action Based on Roofline
| Kernel Type | Typical AI | Bottleneck | Primary Optimization |
|---|---|---|---|
| Elementwise (add, relu) | ~0.25 | Memory-bandwidth | Fusion, vectorization |
| LayerNorm / RMSNorm | ~1-2 | Memory-bandwidth | Fusion, Welford single-pass |
| Softmax | ~2-5 | Mixed | Online softmax, warp reduce |
| Matmul >1024² | ~50-200 | Compute | Tensor Core, tile tuning |
| FlashAttention | ~50-150 | Mixed | Online softmax + MMA + TMA |

> Source: [Triton Documentation - Performance](https://triton-lang.org/main/index.html) · [AMD ROCm Blog - Roofline](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)

---

## GEAK AI-Agent Approach (AMD)

For reference on what AI-assisted kernel optimization can achieve:
- GEAK-OptimAgentv2: multi-offspring evolution + LLM evaluator + hardware profiler feedback → 3.32× avg speedup
- GEAK-OpenEvolve: Quality-Diversity search (MAP-Elites) over 9 optimization dimensions → 3.42×-7.02× speedup
- Optimization dimensions: tile size, num_warps, num_stages, memory layout, instruction scheduling, register allocation, etc.

> Source: [AMD GEAK Blog](https://rocm.blogs.amd.com/artificial-intelligence/geak-agents-family/)

---

## Quick Reference: Operator-Type Optimization Priority

| Operator Type | Priority 1 | Priority 2 | Priority 3 |
|---|---|---|---|
| **Matmul (large)** | Tile size + Tensor Core | Swizzle + split-K | num_stages tuning |
| **Matmul (small/batched)** | Persistent kernel | Grouped ordering | Avoid atomic contention |
| **Attention** | Online softmax fusion | TMA + warp specialization | Group tiling (GQA) |
| **Elementwise** | Operator fusion | Vectorization (`tl.max_contiguous`) | Grid size tuning |
| **Normalization** | Welford single-pass | Fusion with downstream op | Warp reduction |
| **Reduction** | Warp shuffle reduce | Split-K for large K | Persistent for small M×N |
| **Variable-length / Sparse** | `cu_seqlens` indexing | Load balancing | `mask=` for boundary |
