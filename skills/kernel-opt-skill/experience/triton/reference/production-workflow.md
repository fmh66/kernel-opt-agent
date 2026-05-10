# Triton Production Optimization Workflow

Real-world optimization workflows from production deployments, combining profiling, autotuning, and verification.

---

## The Production Optimization Loop

Based on the StepTronOSS workflow used to land Triton kernels in production models.

### Phase 1: Profile First
- Start from a profiler trace, not intuition — identify the actual hot kernels in forward/backward passes
- Use `torch.profiler.profile()` with `activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]`
- Rank kernels by total GPU time; focus on top 3-5 offenders
- Check if the bottleneck is memory-bandwidth or compute using Roofline

### Phase 2: Replace One Op at a Time
- Prefer single-op replacement over large fusions first
- Keep optimized kernels in `model/optimizations/` directory with drop-in replacement interface
- Maintain a naive PyTorch reference implementation as correctness baseline
- Fuse only after single-op replacements are validated — large fusions are harder to debug

### Phase 3: Three-Level Testing
1. **Unit correctness**: `torch.allclose()` with fp64 reference, appropriate atol/rtol for dtype
2. **Microbenchmark**: isolated kernel timing with `torch.cuda.Event` and warmup iterations
3. **Real experiment**: full model forward + backward + optimizer step, verify convergence and throughput

### Phase 4: Benchmark Standard
- Always report forward + backward times + end-to-end throughput
- Test multiple shapes: small (batch=1), medium (typical production batch), large (max GPU memory)
- Cross-GPU validation: A100 and H100 optimal configs usually differ

> Source: [StepTronOSS Triton Acceleration Workflow](https://github.com/stepfun-ai/SteptronOss/blob/dev/docs/TRITON_ACCELERATION_WORKFLOW.md)

---

## Autotuning Best Practices

### Configuration Space Design
- Use `itertools.product()` to generate config cartesian products, but **filter to 5-15 candidates** — exhaustive search makes first-run painfully slow
- Tile sizes should be powers of 2 (16, 32, 64, 128, 256)
- Co-tune `num_warps` (4 or 8) and `num_stages` (2-5) with tile sizes
- Use `key` parameter on `@triton.autotune` to cache by input shape — avoids re-tuning for same shape

### Caching Tuning Results
Autotuning results are NOT auto-saved across runs (each restart = re-tune):
- **Development**: run tuning once, record the best config, hard-code it for production (remove `@triton.autotune`)
- **Custom caching**: wrap autotune with a JSON/config-based cache that persists across runs
- **CI integration**: cache frozen configs with releases so users never pay tuning cost

### Common Pitfalls
- Too many candidates → minutes of tuning before first real computation
- `constexpr` parameters changing dynamically → triggers recompilation on every call (12µs kernel + 0.7ms compile overhead)
- Optimizing for only one shape → config degrades on slightly different dimensions
- Not specifying `key` → same kernel shape re-tuned on every Python process restart

> Source: [Triton Issue #6355 - saving autotune results](https://github.com/triton-lang/triton/issues/6355) · [Triton Issue #4257 - tuning candidates](https://github.com/triton-lang/triton/issues/4257) · [Helion Autotuner](https://github.com/pytorch/helion)

---

## Intelligent Autotuning (Beyond Grid Search)

### Helion Approach (PyTorch)
- **ML-based filtering**: small neural networks / decision trees / random forests to prune search space
- **Evolutionary algorithms**: generate configs adaptively rather than exhaustively
- **Coordinate descent tuning**: optimize one parameter at a time — one case showed 598 GB/s → 971 GB/s (1.6×) for a reduction kernel

### When Brute Force Fails
For large config spaces (e.g., 4 tile dimensions × 3 num_warps × 3 num_stages = 108+ combos), exhaustive tuning is impractical. Use:
1. Run a sparse grid first (5-10 combos) to warm up
2. Profile the best candidate with NCU
3. Only expand the search space for dimensions that NCU flags as bottlenecked

> Source: [Triton Issue #9308 - ML-driven autotuning](https://github.com/triton-lang/triton/issues/9308) · [Helion Autotuner](https://github.com/pytorch/helion)

---

## JIT and Caching Management

### Triton Cache Behavior
- Cache key = hash of (kernel source, compilation flags, GPU architecture, driver version)
- `~/.triton/cache/` is the default location
- Useful environment variables:
  - `TRITON_CACHE_DIR=<path>` — custom cache location
  - `TRITON_ALWAYS_COMPILE=1` — bypass cache, always recompile
  - `TRITON_KERNEL_OVERRIDE=1` — substitute compiled artifacts
  - `TRITON_STORE_BINARY_ONLY=1` — save ~77% disk space (omit IR/PTX intermediate files)

### Avoiding Recompilation Overhead
- Keep `constexpr` parameters stable across calls — changing strides/batch sizes triggers recompilation
- Use CUDA Graphs in production to capture compiled kernels and eliminate JIT latency
- In one production case (SGLang), CUDA Graphs provided 6.4× end-to-end speedup for attention layers

> Source: [Red Hat - Understanding Triton Cache](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/)

---

## IR Inspection for Optimization Verification

Dump intermediate representations to verify optimization passes are firing:

```bash
export TRITON_CACHE_DIR=/tmp/triton_cache
# After kernel run, inspect:
# TTIR (Triton IR)     — high-level structure
# TTGIR (Triton GPU IR) — layout, pipeline, num_stages effect
# LLIR (LLVM IR)       — deep debugging
# PTX                  — key instructions: ld.global.v4, mma.sync, cp.async, wgmma
```

### What to Look For
- **TTGIR**: confirm that num_stages > 1 (pipelining active), check tile layout annotations
- **PTX**: confirm `cp.async` or `cp.async.bulk.tensor` (TMA) for async loads; `mma.sync` or `wgmma` for Tensor Core path
- **PTX**: check for unexpected `st.local` (register spills to local memory) — indicates tile too large
- **Register count**: inspect cubin metadata for register usage per thread

> Source: [AMD ROCm Blog - Triton Optimization](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html) · [Triton Documentation](https://triton-lang.org/main/index.html)

---

## Cross-Platform (NVIDIA + AMD) Considerations

### Key Differences
| Aspect | NVIDIA | AMD |
|---|---|---|
| Tensor Core API | WMMA / WGMMA | MFMA / WMMA |
| Async copy | `cp.async` / TMA | Stream Pipeline pass |
| Shared memory | 164-228 KB (configurable) | LDS (Local Data Share), size varies |
| Compiler passes | Standard Triton pipeline | AMD-specific passes must be enabled |

### AMD-Specific Optimization Passes (ROCm Triton Fork)
- `AMD GPU Accelerate Matmul`: optimize dot-op layouts for MFMA matrix cores
- `AMD GPU Stream Pipeline`: pipeline global loads through registers into shared memory
- `AMD GPU Block Pingpong`: interleave instructions from two warps on same SIMD for better latency hiding
- `AMD GPU Reorder Instructions`: reduce register pressure by reordering conversions
- `AMD GPU Optimize LDS Usage`: minimize shared memory consumption

### Cross-Platform Strategy
1. Develop and validate correctness on NVIDIA first (easier tooling)
2. Profile on AMD and enable AMD-specific passes
3. Dump IRs on both platforms to verify equivalent optimization levels
4. Maintain separate autotune configs per architecture — optimal tile sizes differ

> Source: [AMD ROCm Blog - Triton Kernel Optimizations](https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html)

---

## Production Checklist

Before shipping a Triton kernel to production:
- [ ] Correctness verified with fp64 reference at multiple shapes
- [ ] Boundary cases tested: non-divisible dims, K=0, extreme values, NaN propagation
- [ ] Autotune config frozen and cached — no first-run tuning for users
- [ ] `constexpr` params stable — no dynamic recompilation in hot paths
- [ ] CUDA Graphs wrapper ready (if applicable) to eliminate JIT launch overhead
- [ ] Benchmarked at 3 shape sizes (small/medium/large)
- [ ] Cross-architecture tested (A100 + H100 minimum)
- [ ] Performance regression benchmark in CI
- [ ] NCU profile shows expected instruction paths (TMA, WGMMA, etc.)
