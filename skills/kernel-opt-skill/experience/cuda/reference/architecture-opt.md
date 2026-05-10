# CUDA Architecture-Specific Optimization

Architecture-specific tuning guidance for Ampere, Hopper, and Blackwell GPUs. Always verify with NCU on target hardware — optimal configs differ significantly across generations.

> Source: [NVIDIA Blackwell Tuning Guide](https://docs.nvidia.com/cuda/archive/12.9.1/blackwell-tuning-guide/index.html) · [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.3/ampere-tuning-guide/index.html) · [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

---

## Quick Reference by Architecture

| Resource | Ampere (A100) | Hopper (H100) | Blackwell (B200/GB200) |
|---|---|---|---|
| SM Count | 108 | 132 | 148 (B200) |
| Max Shared Memory / SM | 164 KB | 228 KB | 228 KB |
| L2 Cache | 40 MB | 50 MB | 126 MB (GB200) |
| Max Registers / SM | 64K | 64K | 64K |
| Tensor Core Data Types | FP16/BF16/TF32/INT8 | + FP8 | + NVFP4/MXFP8 |
| NVLink Bandwidth | 600 GB/s | 900 GB/s | 900 GB/s (unidirectional) |

---

## Ampere (A100 / RTX 30) Specific

### Key Hardware Features
- **LDGSTS instruction**: direct global → shared memory copy, bypassing register intermediate
- **TF32 Tensor Cores**: 156 TFLOPS vs 19.5 FP32 CUDA core — always prefer TF32 over FP32 for MMA
- **`cp.async`**: async global → shared transfer without consuming compute units
- **Structured Sparsity (2:4)**: hardware support for ~2× throughput via `cusparseLt`

### Tuning Notes
- Shared memory capacity configurable: `cudaFuncSetAttribute` to shift L1/SMEM ratio
- L2 cache residency controls via `cudaAccessPolicyWindow`
- `cp.async` + multi-stage pipeline (double/triple buffering) is the primary latency-hiding mechanism

---

## Hopper (H100) Specific

### Key Hardware Features
- **TMA (Tensor Memory Accelerator)**: `cp.async.bulk.tensor` — dedicated hardware unit for async tensor transfers with multi-dimensional addressing, lower latency than `cp.async`
- **FP8 Tensor Cores**: `wgmma.fence` + `wgmma.commit_group` for warp-group MMA with FP8
- **Warp Specialization with `setmaxnreg`**: dynamically allocate registers per warp — producer warps get fewer regs, consumer warps get more
- **Distributed Shared Memory (DSMEM)**: SMs within a cluster can directly access each other's shared memory
- **Thread Block Clusters**: up to 16 CTAs per cluster with hardware-accelerated sync

### Tuning Notes
- Prefer TMA over `cp.async` when available — frees registers and compute units
- Warp specialization is a first-class pattern: dedicate warps to TMA load vs MMA compute vs epilogue
- Cluster-level sync via `cuda::cluster::barrier` for DSMEM coordination
- TMA works on both local and peer (NVLink) addresses — use for multi-GPU fused kernels

---

## Blackwell (B200/GB200) Specific

### Key Hardware Features
- **Tensor Memory (TMEM)**: 128×512 per SM, ~200× faster than HBM — dedicated on-chip memory for MMA accumulators
- **Cluster Launch Control (CLC)**: hardware tile scheduler that distributes work in producer-consumer model, reducing software scheduling overhead
- **2-CTA MMA Mode**: two CTAs share tensor core resources on one SM for better utilization
- **Implicit `tcgen05.cp` ↔ `tcgen05.mma` pipelining**: scale copy to TMEM is automatically pipelined with MMA — issue from same thread, drop explicit barrier waits
- **5th-gen NVLink**: 900 GB/s unidirectional with `multimem.red` / `multimem.ld_reduce` — single-shot all-reduce in fabric
- **NVSwitch multicast**: broadcast data to multiple GPUs in one transaction

### Tuning Notes
- **Buffer TMEM stages**: TMEM is 128×512 — buffer two MMA stages (columns 0-255 for stage 0, 256-511 for stage 1) to overlap accumulation with epilogue write-out
- **Double accumulation**: for large GEMM (>2048²), run two MMA pipelines sharing one A tile with different B tiles — halves memory traffic
- **Avoid unnecessary fences**: `tcgen05.fence` and `fence.proxy.async` can cost 20-30 TFLOPS. The chain TMA→mbarrier→MMA is already causally ordered — extra fences are often redundant
- **CLC scheduling**: on Blackwell, launch as many CTAs as SMs and let CLC handle tile distribution — no need for software tile scheduler
- **SM specialization for multi-GPU**: only 8-16 SMs (out of 148) needed to saturate NVLink; assign remaining to compute-only warps

> Source: [Hazy Research - One Kernel for All GPUs](https://hazyresearch.stanford.edu/blog/2025-09-22-pgl) · [ThunderKittens 2.0](https://hazyresearch.stanford.edu/blog/2026-02-19-tk-2) · [Modular - Matrix Multiplication on Blackwell Part 4](https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota)

---

## Cross-Architecture Principles

### Shared Memory Capacity Planning
- Ampere: max 164 KB/SM (configurable split with L1)
- Hopper/Blackwell: max 228 KB/SM
- `num_stages` selection is bounded by tile size × element size × 2 (A+B tiles) < available SMEM
- Rule of thumb: keep at least 2 stages for pipelining; 3-4 on Ampere, 4-6 on Hopper/Blackwell with TMA

### Occupancy Sweet Spots
- Compute-bound kernels (MMA-heavy): target 25-50% occupancy, maximize register count per thread
- Memory-bound kernels (elementwise, reduction): target 50-100% occupancy to hide DRAM latency
- Use `--ptxas-options=-v` to check register usage; use `__launch_bounds__` to guide compiler

### When to Use Which Architecture Feature
| Problem | Ampere | Hopper | Blackwell |
|---|---|---|---|
| Hide DRAM latency | `cp.async` + pipeline | TMA + warp specialization | TMA + CLC |
| Reduce sync overhead | Cooperative groups | Cluster barriers | CLC hardware scheduling |
| MMA throughput | TF32/BF16 WMMA | WGMMA FP8 | TMEM + 2-CTA mode |
| Multi-GPU reduction | NCCL ring | NVSwitch + DSMEM | NVSwitch multicast |
| L2 cache control | Residency hints | Residency + TMA hints | 126 MB — less contention |

---

## NCU Validation per Architecture

When profiling architecture-specific optimizations:
- **Hopper**: confirm TMA instructions appear in PTX; check `device__attribute_tma_utilization`
- **Blackwell**: confirm `tcgen05.mma` and `tcgen05.cp` in SASS; check TMEM utilization
- **Ampere**: confirm `cp.async` and reduced `Long Scoreboard` stalls
- For all: verify that register spills decreased; verify `Achieved Occupancy` aligns with target range for bottleneck type
