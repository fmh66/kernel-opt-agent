---
name: benchmark
description: Benchmark a custom CUDA/Triton kernel against a reference implementation (PyTorch). Measures execution time via triton.testing.do_bench.
---

# benchmark

## Directory Structure

```
benchmark/
├── BENCHMARK.md
└── script/
    └── benchmark.py
```

---

## Overview

Compares solution kernel performance against PyTorch reference (both **eager** and **torch.compile**), outputting:

- **Execution time** (`triton.testing.do_bench`, mean/median/min/max/std over raw samples) — 3-way comparison on preallocated/static tensors
- **Correctness validation** (`torch.allclose` against both eager and compiled reference, run before timing)

> Measurement strategy: execution time is collected only via `triton.testing.do_bench`.
> The default PyTorch baseline reuses static CUDA tensors so it is comparable to custom kernels running in-place on preallocated buffers.

---

## Usage

> **Prerequisites**
> - CUDA: compile `.so` first via nvcc; the script only loads, does not compile
> - Triton: no `.so` compilation required

```bash
# Compile first (CUDA only)
nvcc -shared -std=c++17 -arch=sm_90 -O3 -Xcompiler -fPIC -o kernel.so kernel.cu

# Benchmark (CUDA or Triton)
python script/benchmark.py <solution.{cu,py}> \
    --ref=<ref.py> \
    --output-dir=<dir> \
    --M=<M> --N=<N> \
    [--backend=<auto/cuda/triton>] \
    [--warmup=<n>] \
    [--iters=<n>] \
    [--ptr-size=<n>] \
    [--arch=<sm_XX>] \
    [--gpu=<id>] \
    [--atol=<atol>] [--rtol=<rtol>] \
    [--seed=<seed>]
```

---

## CLI Parameters

| Parameter | Required | Default | Description |
|---|:---:|---|---|
| `solution_file` | ✓ | — | `.cu` or `.py` (Triton) |
| `--ref` | ✓ | — | Reference implementation `.py`, defines `reference(**kwargs)` |
| `--output-dir` | ✓ | — | Output directory |
| `--M/--N/...` | ✓ | — | Integer dimension parameters from kernel signature |
| `--backend` | | `auto` | `auto/cuda/triton` |
| `--warmup` | | 20 | `do_bench` warmup duration in milliseconds |
| `--iters` | | 100 | `do_bench` repetition duration in milliseconds |
| `--ptr-size` | | 0 | Override CUDA pointer buffer element count (ignored for Triton) |
| `--arch` | | auto-detected | e.g. `sm_90` |
| `--gpu` | | 0 | GPU device index |
| `--atol/--rtol` | | 1e-4/1e-3 | Correctness tolerance |
| `--seed` | | 42 | Random seed |

---

## Output Files

| File | Description |
|---|---|
| `benchmark.md` | 3-way comparison report: Solution vs PyTorch Eager vs PyTorch Compile, with correctness, steady-state execution time, and speedup |
