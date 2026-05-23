# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v1.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Correctness** | PASS |
| **Timing Scope** | Steady-state `triton.testing.do_bench` on preallocated/static tensors |
| **Timing Note** | Solution and PyTorch baselines exclude per-call input cloning. |

## Timing (triton.testing.do_bench)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Mean Time (ms)      | 0.0305 | 0.0553 | 0.0222 |
| Median Time (ms)    | 0.0298 | 0.0553 | 0.0225 |
| Min Time (ms)       | 0.0287 | 0.0543 | 0.0205 |
| Max Time (ms)       | 0.0349 | 0.0563 | 0.0236 |
| Std dev (ms)        | 0.0014 | 0.0006 | 0.0006 |
| Speedup (vs PyTorch Eager, mean)   | 1.81x | - | - |
| Speedup (vs PyTorch Eager, median) | 1.86x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.73x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.76x | - | - |
