# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Correctness** | PASS |
| **Timing Scope** | Steady-state `triton.testing.do_bench` on preallocated/static tensors |
| **Timing Note** | Solution and PyTorch baselines exclude per-call input cloning. |

## Timing (triton.testing.do_bench)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Mean Time (ms)      | 8.4474 | 3.0694 | 1.8267 |
| Median Time (ms)    | 8.4449 | 3.0700 | 1.8268 |
| Min Time (ms)       | 8.3988 | 3.0638 | 1.8227 |
| Max Time (ms)       | 8.5166 | 3.0751 | 1.8309 |
| Std dev (ms)        | 0.0361 | 0.0030 | 0.0016 |
| Speedup (vs PyTorch Eager, mean)   | 0.36x | - | - |
| Speedup (vs PyTorch Eager, median) | 0.36x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.22x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.22x | - | - |
