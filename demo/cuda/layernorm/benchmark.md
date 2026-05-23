# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v2.cu` |
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
| Mean Time (ms)      | 1.8646 | 3.0694 | 1.8271 |
| Median Time (ms)    | 1.8627 | 3.0693 | 1.8269 |
| Min Time (ms)       | 1.8575 | 3.0628 | 1.8237 |
| Max Time (ms)       | 1.9497 | 3.0771 | 1.8309 |
| Std dev (ms)        | 0.0134 | 0.0035 | 0.0017 |
| Speedup (vs PyTorch Eager, mean)   | 1.65x | - | - |
| Speedup (vs PyTorch Eager, median) | 1.65x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.98x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.98x | - | - |
