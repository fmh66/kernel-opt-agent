# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Correctness** | PASS |
| **Timing Scope** | Steady-state `triton.testing.do_bench` on preallocated/static tensors |
| **Timing Note** | Solution and PyTorch baselines exclude per-call input cloning. |

## Timing (triton.testing.do_bench)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Mean Time (ms)      | 0.0190 | 0.0079 | 0.0052 |
| Median Time (ms)    | 0.0186 | 0.0082 | 0.0051 |
| Min Time (ms)       | 0.0174 | 0.0072 | 0.0041 |
| Max Time (ms)       | 0.0247 | 0.0092 | 0.0061 |
| Std dev (ms)        | 0.0009 | 0.0005 | 0.0003 |
| Speedup (vs PyTorch Eager, mean)   | 0.41x | - | - |
| Speedup (vs PyTorch Eager, median) | 0.44x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.27x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.28x | - | - |
