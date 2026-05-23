# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0.py` |
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
| Mean Time (ms)      | 0.0150 | 0.0265 | 0.0224 |
| Median Time (ms)    | 0.0154 | 0.0266 | 0.0225 |
| Min Time (ms)       | 0.0143 | 0.0246 | 0.0205 |
| Max Time (ms)       | 0.0164 | 0.0276 | 0.0236 |
| Std dev (ms)        | 0.0006 | 0.0006 | 0.0006 |
| Speedup (vs PyTorch Eager, mean)   | 1.77x | - | - |
| Speedup (vs PyTorch Eager, median) | 1.73x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 1.49x | - | - |
| Speedup (vs PyTorch Compile, median) | 1.47x | - | - |
