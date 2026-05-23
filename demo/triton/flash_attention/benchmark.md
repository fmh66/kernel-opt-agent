# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `flash_attention.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Correctness** | PASS |
| **Timing Scope** | Steady-state `triton.testing.do_bench` on preallocated/static tensors |
| **Timing Note** | Solution and PyTorch baselines exclude per-call input cloning. |

## Timing (triton.testing.do_bench)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Mean Time (ms)      | 4.0838 | 17.1028 | 2.0392 |
| Median Time (ms)    | 4.0704 | 17.1090 | 2.0439 |
| Min Time (ms)       | 3.9956 | 17.0639 | 1.9251 |
| Max Time (ms)       | 4.1421 | 17.1233 | 2.0859 |
| Std dev (ms)        | 0.0371 | 0.0235 | 0.0403 |
| Speedup (vs PyTorch Eager, mean)   | 4.19x | - | - |
| Speedup (vs PyTorch Eager, median) | 4.20x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.50x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.50x | - | - |
