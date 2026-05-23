# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `conv2d.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Correctness** | PASS |
| **Timing Scope** | Steady-state `triton.testing.do_bench` on preallocated/static tensors |
| **Timing Note** | Solution and PyTorch baselines exclude per-call input cloning. |

## Timing (triton.testing.do_bench)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Mean Time (ms)      | 2.4118 | 0.1443 | 0.1749 |
| Median Time (ms)    | 2.4387 | 0.1434 | 0.1732 |
| Min Time (ms)       | 2.2252 | 0.1403 | 0.1710 |
| Max Time (ms)       | 2.4750 | 0.1516 | 0.4260 |
| Std dev (ms)        | 0.0793 | 0.0030 | 0.0188 |
| Speedup (vs PyTorch Eager, mean)   | 0.06x | - | - |
| Speedup (vs PyTorch Eager, median) | 0.06x | - | - |
| Speedup (vs PyTorch Compile, mean)   | 0.07x | - | - |
| Speedup (vs PyTorch Compile, median) | 0.07x | - | - |
