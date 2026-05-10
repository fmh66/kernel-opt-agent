# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v5.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.1749 | 0.2214 | 0.2719 |
| Std dev (ms)        | 0.0022 | 0.0034 | 0.0045 |
| Speedup (vs PyTorch Eager)    | 1.27x | — | — |
| Speedup (vs PyTorch Compile)    | 1.55x | — | — |
