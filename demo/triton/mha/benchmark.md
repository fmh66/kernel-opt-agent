# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v5.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.1939 | 0.9232 | 0.8104 |
| Std dev (ms)        | 0.0008 | 0.2378 | 0.0404 |
| Speedup (vs PyTorch Eager)    | 4.76x | — | — |
| Speedup (vs PyTorch Compile)    | 4.18x | — | — |
