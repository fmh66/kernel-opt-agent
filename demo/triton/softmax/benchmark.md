# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v0.py` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.0388 | 0.0731 | 0.1498 |
| Std dev (ms)        | 0.0018 | 0.0050 | 0.0061 |
| Speedup (vs PyTorch Eager)    | 1.88x | — | — |
| Speedup (vs PyTorch Compile)    | 3.86x | — | — |

## Hardware Metrics (nsight-python)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|----------:|----------:|
| SM Throughput (% peak) | 16.5714 | 16.7501 | 16.9366 |
| Memory Throughput (% peak) | 74.0569 | 73.0752 | 73.7460 |
| DRAM Bandwidth (bytes/s) | 5.37e+11 | 5.31e+11 | 5.37e+11 |
| Achieved Occupancy (%) | 19.7492 | 19.9812 | 20.0335 |
