# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v5.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.5306 | 0.1977 | 0.2446 |
| Std dev (ms)        | 0.0188 | 0.0049 | 0.0079 |
| Speedup (vs PyTorch Eager)    | 0.37x | — | — |
| Speedup (vs PyTorch Compile)    | 0.46x | — | — |

## Hardware Metrics (nsight-python)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|----------:|----------:|
| SM Throughput (% peak) | 76.6065 | 76.6759 | 76.7676 |
| Memory Throughput (% peak) | 3.8846 | 3.8836 | 3.8937 |
| DRAM Bandwidth (bytes/s) | 2.83e+10 | 2.83e+10 | 2.84e+10 |
| Achieved Occupancy (%) | 90.2079 | 90.0954 | 90.2459 |
