# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v2.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.2971 | 0.8112 | 0.7150 |
| Std dev (ms)        | 0.0033 | 0.0021 | 0.0040 |
| Speedup (vs PyTorch Eager)    | 2.73x | — | — |
| Speedup (vs PyTorch Compile)    | 2.41x | — | — |

## Hardware Metrics (nsight-python)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|----------:|----------:|
| SM Throughput (% peak) | 14.4512 | 14.3620 | 14.4416 |
| Memory Throughput (% peak) | 93.5993 | 93.5814 | 93.6182 |
| DRAM Bandwidth (bytes/s) | 6.82e+11 | 6.82e+11 | 6.82e+11 |
| Achieved Occupancy (%) | 80.4286 | 80.5232 | 80.5051 |
