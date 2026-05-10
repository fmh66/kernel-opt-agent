# Benchmark Report

| Field | Value |
|-------|-------|
| **Solution** | `v5.cu` |
| **Reference** | `ref.py` |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Correctness** | PASS |

## Timing (CUDA Events)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|-------------:|----------------:|
| Execution Time (ms) | 0.8944 | 0.4241 | 0.3422 |
| Std dev (ms)        | 0.0429 | 0.0239 | 0.0144 |
| Speedup (vs PyTorch Eager)    | 0.47x | — | — |
| Speedup (vs PyTorch Compile)    | 0.38x | — | — |

## Hardware Metrics (nsight-python)

| Metric | Solution | PyTorch Eager | PyTorch Compile |
|--------|----------:|----------:|----------:|
| SM Throughput (% peak) | 24.1984 | 24.0671 | 24.1983 |
| Memory Throughput (% peak) | 1.6305 | 1.6100 | 1.5939 |
| DRAM Bandwidth (bytes/s) | 1.19e+10 | 1.17e+10 | 1.16e+10 |
| Achieved Occupancy (%) | 65.0883 | 65.0867 | 65.0874 |
