# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | layernorm.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0440 ms ± 0.0032 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 14.5322 |
| Memory Throughput (% of peak) | 84.7243 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.16e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.72e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.43e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.11e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.71e+11 |
| L2 Total Bandwidth (bytes/s) | 8.16e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 45.8984 |
| L2 Hit Rate (%) | 54.3483 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.5399 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1246 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 90.7925 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 39.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 12.7525 |
| Eligible Warps / Cycle | 0.2294 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.5504 |
| Stall: Long Scoreboard | 63.8794 |
| Stall: Short Scoreboard | 3.4599 |
| Stall: Math Pipe Throttle | 0.6133 |
| Stall: Wait | 1.7649 |
| Stall: No Instruction | 0.3401 |
| Stall: Not Selected | 0.7986 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 128.1737 |
| FMUL Throughput (per cycle) | 54.9316 |
| FFMA Throughput (per cycle) | 146.4842 |
| LSU Pipe Utilization (% of peak) | 5.2138 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 8165.0000 |
| Shared Memory Bandwidth (bytes/s) | 9.43e+09 |

**Kernel name:** `layernorm_kernel`