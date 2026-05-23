# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | rmsnorm.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0416 ms ± 0.0031 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 8.0644 |
| Memory Throughput (% of peak) | 85.9991 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.25e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.68e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.57e+11 |
| L1 Global Load Bandwidth (bytes/s) | 7.34e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.67e+11 |
| L2 Total Bandwidth (bytes/s) | 7.76e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 30.5990 |
| L2 Hit Rate (%) | 52.6503 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.9500 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0879 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 100.6040 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 32.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.1956 |
| Eligible Warps / Cycle | 0.1779 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.4611 |
| Stall: Long Scoreboard | 92.0499 |
| Stall: Short Scoreboard | 2.4917 |
| Stall: Math Pipe Throttle | 0.8197 |
| Stall: Wait | 1.8118 |
| Stall: No Instruction | 0.4828 |
| Stall: Not Selected | 0.9391 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 42.1498 |
| FMUL Throughput (per cycle) | 102.3637 |
| FFMA Throughput (per cycle) | 48.1712 |
| LSU Pipe Utilization (% of peak) | 3.2281 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3547.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.66e+09 |

**Kernel name:** `rmsnorm_kernel`