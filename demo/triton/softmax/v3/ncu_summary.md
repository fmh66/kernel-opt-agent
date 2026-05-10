# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0406 ms ± 0.0040 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 12.4279 |
| Memory Throughput (% of peak) | 85.8248 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.25e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.79e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.46e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.79e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.79e+11 |
| L2 Total Bandwidth (bytes/s) | 7.71e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 51.3312 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.8623 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1511 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 77.5005 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 23.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.3633 |
| Eligible Warps / Cycle | 0.2676 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.6040 |
| Stall: Long Scoreboard | 38.4215 |
| Stall: Short Scoreboard | 3.6346 |
| Stall: Math Pipe Throttle | 0.4799 |
| Stall: Wait | 1.6362 |
| Stall: No Instruction | 0.2587 |
| Stall: Not Selected | 0.7152 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 136.0988 |
| FMUL Throughput (per cycle) | 98.9810 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 4.7450 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2981.0000 |
| Shared Memory Bandwidth (bytes/s) | 9.62e+09 |

**Kernel name:** `softmax_kernel_v3`