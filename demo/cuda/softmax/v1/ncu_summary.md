# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 0.4516 ms ± 0.0035 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 9.0802 |
| Memory Throughput (% of peak) | 92.3764 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.73e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.84e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.89e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.63e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.08e+11 |
| L2 Total Bandwidth (bytes/s) | 7.31e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 20.4262 |
| L2 Hit Rate (%) | 47.3847 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.5916 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0577 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.7549 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 64.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 8.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 5.7739 |
| Eligible Warps / Cycle | 0.0791 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 27.1140 |
| Stall: Long Scoreboard | 140.8207 |
| Stall: Short Scoreboard | 4.4616 |
| Stall: Math Pipe Throttle | 0.0652 |
| Stall: Wait | 2.3141 |
| Stall: No Instruction | 0.1173 |
| Stall: Not Selected | 0.3882 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 622592.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 76.7452 |
| FMUL Throughput (per cycle) | 40.2598 |
| FFMA Throughput (per cycle) | 83.0358 |
| LSU Pipe Utilization (% of peak) | 2.3355 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 103900.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.42e+09 |

**Kernel name:** `softmax_v1`