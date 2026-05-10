# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0399 ms ± 0.0029 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 12.0227 |
| Memory Throughput (% of peak) | 85.5663 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.22e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.65e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.58e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.64e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.64e+11 |
| L2 Total Bandwidth (bytes/s) | 7.41e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.7840 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.9103 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1525 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 78.8133 |
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
| Issue Slot Utilization (% of peak) | 15.4986 |
| Eligible Warps / Cycle | 0.2770 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.3655 |
| Stall: Long Scoreboard | 37.2326 |
| Stall: Short Scoreboard | 3.7193 |
| Stall: Math Pipe Throttle | 0.4719 |
| Stall: Wait | 1.6377 |
| Stall: No Instruction | 0.2499 |
| Stall: Not Selected | 0.7072 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 131.6619 |
| FMUL Throughput (per cycle) | 95.7541 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 4.7870 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2839.0000 |
| Shared Memory Bandwidth (bytes/s) | 9.24e+09 |

**Kernel name:** `softmax_kernel`