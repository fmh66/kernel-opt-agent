# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Execution Time** | 2.0619 ms ± 0.0373 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 98.0728 |
| Memory Throughput (% of peak) | 1.4779 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.08e+10 |
| DRAM Read Bandwidth (bytes/s) | 3.83e+09 |
| DRAM Write Bandwidth (bytes/s) | 6.94e+09 |
| L1 Global Load Bandwidth (bytes/s) | 6.49e+12 |
| L1 Global Store Bandwidth (bytes/s) | 7.26e+09 |
| L2 Total Bandwidth (bytes/s) | 9.72e+11 |
| Global Load Efficiency (%) | 68.0514 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 85.2790 |
| L2 Hit Rate (%) | 99.5922 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 13.5778 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6917 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 76.3860 |
| Theoretical Occupancy (%) | 83.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 48.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 39.0095 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 69.1748 |
| Eligible Warps / Cycle | 3.1506 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 1.5830 |
| Stall: Short Scoreboard | 0.0016 |
| Stall: Math Pipe Throttle | 2.3750 |
| Stall: Wait | 1.5251 |
| Stall: No Instruction | 0.0285 |
| Stall: Not Selected | 3.5481 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 7.45e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 575.0279 |
| LSU Pipe Utilization (% of peak) | 24.6645 |
| Warp Execution Efficiency | 28.0543 |
| L1 Bank Conflicts (total) | 3.81e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `conv2d_v3`