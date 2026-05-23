# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Execution Time** | 2.9033 ms ± 0.0102 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 78.7145 |
| Memory Throughput (% of peak) | 0.9669 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 7.05e+09 |
| DRAM Read Bandwidth (bytes/s) | 2.51e+09 |
| DRAM Write Bandwidth (bytes/s) | 4.53e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.53e+12 |
| L1 Global Store Bandwidth (bytes/s) | 4.80e+09 |
| L2 Total Bandwidth (bytes/s) | 5.89e+11 |
| Global Load Efficiency (%) | 30.3250 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 63.9511 |
| L2 Hit Rate (%) | 99.7001 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 22.9714 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6296 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 98.7880 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 1296.0000 |
| Waves / SM | 32.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 62.9820 |
| Eligible Warps / Cycle | 1.7340 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 4.9446 |
| Stall: Long Scoreboard | 5.4985 |
| Stall: Short Scoreboard | 0.2233 |
| Stall: Math Pipe Throttle | 0.3785 |
| Stall: Wait | 2.3744 |
| Stall: No Instruction | 0.3152 |
| Stall: Not Selected | 1.7532 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.49e+08 |
| Divergent Branch Targets (total) | 1.61e+07 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 368.2635 |
| LSU Pipe Utilization (% of peak) | 19.9172 |
| Warp Execution Efficiency | 27.6347 |
| L1 Bank Conflicts (total) | 6.62e+07 |
| Shared Memory Bandwidth (bytes/s) | 3.27e+12 |

**Kernel name:** `conv2d_v2`