# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Execution Time** | 3.1676 ms ± 0.0120 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 72.6523 |
| Memory Throughput (% of peak) | 0.9012 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.57e+09 |
| DRAM Read Bandwidth (bytes/s) | 2.34e+09 |
| DRAM Write Bandwidth (bytes/s) | 4.23e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.38e+12 |
| L1 Global Store Bandwidth (bytes/s) | 4.47e+09 |
| L2 Total Bandwidth (bytes/s) | 5.73e+11 |
| Global Load Efficiency (%) | 31.5229 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 60.3767 |
| L2 Hit Rate (%) | 99.7736 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 29.5593 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7306 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 98.8149 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 1296.0000 |
| Waves / SM | 32.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 73.0682 |
| Eligible Warps / Cycle | 2.2637 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.8446 |
| Stall: Long Scoreboard | 3.6497 |
| Stall: Short Scoreboard | 0.4063 |
| Stall: Math Pipe Throttle | 0.8306 |
| Stall: Wait | 2.7899 |
| Stall: No Instruction | 0.2144 |
| Stall: Not Selected | 2.0966 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.61e+08 |
| Divergent Branch Targets (total) | 7.80e+06 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 343.5741 |
| LSU Pipe Utilization (% of peak) | 17.1235 |
| Warp Execution Efficiency | 29.0725 |
| L1 Bank Conflicts (total) | 5.60e+07 |
| Shared Memory Bandwidth (bytes/s) | 3.05e+12 |

**Kernel name:** `conv2d_v1`