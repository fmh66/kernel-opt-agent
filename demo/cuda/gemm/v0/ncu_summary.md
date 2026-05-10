# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.9445 ms ± 0.0213 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 97.3330 |
| Memory Throughput (% of peak) | 2.1904 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.60e+10 |
| DRAM Read Bandwidth (bytes/s) | 9.75e+09 |
| DRAM Write Bandwidth (bytes/s) | 6.22e+09 |
| L1 Global Load Bandwidth (bytes/s) | 4.97e+12 |
| L1 Global Store Bandwidth (bytes/s) | 4.86e+09 |
| L2 Total Bandwidth (bytes/s) | 6.21e+11 |
| Global Load Efficiency (%) | 56.2500 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 87.5270 |
| L2 Hit Rate (%) | 98.5096 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 19.1739 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2858 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.0356 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 8.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 28.5794 |
| Eligible Warps / Cycle | 1.5942 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 4.2519 |
| Stall: Short Scoreboard | 0.0115 |
| Stall: Math Pipe Throttle | 0.0389 |
| Stall: Wait | 2.0201 |
| Stall: No Instruction | 0.0430 |
| Stall: Not Selected | 4.5737 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.33e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 653.1207 |
| LSU Pipe Utilization (% of peak) | 24.5170 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.28e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_gemm`