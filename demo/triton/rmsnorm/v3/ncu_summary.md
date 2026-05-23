# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0425 ms ± 0.0031 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 7.9974 |
| Memory Throughput (% of peak) | 84.2235 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.12e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.65e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.47e+11 |
| L1 Global Load Bandwidth (bytes/s) | 7.28e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.64e+11 |
| L2 Total Bandwidth (bytes/s) | 7.70e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 30.5990 |
| L2 Hit Rate (%) | 52.6032 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.8988 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0870 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 94.9533 |
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
| Issue Slot Utilization (% of peak) | 9.1011 |
| Eligible Warps / Cycle | 0.1776 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.4146 |
| Stall: Long Scoreboard | 91.8223 |
| Stall: Short Scoreboard | 2.5560 |
| Stall: Math Pipe Throttle | 0.8338 |
| Stall: Wait | 1.8129 |
| Stall: No Instruction | 0.4975 |
| Stall: Not Selected | 0.9566 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 41.7997 |
| FMUL Throughput (per cycle) | 101.5136 |
| FFMA Throughput (per cycle) | 47.7711 |
| LSU Pipe Utilization (% of peak) | 3.1955 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3752.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.62e+09 |

**Kernel name:** `rmsnorm_kernel`