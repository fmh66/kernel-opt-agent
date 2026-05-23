# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0455 ms ± 0.0033 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 14.0823 |
| Memory Throughput (% of peak) | 84.3875 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.13e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.72e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.40e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.11e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.71e+11 |
| L2 Total Bandwidth (bytes/s) | 8.16e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 45.8984 |
| L2 Hit Rate (%) | 54.8371 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.3625 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0989 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 93.0617 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 10.1676 |
| Eligible Warps / Cycle | 0.1564 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.7392 |
| Stall: Long Scoreboard | 77.0083 |
| Stall: Short Scoreboard | 5.0597 |
| Stall: Math Pipe Throttle | 0.4123 |
| Stall: Wait | 2.0814 |
| Stall: No Instruction | 0.5291 |
| Stall: Not Selected | 0.6000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 128.2121 |
| FMUL Throughput (per cycle) | 54.9480 |
| FFMA Throughput (per cycle) | 146.5281 |
| LSU Pipe Utilization (% of peak) | 5.1967 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 8478.0000 |
| Shared Memory Bandwidth (bytes/s) | 9.43e+09 |

**Kernel name:** `layernorm_kernel`