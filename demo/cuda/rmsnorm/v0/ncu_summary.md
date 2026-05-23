# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.5460 ms ± 0.0043 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 0.3115 |
| Memory Throughput (% of peak) | 2.5188 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.84e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.17e+10 |
| DRAM Write Bandwidth (bytes/s) | 6.68e+09 |
| L1 Global Load Bandwidth (bytes/s) | 1.30e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.42e+10 |
| L2 Total Bandwidth (bytes/s) | 8.05e+10 |
| Global Load Efficiency (%) | 12.5000 |
| Global Store Efficiency (%) | 12.5000 |
| L1 Hit Rate (%) | 89.1328 |
| L2 Hit Rate (%) | 85.7940 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 0.6744 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0169 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.5220 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.0079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 1.6911 |
| Eligible Warps / Cycle | 0.0207 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 56.0744 |
| Stall: Short Scoreboard | 0.0051 |
| Stall: Math Pipe Throttle | 0.0005 |
| Stall: Wait | 1.8084 |
| Stall: No Instruction | 0.0486 |
| Stall: Not Selected | 0.2246 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 4512.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0010 |
| FMUL Throughput (per cycle) | 2.0928 |
| FFMA Throughput (per cycle) | 1.0515 |
| LSU Pipe Utilization (% of peak) | 1.6475 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.55e+06 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_rmsnorm`