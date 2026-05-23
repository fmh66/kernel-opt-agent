# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Execution Time** | 0.0190 ms ± 0.0017 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 8.9102 |
| Memory Throughput (% of peak) | 23.7240 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.71e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.52e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.97e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.51e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.51e+11 |
| L2 Total Bandwidth (bytes/s) | 3.38e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 55.3748 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.1667 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1655 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 64.2031 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 400.0000 |
| Registers / Thread | 16.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.7937 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 17.9880 |
| Eligible Warps / Cycle | 0.2723 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 20.8444 |
| Stall: Short Scoreboard | 1.3514 |
| Stall: Math Pipe Throttle | 0.1345 |
| Stall: Wait | 2.7469 |
| Stall: No Instruction | 1.5011 |
| Stall: Not Selected | 0.5036 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6400.0000 |
| Divergent Branch Targets (total) | 3200.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 29.2905 |
| FMUL Throughput (per cycle) | 120.1844 |
| FFMA Throughput (per cycle) | 83.1456 |
| LSU Pipe Utilization (% of peak) | 2.2566 |
| Warp Execution Efficiency | 25.0086 |
| L1 Bank Conflicts (total) | 615.0000 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_gelu`