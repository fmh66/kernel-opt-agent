# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0437 ms ± 0.0061 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 7.8317 |
| Memory Throughput (% of peak) | 85.6781 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.23e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.73e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.50e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.72e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.72e+11 |
| L2 Total Bandwidth (bytes/s) | 7.58e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 50.8711 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.2940 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1218 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 40.2444 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 33.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 8.0000 |
| Waves / SM | 0.7619 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 12.3371 |
| Eligible Warps / Cycle | 0.1713 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.9641 |
| Stall: Long Scoreboard | 26.0661 |
| Stall: Short Scoreboard | 2.4763 |
| Stall: Math Pipe Throttle | 0.2235 |
| Stall: Wait | 1.7676 |
| Stall: No Instruction | 0.1338 |
| Stall: Not Selected | 0.3882 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 112.7155 |
| FMUL Throughput (per cycle) | 97.4837 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 2.5887 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2753.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.09e+09 |

**Kernel name:** `softmax_kernel_v5`