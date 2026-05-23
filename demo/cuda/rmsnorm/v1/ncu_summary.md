# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0306 ms ± 0.0039 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 20.1158 |
| Memory Throughput (% of peak) | 78.3931 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.69e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.31e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.37e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.90e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.30e+11 |
| L2 Total Bandwidth (bytes/s) | 7.14e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 47.2107 |
| L2 Hit Rate (%) | 54.3086 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 11.9114 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2539 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 101.7151 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 30.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 2.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 25.8242 |
| Eligible Warps / Cycle | 0.5762 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 4.0678 |
| Stall: Long Scoreboard | 23.2960 |
| Stall: Short Scoreboard | 1.5495 |
| Stall: Math Pipe Throttle | 0.4574 |
| Stall: Wait | 2.8857 |
| Stall: No Instruction | 0.3266 |
| Stall: Not Selected | 1.2023 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 61440.0000 |
| Divergent Branch Targets (total) | 1024.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 61.1782 |
| FMUL Throughput (per cycle) | 86.9487 |
| FFMA Throughput (per cycle) | 43.6866 |
| LSU Pipe Utilization (% of peak) | 7.6093 |
| Warp Execution Efficiency | 31.5791 |
| L1 Bank Conflicts (total) | 12842.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.06e+09 |

**Kernel name:** `rmsnorm_v1`