# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0295 ms ± 0.0021 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 18.9471 |
| Memory Throughput (% of peak) | 80.2447 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.84e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.48e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.35e+11 |
| L1 Global Load Bandwidth (bytes/s) | 8.67e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.47e+11 |
| L2 Total Bandwidth (bytes/s) | 7.62e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 38.6248 |
| L2 Hit Rate (%) | 54.2713 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 8.6607 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1893 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.9892 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 256.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 19.3228 |
| Eligible Warps / Cycle | 0.4601 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.1599 |
| Stall: Long Scoreboard | 37.5630 |
| Stall: Short Scoreboard | 1.2545 |
| Stall: Math Pipe Throttle | 0.7406 |
| Stall: Wait | 2.2373 |
| Stall: No Instruction | 0.5381 |
| Stall: Not Selected | 1.3558 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 49152.0000 |
| Divergent Branch Targets (total) | 4608.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 64.7514 |
| FMUL Throughput (per cycle) | 91.0476 |
| FFMA Throughput (per cycle) | 52.6369 |
| LSU Pipe Utilization (% of peak) | 6.8838 |
| Warp Execution Efficiency | 31.1138 |
| L1 Bank Conflicts (total) | 7646.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.47e+09 |

**Kernel name:** `rmsnorm_v5`