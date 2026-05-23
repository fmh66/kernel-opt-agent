# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0463 ms ± 0.0038 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 5.7435 |
| Memory Throughput (% of peak) | 74.7938 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.43e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.00e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.43e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.74e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.99e+11 |
| L2 Total Bandwidth (bytes/s) | 6.35e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 7.4653 |
| L2 Hit Rate (%) | 53.6900 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.3696 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0674 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 31.9585 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 6.9092 |
| Eligible Warps / Cycle | 0.0788 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.1024 |
| Stall: Long Scoreboard | 28.0243 |
| Stall: Short Scoreboard | 2.7510 |
| Stall: Math Pipe Throttle | 0.1074 |
| Stall: Wait | 1.7044 |
| Stall: No Instruction | 0.1878 |
| Stall: Not Selected | 0.1372 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 34.3076 |
| FMUL Throughput (per cycle) | 83.3186 |
| FFMA Throughput (per cycle) | 39.2087 |
| LSU Pipe Utilization (% of peak) | 2.6552 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3547.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.80e+09 |

**Kernel name:** `rmsnorm_kernel`