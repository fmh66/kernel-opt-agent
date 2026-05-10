# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0438 ms ± 0.0067 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 10.2190 |
| Memory Throughput (% of peak) | 79.8873 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.81e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.18e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.63e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.17e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.17e+11 |
| L2 Total Bandwidth (bytes/s) | 6.46e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 51.3161 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.5792 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1491 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 52.7837 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 26.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.3354 |
| Eligible Warps / Cycle | 0.2100 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.6721 |
| Stall: Long Scoreboard | 21.9948 |
| Stall: Short Scoreboard | 2.5715 |
| Stall: Math Pipe Throttle | 0.2344 |
| Stall: Wait | 1.4554 |
| Stall: No Instruction | 0.1384 |
| Stall: Not Selected | 0.3600 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 4096.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 114.0207 |
| FMUL Throughput (per cycle) | 82.9242 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 4.4641 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3703.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.06e+09 |

**Kernel name:** `softmax_kernel_v2`