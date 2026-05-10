# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1916 ms ± 0.0058 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 80.8784 |
| Memory Throughput (% of peak) | 12.6398 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.21e+10 |
| DRAM Read Bandwidth (bytes/s) | 6.08e+10 |
| DRAM Write Bandwidth (bytes/s) | 3.13e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.72e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.69e+10 |
| L2 Total Bandwidth (bytes/s) | 1.75e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 96.5639 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 37.7951 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.5727 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 37.1331 |
| Theoretical Occupancy (%) | 41.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 56.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16384.0000 |
| Waves / SM | 2.4381 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 57.3033 |
| Eligible Warps / Cycle | 1.2875 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.0349 |
| Stall: Long Scoreboard | 0.1853 |
| Stall: Short Scoreboard | 1.1644 |
| Stall: Math Pipe Throttle | 0.0293 |
| Stall: Wait | 0.2868 |
| Stall: No Instruction | 0.0125 |
| Stall: Not Selected | 1.2436 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 262144.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 3768.4105 |
| LSU Pipe Utilization (% of peak) | 21.7174 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 46099.0000 |
| Shared Memory Bandwidth (bytes/s) | 6.04e+12 |

**Kernel name:** `_gemm_kernel`