# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0415 ms ± 0.0027 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 21.2397 |
| Memory Throughput (% of peak) | 77.6917 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.66e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.27e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.39e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.26e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.26e+11 |
| L2 Total Bandwidth (bytes/s) | 6.64e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 51.7371 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.5710 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2136 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 98.3372 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 28.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.7131 |
| Eligible Warps / Cycle | 0.4525 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.9229 |
| Stall: Long Scoreboard | 26.9517 |
| Stall: Short Scoreboard | 4.3717 |
| Stall: Math Pipe Throttle | 0.6004 |
| Stall: Wait | 1.8074 |
| Stall: No Instruction | 0.2336 |
| Stall: Not Selected | 1.0860 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 8192.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 161.5817 |
| FMUL Throughput (per cycle) | 86.1769 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 8.9156 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4192.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.59e+10 |

**Kernel name:** `softmax_kernel_v1`