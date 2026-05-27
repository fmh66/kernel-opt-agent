# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0524 ms ± 0.0075 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 22.0915 |
| Memory Throughput (% of peak) | 3.5762 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.61e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.51e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.06e+08 |
| L1 Global Load Bandwidth (bytes/s) | 2.07e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.55e+09 |
| L2 Total Bandwidth (bytes/s) | 1.82e+11 |
| Global Load Efficiency (%) | 81.2500 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 21.1583 |
| L2 Hit Rate (%) | 85.8905 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.7971 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1493 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 24.7627 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 32.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.0414 |
| Eligible Warps / Cycle | 0.1650 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.6047 |
| Stall: Long Scoreboard | 11.1487 |
| Stall: Short Scoreboard | 3.1126 |
| Stall: Math Pipe Throttle | 0.0347 |
| Stall: Wait | 1.9711 |
| Stall: No Instruction | 0.0830 |
| Stall: Not Selected | 0.0918 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17408.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 147.6453 |
| FMUL Throughput (per cycle) | 110.0965 |
| FFMA Throughput (per cycle) | 97.1350 |
| LSU Pipe Utilization (% of peak) | 6.1089 |
| Warp Execution Efficiency | 31.9793 |
| L1 Bank Conflicts (total) | 19.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.60e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`