# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0519 ms ± 0.0038 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 35.6626 |
| Memory Throughput (% of peak) | 3.2775 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.38e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.29e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.61e+08 |
| L1 Global Load Bandwidth (bytes/s) | 1.92e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.03e+09 |
| L2 Total Bandwidth (bytes/s) | 9.29e+10 |
| Global Load Efficiency (%) | 81.6860 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 59.5620 |
| L2 Hit Rate (%) | 74.7686 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 8.7260 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1954 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 25.3364 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 19.6700 |
| Eligible Warps / Cycle | 0.2493 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1452 |
| Stall: Long Scoreboard | 5.5691 |
| Stall: Short Scoreboard | 3.8209 |
| Stall: Math Pipe Throttle | 0.0811 |
| Stall: Wait | 1.6625 |
| Stall: No Instruction | 0.0796 |
| Stall: Not Selected | 0.2766 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17408.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 267.7651 |
| FMUL Throughput (per cycle) | 196.1307 |
| FFMA Throughput (per cycle) | 176.1612 |
| LSU Pipe Utilization (% of peak) | 10.0539 |
| Warp Execution Efficiency | 31.9927 |
| L1 Bank Conflicts (total) | 0.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.90e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`