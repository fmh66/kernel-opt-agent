# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0516 ms ± 0.0044 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 35.8640 |
| Memory Throughput (% of peak) | 3.4172 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.48e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.38e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.93e+08 |
| L1 Global Load Bandwidth (bytes/s) | 1.66e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.25e+09 |
| L2 Total Bandwidth (bytes/s) | 9.65e+10 |
| Global Load Efficiency (%) | 96.3028 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 51.4358 |
| L2 Hit Rate (%) | 77.5602 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.0341 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2149 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 25.3961 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.6292 |
| Eligible Warps / Cycle | 0.3057 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1734 |
| Stall: Long Scoreboard | 4.4521 |
| Stall: Short Scoreboard | 3.1559 |
| Stall: Math Pipe Throttle | 0.0935 |
| Stall: Wait | 1.7852 |
| Stall: No Instruction | 0.0729 |
| Stall: Not Selected | 0.4101 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17408.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 279.5519 |
| FMUL Throughput (per cycle) | 209.6687 |
| FFMA Throughput (per cycle) | 183.9157 |
| LSU Pipe Utilization (% of peak) | 10.0451 |
| Warp Execution Efficiency | 31.9931 |
| L1 Bank Conflicts (total) | 0.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.98e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`