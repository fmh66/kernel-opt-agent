# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0520 ms ± 0.0043 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 18.8333 |
| Memory Throughput (% of peak) | 3.4566 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.51e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.42e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.33e+08 |
| L1 Global Load Bandwidth (bytes/s) | 1.02e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.32e+09 |
| L2 Total Bandwidth (bytes/s) | 9.65e+10 |
| Global Load Efficiency (%) | 81.6860 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 21.0549 |
| L2 Hit Rate (%) | 76.0185 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.2752 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1137 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 12.6147 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 11.4558 |
| Eligible Warps / Cycle | 0.1200 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.4419 |
| Stall: Long Scoreboard | 6.1222 |
| Stall: Short Scoreboard | 2.1709 |
| Stall: Math Pipe Throttle | 0.0215 |
| Stall: Wait | 1.8392 |
| Stall: No Instruction | 0.0584 |
| Stall: Not Selected | 0.0473 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 8704.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 141.3628 |
| FMUL Throughput (per cycle) | 106.0269 |
| FFMA Throughput (per cycle) | 93.0018 |
| LSU Pipe Utilization (% of peak) | 5.2775 |
| Warp Execution Efficiency | 31.9868 |
| L1 Bank Conflicts (total) | 0.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.54e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`