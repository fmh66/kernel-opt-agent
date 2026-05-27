# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0509 ms ± 0.0051 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 19.2443 |
| Memory Throughput (% of peak) | 3.6580 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.66e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.56e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.41e+08 |
| L1 Global Load Bandwidth (bytes/s) | 8.91e+10 |
| L1 Global Store Bandwidth (bytes/s) | 5.65e+09 |
| L2 Total Bandwidth (bytes/s) | 1.04e+11 |
| Global Load Efficiency (%) | 96.3028 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 6.1543 |
| L2 Hit Rate (%) | 77.7487 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.3678 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1140 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 12.7022 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 37.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 11.4828 |
| Eligible Warps / Cycle | 0.1215 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.4462 |
| Stall: Long Scoreboard | 5.4788 |
| Stall: Short Scoreboard | 2.6124 |
| Stall: Math Pipe Throttle | 0.0273 |
| Stall: Wait | 1.8813 |
| Stall: No Instruction | 0.0822 |
| Stall: Not Selected | 0.0567 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 8704.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 149.9576 |
| FMUL Throughput (per cycle) | 112.4734 |
| FFMA Throughput (per cycle) | 98.6563 |
| LSU Pipe Utilization (% of peak) | 5.3841 |
| Warp Execution Efficiency | 31.9860 |
| L1 Bank Conflicts (total) | 2.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.63e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`