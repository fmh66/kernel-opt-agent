# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0497 ms ± 0.0042 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 35.6638 |
| Memory Throughput (% of peak) | 3.2352 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.35e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.28e+10 |
| DRAM Write Bandwidth (bytes/s) | 6.97e+08 |
| L1 Global Load Bandwidth (bytes/s) | 1.92e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.02e+09 |
| L2 Total Bandwidth (bytes/s) | 9.28e+10 |
| Global Load Efficiency (%) | 81.6860 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 59.5244 |
| L2 Hit Rate (%) | 80.9427 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.0339 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2143 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 25.3816 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 39.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.5531 |
| Eligible Warps / Cycle | 0.2996 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1136 |
| Stall: Long Scoreboard | 4.7422 |
| Stall: Short Scoreboard | 2.8634 |
| Stall: Math Pipe Throttle | 0.1132 |
| Stall: Wait | 1.7906 |
| Stall: No Instruction | 0.0653 |
| Stall: Not Selected | 0.4017 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17408.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 267.1195 |
| FMUL Throughput (per cycle) | 200.3442 |
| FFMA Throughput (per cycle) | 175.7365 |
| LSU Pipe Utilization (% of peak) | 9.8641 |
| Warp Execution Efficiency | 31.9935 |
| L1 Bank Conflicts (total) | 9.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.90e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`