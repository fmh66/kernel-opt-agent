# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0734 ms ± 0.0036 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 5.9254 |
| Memory Throughput (% of peak) | 0.9895 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 7.21e+09 |
| DRAM Read Bandwidth (bytes/s) | 7.14e+09 |
| DRAM Write Bandwidth (bytes/s) | 7.07e+07 |
| L1 Global Load Bandwidth (bytes/s) | 5.51e+10 |
| L1 Global Store Bandwidth (bytes/s) | 1.51e+09 |
| L2 Total Bandwidth (bytes/s) | 1.31e+10 |
| Global Load Efficiency (%) | 82.7128 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 86.8251 |
| L2 Hit Rate (%) | 48.9350 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 4.1463 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0913 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3431 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 32.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.1460 |
| Eligible Warps / Cycle | 0.0915 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.0372 |
| Stall: Long Scoreboard | 2.9849 |
| Stall: Short Scoreboard | 3.3332 |
| Stall: Math Pipe Throttle | 0.0025 |
| Stall: Wait | 2.2747 |
| Stall: No Instruction | 0.0584 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17536.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 39.8131 |
| FMUL Throughput (per cycle) | 29.9911 |
| FFMA Throughput (per cycle) | 26.1928 |
| LSU Pipe Utilization (% of peak) | 4.0252 |
| Warp Execution Efficiency | 31.9972 |
| L1 Bank Conflicts (total) | 0.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.36e+09 |

**Kernel name:** `_gqa_paged_decode_kernel`