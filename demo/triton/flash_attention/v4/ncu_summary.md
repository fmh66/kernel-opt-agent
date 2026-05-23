# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 5.2533 ms ± 0.0213 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 50.5598 |
| Memory Throughput (% of peak) | 27.2662 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.99e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.93e+11 |
| DRAM Write Bandwidth (bytes/s) | 5.42e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.31e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.09e+09 |
| L2 Total Bandwidth (bytes/s) | 3.36e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 42.4490 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.0356 |
| Tensor Core Utilization (% of peak) | 52.5482 |
| IPC (instructions per cycle) | 0.2185 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3338 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1536.0000 |
| Registers / Thread | 255.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 65536.0000 |
| Waves / SM | 18.2857 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.8508 |
| Eligible Warps / Cycle | 0.2185 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0654 |
| Stall: Long Scoreboard | 0.0526 |
| Stall: Short Scoreboard | 0.7116 |
| Stall: Math Pipe Throttle | 1.0586 |
| Stall: Wait | 1.5967 |
| Stall: No Instruction | 0.0037 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 786432.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 180.4986 |
| FMUL Throughput (per cycle) | 261.6234 |
| FFMA Throughput (per cycle) | 5.3088 |
| LSU Pipe Utilization (% of peak) | 2.4425 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 406640.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.31e+12 |

**Kernel name:** `flash_attention_kernel`