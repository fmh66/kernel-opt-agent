# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 4.8214 ms ± 0.0959 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 63.4630 |
| Memory Throughput (% of peak) | 60.2081 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 4.39e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.32e+11 |
| DRAM Write Bandwidth (bytes/s) | 6.46e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.80e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.05e+09 |
| L2 Total Bandwidth (bytes/s) | 7.86e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 47.0400 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 18.9495 |
| Tensor Core Utilization (% of peak) | 69.6360 |
| IPC (instructions per cycle) | 0.3579 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 15.2075 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 384.0000 |
| Registers / Thread | 255.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 49152.0000 |
| Waves / SM | 2.2857 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 35.7884 |
| Eligible Warps / Cycle | 0.4379 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1689 |
| Stall: Long Scoreboard | 0.1295 |
| Stall: Short Scoreboard | 0.6066 |
| Stall: Math Pipe Throttle | 1.6228 |
| Stall: Wait | 1.2340 |
| Stall: No Instruction | 0.0035 |
| Stall: Not Selected | 0.2238 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.58e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 226.5628 |
| FMUL Throughput (per cycle) | 328.3911 |
| FFMA Throughput (per cycle) | 6.6636 |
| LSU Pipe Utilization (% of peak) | 4.7367 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 464163.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.09e+12 |

**Kernel name:** `flash_attention_kernel`