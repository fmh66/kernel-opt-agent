# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 6.6973 ms ± 0.6235 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 41.9828 |
| Memory Throughput (% of peak) | 94.7023 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.90e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.86e+11 |
| DRAM Write Bandwidth (bytes/s) | 4.03e+09 |
| L1 Global Load Bandwidth (bytes/s) | 9.66e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.76e+09 |
| L2 Total Bandwidth (bytes/s) | 9.70e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 44.1637 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 11.1642 |
| Tensor Core Utilization (% of peak) | 51.4091 |
| IPC (instructions per cycle) | 0.2711 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 21.5481 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 6144.0000 |
| Registers / Thread | 108.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 49664.0000 |
| Waves / SM | 36.5714 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.1098 |
| Eligible Warps / Cycle | 0.3262 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.4469 |
| Stall: Long Scoreboard | 0.9432 |
| Stall: Short Scoreboard | 0.6842 |
| Stall: Math Pipe Throttle | 0.6075 |
| Stall: Wait | 1.0116 |
| Stall: No Instruction | 0.0157 |
| Stall: Not Selected | 0.1555 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 9.44e+06 |
| Divergent Branch Targets (total) | 6.29e+06 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 173.3769 |
| FMUL Throughput (per cycle) | 216.2051 |
| FFMA Throughput (per cycle) | 16.5121 |
| LSU Pipe Utilization (% of peak) | 9.5346 |
| Warp Execution Efficiency | 31.5883 |
| L1 Bank Conflicts (total) | 6.82e+06 |
| Shared Memory Bandwidth (bytes/s) | 5.86e+12 |

**Kernel name:** `flash_attention_kernel`