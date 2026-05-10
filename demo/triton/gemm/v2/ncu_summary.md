# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1819 ms ± 0.0044 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 47.4735 |
| Memory Throughput (% of peak) | 12.5241 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.13e+10 |
| DRAM Read Bandwidth (bytes/s) | 6.68e+10 |
| DRAM Write Bandwidth (bytes/s) | 2.45e+10 |
| L1 Global Load Bandwidth (bytes/s) | 9.18e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.87e+10 |
| L2 Total Bandwidth (bytes/s) | 9.48e+11 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 93.0474 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 49.1053 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6247 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 42.6369 |
| Theoretical Occupancy (%) | 50.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 80.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32768.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 62.5161 |
| Eligible Warps / Cycle | 1.6075 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.6421 |
| Stall: Long Scoreboard | 0.2173 |
| Stall: Short Scoreboard | 2.6402 |
| Stall: Math Pipe Throttle | 0.0590 |
| Stall: Wait | 0.1738 |
| Stall: No Instruction | 0.0115 |
| Stall: Not Selected | 1.5744 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 131072.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 3996.9650 |
| LSU Pipe Utilization (% of peak) | 14.1266 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 22255.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.61e+12 |

**Kernel name:** `_gemm_kernel`