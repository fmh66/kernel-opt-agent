# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0467 ms ± 0.0039 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 23.7377 |
| Memory Throughput (% of peak) | 78.5360 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.69e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.35e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.34e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.00e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.34e+11 |
| L2 Total Bandwidth (bytes/s) | 7.35e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 45.8984 |
| L2 Hit Rate (%) | 54.3131 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 8.1179 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1796 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 98.8134 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 32.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32.0000 |
| Waves / SM | 2.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 18.3392 |
| Eligible Warps / Cycle | 0.3139 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.3104 |
| Stall: Long Scoreboard | 34.9462 |
| Stall: Short Scoreboard | 5.5273 |
| Stall: Math Pipe Throttle | 0.4240 |
| Stall: Wait | 2.2084 |
| Stall: No Instruction | 0.3508 |
| Stall: Not Selected | 0.7122 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 209.0224 |
| FMUL Throughput (per cycle) | 55.0059 |
| FFMA Throughput (per cycle) | 132.0142 |
| LSU Pipe Utilization (% of peak) | 9.0589 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 14942.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.63e+10 |

**Kernel name:** `layernorm_kernel`