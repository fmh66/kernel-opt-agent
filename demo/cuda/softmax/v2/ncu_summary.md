# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 0.2967 ms ± 0.0032 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 14.3797 |
| Memory Throughput (% of peak) | 93.7228 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.83e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.43e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.40e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.88e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.44e+11 |
| L2 Total Bandwidth (bytes/s) | 7.32e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.1300 |
| L2 Hit Rate (%) | 39.4821 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.8039 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0894 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 81.0669 |
| Theoretical Occupancy (%) | 83.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 64.0000 |
| Dynamic Shared Memory (bytes) | 16384.0000 |
| Waves / SM | 9.7524 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 8.9489 |
| Eligible Warps / Cycle | 0.1492 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 10.6482 |
| Stall: Long Scoreboard | 63.8364 |
| Stall: Short Scoreboard | 5.6345 |
| Stall: Math Pipe Throttle | 0.1280 |
| Stall: Wait | 1.6671 |
| Stall: No Instruction | 0.1438 |
| Stall: Not Selected | 0.6488 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 622592.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 121.5363 |
| FMUL Throughput (per cycle) | 63.7567 |
| FFMA Throughput (per cycle) | 131.4983 |
| LSU Pipe Utilization (% of peak) | 3.6738 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 116332.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.97e+11 |

**Kernel name:** `softmax_v2`