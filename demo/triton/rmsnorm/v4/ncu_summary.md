# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.1645 ms ± 0.0320 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 6.9262 |
| Memory Throughput (% of peak) | 68.7791 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.00e+11 |
| DRAM Read Bandwidth (bytes/s) | 2.89e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.11e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.36e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.89e+11 |
| L2 Total Bandwidth (bytes/s) | 6.41e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 3.7748 |
| L2 Hit Rate (%) | 100.9845 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.6424 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0601 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 19.4209 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 168.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.1667 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 6.1063 |
| Eligible Warps / Cycle | 0.0671 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 10.8896 |
| Stall: Long Scoreboard | 11.3550 |
| Stall: Short Scoreboard | 2.8133 |
| Stall: Math Pipe Throttle | 0.0866 |
| Stall: Wait | 3.3286 |
| Stall: No Instruction | 4.1143 |
| Stall: Not Selected | 0.1005 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 8864.0000 |
| Divergent Branch Targets (total) | 1192.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 33.3027 |
| FMUL Throughput (per cycle) | 85.6355 |
| FFMA Throughput (per cycle) | 38.0602 |
| LSU Pipe Utilization (% of peak) | 2.4660 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 1602.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.31e+09 |

**Kernel name:** `vectorized_elementwise_kernel|vectorized_elementwise_kernel|rmsnorm_kernel`