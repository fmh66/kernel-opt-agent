# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Execution Time** | 1.8617 ms ± 0.0149 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 12.7340 |
| Memory Throughput (% of peak) | 94.0885 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.86e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.58e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.28e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.15e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.29e+11 |
| L2 Total Bandwidth (bytes/s) | 8.36e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 26.9881 |
| L2 Hit Rate (%) | 45.0932 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.0633 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0942 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 90.6323 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 512.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 22.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 136.0000 |
| Waves / SM | 40.6349 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.4260 |
| Eligible Warps / Cycle | 0.1217 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 9.2390 |
| Stall: Long Scoreboard | 101.6212 |
| Stall: Short Scoreboard | 0.2866 |
| Stall: Math Pipe Throttle | 0.1870 |
| Stall: Wait | 2.1105 |
| Stall: No Instruction | 0.1594 |
| Stall: Not Selected | 0.2992 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.93e+06 |
| Divergent Branch Targets (total) | 10240.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 75.4671 |
| FMUL Throughput (per cycle) | 29.8130 |
| FFMA Throughput (per cycle) | 59.6581 |
| LSU Pipe Utilization (% of peak) | 3.1866 |
| Warp Execution Efficiency | 31.9171 |
| L1 Bank Conflicts (total) | 657149.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.19e+09 |

**Kernel name:** `layernorm_v5`