# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 7.7245 ms ± 0.0864 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 87.5208 |
| Memory Throughput (% of peak) | 0.1731 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.26e+09 |
| DRAM Read Bandwidth (bytes/s) | 8.19e+08 |
| DRAM Write Bandwidth (bytes/s) | 4.43e+08 |
| L1 Global Load Bandwidth (bytes/s) | 2.31e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.65e+08 |
| L2 Total Bandwidth (bytes/s) | 2.52e+11 |
| Global Load Efficiency (%) | 17.6471 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 89.0834 |
| L2 Hit Rate (%) | 99.6119 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.2983 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2589 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 62.7332 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 48.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 2048.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 25.8943 |
| Eligible Warps / Cycle | 0.8652 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 14.4114 |
| Stall: Long Scoreboard | 4.9672 |
| Stall: Short Scoreboard | 0.2018 |
| Stall: Math Pipe Throttle | 0.0296 |
| Stall: Wait | 1.8223 |
| Stall: No Instruction | 0.1001 |
| Stall: Not Selected | 2.3424 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 5.77e+07 |
| Divergent Branch Targets (total) | 8192.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.8302 |
| FMUL Throughput (per cycle) | 0.2778 |
| FFMA Throughput (per cycle) | 39.2980 |
| LSU Pipe Utilization (% of peak) | 22.0385 |
| Warp Execution Efficiency | 1.7259 |
| L1 Bank Conflicts (total) | 13612.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.71e+10 |

**Kernel name:** `multi_head_attention_kernel`