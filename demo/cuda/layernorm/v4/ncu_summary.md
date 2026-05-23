# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Execution Time** | 1.8608 ms ± 0.0032 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 11.9446 |
| Memory Throughput (% of peak) | 94.0117 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.85e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.57e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.28e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.14e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.28e+11 |
| L2 Total Bandwidth (bytes/s) | 8.47e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 26.2303 |
| L2 Hit Rate (%) | 45.6183 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 4.1184 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0697 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.1128 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 72.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 6.9705 |
| Eligible Warps / Cycle | 0.1092 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.1504 |
| Stall: Long Scoreboard | 133.4883 |
| Stall: Short Scoreboard | 0.7048 |
| Stall: Math Pipe Throttle | 0.1831 |
| Stall: Wait | 1.4762 |
| Stall: No Instruction | 0.0581 |
| Stall: Not Selected | 0.5211 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.02e+06 |
| Divergent Branch Targets (total) | 10240.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 75.3848 |
| FMUL Throughput (per cycle) | 37.2256 |
| FFMA Throughput (per cycle) | 52.1478 |
| LSU Pipe Utilization (% of peak) | 3.0018 |
| Warp Execution Efficiency | 31.8876 |
| L1 Bank Conflicts (total) | 627188.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.12e+09 |

**Kernel name:** `layernorm_v4`