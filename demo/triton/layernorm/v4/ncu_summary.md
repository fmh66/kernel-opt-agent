# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0443 ms ± 0.0032 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 13.9180 |
| Memory Throughput (% of peak) | 84.5514 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.12e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.68e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.44e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.10e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.67e+11 |
| L2 Total Bandwidth (bytes/s) | 8.07e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 45.8984 |
| L2 Hit Rate (%) | 54.3743 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.3402 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0984 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 94.2733 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 10.1191 |
| Eligible Warps / Cycle | 0.1554 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 7.7902 |
| Stall: Long Scoreboard | 76.7121 |
| Stall: Short Scoreboard | 5.0591 |
| Stall: Math Pipe Throttle | 0.3647 |
| Stall: Wait | 2.0790 |
| Stall: No Instruction | 0.4419 |
| Stall: Not Selected | 0.5329 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 126.7162 |
| FMUL Throughput (per cycle) | 54.3070 |
| FFMA Throughput (per cycle) | 144.8186 |
| LSU Pipe Utilization (% of peak) | 5.1722 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 7390.0000 |
| Shared Memory Bandwidth (bytes/s) | 9.32e+09 |

**Kernel name:** `layernorm_kernel`