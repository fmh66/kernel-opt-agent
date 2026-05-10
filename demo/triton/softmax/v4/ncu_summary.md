# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0646 ms ± 0.0120 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 22.8512 |
| Memory Throughput (% of peak) | 83.1194 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.05e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.49e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.56e+11 |
| L1 Global Load Bandwidth (bytes/s) | 3.49e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.49e+11 |
| L2 Total Bandwidth (bytes/s) | 7.09e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 51.3059 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 8.1544 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2153 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 93.4355 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 20.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32.0000 |
| Waves / SM | 2.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 21.8722 |
| Eligible Warps / Cycle | 0.3989 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 4.6719 |
| Stall: Long Scoreboard | 26.1268 |
| Stall: Short Scoreboard | 4.8393 |
| Stall: Math Pipe Throttle | 0.4060 |
| Stall: Wait | 1.9919 |
| Stall: No Instruction | 0.3147 |
| Stall: Not Selected | 0.8289 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 170.6222 |
| FMUL Throughput (per cycle) | 90.9985 |
| FFMA Throughput (per cycle) | 0.0000 |
| LSU Pipe Utilization (% of peak) | 9.3742 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4452.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.70e+10 |

**Kernel name:** `softmax_kernel_v4`