# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | conv2d.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Execution Time** | 2.5415 ms ± 0.0541 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 76.7997 |
| Memory Throughput (% of peak) | 1.1695 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 8.52e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.05e+09 |
| DRAM Write Bandwidth (bytes/s) | 5.48e+09 |
| L1 Global Load Bandwidth (bytes/s) | 5.18e+12 |
| L1 Global Store Bandwidth (bytes/s) | 5.78e+09 |
| L2 Total Bandwidth (bytes/s) | 6.84e+11 |
| Global Load Efficiency (%) | 69.5597 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 86.9584 |
| L2 Hit Rate (%) | 99.2787 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 31.0920 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7513 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 91.1152 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 37.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 32.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 75.1368 |
| Eligible Warps / Cycle | 2.1068 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 7.8033 |
| Stall: Short Scoreboard | 0.0030 |
| Stall: Math Pipe Throttle | 0.7753 |
| Stall: Wait | 2.4624 |
| Stall: No Instruction | 0.1391 |
| Stall: Not Selected | 1.8092 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.18e+08 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 450.2978 |
| LSU Pipe Utilization (% of peak) | 19.4020 |
| Warp Execution Efficiency | 28.0383 |
| L1 Bank Conflicts (total) | 3.05e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_conv2d`