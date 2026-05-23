# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 8.7068 ms ± 0.1633 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 66.6081 |
| Memory Throughput (% of peak) | 34.0471 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.48e+11 |
| DRAM Read Bandwidth (bytes/s) | 2.45e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.48e+09 |
| L1 Global Load Bandwidth (bytes/s) | 4.21e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.26e+09 |
| L2 Total Bandwidth (bytes/s) | 4.24e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 42.3177 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 17.5345 |
| Tensor Core Utilization (% of peak) | 67.4670 |
| IPC (instructions per cycle) | 0.3420 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.6685 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 3072.0000 |
| Registers / Thread | 148.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32768.0000 |
| Waves / SM | 36.5714 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 34.1987 |
| Eligible Warps / Cycle | 0.4419 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1868 |
| Stall: Long Scoreboard | 0.7770 |
| Stall: Short Scoreboard | 0.2232 |
| Stall: Math Pipe Throttle | 2.2241 |
| Stall: Wait | 0.9247 |
| Stall: No Instruction | 0.0024 |
| Stall: Not Selected | 0.2921 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 3.15e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 237.7910 |
| FMUL Throughput (per cycle) | 344.6658 |
| FFMA Throughput (per cycle) | 6.9939 |
| LSU Pipe Utilization (% of peak) | 4.1187 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 447842.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.62e+12 |

**Kernel name:** `flash_attention_kernel`