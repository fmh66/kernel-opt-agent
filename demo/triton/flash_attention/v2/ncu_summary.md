# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 5.0377 ms ± 0.0246 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 56.9075 |
| Memory Throughput (% of peak) | 90.0174 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.56e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.50e+11 |
| DRAM Write Bandwidth (bytes/s) | 5.87e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.09e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.50e+09 |
| L2 Total Bandwidth (bytes/s) | 7.15e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 9.8778 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 15.2238 |
| Tensor Core Utilization (% of peak) | 57.5975 |
| IPC (instructions per cycle) | 0.2782 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3321 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 3072.0000 |
| Registers / Thread | 255.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 81920.0000 |
| Waves / SM | 36.5714 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.8176 |
| Eligible Warps / Cycle | 0.2782 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0547 |
| Stall: Long Scoreboard | 0.2236 |
| Stall: Short Scoreboard | 0.3673 |
| Stall: Math Pipe Throttle | 0.7639 |
| Stall: Wait | 1.0922 |
| Stall: No Instruction | 0.0017 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 786432.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 197.1846 |
| FMUL Throughput (per cycle) | 243.6798 |
| FFMA Throughput (per cycle) | 2.9876 |
| LSU Pipe Utilization (% of peak) | 3.1992 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 452175.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.28e+12 |

**Kernel name:** `flash_attention_kernel`