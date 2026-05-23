# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 8, 'C_in': 64, 'H': 56, 'W': 56, 'C_out': 128, 'K': 3, 'stride': 1, 'pad': 1} |
| **Execution Time** | 2.0364 ms ± 0.0163 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 80.6300 |
| Memory Throughput (% of peak) | 1.4433 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.05e+10 |
| DRAM Read Bandwidth (bytes/s) | 3.75e+09 |
| DRAM Write Bandwidth (bytes/s) | 6.77e+09 |
| L1 Global Load Bandwidth (bytes/s) | 5.80e+12 |
| L1 Global Store Bandwidth (bytes/s) | 7.15e+09 |
| L2 Total Bandwidth (bytes/s) | 6.11e+11 |
| Global Load Efficiency (%) | 73.9940 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 89.8821 |
| L2 Hit Rate (%) | 100.1847 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 13.0211 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7147 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 59.3798 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 60.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 24.3810 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 71.4769 |
| Eligible Warps / Cycle | 1.8001 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 3.6653 |
| Stall: Short Scoreboard | 0.0053 |
| Stall: Math Pipe Throttle | 1.1520 |
| Stall: Wait | 2.0962 |
| Stall: No Instruction | 0.0327 |
| Stall: Not Selected | 1.5101 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.58e+07 |
| Divergent Branch Targets (total) | 5.51e+06 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 555.1699 |
| LSU Pipe Utilization (% of peak) | 18.0111 |
| Warp Execution Efficiency | 28.8726 |
| L1 Bank Conflicts (total) | 1.69e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `conv2d_v4`