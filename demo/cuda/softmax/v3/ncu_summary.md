# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 0.3018 ms ± 0.0056 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 6.8599 |
| Memory Throughput (% of peak) | 93.3745 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.81e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.45e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.36e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.80e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.40e+11 |
| L2 Total Bandwidth (bytes/s) | 7.20e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0410 |
| L2 Hit Rate (%) | 38.4671 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.3660 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0707 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 80.5332 |
| Theoretical Occupancy (%) | 83.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 64.0000 |
| Dynamic Shared Memory (bytes) | 16384.0000 |
| Waves / SM | 9.7524 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 7.0793 |
| Eligible Warps / Cycle | 0.0993 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 21.3481 |
| Stall: Long Scoreboard | 66.1736 |
| Stall: Short Scoreboard | 17.5771 |
| Stall: Math Pipe Throttle | 0.1172 |
| Stall: Wait | 1.5323 |
| Stall: No Instruction | 0.2590 |
| Stall: Not Selected | 0.4044 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 491520.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 118.8500 |
| FMUL Throughput (per cycle) | 62.3476 |
| FFMA Throughput (per cycle) | 128.5918 |
| LSU Pipe Utilization (% of peak) | 1.3838 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 110853.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.88e+11 |

**Kernel name:** `softmax_v3`