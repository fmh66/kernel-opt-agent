# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 0.2961 ms ± 0.0020 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 14.3169 |
| Memory Throughput (% of peak) | 93.8940 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.84e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.46e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.39e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.86e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.43e+11 |
| L2 Total Bandwidth (bytes/s) | 7.29e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.1213 |
| L2 Hit Rate (%) | 39.0586 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.5297 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0799 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 81.6488 |
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
| Issue Slot Utilization (% of peak) | 7.9982 |
| Eligible Warps / Cycle | 0.1232 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 12.2800 |
| Stall: Long Scoreboard | 72.1182 |
| Stall: Short Scoreboard | 6.5745 |
| Stall: Math Pipe Throttle | 0.0850 |
| Stall: Wait | 1.9534 |
| Stall: No Instruction | 0.1487 |
| Stall: Not Selected | 0.5492 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 622592.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 89.2666 |
| FMUL Throughput (per cycle) | 63.4784 |
| FFMA Throughput (per cycle) | 3.9674 |
| LSU Pipe Utilization (% of peak) | 3.6896 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 114688.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.95e+11 |

**Kernel name:** `softmax_v4`