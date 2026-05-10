# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 0.2960 ms ± 0.0025 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 14.3186 |
| Memory Throughput (% of peak) | 93.7805 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.84e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.45e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.39e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.86e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.43e+11 |
| L2 Total Bandwidth (bytes/s) | 7.29e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.1408 |
| L2 Hit Rate (%) | 39.2313 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.8166 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0896 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 81.3688 |
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
| Issue Slot Utilization (% of peak) | 8.9720 |
| Eligible Warps / Cycle | 0.1476 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 11.4226 |
| Stall: Long Scoreboard | 65.6721 |
| Stall: Short Scoreboard | 5.8170 |
| Stall: Math Pipe Throttle | 0.1292 |
| Stall: Wait | 1.6673 |
| Stall: No Instruction | 0.1434 |
| Stall: Not Selected | 0.6635 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 622592.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 121.0204 |
| FMUL Throughput (per cycle) | 63.4861 |
| FFMA Throughput (per cycle) | 130.9401 |
| LSU Pipe Utilization (% of peak) | 3.6834 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 114405.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.95e+11 |

**Kernel name:** `softmax_v5`