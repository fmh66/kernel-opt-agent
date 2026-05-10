# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 4096, 'D': 4096} |
| **Execution Time** | 3.4769 ms ± 0.2207 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 1.2054 |
| Memory Throughput (% of peak) | 13.3524 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.73e+10 |
| DRAM Read Bandwidth (bytes/s) | 5.85e+10 |
| DRAM Write Bandwidth (bytes/s) | 3.88e+10 |
| L1 Global Load Bandwidth (bytes/s) | 4.68e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.12e+11 |
| L2 Total Bandwidth (bytes/s) | 3.71e+11 |
| Global Load Efficiency (%) | 12.5000 |
| Global Store Efficiency (%) | 12.5000 |
| L1 Hit Rate (%) | 91.8728 |
| L2 Hit Rate (%) | 84.2183 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.0355 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0392 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.6578 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 16.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 3.9165 |
| Eligible Warps / Cycle | 0.0394 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 46.3266 |
| Stall: Short Scoreboard | 1.3240 |
| Stall: Math Pipe Throttle | 0.0004 |
| Stall: Wait | 1.8598 |
| Stall: No Instruction | 0.1009 |
| Stall: Not Selected | 0.0049 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 820608.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 7.6275 |
| FMUL Throughput (per cycle) | 2.5425 |
| FFMA Throughput (per cycle) | 22.8824 |
| LSU Pipe Utilization (% of peak) | 1.2476 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 6.89e+07 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_softmax`