# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0459 ms ± 0.0030 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 9.8482 |
| Memory Throughput (% of peak) | 76.0981 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.53e+11 |
| DRAM Read Bandwidth (bytes/s) | 2.98e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.56e+11 |
| L1 Global Load Bandwidth (bytes/s) | 4.45e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.97e+11 |
| L2 Total Bandwidth (bytes/s) | 6.52e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 13.4375 |
| L2 Hit Rate (%) | 54.8656 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.1070 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1205 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 32.3889 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 12.2639 |
| Eligible Warps / Cycle | 0.1414 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.9035 |
| Stall: Long Scoreboard | 13.1298 |
| Stall: Short Scoreboard | 2.9799 |
| Stall: Math Pipe Throttle | 0.0777 |
| Stall: Wait | 1.8284 |
| Stall: No Instruction | 0.1016 |
| Stall: Not Selected | 0.1543 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 7168.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 102.0016 |
| FMUL Throughput (per cycle) | 48.5722 |
| FFMA Throughput (per cycle) | 116.5732 |
| LSU Pipe Utilization (% of peak) | 4.5675 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 5622.0000 |
| Shared Memory Bandwidth (bytes/s) | 7.53e+09 |

**Kernel name:** `layernorm_kernel`