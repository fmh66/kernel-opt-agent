# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 3.8912 ms ± 0.0333 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 26.7462 |
| Memory Throughput (% of peak) | 90.4642 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.59e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.58e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.55e+11 |
| L1 Global Load Bandwidth (bytes/s) | 2.62e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.16e+11 |
| L2 Total Bandwidth (bytes/s) | 2.51e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 3.5523 |
| L2 Hit Rate (%) | 74.3433 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.3808 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2693 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 101.7355 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 16384.0000 |
| Registers / Thread | 80.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 1024.0000 |
| Waves / SM | 32.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.5759 |
| Eligible Warps / Cycle | 0.4864 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.0707 |
| Stall: Long Scoreboard | 24.1716 |
| Stall: Short Scoreboard | 1.6919 |
| Stall: Math Pipe Throttle | 0.5906 |
| Stall: Wait | 2.7753 |
| Stall: No Instruction | 0.3808 |
| Stall: Not Selected | 0.7539 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 983040.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 294.5957 |
| FMUL Throughput (per cycle) | 149.7069 |
| FFMA Throughput (per cycle) | 311.1151 |
| LSU Pipe Utilization (% of peak) | 6.5193 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.05e+07 |
| Shared Memory Bandwidth (bytes/s) | 1.45e+11 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`