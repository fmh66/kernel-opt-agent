# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.4067 ms ± 0.0075 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 19.0248 |
| Memory Throughput (% of peak) | 79.6528 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.80e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.09e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.72e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.99e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.08e+11 |
| L2 Total Bandwidth (bytes/s) | 1.02e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 79.4485 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.4320 |
| Tensor Core Utilization (% of peak) | 21.9551 |
| IPC (instructions per cycle) | 0.2708 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 103.4720 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 206.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 57344.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.7303 |
| Eligible Warps / Cycle | 0.4899 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.6059 |
| Stall: Long Scoreboard | 24.2986 |
| Stall: Short Scoreboard | 0.8799 |
| Stall: Math Pipe Throttle | 0.7093 |
| Stall: Wait | 2.7762 |
| Stall: No Instruction | 0.3816 |
| Stall: Not Selected | 0.7659 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 147456.0000 |
| Divergent Branch Targets (total) | 65536.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 73.9113 |
| FMUL Throughput (per cycle) | 86.8957 |
| FFMA Throughput (per cycle) | 3.9952 |
| LSU Pipe Utilization (% of peak) | 5.3654 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 524288.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.78e+12 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`