# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | mha.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 141.2299 ms ± 0.9015 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 37.7971 |
| Memory Throughput (% of peak) | 77.6560 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.65e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.10e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.58e+11 |
| L1 Global Load Bandwidth (bytes/s) | 2.21e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.09e+11 |
| L2 Total Bandwidth (bytes/s) | 2.19e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.6399 |
| L2 Hit Rate (%) | 94.4359 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.9512 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2732 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 102.0523 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1.05e+06 |
| Registers / Thread | 80.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 512.0000 |
| Waves / SM | 2080.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.9394 |
| Eligible Warps / Cycle | 0.4913 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.1638 |
| Stall: Long Scoreboard | 30.1815 |
| Stall: Short Scoreboard | 1.4641 |
| Stall: Math Pipe Throttle | 0.6036 |
| Stall: Wait | 2.7779 |
| Stall: No Instruction | 0.3826 |
| Stall: Not Selected | 0.7724 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.29e+07 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 479.3191 |
| FMUL Throughput (per cycle) | 231.1003 |
| FFMA Throughput (per cycle) | 232.8121 |
| LSU Pipe Utilization (% of peak) | 9.3385 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 7.31e+08 |
| Shared Memory Bandwidth (bytes/s) | 7.04e+10 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`