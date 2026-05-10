# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.2640 ms ± 0.0061 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 41.9697 |
| Memory Throughput (% of peak) | 78.4528 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.71e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.16e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.57e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.10e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.15e+11 |
| L2 Total Bandwidth (bytes/s) | 1.15e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 83.9006 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.3574 |
| Tensor Core Utilization (% of peak) | 50.7276 |
| IPC (instructions per cycle) | 0.2686 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 100.2921 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 255.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 49152.0000 |
| Waves / SM | 4.0635 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.5128 |
| Eligible Warps / Cycle | 0.4841 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.5242 |
| Stall: Long Scoreboard | 24.3985 |
| Stall: Short Scoreboard | 0.2793 |
| Stall: Math Pipe Throttle | 1.8659 |
| Stall: Wait | 2.7771 |
| Stall: No Instruction | 0.3814 |
| Stall: Not Selected | 0.7544 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 147456.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 145.4251 |
| FMUL Throughput (per cycle) | 183.4339 |
| FFMA Throughput (per cycle) | 2.2034 |
| LSU Pipe Utilization (% of peak) | 11.6800 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 78081.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.69e+12 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`