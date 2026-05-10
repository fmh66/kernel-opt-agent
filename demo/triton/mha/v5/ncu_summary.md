# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.1933 ms ± 0.0009 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 42.7641 |
| Memory Throughput (% of peak) | 80.0071 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.81e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.17e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.80e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.10e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.16e+11 |
| L2 Total Bandwidth (bytes/s) | 1.14e+12 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 83.9954 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.3336 |
| Tensor Core Utilization (% of peak) | 51.5225 |
| IPC (instructions per cycle) | 0.2262 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 77.9126 |
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
| Issue Slot Utilization (% of peak) | 22.6307 |
| Eligible Warps / Cycle | 0.3574 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.6017 |
| Stall: Long Scoreboard | 40.0721 |
| Stall: Short Scoreboard | 0.3148 |
| Stall: Math Pipe Throttle | 1.7808 |
| Stall: Wait | 2.7562 |
| Stall: No Instruction | 0.3767 |
| Stall: Not Selected | 0.6465 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 147456.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 148.1777 |
| FMUL Throughput (per cycle) | 119.5525 |
| FFMA Throughput (per cycle) | 2.2451 |
| LSU Pipe Utilization (% of peak) | 11.9573 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 80790.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.67e+12 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`