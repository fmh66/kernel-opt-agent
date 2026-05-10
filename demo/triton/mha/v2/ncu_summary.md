# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.2838 ms ± 0.0076 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 35.2645 |
| Memory Throughput (% of peak) | 78.0805 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.67e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.12e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.56e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.31e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.11e+11 |
| L2 Total Bandwidth (bytes/s) | 9.61e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 78.7204 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.5085 |
| Tensor Core Utilization (% of peak) | 45.2534 |
| IPC (instructions per cycle) | 0.2730 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 100.1145 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 140.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 81920.0000 |
| Waves / SM | 4.0635 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.9706 |
| Eligible Warps / Cycle | 0.4909 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1562 |
| Stall: Long Scoreboard | 24.1921 |
| Stall: Short Scoreboard | 0.9428 |
| Stall: Math Pipe Throttle | 0.6007 |
| Stall: Wait | 2.7753 |
| Stall: No Instruction | 0.3834 |
| Stall: Not Selected | 0.7677 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 147456.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 125.8943 |
| FMUL Throughput (per cycle) | 185.6015 |
| FFMA Throughput (per cycle) | 3.7028 |
| LSU Pipe Utilization (% of peak) | 10.2430 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 8167.0000 |
| Shared Memory Bandwidth (bytes/s) | 5.16e+12 |

**Kernel name:** `elementwise_kernel|elementwise_kernel|elementwise_kernel|fused_mha_kernel|elementwise_kernel`