# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Execution Time** | 0.0191 ms ± 0.0019 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 8.6375 |
| Memory Throughput (% of peak) | 23.3143 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.67e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.50e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.73e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.49e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.49e+11 |
| L2 Total Bandwidth (bytes/s) | 3.36e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 55.1806 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.6347 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1533 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 56.0973 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 800.0000 |
| Registers / Thread | 16.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.7937 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 16.3598 |
| Eligible Warps / Cycle | 0.2370 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 21.9245 |
| Stall: Short Scoreboard | 1.2105 |
| Stall: Math Pipe Throttle | 0.0934 |
| Stall: Wait | 2.7455 |
| Stall: No Instruction | 1.1581 |
| Stall: Not Selected | 0.4168 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6400.0000 |
| Divergent Branch Targets (total) | 3200.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 28.9187 |
| FMUL Throughput (per cycle) | 118.6591 |
| FFMA Throughput (per cycle) | 82.0903 |
| LSU Pipe Utilization (% of peak) | 2.0903 |
| Warp Execution Efficiency | 25.0086 |
| L1 Bank Conflicts (total) | 685.0000 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `gelu_kernel`