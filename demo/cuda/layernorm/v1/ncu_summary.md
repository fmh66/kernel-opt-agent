# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Execution Time** | 2.4509 ms ± 0.0037 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 10.7341 |
| Memory Throughput (% of peak) | 94.7626 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.91e+11 |
| DRAM Read Bandwidth (bytes/s) | 5.19e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.72e+11 |
| L1 Global Load Bandwidth (bytes/s) | 8.62e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.72e+11 |
| L2 Total Bandwidth (bytes/s) | 7.76e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 25.4163 |
| L2 Hit Rate (%) | 33.1153 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.6540 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0883 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 98.1196 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 23.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 72.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 8.8362 |
| Eligible Warps / Cycle | 0.1060 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1735 |
| Stall: Long Scoreboard | 126.3261 |
| Stall: Short Scoreboard | 0.1781 |
| Stall: Math Pipe Throttle | 0.1083 |
| Stall: Wait | 2.4513 |
| Stall: No Instruction | 0.1735 |
| Stall: Not Selected | 0.2000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.02e+07 |
| Divergent Branch Targets (total) | 20480.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 73.2062 |
| FMUL Throughput (per cycle) | 22.4757 |
| FFMA Throughput (per cycle) | 44.9733 |
| LSU Pipe Utilization (% of peak) | 2.7096 |
| Warp Execution Efficiency | 31.9281 |
| L1 Bank Conflicts (total) | 642676.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.42e+08 |

**Kernel name:** `layernorm_v1`