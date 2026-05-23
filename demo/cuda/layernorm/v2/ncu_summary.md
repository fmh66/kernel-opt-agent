# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Execution Time** | 1.8653 ms ± 0.0035 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 11.9070 |
| Memory Throughput (% of peak) | 93.9253 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.85e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.57e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.27e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.11e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.28e+11 |
| L2 Total Bandwidth (bytes/s) | 7.69e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 32.3743 |
| L2 Hit Rate (%) | 40.8356 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.9445 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0908 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.8553 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 22.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 72.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.0862 |
| Eligible Warps / Cycle | 0.1140 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.7806 |
| Stall: Long Scoreboard | 122.8387 |
| Stall: Short Scoreboard | 0.1557 |
| Stall: Math Pipe Throttle | 0.1591 |
| Stall: Wait | 2.1223 |
| Stall: No Instruction | 0.1568 |
| Stall: Not Selected | 0.2531 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.77e+06 |
| Divergent Branch Targets (total) | 10240.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 67.7258 |
| FMUL Throughput (per cycle) | 29.6868 |
| FFMA Throughput (per cycle) | 59.4054 |
| LSU Pipe Utilization (% of peak) | 3.0063 |
| Warp Execution Efficiency | 31.9136 |
| L1 Bank Conflicts (total) | 618055.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.11e+09 |

**Kernel name:** `layernorm_v2`