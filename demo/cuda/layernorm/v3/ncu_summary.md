# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 10240, 'D': 10240} |
| **Execution Time** | 1.8648 ms ± 0.0037 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 11.9781 |
| Memory Throughput (% of peak) | 94.0074 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.85e+11 |
| DRAM Read Bandwidth (bytes/s) | 4.58e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.27e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.12e+11 |
| L1 Global Store Bandwidth (bytes/s) | 2.28e+11 |
| L2 Total Bandwidth (bytes/s) | 7.69e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 32.0812 |
| L2 Hit Rate (%) | 40.4907 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 2.9515 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0912 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 97.6664 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 10240.0000 |
| Registers / Thread | 20.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 72.0000 |
| Waves / SM | 20.3175 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.1204 |
| Eligible Warps / Cycle | 0.1118 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.6658 |
| Stall: Long Scoreboard | 122.7421 |
| Stall: Short Scoreboard | 0.1524 |
| Stall: Math Pipe Throttle | 0.1227 |
| Stall: Wait | 2.1157 |
| Stall: No Instruction | 0.1504 |
| Stall: Not Selected | 0.2264 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6.77e+06 |
| Divergent Branch Targets (total) | 10240.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 67.7768 |
| FMUL Throughput (per cycle) | 29.7091 |
| FFMA Throughput (per cycle) | 59.4501 |
| LSU Pipe Utilization (% of peak) | 3.0285 |
| Warp Execution Efficiency | 31.9107 |
| L1 Bank Conflicts (total) | 631584.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.11e+09 |

**Kernel name:** `layernorm_v3`