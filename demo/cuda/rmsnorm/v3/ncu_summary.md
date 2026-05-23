# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0307 ms ± 0.0020 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 15.8122 |
| Memory Throughput (% of peak) | 71.8347 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.22e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.11e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.11e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.30e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.10e+11 |
| L2 Total Bandwidth (bytes/s) | 6.67e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 47.0886 |
| L2 Hit Rate (%) | 54.1175 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.3192 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1743 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 100.0627 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 30.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 17.7520 |
| Eligible Warps / Cycle | 0.4002 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.2204 |
| Stall: Long Scoreboard | 46.2210 |
| Stall: Short Scoreboard | 1.0328 |
| Stall: Math Pipe Throttle | 0.3198 |
| Stall: Wait | 2.6145 |
| Stall: No Instruction | 0.2584 |
| Stall: Not Selected | 1.2001 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 40960.0000 |
| Divergent Branch Targets (total) | 1024.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 31.8535 |
| FMUL Throughput (per cycle) | 81.4432 |
| FFMA Throughput (per cycle) | 40.9205 |
| LSU Pipe Utilization (% of peak) | 6.1399 |
| Warp Execution Efficiency | 31.4103 |
| L1 Bank Conflicts (total) | 10218.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.93e+09 |

**Kernel name:** `rmsnorm_v3`