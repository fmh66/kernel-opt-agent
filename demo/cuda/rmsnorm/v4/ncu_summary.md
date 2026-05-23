# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0309 ms ± 0.0020 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 19.2288 |
| Memory Throughput (% of peak) | 74.2839 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.40e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.18e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.23e+11 |
| L1 Global Load Bandwidth (bytes/s) | 9.50e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.17e+11 |
| L2 Total Bandwidth (bytes/s) | 6.79e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 47.2717 |
| L2 Hit Rate (%) | 53.2336 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.5303 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2396 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 106.5098 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 26.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 2.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 24.3874 |
| Eligible Warps / Cycle | 0.5088 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 3.8734 |
| Stall: Long Scoreboard | 31.2270 |
| Stall: Short Scoreboard | 1.1074 |
| Stall: Math Pipe Throttle | 0.5253 |
| Stall: Wait | 2.1398 |
| Stall: No Instruction | 0.3798 |
| Stall: Not Selected | 1.1499 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 69632.0000 |
| Divergent Branch Targets (total) | 1024.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 79.2594 |
| FMUL Throughput (per cycle) | 103.8935 |
| FFMA Throughput (per cycle) | 20.9816 |
| LSU Pipe Utilization (% of peak) | 7.6396 |
| Warp Execution Efficiency | 31.5521 |
| L1 Bank Conflicts (total) | 17024.0000 |
| Shared Memory Bandwidth (bytes/s) | 7.73e+09 |

**Kernel name:** `rmsnorm_v4`