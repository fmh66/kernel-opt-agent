# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0481 ms ± 0.0046 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 6.8415 |
| Memory Throughput (% of peak) | 81.5351 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.93e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.40e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.52e+11 |
| L1 Global Load Bandwidth (bytes/s) | 5.09e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.40e+11 |
| L2 Total Bandwidth (bytes/s) | 7.19e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 16.7188 |
| L2 Hit Rate (%) | 53.1994 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.5881 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0748 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 50.7831 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 37.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 7.8493 |
| Eligible Warps / Cycle | 0.1119 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 4.8413 |
| Stall: Long Scoreboard | 46.3223 |
| Stall: Short Scoreboard | 2.5184 |
| Stall: Math Pipe Throttle | 0.3735 |
| Stall: Wait | 1.6223 |
| Stall: No Instruction | 0.3219 |
| Stall: Not Selected | 0.4279 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 0.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 39.0088 |
| FMUL Throughput (per cycle) | 94.7357 |
| FFMA Throughput (per cycle) | 44.5815 |
| LSU Pipe Utilization (% of peak) | 2.8880 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 5498.0000 |
| Shared Memory Bandwidth (bytes/s) | 4.31e+09 |

**Kernel name:** `rmsnorm_kernel`