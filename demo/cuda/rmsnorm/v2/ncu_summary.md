# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0295 ms ± 0.0016 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 10.6745 |
| Memory Throughput (% of peak) | 79.9215 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.80e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.41e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.40e+11 |
| L1 Global Load Bandwidth (bytes/s) | 1.02e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.40e+11 |
| L2 Total Bandwidth (bytes/s) | 7.24e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 47.4487 |
| L2 Hit Rate (%) | 53.1977 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.9570 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1635 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 100.9929 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 1024.0000 |
| Registers / Thread | 30.0000 |
| Static Shared Memory (bytes) | 128.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 2.0317 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 16.8968 |
| Eligible Warps / Cycle | 0.3441 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 9.4404 |
| Stall: Long Scoreboard | 41.3041 |
| Stall: Short Scoreboard | 1.8108 |
| Stall: Math Pipe Throttle | 0.4475 |
| Stall: Wait | 2.1304 |
| Stall: No Instruction | 0.4907 |
| Stall: Not Selected | 1.0561 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 53248.0000 |
| Divergent Branch Targets (total) | 1024.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 73.8655 |
| FMUL Throughput (per cycle) | 100.2864 |
| FFMA Throughput (per cycle) | 33.6465 |
| LSU Pipe Utilization (% of peak) | 4.3809 |
| Warp Execution Efficiency | 31.2315 |
| L1 Bank Conflicts (total) | 11788.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.29e+09 |

**Kernel name:** `rmsnorm_v2`