# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0572 ms ± 0.0144 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 36.0872 |
| Memory Throughput (% of peak) | 3.4209 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.49e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.40e+10 |
| DRAM Write Bandwidth (bytes/s) | 9.08e+08 |
| L1 Global Load Bandwidth (bytes/s) | 1.67e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.28e+09 |
| L2 Total Bandwidth (bytes/s) | 1.73e+11 |
| Global Load Efficiency (%) | 96.3028 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 7.6752 |
| L2 Hit Rate (%) | 89.0534 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 10.2057 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2187 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 25.4835 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 128.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.2540 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 22.0103 |
| Eligible Warps / Cycle | 0.3122 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.1435 |
| Stall: Long Scoreboard | 4.4031 |
| Stall: Short Scoreboard | 3.1255 |
| Stall: Math Pipe Throttle | 0.1023 |
| Stall: Wait | 1.7752 |
| Stall: No Instruction | 0.0644 |
| Stall: Not Selected | 0.4164 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 17408.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 281.2916 |
| FMUL Throughput (per cycle) | 210.9735 |
| FFMA Throughput (per cycle) | 185.0602 |
| LSU Pipe Utilization (% of peak) | 10.1980 |
| Warp Execution Efficiency | 31.9931 |
| L1 Bank Conflicts (total) | 9.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.00e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`