# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | kernel.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'batch_size': 16, 'num_pages': 256} |
| **Execution Time** | 0.0530 ms ± 0.0092 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 41.8265 |
| Memory Throughput (% of peak) | 3.4187 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.49e+10 |
| DRAM Read Bandwidth (bytes/s) | 2.37e+10 |
| DRAM Write Bandwidth (bytes/s) | 1.14e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.91e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.24e+09 |
| L2 Total Bandwidth (bytes/s) | 1.72e+11 |
| Global Load Efficiency (%) | 81.2500 |
| Global Store Efficiency (%) | 90.2778 |
| L1 Hit Rate (%) | 60.0937 |
| L2 Hit Rate (%) | 85.8307 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 13.1096 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2875 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 50.2587 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 32.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 28.9176 |
| Eligible Warps / Cycle | 0.4226 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.0283 |
| Stall: Long Scoreboard | 10.3002 |
| Stall: Short Scoreboard | 3.5946 |
| Stall: Math Pipe Throttle | 0.1509 |
| Stall: Wait | 2.0099 |
| Stall: No Instruction | 0.0996 |
| Stall: Not Selected | 0.5232 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 34816.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 279.6939 |
| FMUL Throughput (per cycle) | 208.5533 |
| FFMA Throughput (per cycle) | 184.0092 |
| LSU Pipe Utilization (% of peak) | 11.7182 |
| Warp Execution Efficiency | 31.9897 |
| L1 Bank Conflicts (total) | 19.0000 |
| Shared Memory Bandwidth (bytes/s) | 1.98e+10 |

**Kernel name:** `_gqa_paged_decode_kernel`