# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1587 ms ± 0.0031 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 53.4235 |
| Memory Throughput (% of peak) | 13.3111 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.70e+10 |
| DRAM Read Bandwidth (bytes/s) | 7.00e+10 |
| DRAM Write Bandwidth (bytes/s) | 2.70e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.08e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.38e+10 |
| L2 Total Bandwidth (bytes/s) | 1.12e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 93.8152 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 57.3187 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6980 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 23.4831 |
| Theoretical Occupancy (%) | 33.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 96.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 24576.0000 |
| Waves / SM | 0.7619 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 69.8247 |
| Eligible Warps / Cycle | 1.5101 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.3355 |
| Stall: Long Scoreboard | 0.0815 |
| Stall: Short Scoreboard | 0.5725 |
| Stall: Math Pipe Throttle | 0.0207 |
| Stall: Wait | 0.1247 |
| Stall: No Instruction | 0.0188 |
| Stall: Not Selected | 1.1612 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 131072.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 4708.3113 |
| LSU Pipe Utilization (% of peak) | 13.3653 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 20455.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.86e+12 |

**Kernel name:** `_gemm_kernel`