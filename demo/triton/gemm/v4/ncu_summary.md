# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1699 ms ± 0.0025 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 46.7693 |
| Memory Throughput (% of peak) | 20.0276 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.46e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.11e+11 |
| DRAM Write Bandwidth (bytes/s) | 3.52e+10 |
| L1 Global Load Bandwidth (bytes/s) | 9.91e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.10e+10 |
| L2 Total Bandwidth (bytes/s) | 1.03e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 89.4098 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 51.9159 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.6067 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 8.3199 |
| Theoretical Occupancy (%) | 8.3333 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 128.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 65536.0000 |
| Waves / SM | 3.0476 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 60.6791 |
| Eligible Warps / Cycle | 0.6068 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0410 |
| Stall: Long Scoreboard | 0.0939 |
| Stall: Short Scoreboard | 0.2383 |
| Stall: Math Pipe Throttle | 0.0011 |
| Stall: Wait | 0.0674 |
| Stall: No Instruction | 0.0282 |
| Stall: Not Selected | 0.0000 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 32768.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 4257.6291 |
| LSU Pipe Utilization (% of peak) | 10.8872 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 0.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.67e+12 |

**Kernel name:** `_gemm_kernel`