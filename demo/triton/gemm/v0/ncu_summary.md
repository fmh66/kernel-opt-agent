# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | gemm.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1613 ms ± 0.0039 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 51.7071 |
| Memory Throughput (% of peak) | 13.9428 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.02e+11 |
| DRAM Read Bandwidth (bytes/s) | 7.45e+10 |
| DRAM Write Bandwidth (bytes/s) | 2.72e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.06e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.33e+10 |
| L2 Total Bandwidth (bytes/s) | 1.10e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 93.3052 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 59.5270 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7074 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 22.9819 |
| Theoretical Occupancy (%) | 25.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 256.0000 |
| Registers / Thread | 96.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 32768.0000 |
| Waves / SM | 1.0159 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 70.7626 |
| Eligible Warps / Cycle | 1.5104 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.2222 |
| Stall: Long Scoreboard | 0.1105 |
| Stall: Short Scoreboard | 0.6988 |
| Stall: Math Pipe Throttle | 0.0368 |
| Stall: Wait | 0.0858 |
| Stall: No Instruction | 0.0249 |
| Stall: Not Selected | 1.1350 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 65536.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 4633.5244 |
| LSU Pipe Utilization (% of peak) | 12.9622 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 24355.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.96e+12 |

**Kernel name:** `_gemm_kernel`