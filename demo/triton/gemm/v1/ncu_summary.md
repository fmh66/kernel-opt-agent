# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.1645 ms ± 0.0056 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 51.7558 |
| Memory Throughput (% of peak) | 13.5999 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.91e+10 |
| DRAM Read Bandwidth (bytes/s) | 7.26e+10 |
| DRAM Write Bandwidth (bytes/s) | 2.65e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.04e+12 |
| L1 Global Store Bandwidth (bytes/s) | 3.24e+10 |
| L2 Total Bandwidth (bytes/s) | 1.07e+12 |
| Global Load Efficiency (%) | 0.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 93.2308 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 59.2475 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.7120 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 23.0236 |
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
| Issue Slot Utilization (% of peak) | 71.2207 |
| Eligible Warps / Cycle | 1.5040 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.1860 |
| Stall: Long Scoreboard | 0.1158 |
| Stall: Short Scoreboard | 0.7486 |
| Stall: Math Pipe Throttle | 0.0438 |
| Stall: Wait | 0.0869 |
| Stall: No Instruction | 0.0208 |
| Stall: Not Selected | 1.1114 |

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
| FFMA Throughput (per cycle) | 4544.4168 |
| LSU Pipe Utilization (% of peak) | 13.1275 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 21923.0000 |
| Shared Memory Bandwidth (bytes/s) | 2.91e+12 |

**Kernel name:** `_gemm_kernel`