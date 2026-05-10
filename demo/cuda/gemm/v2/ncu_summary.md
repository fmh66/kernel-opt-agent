# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.6912 ms ± 0.0100 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 95.6884 |
| Memory Throughput (% of peak) | 2.8837 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.10e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.29e+10 |
| DRAM Write Bandwidth (bytes/s) | 8.14e+09 |
| L1 Global Load Bandwidth (bytes/s) | 8.17e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.39e+09 |
| L2 Total Bandwidth (bytes/s) | 8.12e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 1.6196 |
| L2 Hit Rate (%) | 98.3753 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.5850 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2709 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.0327 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 39.0000 |
| Static Shared Memory (bytes) | 4096.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 8.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 27.1050 |
| Eligible Warps / Cycle | 1.3447 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 6.3002 |
| Stall: Long Scoreboard | 3.3081 |
| Stall: Short Scoreboard | 0.2289 |
| Stall: Math Pipe Throttle | 0.1489 |
| Stall: Wait | 1.7783 |
| Stall: No Instruction | 0.0430 |
| Stall: Not Selected | 3.9579 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.08e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 855.6966 |
| LSU Pipe Utilization (% of peak) | 24.1026 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.20e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.50e+12 |

**Kernel name:** `tiled_gemm_v2`