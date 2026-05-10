# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.7200 ms ± 0.0063 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 96.5941 |
| Memory Throughput (% of peak) | 2.7559 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.01e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.22e+10 |
| DRAM Write Bandwidth (bytes/s) | 7.85e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.79e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.09e+09 |
| L2 Total Bandwidth (bytes/s) | 7.73e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 1.6232 |
| L2 Hit Rate (%) | 98.5008 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.4279 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2616 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.0675 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 8192.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 8.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 26.1668 |
| Eligible Warps / Cycle | 1.2612 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 8.5069 |
| Stall: Long Scoreboard | 2.5808 |
| Stall: Short Scoreboard | 0.1701 |
| Stall: Math Pipe Throttle | 0.0592 |
| Stall: Wait | 1.5960 |
| Stall: No Instruction | 0.0212 |
| Stall: Not Selected | 3.8235 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.61e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 813.0781 |
| LSU Pipe Utilization (% of peak) | 24.3334 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 6.30e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.29e+12 |

**Kernel name:** `tiled_gemm_v4`