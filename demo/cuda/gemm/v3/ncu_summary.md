# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.6842 ms ± 0.0112 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 96.3552 |
| Memory Throughput (% of peak) | 2.9187 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.13e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.29e+10 |
| DRAM Write Bandwidth (bytes/s) | 8.34e+09 |
| L1 Global Load Bandwidth (bytes/s) | 8.24e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.44e+09 |
| L2 Total Bandwidth (bytes/s) | 8.18e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 1.5456 |
| L2 Hit Rate (%) | 98.4722 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.9940 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2582 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 94.9276 |
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
| Issue Slot Utilization (% of peak) | 25.8242 |
| Eligible Warps / Cycle | 1.4318 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.7557 |
| Stall: Long Scoreboard | 1.9847 |
| Stall: Short Scoreboard | 0.0404 |
| Stall: Math Pipe Throttle | 0.1455 |
| Stall: Wait | 1.4156 |
| Stall: No Instruction | 0.0230 |
| Stall: Not Selected | 4.5519 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 557056.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 861.6595 |
| LSU Pipe Utilization (% of peak) | 24.2775 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.19e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.53e+12 |

**Kernel name:** `tiled_gemm_v3`