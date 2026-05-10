# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.5232 ms ± 0.0183 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 76.7014 |
| Memory Throughput (% of peak) | 3.8936 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.84e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.74e+10 |
| DRAM Write Bandwidth (bytes/s) | 1.10e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.65e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.73e+10 |
| L2 Total Bandwidth (bytes/s) | 1.10e+12 |
| Global Load Efficiency (%) | 66.6667 |
| Global Store Efficiency (%) | 50.0000 |
| L1 Hit Rate (%) | 34.5373 |
| L2 Hit Rate (%) | 98.5678 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 12.4557 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2918 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 90.2146 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 2048.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 4.0635 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 29.1932 |
| Eligible Warps / Cycle | 1.0573 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 8.4402 |
| Stall: Long Scoreboard | 3.9587 |
| Stall: Short Scoreboard | 1.1354 |
| Stall: Math Pipe Throttle | 0.0449 |
| Stall: Wait | 1.1719 |
| Stall: No Instruction | 0.0300 |
| Stall: Not Selected | 2.6296 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.06e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 1175.5095 |
| LSU Pipe Utilization (% of peak) | 19.3492 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 1.69e+07 |
| Shared Memory Bandwidth (bytes/s) | 3.88e+12 |

**Kernel name:** `tiled_gemm_v5`