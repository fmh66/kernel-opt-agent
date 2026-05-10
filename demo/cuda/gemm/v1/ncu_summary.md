# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'M': 1024, 'K': 1024, 'N': 1024} |
| **Execution Time** | 0.7025 ms ± 0.0118 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 94.5259 |
| Memory Throughput (% of peak) | 2.8402 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 2.07e+10 |
| DRAM Read Bandwidth (bytes/s) | 1.27e+10 |
| DRAM Write Bandwidth (bytes/s) | 8.04e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.99e+11 |
| L1 Global Store Bandwidth (bytes/s) | 6.31e+09 |
| L2 Total Bandwidth (bytes/s) | 7.87e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 3.3100 |
| L2 Hit Rate (%) | 98.3387 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.9744 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.2945 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 95.2064 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 4096.0000 |
| Registers / Thread | 38.0000 |
| Static Shared Memory (bytes) | 2048.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 8.1270 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 29.4573 |
| Eligible Warps / Cycle | 1.2993 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 5.9638 |
| Stall: Long Scoreboard | 5.4581 |
| Stall: Short Scoreboard | 0.3095 |
| Stall: Math Pipe Throttle | 0.1372 |
| Stall: Wait | 2.1071 |
| Stall: No Instruction | 0.0749 |
| Stall: Not Selected | 3.4102 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.13e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 0.0000 |
| FMUL Throughput (per cycle) | 0.0000 |
| FFMA Throughput (per cycle) | 845.3008 |
| LSU Pipe Utilization (% of peak) | 23.8138 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.43e+06 |
| Shared Memory Bandwidth (bytes/s) | 4.44e+12 |

**Kernel name:** `tiled_gemm`