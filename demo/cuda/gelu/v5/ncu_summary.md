# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Execution Time** | 0.0193 ms ± 0.0018 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 8.2229 |
| Memory Throughput (% of peak) | 18.6369 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.34e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.20e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.41e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.19e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.19e+11 |
| L2 Total Bandwidth (bytes/s) | 2.72e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 55.5495 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.6889 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1446 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 29.2495 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 168.0000 |
| Registers / Thread | 24.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.3333 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 15.0767 |
| Eligible Warps / Cycle | 0.1999 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 11.3845 |
| Stall: Short Scoreboard | 1.0441 |
| Stall: Math Pipe Throttle | 0.1738 |
| Stall: Wait | 2.8994 |
| Stall: No Instruction | 0.5740 |
| Stall: Not Selected | 0.3322 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 10944.0000 |
| Divergent Branch Targets (total) | 3200.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 22.8746 |
| FMUL Throughput (per cycle) | 93.8588 |
| FFMA Throughput (per cycle) | 64.9331 |
| LSU Pipe Utilization (% of peak) | 1.2911 |
| Warp Execution Efficiency | 26.3264 |
| L1 Bank Conflicts (total) | 280.0000 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `gelu_kernel`