# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 1024, 'D': 1024} |
| **Execution Time** | 0.0483 ms ± 0.0031 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 10.9410 |
| Memory Throughput (% of peak) | 76.1953 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.55e+11 |
| DRAM Read Bandwidth (bytes/s) | 3.16e+11 |
| DRAM Write Bandwidth (bytes/s) | 2.39e+11 |
| L1 Global Load Bandwidth (bytes/s) | 6.30e+11 |
| L1 Global Store Bandwidth (bytes/s) | 3.15e+11 |
| L2 Total Bandwidth (bytes/s) | 6.93e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 27.8646 |
| L2 Hit Rate (%) | 55.2795 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.2606 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1260 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 51.3800 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 512.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 16.0000 |
| Waves / SM | 0.5079 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 12.9805 |
| Eligible Warps / Cycle | 0.1986 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.4255 |
| Stall: Long Scoreboard | 25.9622 |
| Stall: Short Scoreboard | 2.9570 |
| Stall: Math Pipe Throttle | 0.3084 |
| Stall: Wait | 1.8586 |
| Stall: No Instruction | 0.1933 |
| Stall: Not Selected | 0.5265 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6144.0000 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 108.3502 |
| FMUL Throughput (per cycle) | 51.5953 |
| FFMA Throughput (per cycle) | 123.8288 |
| LSU Pipe Utilization (% of peak) | 4.8353 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3431.0000 |
| Shared Memory Bandwidth (bytes/s) | 8.00e+09 |

**Kernel name:** `layernorm_kernel`