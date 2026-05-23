# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | flash_attention.py |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'B': 4, 'H': 12, 'N': 4096, 'd': 64} |
| **Execution Time** | 4.1847 ms ± 0.0383 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 63.4922 |
| Memory Throughput (% of peak) | 89.6618 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 6.54e+11 |
| DRAM Read Bandwidth (bytes/s) | 6.47e+11 |
| DRAM Write Bandwidth (bytes/s) | 6.38e+09 |
| L1 Global Load Bandwidth (bytes/s) | 7.72e+11 |
| L1 Global Store Bandwidth (bytes/s) | 5.98e+09 |
| L2 Total Bandwidth (bytes/s) | 7.78e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 17.5207 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 17.0821 |
| Tensor Core Utilization (% of peak) | 64.4187 |
| IPC (instructions per cycle) | 0.3226 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 16.5432 |
| Theoretical Occupancy (%) | 16.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 128.0000 |
| Grid Size | 3072.0000 |
| Registers / Thread | 204.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 49152.0000 |
| Waves / SM | 18.2857 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 32.2628 |
| Eligible Warps / Cycle | 0.3811 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.2041 |
| Stall: Long Scoreboard | 1.4064 |
| Stall: Short Scoreboard | 0.5439 |
| Stall: Math Pipe Throttle | 1.4562 |
| Stall: Wait | 1.2555 |
| Stall: No Instruction | 0.0054 |
| Stall: Not Selected | 0.1850 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 1.57e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 226.6671 |
| FMUL Throughput (per cycle) | 328.5422 |
| FFMA Throughput (per cycle) | 6.6667 |
| LSU Pipe Utilization (% of peak) | 4.4670 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 500525.0000 |
| Shared Memory Bandwidth (bytes/s) | 3.98e+12 |

**Kernel name:** `flash_attention_kernel`