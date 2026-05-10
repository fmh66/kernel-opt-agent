# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v4.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.8696 ms ± 0.0160 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 24.2353 |
| Memory Throughput (% of peak) | 1.6298 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.19e+10 |
| DRAM Read Bandwidth (bytes/s) | 7.91e+09 |
| DRAM Write Bandwidth (bytes/s) | 3.97e+09 |
| L1 Global Load Bandwidth (bytes/s) | 4.00e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.58e+09 |
| L2 Total Bandwidth (bytes/s) | 2.28e+12 |
| Global Load Efficiency (%) | 66.3265 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 44.1598 |
| L2 Hit Rate (%) | 99.4548 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.2198 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1413 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 65.3090 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 44.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 2056.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 14.1341 |
| Eligible Warps / Cycle | 0.1823 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.2409 |
| Stall: Long Scoreboard | 44.6695 |
| Stall: Short Scoreboard | 0.9550 |
| Stall: Math Pipe Throttle | 0.0851 |
| Stall: Wait | 2.1639 |
| Stall: No Instruction | 0.1292 |
| Stall: Not Selected | 0.2862 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 3.59e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 54.6285 |
| FMUL Throughput (per cycle) | 51.8971 |
| FFMA Throughput (per cycle) | 317.5280 |
| LSU Pipe Utilization (% of peak) | 6.3204 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.38e+07 |
| Shared Memory Bandwidth (bytes/s) | 1.45e+11 |

**Kernel name:** `multi_head_attention_kernel`