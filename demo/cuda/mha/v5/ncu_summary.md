# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v5.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 0.8680 ms ± 0.0166 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 24.3379 |
| Memory Throughput (% of peak) | 1.6802 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.22e+10 |
| DRAM Read Bandwidth (bytes/s) | 7.99e+09 |
| DRAM Write Bandwidth (bytes/s) | 4.26e+09 |
| L1 Global Load Bandwidth (bytes/s) | 4.03e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.59e+09 |
| L2 Total Bandwidth (bytes/s) | 2.28e+12 |
| Global Load Efficiency (%) | 66.3265 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 43.9922 |
| L2 Hit Rate (%) | 100.5208 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 9.2150 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1412 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 65.0503 |
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
| Issue Slot Utilization (% of peak) | 14.1269 |
| Eligible Warps / Cycle | 0.1818 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.2490 |
| Stall: Long Scoreboard | 44.5087 |
| Stall: Short Scoreboard | 0.9776 |
| Stall: Math Pipe Throttle | 0.0842 |
| Stall: Wait | 2.1629 |
| Stall: No Instruction | 0.1289 |
| Stall: Not Selected | 0.2845 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 3.59e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 54.8597 |
| FMUL Throughput (per cycle) | 52.1167 |
| FFMA Throughput (per cycle) | 318.8722 |
| LSU Pipe Utilization (% of peak) | 6.3171 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 4.37e+07 |
| Shared Memory Bandwidth (bytes/s) | 1.46e+11 |

**Kernel name:** `multi_head_attention_kernel`