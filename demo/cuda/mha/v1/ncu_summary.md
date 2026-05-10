# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v1.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 1.8770 ms ± 0.0951 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 19.1110 |
| Memory Throughput (% of peak) | 0.7181 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.23e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.44e+09 |
| DRAM Write Bandwidth (bytes/s) | 1.80e+09 |
| L1 Global Load Bandwidth (bytes/s) | 5.38e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.14e+09 |
| L2 Total Bandwidth (bytes/s) | 1.36e+12 |
| Global Load Efficiency (%) | 21.9595 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 75.1886 |
| L2 Hit Rate (%) | 99.7264 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.7605 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0792 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 64.4114 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 40.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 2056.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 7.9206 |
| Eligible Warps / Cycle | 0.1770 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.3601 |
| Stall: Long Scoreboard | 30.3667 |
| Stall: Short Scoreboard | 3.2760 |
| Stall: Math Pipe Throttle | 0.0156 |
| Stall: Wait | 1.7460 |
| Stall: No Instruction | 0.1097 |
| Stall: Not Selected | 1.2347 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.61e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 4.7731 |
| FMUL Throughput (per cycle) | 3.5798 |
| FFMA Throughput (per cycle) | 157.8103 |
| LSU Pipe Utilization (% of peak) | 4.8093 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.27e+08 |
| Shared Memory Bandwidth (bytes/s) | 6.42e+10 |

**Kernel name:** `multi_head_attention_kernel`