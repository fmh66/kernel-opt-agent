# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v3.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 1.0956 ms ± 0.0308 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 19.8352 |
| Memory Throughput (% of peak) | 1.3203 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 9.62e+09 |
| DRAM Read Bandwidth (bytes/s) | 6.36e+09 |
| DRAM Write Bandwidth (bytes/s) | 3.27e+09 |
| L1 Global Load Bandwidth (bytes/s) | 3.28e+12 |
| L1 Global Store Bandwidth (bytes/s) | 2.10e+09 |
| L2 Total Bandwidth (bytes/s) | 2.64e+12 |
| Global Load Efficiency (%) | 66.3265 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 19.9190 |
| L2 Hit Rate (%) | 99.8937 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 6.6920 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0991 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 64.2548 |
| Theoretical Occupancy (%) | 66.6667 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 64.0000 |
| Grid Size | 8192.0000 |
| Registers / Thread | 56.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 2056.0000 |
| Waves / SM | 6.0952 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 9.9176 |
| Eligible Warps / Cycle | 0.1693 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 2.0927 |
| Stall: Long Scoreboard | 35.0222 |
| Stall: Short Scoreboard | 5.0544 |
| Stall: Math Pipe Throttle | 0.0300 |
| Stall: Wait | 1.9741 |
| Stall: No Instruction | 0.1384 |
| Stall: Not Selected | 0.7059 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.33e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 44.7103 |
| FMUL Throughput (per cycle) | 42.4748 |
| FFMA Throughput (per cycle) | 259.8788 |
| LSU Pipe Utilization (% of peak) | 4.9837 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 3.68e+07 |
| Shared Memory Bandwidth (bytes/s) | 1.18e+11 |

**Kernel name:** `multi_head_attention_kernel`