# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 512, 'd_model': 1024, 'num_heads': 16} |
| **Execution Time** | 1.9298 ms ± 0.0067 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 18.8038 |
| Memory Throughput (% of peak) | 0.7049 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 5.14e+09 |
| DRAM Read Bandwidth (bytes/s) | 3.38e+09 |
| DRAM Write Bandwidth (bytes/s) | 1.76e+09 |
| L1 Global Load Bandwidth (bytes/s) | 5.30e+12 |
| L1 Global Store Bandwidth (bytes/s) | 1.12e+09 |
| L2 Total Bandwidth (bytes/s) | 1.76e+12 |
| Global Load Efficiency (%) | 21.9595 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 67.1794 |
| L2 Hit Rate (%) | 99.8533 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 3.6918 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.0777 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 64.3865 |
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
| Issue Slot Utilization (% of peak) | 7.7711 |
| Eligible Warps / Cycle | 0.1719 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 1.5018 |
| Stall: Long Scoreboard | 32.3078 |
| Stall: Short Scoreboard | 2.8259 |
| Stall: Math Pipe Throttle | 0.0174 |
| Stall: Wait | 1.8193 |
| Stall: No Instruction | 0.1015 |
| Stall: Not Selected | 1.2122 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 2.62e+06 |
| Divergent Branch Targets (total) | 0.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 4.6964 |
| FMUL Throughput (per cycle) | 3.5223 |
| FFMA Throughput (per cycle) | 155.2737 |
| LSU Pipe Utilization (% of peak) | 4.7352 |
| Warp Execution Efficiency | 32.0000 |
| L1 Bank Conflicts (total) | 2.17e+08 |
| Shared Memory Bandwidth (bytes/s) | 6.31e+10 |

**Kernel name:** `multi_head_attention_kernel`