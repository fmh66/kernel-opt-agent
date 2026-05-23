# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v2.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Execution Time** | 0.0192 ms ± 0.0019 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 5.8293 |
| Memory Throughput (% of peak) | 21.8215 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.58e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.40e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.78e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.38e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.38e+11 |
| L2 Total Bandwidth (bytes/s) | 3.20e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 55.9808 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 5.5009 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1097 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 18.5456 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 100.0000 |
| Registers / Thread | 19.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.1984 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 11.3732 |
| Eligible Warps / Cycle | 0.1281 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 7.6408 |
| Stall: Short Scoreboard | 1.0724 |
| Stall: Math Pipe Throttle | 0.0146 |
| Stall: Wait | 2.6804 |
| Stall: No Instruction | 0.9130 |
| Stall: Not Selected | 0.1347 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 7200.0000 |
| Divergent Branch Targets (total) | 3200.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 26.3894 |
| FMUL Throughput (per cycle) | 108.2807 |
| FFMA Throughput (per cycle) | 74.9103 |
| LSU Pipe Utilization (% of peak) | 0.4912 |
| Warp Execution Efficiency | 22.8173 |
| L1 Bank Conflicts (total) | 98.0000 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `gelu_kernel`