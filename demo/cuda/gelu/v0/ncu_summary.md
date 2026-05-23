# NCU Profile Summary

| Field | Value |
|-------|-------|
| **Kernel** | v0.cu |
| **GPU** | NVIDIA RTX A6000 |
| **Arch** | sm_86 |
| **Dims** | {'N': 102400} |
| **Execution Time** | 0.0188 ms ± 0.0015 ms |

## Speed of Light

| Metric | Value |
|--------|------:|
| SM Throughput (% of peak) | 9.0265 |
| Memory Throughput (% of peak) | 23.4108 |

## Memory Workload Analysis

| Metric | Value |
|--------|------:|
| DRAM Total Bandwidth (bytes/s) | 1.69e+11 |
| DRAM Read Bandwidth (bytes/s) | 1.55e+11 |
| DRAM Write Bandwidth (bytes/s) | 1.35e+10 |
| L1 Global Load Bandwidth (bytes/s) | 1.54e+11 |
| L1 Global Store Bandwidth (bytes/s) | 1.54e+11 |
| L2 Total Bandwidth (bytes/s) | 3.47e+11 |
| Global Load Efficiency (%) | 100.0000 |
| Global Store Efficiency (%) | 100.0000 |
| L1 Hit Rate (%) | 0.0000 |
| L2 Hit Rate (%) | 59.5659 |

## Compute Workload Analysis

| Metric | Value |
|--------|------:|
| FMA Pipe Utilization (% of peak) | 7.0262 |
| Tensor Core Utilization (% of peak) | 0.0000 |
| IPC (instructions per cycle) | 0.1640 |

## Occupancy

| Metric | Value |
|--------|------:|
| Achieved Occupancy (%) | 60.6666 |
| Theoretical Occupancy (%) | 100.0000 |

## Launch Statistics

| Metric | Value |
|--------|------:|
| Block Size | 256.0000 |
| Grid Size | 400.0000 |
| Registers / Thread | 16.0000 |
| Static Shared Memory (bytes) | 0.0000 |
| Dynamic Shared Memory (bytes) | 0.0000 |
| Waves / SM | 0.7937 |

## Scheduler Statistics

| Metric | Value |
|--------|------:|
| Issue Slot Utilization (% of peak) | 17.8111 |
| Eligible Warps / Cycle | 0.2679 |

## Warp State / Stall Reasons

| Metric | Value |
|--------|------:|
| Stall: Barrier | 0.0000 |
| Stall: Long Scoreboard | 20.8174 |
| Stall: Short Scoreboard | 1.3547 |
| Stall: Math Pipe Throttle | 0.1342 |
| Stall: Wait | 2.7470 |
| Stall: No Instruction | 1.5041 |
| Stall: Not Selected | 0.5099 |

## Branch Divergence

| Metric | Value |
|--------|------:|
| Branch Targets (total) | 6400.0000 |
| Divergent Branch Targets (total) | 3200.0000 |

## Additional Pipe Utilization

| Metric | Value |
|--------|------:|
| FADD Throughput (per cycle) | 29.6954 |
| FMUL Throughput (per cycle) | 121.8460 |
| FFMA Throughput (per cycle) | 84.2951 |
| LSU Pipe Utilization (% of peak) | 2.2361 |
| Warp Execution Efficiency | 25.0086 |
| L1 Bank Conflicts (total) | 716.0000 |
| Shared Memory Bandwidth (bytes/s) | 0.0000 |

**Kernel name:** `naive_gelu`