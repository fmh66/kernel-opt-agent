# KBS Evidence - v2

## NCU Facts

- Runtime: 0.0530 ms ± 0.0092 ms (neutral vs v0 0.0524 ms)
- Bottleneck: Still Latency-Bound despite improved metrics
- Key symptoms:
  - Occupancy doubled to 50.26% (was 24.76%)
  - SM Throughput doubled to 41.83% (was 22.09%)
  - IPC doubled to 0.288 (was 0.149)
  - L1 Hit Rate tripled to 60.09% (was 21.16%)
  - Long Scoreboard Stall unchanged at 10.30% (was 11.15%)
  - Warp Execution Efficiency flat at 31.99%
- Metrics driving next decision: Despite 2x better occupancy and compute utilization, no speedup — the bottleneck is now clearly memory latency from scattered KV page loads with zero data reuse across programs

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton num_warps occupancy threads block size` | verify num_warps impact | technique-ampere-optimization |
| 2 | `warp execution efficiency reduction scatter` | investigate 32% warp efficiency | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | source-reported | Confirms that num_warps tuning alone (occupancy increase without data reuse) yields diminishing returns — need to address the memory access pattern directly |

## Rejected / Limits

- num_warps=8 improved all measured metrics but didn't translate to speedup — the fundamental issue is 0% KV data reuse between programs; each (batch,head) program loads identical KV pages independently

## Decision Link

`Doubled occupancy but 0% speedup + 32% warp execution efficiency unchanged` -> `Memory latency dominates, data reuse is zero` -> `Fuse Q-heads within a block so KV pages are loaded once and reused across Q-heads` -> `v3: page-outer loop with 2 Q-heads processed simultaneously within each block`
