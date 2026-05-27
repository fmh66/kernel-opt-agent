# KBS Evidence - v1

## NCU Facts

- Runtime: 0.0734 ms ± 0.0036 ms (REGRESSION: +40% vs v0 0.0524 ms)
- Bottleneck: Severe Latency-Bound, worse than v0 (SM SOL 5.93%, Memory SOL 0.99%)
- Key symptoms:
  - Achieved Occupancy: 8.34% (was 24.76% in v0) — grid starvation
  - Waves/SM: 0.0317 (was 0.254) — 8x fewer waves, GPU mostly idle
  - Grid Size: 32 blocks (was 256) — cannot fill 84 SMs
  - L1 Hit Rate: 86.8% (was 21.2%) — better cache reuse doesn't compensate for lost parallelism
  - Issue Slot Utilization: 9.15% (was 15.04%)
- Metrics driving next decision: Lower grid → fewer concurrent blocks → lower occupancy → worse latency hiding → regression

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton num_warps occupancy threads block size` | need more warps per block to hide latency | technique-ampere-optimization |
| 2 | `warp execution efficiency reduction scatter` | investigate reduction bottleneck | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | source-reported | SM86 optimization guide covers occupancy tuning: "Low SM occupancy on small tiles" → "Reduce per-thread register count via register budgeting" — but v1 issue is different: grid starvation, not register pressure |

## Rejected / Limits

- Reducing grid was the wrong direction for this latency-bound kernel — L2 cache (86% hit rate in v0) was already masking memory latency; grid starvation (32 blocks / 84 SMs) caused the regression

## Decision Link

`Grid starvation: 32 blocks cannot fill 84 SMs` -> `Occupancy dropped from 24.76% to 8.34%` -> `v2 must restore grid density while adding warps per block for latency hiding` -> `num_warps=8 on v0-style grid`
