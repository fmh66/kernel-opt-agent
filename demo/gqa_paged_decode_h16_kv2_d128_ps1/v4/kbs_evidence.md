# KBS Evidence - v4

## NCU Facts

- Runtime: 0.0510 ms ± 0.0041 ms (REGRESSION vs v3 0.0497 ms)
- Bottleneck: Grid starvation returns — 64 blocks insufficient for 84 SMs
- Key symptoms:
  - 4x KV reuse achieved but 64 blocks cause occupancy drop
  - Optimal grid seems to be >=128 blocks
- Metrics driving next decision: Grid size is the critical lever; need optimizations that preserve high block count while reducing per-block latency

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton flash attention decode v1 v2 split kv` | alternative decode optimization patterns | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Grid starvation pattern confirmed: diminishing returns beyond 2-head fusion |

## Rejected / Limits

- 4-Q-head fusion: rejected due to grid starvation at 64 blocks on 84-SM GPU

## Decision Link

`4x KV reuse loses to grid starvation` -> `Keep grid >=128 blocks` -> `Optimize per-block math: use fast exp approximation to reduce compute latency`
