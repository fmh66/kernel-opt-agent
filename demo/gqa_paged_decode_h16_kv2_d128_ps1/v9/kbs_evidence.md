# KBS Evidence - v9

## NCU Facts

- Runtime: 0.0509 ms (near v3's 0.0497 ms but slightly slower)
- Bottleneck: Memory-latency-bound — optimization space nearly exhausted
- Key symptoms: v6 structure with num_warps=4 performs similarly to v3; no clear winner between warp configurations
- Metrics driving next decision: v3 remains the best performing version at 0.0497 ms

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton flash attention decode v1 v2 split kv` | final alternative patterns | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Optimization pipeline complete for SM86 decode kernel |

## Rejected / Limits

- v3 at 0.0497 ms represents the optimal balance of grid parallelism (128 blocks) and KV data reuse (2x)

## Decision Link

`v3 = 0.0497 ms is best` -> `v10 preserves v3 structure with micro-optimizations (precomputed offsets) as final version`
