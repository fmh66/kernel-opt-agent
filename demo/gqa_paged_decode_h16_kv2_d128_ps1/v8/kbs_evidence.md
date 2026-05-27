# KBS Evidence - v8

## NCU Facts

- Runtime: 0.0520 ms (neutral — num_warps=4 doesn't help with 2-head fusion)
- Bottleneck: Memory-latency-bound — occupancy tuning saturated
- Key symptoms: Fewer warps per block doesn't help when grid already provides sufficient occupancy
- Metrics driving next decision: Address computation + warps tuning combination may help

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton num_warps occupancy threads block size` | warp tuning impact | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Occupancy tuning complete — need different optimization dimension |

## Rejected / Limits

- num_warps=4: neutral result, no benefit over num_warps=8 for this kernel

## Decision Link

`num_warps=4 neutral` -> `Combine precomputed offsets with tuned warps` -> `v6 structure + num_warps=4`
