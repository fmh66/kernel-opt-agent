# KBS Evidence - v6

## NCU Facts

- Runtime: 0.0499 ms (essentially tied with v3 at 0.0497 ms)
- Bottleneck: Memory-latency-bound — address optimization has minimal impact
- Key symptoms: Precomputing page base offsets reduced instruction count but didn't change memory latency profile
- Metrics driving next decision: Further latency reduction requires caching or data layout changes

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton cache modifier cg cs eviction policy` | cache hint optimization | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | source-reported | SM86 optimization: memory-bound kernels benefit from cache policy selection |

## Rejected / Limits

- Address computation optimization alone: insufficient to move the bottleneck

## Decision Link

`Address precomputation is neutral` -> `Try cache hint optimization` -> `Add cache_modifier to K/V loads for streaming access pattern`
