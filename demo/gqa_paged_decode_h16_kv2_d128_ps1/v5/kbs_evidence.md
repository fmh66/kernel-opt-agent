# KBS Evidence - v5

## NCU Facts

- Runtime: 0.0519 ms (regression vs v3 0.0497 ms)
- Bottleneck: Memory-latency-bound — compute optimizations ineffective
- Key symptoms: Fast exp2 provided no benefit; kernel bottleneck is memory latency, not math
- Metrics driving next decision: Need to reduce memory/address computation latency, not math

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton flash attention decode v1 v2 split kv` | decode math-to-memory ratio | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | Confirms memory-latency-bound: "Memory-Bound" kernels need "Coalesce/vectorize loads; choose cache policy by reuse" |

## Rejected / Limits

- Fast exp2: rejected for compute-bound optimization on memory-bound kernel

## Decision Link

`Fast exp2 no benefit on memory-bound kernel` -> `Optimize address computation and load pattern` -> `Precompute per-page KV offsets to reduce address computation instructions per iteration`
