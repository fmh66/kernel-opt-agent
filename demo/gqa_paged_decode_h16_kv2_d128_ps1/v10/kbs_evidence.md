# KBS Evidence - v10

## NCU Facts

- Runtime: 0.0516 ms (near v3's 0.0497 ms and v6's 0.0499 ms, within noise)
- Bottleneck: Memory-latency-bound — optimization space exhausted after 10 iterations
- Key symptoms: v3 structure + v6 precomputed offsets = performance within noise of both parents; no further architectural headroom
- Metrics driving next decision: Optimization pipeline complete for SM86 decode kernel with given dimensions

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton flash attention decode paged kv cache sm86` | final performance ceiling | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | All SM86 optimization dimensions explored for this decode kernel |

## Rejected / Limits

- Further grid/occupancy tuning: v3's 128 blocks at 2-head fusion is optimal; 64 blocks (v4) causes grid starvation, 256 blocks (v0) dilutes KV reuse
- Compute optimizations (fast math, address precomputation): memory-latency-bound kernel, compute-side savings don't translate to runtime gains
- Cache hints (.cg): counterproductive when intra-block K/V data is reused across Q-head pairs
- num_warps tuning: num_warps=4 (v8, v9) reduces occupancy without benefit; num_warps=8 (v3, v6) is optimal

## Decision Link

`v3 = 0.0497 ms is best overall` -> `v10 preserves v3 architecture with v6 micro-optimization as final combination` -> `optimization space saturated for this kernel configuration`
