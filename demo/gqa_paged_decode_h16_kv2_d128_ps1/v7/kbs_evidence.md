# KBS Evidence - v7

## NCU Facts

- Runtime: 0.0572 ms (REGRESSION — .cg cache hint harmful)
- Bottleneck: L1 bypass hurts within-block KV reuse (2 Q-heads sharing same K/V)
- Key symptoms: cache_modifier=".cg" prevents L1 caching of data reused within block
- Metrics driving next decision: Cache hints are counterproductive when intra-block data sharing occurs

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton cache modifier cg cs eviction policy` | verify cache semantics | 0 results |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | source-reported | .cg appropriate only for true streaming data; intra-block reuse needs L1 |

## Rejected / Limits

- cache_modifier=".cg": rejected — prevents L1 cache hits for data reused across 2 Q-heads within same block

## Decision Link

`.cg harmful for intra-block reuse` -> `Try num_warps=4 to reduce register pressure` -> `Lower occupancy per block may increase blocks/SM`
