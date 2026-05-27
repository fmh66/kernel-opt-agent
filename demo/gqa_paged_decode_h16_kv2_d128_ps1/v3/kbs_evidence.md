# KBS Evidence - v3

## NCU Facts

- Runtime: 0.0497 ms ± 0.0042 ms (5.2% speedup vs v0 0.0524 ms)
- Bottleneck: Mixed — improved from Latency-Bound toward Memory-Bound
- Key symptoms:
  - SM Throughput: 35.66% (v0: 22.09%) — compute utilization improved
  - 2x KV data reuse (2 Q-heads sharing KV loads within block)
  - Grid: 128 blocks (v0: 256, v1: 32, v2: 256)
- Metrics driving next decision: KV reuse of 2x yielded 5% speedup; increasing to 4x and 8x should yield additional gains

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `gqa grouped query attention kernel triton` | verify GQA fusion strategy | kernel-nsa |
| 2 | `attention decode kernel occupancy warp` | check decode optimization patterns | technique-ampere-optimization |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | verified | KV data reuse strategy validated: 2x reuse → 5% speedup confirms memory latency reduction hypothesis |

## Rejected / Limits

- kernel-nsa: rejected because targets sparse attention, not standard dense GQA decode

## Decision Link

`2x KV reuse gave 5% speedup` -> `Continue fusion: 4 Q-heads per block (4x KV reuse)` -> `Expected: additional speedup from reduced memory traffic`
