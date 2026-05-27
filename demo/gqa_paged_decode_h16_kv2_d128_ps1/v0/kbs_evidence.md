# KBS Evidence - v0

## NCU Facts

- Runtime: 0.0524 ms ± 0.0075 ms
- Bottleneck: Latency-Bound (SM SOL 22.09%, Memory SOL 3.58% — both well below 40%)
- Key symptoms:
  - Achieved Occupancy: 24.76%, only 0.254 waves/SM
  - Block Size: 128 threads (4 warps), Grid Size: 256 blocks
  - Registers/Thread: 32
  - Long Scoreboard Stall: 11.15% (global memory latency)
  - L1 Hit Rate: 21.16% (repeated scatter-gather KV cache reads)
  - Issue Slot Utilization: 15.04%
  - Eligible Warps/Cycle: 0.165
- Metrics driving next decision: Low occupancy + low L1 hit rate → need to increase work per block and reduce redundant KV loads

## Queries

| # | Query | Reason | Result |
|---|---|---|---|
| 1 | `triton decode attention GQA paged KV cache occupancy` | kernel-specific tactic | 0 results |
| 2 | `attention decode kernel occupancy warp` | broader kernel pattern | technique-ampere-optimization |
| 3 | `latency bound kernel optimization vectorize memory coalesce` | bottleneck tactic | 0 results |
| 4 | `flash attention decode phase kernel` | fallback domain search | pr-flashinfer-3276, blog-vllm-deepseek-v3-sparse |

## Selected Evidence

| Doc ID | Path | Confidence | Applies Because |
|---|---|---|---|
| technique-ampere-optimization | store/docs/wiki/techniques/ampere-optimization.md | source-reported | Targets SM86 (our GPU), covers occupancy tuning via regs/block-size, and low-SM-occupancy remediation with work-per-CTA increase |

## Rejected / Limits

- pr-flashinfer-3276: rejected because targets SM90 Hopper FP8 FMHAv2 pipeline, not applicable to SM86 Triton
- blog-vllm-deepseek-v3-sparse: rejected because targets sparse attention (DSA), not standard dense GQA decode

## Decision Link

`Low occupancy (24.76%) + L1 hit rate (21.16%) + redundant KV loads (8 Q-heads reload same KV data)` -> `technique-ampere-optimization: increase work per CTA` -> `fuse all Q-heads sharing a KV head into a single program instance (1→8x work/block)` -> `expected: higher occupancy, better L1 reuse, reduced total memory traffic`
