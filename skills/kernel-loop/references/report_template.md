# Report Template

Use this template when writing `<output_dir>/final_report.md`.

Fill values from measured artifacts only:

| Artifact | Use |
|---|---|
| `env_check.md` | GPU, CUDA, nvcc, ncu, nsight-python, Triton, PyTorch |
| `v*/correctness.md` | Correctness status |
| `v*/ncu_summary.md` | Runtime, throughput, occupancy, bottleneck, high-level stalls |
| `v*/ncu_details.md` | Detailed metrics not present in the summary |
| `v*/hypothesis.txt` | Strategy, rationale, expected gain, decision rule |
| `benchmark.md` | Final timing and baseline speedups |

Use `N/A` for metrics that were not collected. Use `unknown` for expected values that cannot be located in artifacts. Do not invent missing measurements.

In strategy tables, list only strategies that were actually used in at least one version. Use `yes` for the version where the listed strategy was applied and `no` for versions where that same strategy was not applied. Do not include unused strategy rows.

```markdown
# CUDA Optimization Final Report - `<kernel_name>` (`<date>`)

## Environment

| Item | Value |
|---|---|
| GPU | `<name>` (CC `<x.y>`) |
| CUDA / nvcc | `<version>` |
| Kernel file | `<path>` |

---

## Version Iteration Comparison

| Metric | v0 (baseline) | v1 | v2 | v3 | ... | best |
|---|---|---|---|---|---|---|
| Correctness | | | | | | |
| Execution Time (ms) | | | | | | |
| Speedup (x) | 1.00 | | | | | |
| Memory Throughput (%) | | | | | | |
| Compute Throughput (%) | | | | | | |
| SM Active Cycles (%) | | | | | | |
| Bottleneck | | | | | | |
| Achieved Occupancy (%) | | | | | | |
| Active Warps / SM | | | | | | |
| Registers / Thread | | | | | | |
| Warp Stall - Long SB (%) | | | | | | |
| Warp Stall - Short SB (%) | | | | | | |
| Branch Divergence (%) | | | | | | |
| Key metric notes | | | | | | |

---

## Optimization Strategies per Version

Only include strategies that were actually used in at least one version. Add rows dynamically from `hypothesis.txt` and code changes instead of keeping a fixed full catalog.

| Strategy | v1 | v2 | v3 | ... |
|---|---|---|---|---|
| `<used strategy>` | yes/no | yes/no | yes/no | |
| `<used strategy>` | yes/no | yes/no | yes/no | |

**Decision rationale per version:**

- **v1:** `<strategy selection rationale and expected gain>`
- **v2:** `<strategy selection rationale and expected gain>`
- **v3:** `<strategy selection rationale and expected gain>`

---

## Hypothesis Outcomes

| Transition | Hypothesis | Result | Evidence |
|---|---|---|---|
| v0 -> v1 | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |
| v1 -> v2 | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |
| v2 -> v3 | `<single change>` | improved/regressed/neutral/invalid | `<metrics>` |

---

## KBS Evidence

| Version | Doc ID | Canonical path | Used for |
|---|---|---|---|
| v1 | `<doc-id or N/A>` | `<path or N/A>` | `<strategy/rationale>` |
| v2 | `<doc-id or N/A>` | `<path or N/A>` | `<strategy/rationale>` |

---

## Final Benchmark

| Item | Value |
|---|---|
| Best kernel | `<path>` |
| Baselines | `<pytorch-eager, torch-compile, flashinfer, ...>` |
| Best execution time | `<ms>` |
| Baseline execution time | `<ms>` |
| Benchmark speedup | `<x>` |
| Benchmark artifact | `benchmark.md` |

---

## Best Version Conclusion

**Best version:** `v<N>` - execution time reduced from `<v0>` ms to `<vN>` ms, speedup `<x>`.

Key gains: `<primary optimization strategies>`.

Stopping reason: `<max iterations reached / performance target met / bottleneck saturated>`.

**Remaining optimization opportunities:** `<potential improvements for the next round, or N/A>`
```
