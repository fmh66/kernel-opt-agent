---
name: kernel-loop
description: Iterative GPU kernel optimization orchestrator for the local kernel-loop skill that chains kernel_profile, kernel_KBS, and kernel_benchmark. Use when the user wants an end-to-end CUDA/CUTLASS/CuTe DSL/Triton optimization loop with environment checks, correctness validation, Nsight Compute profiling, KBS-guided hypotheses, one-change kernel iterations, final benchmarking, and optimization reports.
---

# kernel-loop

Use this skill to run a bottleneck-driven kernel optimization loop by orchestrating the existing skills:

| Skill | Role |
|---|---|
| `kernel-profile` | Environment readiness, optional GPU clock config, correctness checks, NCU profiling, bottleneck evidence |
| `kernel-KBS` | Read-only evidence search for optimization patterns, hardware features, prior PRs, and implementation examples |
| `kernel-benchmark` | Final correctness and timing comparison against PyTorch eager, torch.compile, or FlashInfer baselines |

This skill does not replace those skills or duplicate their scripts. It defines the loop, artifact layout, decision points, and strict iteration discipline.

## Required Inputs

| Input | Meaning |
|---|---|
| `<kernel>` | Candidate `.cu` or `.py` implementation |
| `<ref.py>` | Python reference defining `reference(**kwargs)` |
| `<implementation>` | `auto`, `cuda-cpp`, `cutlass`, `cute-dsl`, or `triton` |
| `<dims>` | Stable problem parameters such as `--M=1024 --N=1024` |
| `<gpu>` | Target CUDA device, default `0` |
| `<N>` | Max optimization iterations, default `3`; once selected, keep fixed |

Keep implementation, dimensions, seed, tolerances, GPU, pointer sizing, warmups, and timing trials fixed across all versions.

## Artifact Layout

Create `N+1` version directories:

```text
<output_dir>/
+-- ref.py
+-- env_check.md
+-- v0/
|   +-- ncu_summary.md
|   +-- ncu_details.md
|   +-- hypothesis.txt
|   +-- <kernel version>
+-- v1/
+-- ...
+-- vN/
+-- final_report.md
+-- benchmark.md
```

`v0` is the starting implementation. `v1` through `vN` are successive one-change optimization attempts. Preserve every version, even when it regresses.

## Workflow

1. Run environment check and optional clock config.
2. Copy or place `ref.py` and the initial kernel in `<output_dir>/v0`.
3. Run correctness for the current version.
4. If correctness fails, fix that version until correctness passes before profiling.
5. Run NCU profiling for the current version.
6. Read `ncu_summary.md` first, then `ncu_details.md` and `kernel_profile/reference/NCU.md` only as needed.
7. Classify bottleneck: `Memory-Bound`, `Compute-Bound`, `Latency-Bound`, `Occupancy-Bound`, or `Mixed`.
8. Query `kernel_KBS` for evidence-backed tactics relevant to kernel type, backend, architecture, and bottleneck.
9. Write a one-variable hypothesis to `<version_dir>/hypothesis.txt` before editing code.
10. Generate exactly one next kernel version from the hypothesis.
11. Re-run correctness and NCU for the new version.
12. Compare against the previous and best versions.
13. Repeat until `N` iterations are complete.
14. Select the best correct version, run `kernel_benchmark`, and write `final_report.md`.

Do not code multiple optimization ideas into one iteration. If two variables change, the measurement cannot explain the outcome.

## Hypothesis

Before editing code, write `<version_dir>/hypothesis.txt` using the one-variable rule. Read `references/hypothesis.md` for the required format and examples.

The hypothesis must include:

| Field | Required content |
|---|---|
| Version | Source version and proposed destination version |
| Bottleneck | Current classification and metric evidence |
| Single change | Exactly one intended code or configuration change |
| Rationale | Why this change should move the measured bottleneck |
| Expected metric movement | Specific metrics expected to improve or regress |
| Risk | Correctness, occupancy, register pressure, memory traffic, or stability risks |
| Evidence | NCU artifacts and any KBS doc ids used |

Only implement the change after the hypothesis is written.

## Iteration Comparison

After each version, compare:

| Item | Source |
|---|---|
| Runtime | `ncu_summary.md` CUDA event timing |
| Bottleneck class | `ncu_summary.md` plus metric interpretation |
| Key metric changes | `ncu_summary.md` and `ncu_details.md` |
| Hypothesis result | improved, regressed, neutral, or invalid |

A faster but incorrect version is not eligible. A slower version can still be useful if it validates a metric hypothesis; keep it, record it, and move on.

## Final Report

Write `<output_dir>/final_report.md` from `references/report_template.md`.

Fill the template from `env_check.md`, every version's `correctness.md`, `ncu_summary.md`, `ncu_details.md`, `hypothesis.txt`, and final `benchmark.md`. If a metric was not collected, write `N/A`. If a value should exist but cannot be found, write `unknown` and do not infer it.

Keep claims tied to artifacts. Do not state a speedup unless it appears in profiling or benchmark output.

## When To Load Other Files

| Need | Load |
|---|---|
| Detailed correctness or profiling options | `kernel_profile/scripts/scripts.md` |
| NCU metric definitions | `kernel_profile/reference/NCU.md` |
| KBS query syntax or schema | `kernel_KBS/SKILL.md`, then `kernel_KBS/references/examples.md` if needed |
| Benchmark options | `kernel_benchmark/scripts/scripts.md` |
| Hypothesis format | `references/hypothesis.md` |
| Final report template | `references/report_template.md` |
