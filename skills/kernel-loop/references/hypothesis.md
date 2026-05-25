# Hypothesis Format

Use this file when `kernel_loop` reaches the "write hypothesis before code" step.

## One-Variable Rule

Each iteration changes exactly one independent variable. Examples:

| Valid single change | Invalid mixed change |
|---|---|
| Change vector width from 1 to 4 | Change vector width and add shared memory tiling |
| Add shared memory staging | Add staging, change block size, and unroll loops |
| Increase block size from 128 to 256 | Increase block size and change memory layout |
| Add `num_warps=8` for Triton | Change `num_warps`, `BLOCK_M`, and `BLOCK_N` together |

Mechanical edits required to support the single change are allowed, but do not alter a second performance variable.

## Required File

Write the hypothesis to the current version directory before editing the next version:

```text
<output_dir>/vK/hypothesis.txt
```

## Template

```text
Version:
- Source: vK
- Target: vK+1

Current evidence:
- Correctness: pass|fail
- Runtime:
- Bottleneck:
- Key metrics:

Single change:
- Change:
- Scope:
- Things intentionally not changed:

Rationale:
- Why this should address the measured bottleneck:

Expected metric movement:
- Improve:
- Watch for regressions:

Risk:
- Correctness risk:
- Performance risk:

Evidence used:
- NCU artifacts:
- KBS docs:

Decision rule:
- Accept if:
- Reject if:
```
