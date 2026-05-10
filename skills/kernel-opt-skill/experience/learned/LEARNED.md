---
name: learned
description: How to record, query, and merge CUDA/Triton kernel optimization outcomes. Use experience_log.py to persist what worked and avoid repeating dead ends.
---

# Learned Experience

Records per-iteration outcomes (success/failure/neutral) with full NCU metrics, and retrieves the most relevant past experiences before composing a hypothesis.

**Persistence**: `reference/experiences.json` (append-only, survives across sessions).

**Script**: `scripts/experience_log.py` — all commands below are run from the skill root (`$SKILL_ROOT`).

---

## Record an iteration outcome

```bash
# Successful iteration:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py add \
  --kernel matmul --backend cuda --chip sm_86 --dims "M=1024,K=512,N=512" \
  --iteration v1 --bottleneck memory_bound \
  --memory-sol-before 72 --sm-sol-before 22 --latency-before 4.2 --occupancy-before 33 \
  --memory-sol-after 48 --sm-sol-after 35 --latency-after 1.8 --occupancy-after 42 \
  --optimization "shared-memory-tiling{tile-size=128}" \
  --hypothesis "Add shared memory tiling to reduce global memory latency" \
  --outcome success \
  --notes "Memory SOL dropped from 72% to 48%, Long Scoreboard stall reduced"

# Failed iteration (prevents retrying the same dead end):
python $SKILL_ROOT/experience/learned/scripts/experience_log.py add \
  --kernel matmul --backend cuda --chip sm_86 --iteration v3 --bottleneck compute_bound \
  --latency-before 1.8 --latency-after 2.1 \
  --optimization "loop-unrolling{factor=8}" \
  --outcome failure \
  --notes "register spills at unroll-factor=8, 149 regs/thread"
```

### Outcome thresholds

| Outcome | Condition |
|---|---|
| `success` | > 5% speedup AND correctness PASS |
| `failure` | regression, correctness FAIL, or < 2% change |
| `neutral` | 2–5% change (record anyway — patterns emerge over time) |

**Record every outcome immediately after each iteration.** Successes narrow toward the optimum; failures prevent revisiting dead ends.

---

## Query relevant past experiences

```bash
# Before composing the next hypothesis — get what worked and what failed:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py recommend \
  --kernel matmul --backend cuda --chip sm_86 --bottleneck memory_bound

# Search by keyword:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py search "tiling"

# List all with optional filters:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py list \
  [--kernel matmul] [--backend cuda] [--outcome success] [--bottleneck memory_bound]
```

### `recommend` output

Top-N past entries ranked by kernel/chip/bottleneck match + speedup, followed by:

- **Most successful optimization** recommendation (most frequent success pattern for this context)
- **Known failures** warning (strategies that previously failed — avoid these these)
- Empty result → rely on the strategy guides in `cuda/CUDA.md` or `triton/TRITON.md`

---

## Merge workflow (prevent bloat)

Two-phase LLM-driven merge to collapse duplicate entries that represent the same experiment repeated across sessions.

```bash
# Phase 1: review all entries (LLM reads and decides what to merge)
python $SKILL_ROOT/experience/learned/scripts/experience_log.py merge

# Phase 2: apply explicit merge groups after review
python $SKILL_ROOT/experience/learned/scripts/experience_log.py merge --groups "1,3" "2,4,5"
```

**Phase 1** prints every entry grouped by kernel with full metrics (latency before/after, SOL%, optimization, hypothesis, notes). The LLM reads them and decides which entries represent the same experiment. **Phase 2** collapses each group mechanically: best metrics kept (highest speedup for successes), notes concatenated, longest hypothesis kept, `merged_count` recorded.

---

## Sync and statistics

```bash
# Regenerate experiences.md from JSON:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py sync

# Per-kernel success rate and top optimizations:
python $SKILL_ROOT/experience/learned/scripts/experience_log.py stats
```

Run `sync` and `stats` when max iterations are reached, before generating the final report.

---

## Flags reference

### `add` command

| Flag | Required | Notes |
|---|---|---|
| `--kernel` | recommended | matmul, convolution, reduction, attention, elementwise, … |
| `--backend` | required | `cuda` or `triton` — tracked separately |
| `--chip` | recommended | sm_86, sm_80, … (from `env_check.md`) |
| `--dims` | optional | Free-form, e.g. `"M=1024,K=512,N=512"` |
| `--iteration` | recommended | vN label |
| `--bottleneck` | recommended | memory_bound, compute_bound, latency_bound, occupancy_bound, grid_anomaly |
| `--stall` | optional | Dominant NCU warp stall reason |
| `--{memory,sm}-sol-{before,after}` | optional | Speed of Light % |
| `--latency-{before,after}` | optional | ms |
| `--occupancy-{before,after}` | optional | Achieved occupancy % |
| `--optimization` | recommended | Optimization strategy applied this iteration |
| `--hypothesis` | optional | Content of hypothesis.txt |
| `--outcome` | **required** | success / failure / neutral |
| `--notes` | optional | Why it worked or failed |

### `recommend` command

| Flag | Required | Notes |
|---|---|---|
| `--kernel` | recommended | Current kernel type |
| `--backend` | recommended | `cuda` or `triton` |
| `--chip` | recommended | Current GPU chip |
| `--bottleneck` | recommended | Current diagnosed bottleneck |
| `-n` | optional | Max results to show (default 5) |
