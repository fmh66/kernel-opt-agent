---
name: experience
description: Index of accumulated CUDA/Triton kernel optimization experience. Routes to strategy guides (cuda/triton), learned outcome recorder, and hypothesis formulation rules.
---

# Experience Layer

All accumulated optimization knowledge lives here — strategy guides, recorded outcomes, and the tooling to retrieve and apply past learnings.

## Directory Map

```text
experience/
├── EXPERIENCE.md              ← this file (index)
├── cuda/
│   ├── CUDA.md                CUDA optimization experience by bottleneck type
│   └── reference/             memory-opt / compute-opt / latency-opt / architecture-opt / community-opt
├── triton/
│   ├── TRITON.md              Triton optimization experience by bottleneck type
│   └── reference/             triton-opt / production-workflow / community-experience
└── learned/
    ├── LEARNED.md             How to record, query, and merge optimization outcomes
    ├── scripts/
    │   └── experience_log.py  CLI tool: add / recommend / search / merge / sync / stats
    └── reference/
        ├── experiences.json   Append-only outcome database (survives across sessions)
        └── experiences.md     Human-readable markdown view (auto-generated via sync)
```

## When to use each path

| Scenario | Go to |
|---|---|
| I know the bottleneck type, need optimization ideas | `cuda/CUDA.md` or `triton/TRITON.md` |
| I need detailed implementation guidance for one technique | `cuda/reference/*.md` or `triton/reference/*.md` |
| I need architecture-specific tuning (Ampere/Hopper/Blackwell) | `cuda/reference/architecture-opt.md` |
| I need production deployment workflow or autotuning guidance | `triton/reference/production-workflow.md` |
| I need advanced patterns (persistent, TLX, split-K, grouped ordering) | `triton/reference/community-experience.md` |
| I need community advanced CUDA techniques (warp spec, persistent, PTX) | `cuda/reference/community-opt.md` |
| I want to record an iteration outcome | `learned/LEARNED.md` → `experience_log.py add` |
| I want to check what worked before on similar kernels | `learned/LEARNED.md` → `experience_log.py recommend` |
| I need to clean up duplicate entries in the experience DB | `learned/LEARNED.md` → `experience_log.py merge` |
| I need to formulate a hypothesis for the next iteration | `../reference/hypothesis.md` |

## How this fits into the optimization loop

1. **Before writing code** — query past experience to bias toward strategies that worked and avoid those that failed (see `learned/LEARNED.md`)
2. **After each iteration** — record the outcome immediately so successes narrow toward the optimum and failures prevent dead-end retries
3. **When stuck** — consult `cuda/CUDA.md` or `triton/TRITON.md` for strategy options organized by bottleneck type
