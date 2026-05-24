# Kernel Opt Agent

This repository provides two Claude Code / Codex skills for GPU kernel work:

- [`kernel_KBS`](skills/kernel_KBS/SKILL.md): a read/query knowledge base for CUDA, Triton, CuTe DSL, CUTLASS, and Ampere/Hopper/Blackwell kernel research.
- [`kernel_profile`](skills/kernel_profile/SKILL.md): a local profiling workflow for environment checks, correctness validation, Nsight Compute metrics, and bottleneck diagnosis.

## Skills

| Skill | Purpose | Main entry points |
|---|---|---|
| `kernel_KBS` | Search evidence-backed kernel knowledge from PRs, docs, blogs, contests, curated wiki pages, code artifacts, and provenance records. | `skills/kernel_KBS/scripts/kbs.py` |
| `kernel_profile` | Validate and profile concrete CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels, then classify bottlenecks from NCU evidence. | `skills/kernel_profile/env/scripts/env_check.py`, `skills/kernel_profile/scripts/correctness_check.py`, `skills/kernel_profile/scripts/ncu_profile.py` |

`kernel_KBS` is read-only by default and should be used for retrieval and source-backed implementation ideas. It does not run kernels or collect performance data.

`kernel_profile` runs local checks and profiling. It produces artifacts such as `env_check.md`, `correctness.md`, `ncu_summary.md`, and `ncu_details.md`.

## Layout

```text
skills/
├── kernel_KBS/
│   ├── SKILL.md
│   ├── scripts/
│   ├── references/
│   └── store/
└── kernel_profile/
    ├── SKILL.md
    ├── env/
    ├── scripts/
    └── reference/
```

## Install

Claude Code plugin:

```text
/plugin marketplace add fmh66/kernel-opt-agent
/plugin install kernel-opt-agent@fmh66
```

Codex plugin marketplace:

```text
/plugin marketplace add fmh66/kernel-opt-agent
/plugin install kernel-opt-agent@fmh66
```

This repository includes both `.claude-plugin/plugin.json` and `.codex-plugin/plugin.json`, so the same GitHub marketplace URL works for both tools.

Directly from this repository:

```bash
python3 install_skills.py --target all        # Claude Code + Codex
python3 install_skills.py --target claude     # Claude Code only
python3 install_skills.py --target codex      # Codex only
```

Use `--dry-run` to preview and `--force` to replace existing installed copies. Use `--mode symlink` for local development.

## Typical Use

1. Use `kernel_KBS` to find relevant implementation patterns and source evidence.
2. Implement or revise the kernel.
3. Use `kernel_profile` to validate correctness and collect NCU metrics.
4. Compare measured bottlenecks with the KBS guidance and iterate.

See each skill's `SKILL.md` for the full workflow and command details.
