# Kernel Opt Agent

[中文](./README-zh.md)

This repository provides four Claude Code / Codex skills for GPU kernel work:

- [`kernel-KBS`](skills/kernel-KBS/SKILL.md): a read/query knowledge base for CUDA, Triton, CuTe DSL, CUTLASS, and Ampere/Hopper/Blackwell kernel research.
- [`kernel-benchmark`](skills/kernel-benchmark/SKILL.md): a standalone benchmark workflow for comparing CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels against PyTorch eager, `torch.compile`, and FlashInfer references.
- [`kernel-profile`](skills/kernel-profile/SKILL.md): a local profiling workflow for environment checks, correctness validation, Nsight Compute metrics, and bottleneck diagnosis.
- [`kernel-loop`](skills/kernel-loop/SKILL.md): an iterative optimization orchestrator that chains profiling, KBS-guided hypotheses, one-change kernel iterations, final benchmarking, and reports.

## Skills

| Skill | Purpose | Main entry points |
|---|---|---|
| `kernel-KBS` | Search evidence-backed kernel knowledge from PRs, docs, blogs, contests, curated wiki pages, code artifacts, and provenance records. | `skills/kernel-KBS/scripts/kbs.py` |
| `kernel-benchmark` | Compare custom CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels against PyTorch eager, `torch.compile`, and FlashInfer baselines for correctness and latency. | `skills/kernel-benchmark/scripts/benchmark.py` |
| `kernel-profile` | Validate and profile concrete CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels, then classify bottlenecks from NCU evidence. | `skills/kernel-profile/env/scripts/env_check.py`, `skills/kernel-profile/env/scripts/enc_config.py`, `skills/kernel-profile/scripts/correctness_check.py`, `skills/kernel-profile/scripts/ncu_profile.py` |
| `kernel-loop` | Run a fixed-iteration optimization loop with environment checks, correctness, NCU profiling, bottleneck classification, KBS evidence, one-variable hypotheses, final benchmark, and report. | `skills/kernel-loop/SKILL.md`, `skills/kernel-loop/references/hypothesis.md`, `skills/kernel-loop/references/report_template.md` |

`kernel-KBS` is read-only by default and should be used for retrieval and source-backed implementation ideas. It does not run kernels or collect performance data.

`kernel-benchmark` runs standalone benchmarks and writes `benchmark.md`, including correctness results, timing statistics, and speedups versus selected baselines. It uses KernelBench-style CUDA event timing by default.

`kernel-profile` runs local checks and profiling. It produces artifacts such as `env_check.md`, `correctness.md`, `ncu_summary.md`, and `ncu_details.md`.

`kernel-loop` orchestrates the other skills for an end-to-end optimization loop. It preserves every version, requires one hypothesis before each code change, keeps dimensions and measurement settings fixed, and writes `final_report.md` from measured artifacts.

## Layout

```text
skills/
├── kernel-KBS/
│   ├── SKILL.md
│   ├── requirements.txt
│   ├── references/
│   ├── scripts/
│   └── store/
├── kernel-benchmark/
│   ├── SKILL.md
│   ├── requirements.txt
│   └── scripts/
├── kernel-loop/
│   ├── SKILL.md
│   └── references/
└── kernel-profile/
    ├── SKILL.md
    ├── requirements.txt
    ├── env/
    ├── reference/
    └── scripts/
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

By default, the installer installs `kernel-KBS`, `kernel-benchmark`, `kernel-profile`, and `kernel-loop` into user-level skill directories. Use `--scope project` to install into this repository's `.claude/skills` or `.codex/skills` directories instead.

Useful installer options:

```bash
python3 install_skills.py --dry-run
python3 install_skills.py --force
python3 install_skills.py --mode symlink
python3 install_skills.py --target codex --skill kernel-benchmark
python3 install_skills.py --target all --all-skills
```

## Dependencies

Each skill owns its Python dependencies in its own `requirements.txt`:

```bash
python3 -m pip install -r skills/kernel-KBS/requirements.txt
python3 -m pip install -r skills/kernel-benchmark/requirements.txt
python3 -m pip install -r skills/kernel-profile/requirements.txt
```

GPU runtime dependencies are environment-specific. Benchmarking and profiling may require PyTorch with CUDA support, NVIDIA drivers/runtime, `nvcc`, Nsight Compute / `ncu`, `nsight-python`, Triton, CuTe DSL, or CUTLASS headers depending on the implementation being measured. Run `kernel-profile`'s environment check before profiling.

`kernel-loop` has no separate Python dependency file; it uses the profiling, KBS, and benchmark skills.

## Typical Use

1. Use `kernel-KBS` to find relevant implementation patterns and source evidence.
2. Implement or revise the kernel.
3. Use `kernel-benchmark` to compare the custom kernel against selected PyTorch eager, `torch.compile`, or FlashInfer baselines, confirming correctness and speedup.
4. Use `kernel-profile` to collect NCU metrics and classify bottlenecks for important versions.
5. Compare measured bottlenecks with KBS guidance and iterate.

For a fully managed loop, use `kernel-loop` with a fixed iteration budget. It runs the same profile, evidence, hypothesis, one-change iteration, and final benchmark/report sequence as a single workflow.

See each skill's `SKILL.md` for workflow and command details:

- [`skills/kernel-KBS/SKILL.md`](skills/kernel-KBS/SKILL.md)
- [`skills/kernel-benchmark/SKILL.md`](skills/kernel-benchmark/SKILL.md)
- [`skills/kernel-profile/SKILL.md`](skills/kernel-profile/SKILL.md)
- [`skills/kernel-loop/SKILL.md`](skills/kernel-loop/SKILL.md)
