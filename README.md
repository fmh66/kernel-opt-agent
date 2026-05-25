# Kernel Opt Agent

Kernel Opt Agent packages three Claude Code / Codex skills for GPU kernel work:

- [`kernel_KBS`](skills/kernel_KBS/SKILL.md): a read/query knowledge base for CUDA, Triton, CuTe DSL, CUTLASS, and Ampere/Hopper/Blackwell kernel research.
- [`kernel_benchmark`](skills/kernel_benchmark/SKILL.md): a standalone benchmark workflow for comparing CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels against PyTorch eager, `torch.compile`, and FlashInfer references.
- [`kernel_profile`](skills/kernel_profile/SKILL.md): a local profiling workflow for environment checks, correctness validation, Nsight Compute metrics, and bottleneck diagnosis.

## Skills

| Skill | Purpose | Main entry points |
|---|---|---|
| `kernel_KBS` | Search evidence-backed kernel knowledge from PRs, docs, blogs, contests, curated wiki pages, code artifacts, and provenance records. | `skills/kernel_KBS/scripts/kbs.py` |
| `kernel_benchmark` | Compare custom CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels against PyTorch eager, `torch.compile`, and FlashInfer baselines for correctness and latency. | `skills/kernel_benchmark/scripts/benchmark.py` |
| `kernel_profile` | Validate and profile concrete CUDA-C++, CUTLASS, CuTe DSL, or Triton kernels, then classify bottlenecks from NCU evidence. | `skills/kernel_profile/env/scripts/env_check.py`, `skills/kernel_profile/env/scripts/enc_config.py`, `skills/kernel_profile/scripts/correctness_check.py`, `skills/kernel_profile/scripts/ncu_profile.py` |

`kernel_KBS` is read-only by default and should be used for retrieval and source-backed implementation ideas. It does not run kernels or collect performance data.

`kernel_benchmark` runs standalone benchmarks and writes `benchmark.md`, including correctness results, timing statistics, and speedups versus selected baselines. It uses KernelBench-style CUDA event timing by default.

`kernel_profile` runs local checks and profiling. It produces artifacts such as `env_check.md`, `correctness.md`, `ncu_summary.md`, and `ncu_details.md`.

## Layout

```text
skills/
├── kernel_KBS/
│   ├── README.md
│   ├── SKILL.md
│   ├── requirements.txt
│   ├── references/
│   ├── scripts/
│   └── store/
├── kernel_benchmark/
│   ├── README.md
│   ├── SKILL.md
│   ├── requirements.txt
│   └── scripts/
└── kernel_profile/
    ├── README.md
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

By default, the installer installs `kernel_KBS`, `kernel_benchmark`, and `kernel_profile` into user-level skill directories. Use `--scope project` to install into this repository's `.claude/skills` or `.codex/skills` directories instead.

Useful installer options:

```bash
python3 install_skills.py --dry-run
python3 install_skills.py --force
python3 install_skills.py --mode symlink
python3 install_skills.py --target codex --skill kernel_benchmark
python3 install_skills.py --target all --all-skills
```

## Dependencies

Each skill owns its Python dependencies in its own `requirements.txt`:

```bash
python3 -m pip install -r skills/kernel_KBS/requirements.txt
python3 -m pip install -r skills/kernel_benchmark/requirements.txt
python3 -m pip install -r skills/kernel_profile/requirements.txt
```

GPU runtime dependencies are environment-specific. Benchmarking and profiling may require PyTorch with CUDA support, NVIDIA drivers/runtime, `nvcc`, Nsight Compute / `ncu`, `nsight-python`, Triton, CuTe DSL, or CUTLASS headers depending on the implementation being measured. Run `kernel_profile`'s environment check before profiling.

## Typical Use

1. Use `kernel_KBS` to find relevant implementation patterns and source evidence.
2. Implement or revise the kernel.
3. Use `kernel_benchmark` to compare the custom kernel against selected PyTorch eager, `torch.compile`, or FlashInfer baselines, confirming correctness and speedup.
4. Use `kernel_profile` to collect NCU metrics and classify bottlenecks for important versions.
5. Compare measured bottlenecks with KBS guidance and iterate.

See each skill's `README.md` and `SKILL.md` for workflow and command details:

- [`skills/kernel_KBS/README.md`](skills/kernel_KBS/README.md)
- [`skills/kernel_benchmark/README.md`](skills/kernel_benchmark/README.md)
- [`skills/kernel_profile/README.md`](skills/kernel_profile/README.md)
