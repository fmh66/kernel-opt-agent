# Kernel Opt Agent

[English](./README.md)

本仓库提供四个用于 GPU Kernel 工作的 Claude Code / Codex skill：

- [`kernel-KBS`](skills/kernel-KBS/SKILL.md)：面向 CUDA、Triton、CuTe DSL、CUTLASS 以及 Ampere/Hopper/Blackwell Kernel 研究的只读知识库检索 skill。
- [`kernel-benchmark`](skills/kernel-benchmark/SKILL.md)：面向 CUDA-C++、CUTLASS、CuTe DSL 和 Triton Kernel 的独立 benchmark workflow，用 PyTorch eager、`torch.compile` 和 FlashInfer reference 做正确性和延迟对比。
- [`kernel-profile`](skills/kernel-profile/SKILL.md)：面向本地 Kernel 的环境检查、正确性验证、Nsight Compute profiling 和瓶颈诊断 workflow。
- [`kernel-loop`](skills/kernel-loop/SKILL.md)：迭代式优化编排 skill，串联 profiling、KBS 证据检索、单变量假设、逐版本 Kernel 修改、最终 benchmark 和报告生成。

## Skills

| Skill | 用途 | 主要入口 |
|---|---|---|
| `kernel-KBS` | 从 PR、文档、博客、竞赛、KernelWiki、代码 artifact 和 provenance 记录中检索有证据支撑的 Kernel 知识。 | `skills/kernel-KBS/scripts/kbs.py` |
| `kernel-benchmark` | 将自定义 CUDA-C++、CUTLASS、CuTe DSL 或 Triton Kernel 与 PyTorch eager、`torch.compile` 和 FlashInfer baseline 做 correctness 和 latency 对比。 | `skills/kernel-benchmark/scripts/benchmark.py` |
| `kernel-profile` | 验证并 profiling CUDA-C++、CUTLASS、CuTe DSL 或 Triton Kernel，再根据 NCU 证据判断瓶颈。 | `skills/kernel-profile/env/scripts/env_check.py`, `skills/kernel-profile/env/scripts/enc_config.py`, `skills/kernel-profile/scripts/correctness_check.py`, `skills/kernel-profile/scripts/ncu_profile.py` |
| `kernel-loop` | 运行固定迭代次数的优化循环，包含环境检查、正确性验证、NCU profiling、瓶颈分类、KBS 证据、单变量 hypothesis、最终 benchmark 和报告。 | `skills/kernel-loop/SKILL.md`, `skills/kernel-loop/references/hypothesis.md`, `skills/kernel-loop/references/report_template.md` |

`kernel-KBS` 默认只读，用于资料检索和实现思路参考；它不运行 Kernel，也不采集性能数据。

`kernel-benchmark` 用于独立 benchmark，会生成 `benchmark.md`，包含 correctness、计时统计和相对所选 baseline 的 speedup。默认使用 KernelBench 风格 CUDA event timing。

`kernel-profile` 用于本地检查和 profiling，会生成 `env_check.md`、`correctness.md`、`ncu_summary.md`、`ncu_details.md` 等结果文件。

`kernel-loop` 用于端到端优化循环。它会保留每个版本，要求每次改代码前先写一个 hypothesis，保持维度和测量配置固定，并基于实测 artifact 生成 `final_report.md`。

## 目录结构

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

## 安装

Claude Code plugin：

```text
/plugin marketplace add fmh66/kernel-opt-agent
/plugin install kernel-opt-agent@fmh66
```

Codex plugin marketplace：

```text
/plugin marketplace add fmh66/kernel-opt-agent
/plugin install kernel-opt-agent@fmh66
```

本仓库同时包含 `.claude-plugin/plugin.json` 和 `.codex-plugin/plugin.json`，因此 Claude Code 和 Codex 可以使用同一个 GitHub marketplace URL。

也可以从当前仓库直接安装：

```bash
python3 install_skills.py --target all        # Claude Code + Codex
python3 install_skills.py --target claude     # 仅 Claude Code
python3 install_skills.py --target codex      # 仅 Codex
```

默认会把 `kernel-KBS`、`kernel-benchmark`、`kernel-profile` 和 `kernel-loop` 安装到用户级 skill 目录。使用 `--scope project` 可改为安装到当前仓库的 `.claude/skills` 或 `.codex/skills` 目录。

常用安装参数：

```bash
python3 install_skills.py --dry-run
python3 install_skills.py --force
python3 install_skills.py --mode symlink
python3 install_skills.py --target codex --skill kernel-benchmark
python3 install_skills.py --target all --all-skills
```

## 依赖

每个 skill 在自己的 `requirements.txt` 中维护 Python 依赖：

```bash
python3 -m pip install -r skills/kernel-KBS/requirements.txt
python3 -m pip install -r skills/kernel-benchmark/requirements.txt
python3 -m pip install -r skills/kernel-profile/requirements.txt
```

GPU runtime 依赖取决于本地环境。Benchmark 和 profiling 可能需要支持 CUDA 的 PyTorch、NVIDIA driver/runtime、`nvcc`、Nsight Compute / `ncu`、`nsight-python`、Triton、CuTe DSL 或 CUTLASS headers，具体取决于被测实现。正式 profiling 前建议先运行 `kernel-profile` 的环境检查。

`kernel-loop` 没有单独的 Python 依赖文件；它复用 profiling、KBS 和 benchmark skills。

## 典型用法

1. 用 `kernel-KBS` 查找相关实现模式和来源证据。
2. 实现或修改 Kernel。
3. 用 `kernel-benchmark` 对比自定义 Kernel 和所选 PyTorch eager、`torch.compile` 或 FlashInfer baseline，确认 correctness 和 speedup。
4. 用 `kernel-profile` 对关键版本采集 NCU 指标并判断瓶颈。
5. 将实测瓶颈和 KBS 中的经验对照，再继续迭代。

如果需要托管式完整优化流程，可以使用 `kernel-loop` 并指定固定迭代预算。它会把 profile、证据检索、hypothesis、单次变量修改、最终 benchmark 和报告串成一个 workflow。

完整流程和命令细节见各 skill 的 `SKILL.md`：

- [`skills/kernel-KBS/SKILL.md`](skills/kernel-KBS/SKILL.md)
- [`skills/kernel-benchmark/SKILL.md`](skills/kernel-benchmark/SKILL.md)
- [`skills/kernel-profile/SKILL.md`](skills/kernel-profile/SKILL.md)
- [`skills/kernel-loop/SKILL.md`](skills/kernel-loop/SKILL.md)
