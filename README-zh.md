# Kernel Opt Agent

本仓库提供两个用于 GPU Kernel 工作的 Claude Code / Codex skill：

- [`kernel_KBS`](skills/kernel_KBS/SKILL.md)：面向 CUDA、Triton、CuTe DSL、CUTLASS 以及 Ampere/Hopper/Blackwell Kernel 研究的只读知识库检索 skill。
- [`kernel_profile`](skills/kernel_profile/SKILL.md)：面向本地 Kernel 的环境检查、正确性验证、Nsight Compute profiling 和瓶颈诊断 skill。

## Skills

| Skill | 用途 | 主要入口 |
|---|---|---|
| `kernel_KBS` | 从 PR、文档、博客、竞赛、KernelWiki、代码 artifact 和 provenance 记录中检索有证据支撑的 Kernel 知识。 | `skills/kernel_KBS/scripts/kbs.py` |
| `kernel_profile` | 验证并 profiling CUDA-C++、CUTLASS、CuTe DSL 或 Triton Kernel，再根据 NCU 证据判断瓶颈。 | `skills/kernel_profile/env/scripts/env_check.py`, `skills/kernel_profile/scripts/correctness_check.py`, `skills/kernel_profile/scripts/ncu_profile.py` |

`kernel_KBS` 默认只读，用于资料检索和实现思路参考；它不运行 Kernel，也不采集性能数据。

`kernel_profile` 用于本地检查和 profiling，会生成 `env_check.md`、`correctness.md`、`ncu_summary.md`、`ncu_details.md` 等结果文件。

## 目录结构

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

或从当前仓库直接安装：

```bash
python3 install_skills.py --target all        # Claude Code + Codex
python3 install_skills.py --target claude     # 仅 Claude Code
python3 install_skills.py --target codex      # 仅 Codex
```

可用 `--dry-run` 预览安装动作，用 `--force` 替换已有安装；本地开发可用 `--mode symlink`。

## 典型用法

1. 用 `kernel_KBS` 查找相关实现模式和来源证据。
2. 实现或修改 Kernel。
3. 用 `kernel_profile` 验证正确性并采集 NCU 指标。
4. 将实测瓶颈和 KBS 中的经验对照，再继续迭代。

完整流程和命令细节见各 skill 的 `SKILL.md`。
