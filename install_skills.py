#!/usr/bin/env python3
"""Install this repository's skills for Claude Code and Codex."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


DEFAULT_SKILLS = ("kernel-KBS", "kernel-benchmark", "kernel-profile", "kernel-loop")
TARGETS = ("claude", "codex")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def available_skills(root: Path) -> list[str]:
    skills_root = root / "skills"
    if not skills_root.is_dir():
        return []
    return sorted(
        item.name
        for item in skills_root.iterdir()
        if item.is_dir() and (item / "SKILL.md").is_file()
    )


def default_user_dir(target: str) -> Path:
    if target == "claude":
        override = os.environ.get("CLAUDE_SKILLS_DIR")
        if override:
            return Path(override).expanduser()
        home = os.environ.get("CLAUDE_HOME")
        return Path(home).expanduser() / "skills" if home else Path.home() / ".claude" / "skills"

    if target == "codex":
        override = os.environ.get("CODEX_SKILLS_DIR")
        if override:
            return Path(override).expanduser()
        home = os.environ.get("CODEX_HOME")
        return Path(home).expanduser() / "skills" if home else Path.home() / ".codex" / "skills"

    raise ValueError(f"unknown target: {target}")


def default_project_dir(root: Path, target: str) -> Path:
    if target == "claude":
        return root / ".claude" / "skills"
    if target == "codex":
        return root / ".codex" / "skills"
    raise ValueError(f"unknown target: {target}")


def destination_dir(args: argparse.Namespace, root: Path, target: str) -> Path:
    if target == "claude" and args.claude_dir:
        return Path(args.claude_dir).expanduser()
    if target == "codex" and args.codex_dir:
        return Path(args.codex_dir).expanduser()
    if args.scope == "project":
        return default_project_dir(root, target)
    return default_user_dir(target)


def copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
    return {
        name
        for name in names
        if name in ignored or name.endswith(".pyc") or name.endswith(".pyo")
    }


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def install_skill(
    source: Path,
    destination_root: Path,
    *,
    mode: str,
    force: bool,
    dry_run: bool,
) -> None:
    destination = destination_root / source.name

    if source.resolve() == destination.resolve():
        print(f"skip {source.name}: source and destination are the same")
        return

    action = "link" if mode == "symlink" else "copy"
    if dry_run:
        print(f"would {action} {source} -> {destination}")
        return

    if destination.exists() or destination.is_symlink():
        if not force:
            raise RuntimeError(f"destination exists: {destination} (use --force to replace it)")
        remove_existing(destination)

    destination_root.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source, target_is_directory=True)
    else:
        shutil.copytree(source, destination, symlinks=True, ignore=copy_ignore)

    print(f"{action}ed {source.name} -> {destination}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install kernel-KBS, kernel-benchmark, kernel-profile, and kernel-loop for Claude Code and Codex."
    )
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS),
        default="all",
        help="tool to install for; default installs for both Claude Code and Codex",
    )
    parser.add_argument(
        "--scope",
        choices=("user", "project"),
        default="user",
        help="user installs to ~/.claude/skills and ~/.codex/skills; project installs to .claude/skills and .codex/skills",
    )
    parser.add_argument(
        "--skill",
        action="append",
        dest="skills",
        help="skill directory name to install; may be passed multiple times",
    )
    parser.add_argument(
        "--all-skills",
        action="store_true",
        help="install every skills/* directory that contains SKILL.md",
    )
    parser.add_argument(
        "--mode",
        choices=("copy", "symlink"),
        default="copy",
        help="copy skill directories or install symlinks for local development",
    )
    parser.add_argument("--claude-dir", help="override Claude Code skills directory")
    parser.add_argument("--codex-dir", help="override Codex skills directory")
    parser.add_argument("--force", action="store_true", help="replace existing installed skill directories")
    parser.add_argument("--dry-run", action="store_true", help="print actions without changing files")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    root = repo_root()
    all_skill_names = available_skills(root)

    if args.all_skills:
        selected = all_skill_names
    elif args.skills:
        selected = args.skills
    else:
        selected = list(DEFAULT_SKILLS)

    missing = [name for name in selected if name not in all_skill_names]
    if missing:
        print(f"missing skill(s): {', '.join(missing)}", file=sys.stderr)
        print(f"available skill(s): {', '.join(all_skill_names) or '(none)'}", file=sys.stderr)
        return 2

    targets = TARGETS if args.target == "all" else (args.target,)

    try:
        for target in targets:
            dest_root = destination_dir(args, root, target)
            for name in selected:
                install_skill(
                    root / "skills" / name,
                    dest_root,
                    mode=args.mode,
                    force=args.force,
                    dry_run=args.dry_run,
                )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.dry_run:
        print("done. Restart Claude Code or Codex if they were already running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
