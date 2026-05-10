#!/usr/bin/env python3
"""
Kernel optimization experience logger — records what worked, retrieves what's relevant.

Usage:
    # Record a successful optimization:
    python scripts/experience_log.py add \
      --kernel matmul --chip sm_86 --dims "M=1024,K=512,N=512" \
      --iteration v1 --bottleneck latency_bound \
      --latency-before 4.2 --latency-after 1.8 \
      --optimization "shared-memory-tiling{tile-size=128}" \
      --outcome success --notes "Fixed 1-thread block pathology"

    # Record a failed experiment:
    python scripts/experience_log.py add ... --outcome failure \
      --notes "register spills with unroll-factor=8"

    # Query relevant past experiences:
    python scripts/experience_log.py recommend \
      --kernel matmul --backend cuda --chip sm_86 --bottleneck memory_bound

    # Search by keyword:
    python scripts/experience_log.py search "tiling"

    # List all experiences (with optional filters):
    python scripts/experience_log.py list [--kernel matmul] [--outcome success]

    # Review all entries so the LLM can decide what to merge:
    python scripts/experience_log.py merge

    # Apply explicit merge groups decided after review:
    python scripts/experience_log.py merge --groups "1,3" "2,4,5"

    # Sync to markdown view:
    python scripts/experience_log.py sync

    # Show per-kernel/bottleneck success stats:
    python scripts/experience_log.py stats
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
JSON_PATH = os.path.join(SKILL_DIR, "experience", "learned", "reference", "experiences.json")
MD_PATH = os.path.join(SKILL_DIR, "experience", "learned", "reference", "experiences.md")

VALID_OUTCOMES = {"success", "failure", "neutral"}
VALID_BOTTLENECKS = {"memory_bound", "compute_bound", "latency_bound", "occupancy_bound", "grid_anomaly", "unknown"}


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load():
    if not os.path.exists(JSON_PATH):
        return []
    try:
        with open(JSON_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save(entries):
    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)


def _speedup(latency_before, latency_after):
    if latency_before and latency_after and latency_after > 0:
        return round(latency_before / latency_after, 2)
    return None


# ── Scoring for recommend ─────────────────────────────────────────────────────

def _relevance_score(entry, kernel, backend, chip, bottleneck):
    score = 0
    if entry.get("kernel") == kernel:
        score += 40
    if entry.get("backend", "cuda") == backend:
        score += 50
    if entry.get("chip") == chip:
        score += 20
    if entry.get("bottleneck_before") == bottleneck:
        score += 30
    speedup = entry.get("speedup")
    if speedup and entry.get("outcome") == "success":
        import math
        score += min(10, int(math.log(speedup + 1) * 5))
    score += min(5, entry.get("id", 0) // 10)
    return score


# ── Merge helper ──────────────────────────────────────────────────────────────

def _collapse_group(group):
    """
    Mechanically collapse a list of entries (chosen by the LLM) into one canonical.
    - Canonical = entry with highest speedup (successes) or lowest latency_after
    - Preserves canonical's metrics_before / metrics_after
    - Merges notes: concatenates unique non-empty strings
    - Keeps longest hypothesis
    - Records merged_count and merged_from_iters for traceability
    """
    if len(group) == 1:
        return group[0]

    successes = [e for e in group if e.get("outcome") == "success" and e.get("speedup")]
    if successes:
        canonical = dict(max(successes, key=lambda e: e.get("speedup") or 0))
    else:
        with_lat = [e for e in group if e.get("metrics_after", {}).get("latency_ms")]
        canonical = dict(
            min(with_lat, key=lambda e: e["metrics_after"]["latency_ms"])
            if with_lat else group[0]
        )

    all_notes = list(dict.fromkeys(
        (e.get("notes") or "").strip()
        for e in group
        if (e.get("notes") or "").strip()
    ))
    canonical["notes"] = " | ".join(all_notes) if all_notes else ""
    canonical["hypothesis"] = max(
        ((e.get("hypothesis") or "") for e in group), key=len
    )
    canonical["merged_count"] = len(group)
    canonical["merged_from_iters"] = [
        e.get("iteration", "—") for e in group
        if e.get("iteration") != canonical.get("iteration")
    ]
    return canonical


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_add(args):
    entries = _load()

    outcome = args.outcome.lower()
    if outcome not in VALID_OUTCOMES:
        sys.exit(f"--outcome must be one of: {', '.join(sorted(VALID_OUTCOMES))}")

    bottleneck = (args.bottleneck or "unknown").lower()
    if bottleneck not in VALID_BOTTLENECKS:
        print(f"Warning: --bottleneck '{bottleneck}' not in known list {sorted(VALID_BOTTLENECKS)}")

    speedup = _speedup(args.latency_before, args.latency_after)

    if outcome == "success" and speedup is not None and speedup < 1.02:
        print("⚠  speedup < 2% — consider using --outcome neutral instead of success")

    backend = (args.backend or "cuda").lower()
    if backend not in ("cuda", "triton"):
        sys.exit("--backend must be 'cuda' or 'triton'")

    entry = {
        "id": max((e["id"] for e in entries), default=0) + 1,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "iteration": args.iteration or "—",
        "kernel": args.kernel or "unknown",
        "backend": backend,
        "chip": args.chip or "unknown",
        "dims": args.dims or "",
        "bottleneck_before": bottleneck,
        "dominant_stall": args.stall or "",
        "metrics_before": {
            "memory_sol_pct": args.memory_sol_before,
            "sm_sol_pct": args.sm_sol_before,
            "latency_ms": args.latency_before,
            "achieved_occupancy": args.occupancy_before,
        },
        "metrics_after": {
            "memory_sol_pct": args.memory_sol_after,
            "sm_sol_pct": args.sm_sol_after,
            "latency_ms": args.latency_after,
            "achieved_occupancy": args.occupancy_after,
        },
        "optimization": args.optimization or "",
        "hypothesis": args.hypothesis or "",
        "outcome": outcome,
        "speedup": speedup,
        "notes": args.notes or "",
    }
    entries.append(entry)
    _save(entries)

    icon = "✓" if outcome == "success" else ("✗" if outcome == "failure" else "~")
    speedup_str = f" ({speedup}x)" if speedup else ""
    print(f"{icon} Recorded id={entry['id']} — {outcome}{speedup_str}")
    print(f"  {args.kernel or 'unknown'} / {args.chip or 'unknown'} / {bottleneck}")
    if args.optimization:
        print(f"  Optimization: {args.optimization[:80]}")
    print("  Run `experience_log.py sync` to update experiences.md")


def cmd_recommend(args):
    entries = _load()

    if not entries:
        print("(no experiences recorded yet — run `experience_log.py add` after each iteration)")
        return

    kernel = args.kernel or ""
    backend = args.backend or ""
    chip = args.chip or ""
    bottleneck = (args.bottleneck or "").lower()

    scored = []
    for e in entries:
        score = _relevance_score(e, kernel, backend, chip, bottleneck)
        if score > 0:
            scored.append((score, e))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        print(f"No relevant experiences found for: {kernel} / {chip} / {bottleneck}")
        print("All recorded experiences:")
        for e in entries[-5:]:
            print(f"  id={e['id']} {e['kernel']} {e['chip']} {e['bottleneck_before']} → {e['outcome']}")
        return

    limit = args.n or 5
    print(f"## Past experiences — {kernel or 'any'} / {backend or 'any'} / {chip or 'any'} / {bottleneck or 'any'}\n")

    successes = [e for _, e in scored if e["outcome"] == "success"]
    failures = [e for _, e in scored if e["outcome"] == "failure"]

    shown = 0
    for score, e in scored:
        if shown >= limit:
            break
        icon = "✓" if e["outcome"] == "success" else ("✗" if e["outcome"] == "failure" else "~")
        speedup_str = f" | speedup={e['speedup']}x" if e.get("speedup") else ""
        print(f"[id={e['id']}] {e['date']} iter={e['iteration']} {icon}{speedup_str}")
        print(f"  Context : {e['kernel']} / {e.get('backend', 'cuda')} / {e['chip']} | dims={e['dims']}")
        b = e.get("metrics_before", {})
        a = e.get("metrics_after", {})
        if any(v for v in b.values() if v is not None):
            b_str = f"Memory={b.get('memory_sol_pct')}% SM={b.get('sm_sol_pct')}% lat={b.get('latency_ms')}ms occ={b.get('achieved_occupancy')}%"
            a_str = f"Memory={a.get('memory_sol_pct')}% SM={a.get('sm_sol_pct')}% lat={a.get('latency_ms')}ms occ={a.get('achieved_occupancy')}%"
            print(f"  Before  : {b_str}")
            print(f"  After   : {a_str}")
        if e.get("dominant_stall"):
            print(f"  Stall   : {e['dominant_stall']}")
        print(f"  Opt     : {e['optimization']}")
        print(f"  Hyp     : {e['hypothesis']}")
        if e.get("notes"):
            print(f"  Notes   : {e['notes']}")
        print()
        shown += 1

    if successes:
        opt_counts = {}
        for e in successes:
            key = e["optimization"].split("{")[0].strip()
            opt_counts[key] = opt_counts.get(key, 0) + 1
        top_opt, top_count = max(opt_counts.items(), key=lambda x: x[1])
        total_success = len(successes)
        total_fail = len(failures)
        print(f"── Recommendation ──────────────────────────────────────────")
        print(f"  Most successful optimization: '{top_opt}' ({top_count}/{total_success} successes)")
        if total_fail > 0:
            fail_opts = [e["optimization"].split("{")[0].strip() for e in failures]
            print(f"  Known failures: {', '.join(set(fail_opts))}")
        print(f"  Total: {total_success} success, {total_fail} failure in matched history")
    else:
        print("── No successes found for this context ─────────────────────")
        if failures:
            fail_opts = list({e["optimization"].split("{")[0].strip() for e in failures})
            print(f"  Avoid: {', '.join(fail_opts)} (previously failed)")


def cmd_search(args):
    entries = _load()
    query = args.query.lower()
    hits = [e for e in entries if query in json.dumps(e).lower()]
    if not hits:
        print(f"No entries matching '{args.query}'")
        return
    for e in hits:
        icon = "✓" if e["outcome"] == "success" else ("✗" if e["outcome"] == "failure" else "~")
        speedup_str = f" {e['speedup']}x" if e.get("speedup") else ""
        print(f"[id={e['id']}] {e['date']} {e['kernel']} / {e['chip']} / {e['bottleneck_before']} {icon}{speedup_str}")
        print(f"  Opt   : {e['optimization']}")
        print(f"  Hyp   : {e['hypothesis']}")
        if e.get("notes"):
            print(f"  Notes : {e['notes']}")
        print()


def cmd_list(args):
    entries = _load()
    if not entries:
        print("(no experiences recorded yet)")
        return

    if args.kernel:
        entries = [e for e in entries if e.get("kernel") == args.kernel]
    if args.backend:
        entries = [e for e in entries if e.get("backend", "cuda") == args.backend]
    if args.outcome:
        entries = [e for e in entries if e.get("outcome") == args.outcome]
    if args.bottleneck:
        entries = [e for e in entries if e.get("bottleneck_before") == args.bottleneck]

    if not entries:
        print("(no entries match filters)")
        return

    print(f"{'ID':>4}  {'Date':10}  {'Iter':6}  {'Kernel':12}  {'Backend':8}  {'Chip':8}  {'Bottleneck':15}  {'Out':7}  {'Speedup':8}  Optimization")
    print("-" * 120)
    for e in entries:
        speedup_str = f"{e['speedup']}x" if e.get("speedup") else "—"
        icon = "SUCCESS" if e["outcome"] == "success" else ("FAILURE" if e["outcome"] == "failure" else "NEUTRAL")
        opt_short = e.get("optimization", "")[:35]
        backend = e.get("backend", "cuda")
        print(f"{e['id']:>4}  {e['date']:10}  {e['iteration']:6}  {e['kernel']:12}  {backend:8}  {e['chip']:8}  {e['bottleneck_before']:15}  {icon:7}  {speedup_str:8}  {opt_short}")


def cmd_merge(args):
    """
    Two-phase LLM-driven merge.

    Phase 1 — review (no --groups):
      Prints every entry grouped by kernel so the LLM can read them and
      identify redundant rows. No file changes.

    Phase 2 — apply (--groups "1,3" "2,4,5"):
      Collapses each specified group into one canonical entry. The LLM
      decides which IDs belong together; this function handles the mechanical
      collapse (best metrics, merged notes, longest hypothesis).
    """
    entries = _load()
    if not entries:
        print("(no experiences recorded yet)")
        return

    by_id = {e["id"]: e for e in entries}

    # Phase 1: print all entries for LLM review
    if not args.groups:
        print(f"## All experiences ({len(entries)} total) — review and decide which to merge\n")
        by_kernel_backend = {}
        for e in entries:
            key = (e.get("kernel", "unknown"), e.get("backend", "cuda"))
            by_kernel_backend.setdefault(key, []).append(e)
        for (op, be), op_entries in sorted(by_kernel_backend.items()):
            print(f"### {op} [{be}]")
            for e in op_entries:
                b = e.get("metrics_before", {})
                a = e.get("metrics_after", {})
                icon = "✓" if e["outcome"] == "success" else ("✗" if e["outcome"] == "failure" else "~")
                sp = f"  speedup={e['speedup']}x" if e.get("speedup") else ""
                print(f"  [id={e['id']}] {e['date']}  iter={e.get('iteration','—')}  "
                      f"{icon}{sp}  chip={e.get('chip','?')}  bt={e.get('bottleneck_before','?')}")
                print(f"    optimization: {e.get('optimization','')}")
                print(f"    hypothesis  : {(e.get('hypothesis') or '')[:80]}")
                lat_b = b.get("latency_ms")
                lat_a = a.get("latency_ms")
                if lat_b or lat_a:
                    print(f"    latency     : {lat_b}ms → {lat_a}ms  "
                          f"mem={b.get('memory_sol_pct')}%→{a.get('memory_sol_pct')}%  "
                          f"sm={b.get('sm_sol_pct')}%→{a.get('sm_sol_pct')}%")
                if e.get("notes"):
                    print(f"    notes       : {e['notes'][:80]}")
                print()
        print("─" * 60)
        print("To merge, call:")
        print('  experience_log.py merge --groups "ID,ID" "ID,ID,ID" ...')
        print("Each quoted group collapses to one entry. Unmentioned IDs are kept as-is.")
        return

    # Phase 2: apply explicit groups
    merged_ids = set()
    result_entries = []

    for group_spec in args.groups:
        raw_ids = [s.strip() for s in group_spec.split(",")]
        try:
            ids = [int(x) for x in raw_ids if x]
        except ValueError:
            sys.exit(f"Invalid group '{group_spec}': IDs must be integers.")

        missing = [i for i in ids if i not in by_id]
        if missing:
            sys.exit(f"Unknown ids {missing} in group '{group_spec}'.")

        overlap = [i for i in ids if i in merged_ids]
        if overlap:
            sys.exit(f"ids {overlap} appear in more than one group.")

        group = [by_id[i] for i in ids]
        merged_ids.update(ids)

        outcomes = set(e.get("outcome") for e in group)
        if len(outcomes) > 1:
            print(f"  ⚠  Group {ids}: outcome conflict {sorted(outcomes)} — "
                  f"will note conflict in merged entry")

        if len(group) == 1:
            result_entries.append(group[0])
            print(f"  Group {ids}: single entry, kept as-is.")
        else:
            canonical = _collapse_group(group)
            result_entries.append(canonical)
            sp = f"  speedup={canonical.get('speedup')}x" if canonical.get("speedup") else ""
            print(f"  Group {ids}: merged → id={canonical['id']}{sp}  "
                  f"merged_count={canonical.get('merged_count', 1)}")
            if canonical.get("notes"):
                print(f"    notes: {canonical['notes'][:80]}")

    kept = [e for e in entries if e["id"] not in merged_ids]
    result_entries.extend(kept)

    result_entries.sort(key=lambda e: e["id"])
    for new_id, e in enumerate(result_entries, start=1):
        e["id"] = new_id

    before = len(entries)
    after = len(result_entries)
    _save(result_entries)
    print(f"\n✓ {before} entries → {after} entries (removed {before - after})")
    print("  Run `experience_log.py sync` to update experiences.md.")


def cmd_sync(_args):
    entries = _load()
    lines = [
        "# Optimization Experience Log\n",
        "\n",
        "Auto-generated by `experience_log.py sync`. Edit via CLI, not directly.\n",
        "\n",
        f"**Total: {len(entries)} entries** | "
        f"Success: {sum(1 for e in entries if e['outcome']=='success')} | "
        f"Failure: {sum(1 for e in entries if e['outcome']=='failure')} | "
        f"Neutral: {sum(1 for e in entries if e['outcome']=='neutral')}\n",
        "\n",
        "---\n",
        "\n",
    ]

    if not entries:
        lines.append("*(no experiences recorded yet — run `experience_log.py add` after each iteration)*\n")
    else:
        by_kernel_backend = {}
        for e in entries:
            key = (e["kernel"], e.get("backend", "cuda"))
            by_kernel_backend.setdefault(key, []).append(e)

        for (op, be), op_entries in sorted(by_kernel_backend.items()):
            successes = [e for e in op_entries if e["outcome"] == "success"]
            failures = [e for e in op_entries if e["outcome"] == "failure"]
            lines.append(f"## {op} [{be}]  ({len(successes)} ✓ / {len(failures)} ✗)\n\n")
            lines.append("| ID | Date | Iter | Chip | Backend | Bottleneck | Speedup | Optimization | Outcome | Notes |\n")
            lines.append("|---|---|---|---|---|---|---|---|---|---|\n")
            for e in op_entries:
                speedup_str = f"{e['speedup']}x" if e.get("speedup") else "—"
                icon = "✓" if e["outcome"] == "success" else ("✗" if e["outcome"] == "failure" else "~")
                opt_short = e.get("optimization", "")
                notes_short = e.get("notes", "")
                mc = f" (×{e['merged_count']})" if e.get("merged_count", 1) > 1 else ""
                backend = e.get("backend", "cuda")
                lines.append(
                    f"| {e['id']} | {e['date']} | {e['iteration']} | {e['chip']} "
                    f"| {backend} | {e['bottleneck_before']} | {speedup_str}{mc} | `{opt_short}` | {icon} | {notes_short} |\n"
                )
            lines.append("\n")

        lines.append("## Lessons learned\n\n")
        for (op, be), op_entries in sorted(by_kernel_backend.items()):
            successes = [e for e in op_entries if e["outcome"] == "success"]
            if not successes:
                continue
            best = max(successes, key=lambda e: e.get("speedup") or 0)
            lines.append(f"**{op}** [{be}]: best optimization `{best['optimization'][:60]}` → {best.get('speedup', '?')}x ({best['chip']})\n\n")

    os.makedirs(os.path.dirname(MD_PATH), exist_ok=True)
    with open(MD_PATH, "w") as f:
        f.writelines(lines)
    print(f"✓ Synced {len(entries)} entries → {MD_PATH}")


def cmd_stats(_args):
    entries = _load()
    if not entries:
        print("(no experiences recorded yet)")
        return

    print(f"Total: {len(entries)} entries")
    print(f"  Success: {sum(1 for e in entries if e['outcome']=='success')}")
    print(f"  Failure: {sum(1 for e in entries if e['outcome']=='failure')}")
    print(f"  Neutral: {sum(1 for e in entries if e['outcome']=='neutral')}")
    merged = [e for e in entries if e.get("merged_count", 1) > 1]
    if merged:
        print(f"  Merged entries: {len(merged)} (representing {sum(e['merged_count'] for e in merged)} original runs)")
    print()

    # By backend
    by_backend = {}
    for e in entries:
        by_backend.setdefault(e.get("backend", "cuda"), []).append(e)
    print("By backend:")
    for be, be_entries in sorted(by_backend.items()):
        s = sum(1 for e in be_entries if e["outcome"] == "success")
        f = sum(1 for e in be_entries if e["outcome"] == "failure")
        print(f"  {be:8} {len(be_entries)} entries — {s} success / {f} failure")
    print()

    by_kernel = {}
    for e in entries:
        by_kernel.setdefault((e["kernel"], e.get("backend", "cuda")), []).append(e)
    print("By kernel + backend:")
    for (op, be), op_entries in sorted(by_kernel.items()):
        s = sum(1 for e in op_entries if e["outcome"] == "success")
        f = sum(1 for e in op_entries if e["outcome"] == "failure")
        speedups = [e["speedup"] for e in op_entries if e.get("speedup") and e["outcome"] == "success"]
        avg_speedup = round(sum(speedups) / len(speedups), 2) if speedups else None
        avg_str = f" avg_speedup={avg_speedup}x" if avg_speedup else ""
        print(f"  {op:15} [{be}] {s} success / {f} failure{avg_str}")

    by_bt = {}
    for e in entries:
        by_bt.setdefault(e["bottleneck_before"], []).append(e)
    print("\nBy bottleneck:")
    for bt, bt_entries in sorted(by_bt.items()):
        s = sum(1 for e in bt_entries if e["outcome"] == "success")
        f = sum(1 for e in bt_entries if e["outcome"] == "failure")
        print(f"  {bt:20} {s} success / {f} failure")

    opt_counts = {}
    for e in entries:
        if e["outcome"] == "success" and e.get("optimization"):
            key = e["optimization"].split("{")[0].strip()
            opt_counts[key] = opt_counts.get(key, 0) + 1
    if opt_counts:
        top = sorted(opt_counts.items(), key=lambda x: -x[1])[:3]
        print("\nTop successful optimizations:")
        for opt_name, count in top:
            print(f"  {count}x  {opt_name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Kernel optimization experience logger — record what worked, retrieve what's relevant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    add_p = sub.add_parser("add", help="Record a new optimization experience")
    add_p.add_argument("--kernel", help="Kernel type (matmul, convolution, reduction, attention, elementwise, …)")
    add_p.add_argument("--backend", choices=["cuda", "triton"], default="cuda",
                       help="Backend framework (cuda or triton, default: cuda)")
    add_p.add_argument("--chip", help="GPU chip (e.g. sm_86, sm_80)")
    add_p.add_argument("--dims", help="Tensor dimensions, free-form (e.g. 'M=1024,K=512,N=512')")
    add_p.add_argument("--iteration", "-i", help="Iteration label (e.g. v1, v2)")
    add_p.add_argument("--bottleneck", help="Bottleneck BEFORE this optimization (memory_bound|compute_bound|latency_bound|occupancy_bound|grid_anomaly)")
    add_p.add_argument("--stall", help="Dominant NCU warp stall (e.g. 'Long Scoreboard')")
    add_p.add_argument("--memory-sol-before", type=float, metavar="PCT", dest="memory_sol_before")
    add_p.add_argument("--sm-sol-before", type=float, metavar="PCT", dest="sm_sol_before")
    add_p.add_argument("--latency-before", type=float, metavar="MS", dest="latency_before")
    add_p.add_argument("--occupancy-before", type=float, metavar="PCT", dest="occupancy_before")
    add_p.add_argument("--memory-sol-after", type=float, metavar="PCT", dest="memory_sol_after")
    add_p.add_argument("--sm-sol-after", type=float, metavar="PCT", dest="sm_sol_after")
    add_p.add_argument("--latency-after", type=float, metavar="MS", dest="latency_after")
    add_p.add_argument("--occupancy-after", type=float, metavar="PCT", dest="occupancy_after")
    add_p.add_argument("--optimization", "-o", dest="optimization", metavar="OPT",
                       help="Optimization strategy applied in this iteration")
    add_p.add_argument("--hypothesis", help="Hypothesis text from hypothesis.txt")
    add_p.add_argument("--outcome", required=True, choices=["success", "failure", "neutral"],
                       help="success (>5%% speedup), failure (regression or <2%%), neutral (2–5%%)")
    add_p.add_argument("--notes", help="Additional notes or observations")

    rec_p = sub.add_parser("recommend", help="Find relevant past experiences for current context")
    rec_p.add_argument("--kernel", help="Current kernel type")
    rec_p.add_argument("--backend", choices=["cuda", "triton"], help="Current backend framework")
    rec_p.add_argument("--chip", help="Current GPU chip")
    rec_p.add_argument("--bottleneck", help="Current diagnosed bottleneck")
    rec_p.add_argument("-n", type=int, default=5, help="Max results to show (default 5)")

    srch_p = sub.add_parser("search", help="Search experiences by keyword")
    srch_p.add_argument("query", help="Keyword to search for")

    list_p = sub.add_parser("list", help="List all recorded experiences")
    list_p.add_argument("--kernel", help="Filter by kernel type")
    list_p.add_argument("--backend", choices=["cuda", "triton"], help="Filter by backend (cuda or triton)")
    list_p.add_argument("--outcome", choices=["success", "failure", "neutral"])
    list_p.add_argument("--bottleneck", help="Filter by bottleneck")

    merge_p = sub.add_parser(
        "merge",
        help="Review all entries (no args) or apply explicit merge groups (--groups)"
    )
    merge_p.add_argument(
        "--groups", nargs="+", metavar="ID,ID,...",
        help='Groups to merge, each as "id1,id2,..." (e.g. --groups "1,3" "2,4,5")'
    )

    sub.add_parser("sync", help="Regenerate experiences.md from JSON")
    sub.add_parser("stats", help="Show per-kernel/bottleneck statistics")

    args = parser.parse_args()

    dispatch = {
        "add": cmd_add,
        "recommend": cmd_recommend,
        "search": cmd_search,
        "list": cmd_list,
        "merge": cmd_merge,
        "sync": cmd_sync,
        "stats": cmd_stats,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
    else:
        fn(args)


if __name__ == "__main__":
    main()
