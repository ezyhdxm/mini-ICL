"""
Checkpoint pruning script.

Strategy:
  - Keep ALL checkpoints with step < KEEP_ALL_BELOW
  - Keep only step % PRUNE_INTERVAL == 0 for step >= KEEP_ALL_BELOW
  - ALWAYS keep: model_final_*.pt, checkpoint.pt (optimizer state), all non-.pt files

Usage:
  python prune_checkpoints.py             # dry run — shows savings, deletes nothing
  python prune_checkpoints.py --apply     # actually delete (asks for confirmation first)
  python prune_checkpoints.py --task coin # dry run for one task only
"""

import re
import sys
import argparse
from pathlib import Path

KEEP_ALL_BELOW = 10_000   # keep every checkpoint with step strictly below this
PRUNE_INTERVAL = 1_000    # for step >= KEEP_ALL_BELOW, keep only multiples of this
RESULTS_DIR = Path("results")


def parse_step(filename: str):
    """
    Return the step number for a prunable checkpoint, or None meaning 'always keep'.

    Always keep:  model_final_STEP.pt  (matches model_final_\d+)
                  checkpoint.pt and any other non-model_STEP.pt name
    Prunable:     model_STEP.pt        (matches model_\d+ exactly)
    """
    stem = Path(filename).stem
    # Final checkpoint — always keep
    if re.match(r"^model_final_\d+$", stem):
        return None
    # Regular checkpoint — prunable
    m = re.match(r"^model_(\d+)$", stem)
    if m:
        return int(m.group(1))
    # Anything else (checkpoint.pt, etc.) — always keep
    return None


def should_keep(step: int) -> bool:
    if step < KEEP_ALL_BELOW:
        return True
    return step % PRUNE_INTERVAL == 0


def scan_experiment(exp_path: Path):
    """
    Return (to_keep, to_delete) lists of Path objects for .pt checkpoint files.
    Only plain files matching model_STEP.pt are candidates for deletion.
    Directories are never touched.
    """
    to_keep, to_delete = [], []

    # coin/dyck/latent use a checkpoints/ subdirectory;
    # linear stores model files in the experiment root.
    ckpt_dir = exp_path / "checkpoints"
    if not ckpt_dir.exists():
        ckpt_dir = exp_path

    for f in sorted(ckpt_dir.iterdir()):
        if f.is_dir():          # never touch directories
            continue
        if f.suffix != ".pt":   # only .pt files are considered
            continue
        step = parse_step(f.name)
        if step is None or should_keep(step):
            to_keep.append(f)
        else:
            to_delete.append(f)

    return to_keep, to_delete


def fmt_size(b: int) -> str:
    if b >= 1e9:
        return f"{b / 1e9:.1f} GB"
    return f"{b / 1e6:.0f} MB"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually delete files (default: dry run). Will ask for confirmation.",
    )
    parser.add_argument(
        "--task",
        help="Only process this task, e.g. coin, linear, dyck, latent",
    )
    args = parser.parse_args()

    dry_run = not args.apply

    print(f"Strategy : keep all steps < {KEEP_ALL_BELOW:,}, "
          f"then keep every {PRUNE_INTERVAL:,} steps")
    print(f"Mode     : {'DRY RUN — nothing will be deleted' if dry_run else 'APPLY'}\n")

    # ── Collect tasks ──────────────────────────────────────────────────────────
    tasks = sorted([
        d for d in RESULTS_DIR.iterdir()
        if d.is_dir()
        and any(e.name.startswith("train_") for e in d.iterdir() if e.is_dir())
    ])
    if args.task:
        tasks = [t for t in tasks if t.name == args.task]
        if not tasks:
            print(f"No task named '{args.task}' found in {RESULTS_DIR}/")
            sys.exit(1)

    # ── Scan everything first (before touching a single file) ─────────────────
    plan = {}   # task_name -> list of (exp_path, to_keep, to_delete)
    warnings = []

    for task_dir in tasks:
        exps = sorted([e for e in task_dir.iterdir()
                       if e.is_dir() and e.name.startswith("train_")])
        entries = []
        for exp in exps:
            to_keep, to_delete = scan_experiment(exp)

            # Sanity check: every experiment must keep at least one checkpoint
            if not to_keep:
                warnings.append(f"  WARNING: {exp} would have NO checkpoints left — skipping it")
                to_delete = []   # protect this experiment

            entries.append((exp, to_keep, to_delete))
        plan[task_dir.name] = entries

    # ── Print warnings ─────────────────────────────────────────────────────────
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(w)
        print()

    # ── Summary table ──────────────────────────────────────────────────────────
    grand_before = grand_after = grand_bytes = 0

    for task_name, entries in plan.items():
        t_before = t_after = t_bytes = 0
        for _, to_keep, to_delete in entries:
            t_before += len(to_keep) + len(to_delete)
            t_after  += len(to_keep)
            t_bytes  += sum(f.stat().st_size for f in to_delete)
        grand_before += t_before
        grand_after  += t_after
        grand_bytes  += t_bytes
        verb = "Would delete" if dry_run else "Will delete"
        print(f"  {task_name:<10}  {len(entries)} exps  |  "
              f"{t_before} → {t_after} checkpoints  |  "
              f"{verb}: {t_before - t_after} files  |  "
              f"Save: {fmt_size(t_bytes)}")

    print()
    print(f"  {'TOTAL':<10}  {'':6}  |  "
          f"{grand_before} → {grand_after} checkpoints  |  "
          f"{'Would save' if dry_run else 'Will save'}: {fmt_size(grand_bytes)}")
    print()

    if dry_run:
        print("Run with --apply to delete (you will be asked to confirm first).")
        return

    # ── Confirmation prompt ────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  This will PERMANENTLY DELETE {grand_before - grand_after} files")
    print(f"  freeing {fmt_size(grand_bytes)}. There is NO undo.")
    print("=" * 60)
    answer = input("Type YES to proceed, anything else to cancel: ").strip()
    if answer != "YES":
        print("Cancelled — no files were deleted.")
        sys.exit(0)

    # ── Delete ─────────────────────────────────────────────────────────────────
    print()
    deleted_count = 0
    failed_count  = 0

    for task_name, entries in plan.items():
        for exp, _, to_delete in entries:
            for f in to_delete:
                try:
                    f.unlink()
                    deleted_count += 1
                except OSError as e:
                    print(f"  FAILED to delete {f}: {e}")
                    failed_count += 1

    print(f"\nDeleted {deleted_count} files.")
    if failed_count:
        print(f"WARNING: {failed_count} files could not be deleted (see above).")
    print("\nRe-index to update the experiment browser:")
    print("  ./exp.sh index")


if __name__ == "__main__":
    main()
