#!/usr/bin/env python3
"""
Clean experiment-generated files and folders in the repository root.

Removes (by default):
- ckpt/
- runs/
- saved_models/
- test_output.txt
- files matching "comparison*"

Supports `--dry-run` to list what would be removed and `-y/--yes` to skip confirmation.
"""
from pathlib import Path
import shutil
import argparse
import sys


DEFAULT_DIRS = ["ckpt", "runs", "saved_models"]
DEFAULT_FILES = ["test_output.txt"]
DEFAULT_PATTERNS = ["comparison*"]


def find_targets(root: Path, extra_patterns=None):
    extra_patterns = extra_patterns or []
    targets = []
    # directories
    for d in DEFAULT_DIRS:
        p = root / d
        if p.exists():
            targets.append(p)

    # specific files
    for f in DEFAULT_FILES:
        p = root / f
        if p.exists():
            targets.append(p)

    # glob patterns
    for pat in DEFAULT_PATTERNS + extra_patterns:
        for p in root.glob(pat):
            if p.exists():
                targets.append(p)

    # deduplicate and sort
    unique = []
    seen = set()
    for p in targets:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return sorted(unique, key=lambda p: p.as_posix())


def remove_path(p: Path, dry_run: bool):
    if dry_run:
        print("[dry-run] Would remove:", p)
        return
    try:
        if p.is_dir():
            shutil.rmtree(p)
            print("Removed directory:", p)
        else:
            p.unlink()
            print("Removed file:", p)
    except Exception as e:
        print(f"Failed to remove {p}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Clean experiment-generated files in repo root")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed")
    parser.add_argument("-y", "--yes", action="store_true", help="Don't prompt; perform removals")
    parser.add_argument("--extra", nargs="*", default=[], help="Extra glob patterns to remove (relative to repo root)")
    args = parser.parse_args()

    # assume this script lives in scripts/ at repo_root/scripts/clean_experiments.py
    repo_root = Path(__file__).resolve().parent.parent
    if not repo_root.exists():
        repo_root = Path.cwd()

    targets = find_targets(repo_root, extra_patterns=args.extra)

    if not targets:
        print("No experiment-generated files or directories found to remove.")
        return 0

    print("Repository root:", repo_root)
    print("Found the following targets to remove:")
    for p in targets:
        try:
            print(" -", p.relative_to(repo_root))
        except Exception:
            print(" -", p)

    if args.dry_run:
        print("\nDry-run mode; no changes made.")
        return 0

    if not args.yes:
        ans = input("Proceed to delete the items listed above? [y/N]: ").strip().lower()
        if ans != "y" and ans != "yes":
            print("Aborted by user.")
            return 1

    for p in targets:
        remove_path(p, dry_run=False)

    print("Cleanup complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
