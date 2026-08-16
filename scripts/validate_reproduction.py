"""Validate the regenerated event tables and the pinned notebook numbers.

Thin CLI over :func:`calcium2p.cohort.event_tables.build_event_tables` and
:mod:`calcium2p.cohort.validation`. Regenerates the event tables from the
converted deltaF traces, grades them against the original reference CSVs, and
prints per-group match reports plus the groups that are structurally not
regenerable from V10 data. Exits nonzero when any group falls below its floor.

Usage::

    uv run python scripts/validate_reproduction.py
    uv run python scripts/validate_reproduction.py --alignment stimulus --floor 0.99
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from calcium2p.cohort.event_tables import GROUP_RECIPES, build_event_tables
from calcium2p.cohort.validation import NOT_REGENERABLE, compare_event_tables
from calcium2p.core.cohort_config import EventTableParams
from calcium2p.core.paths import data_root
from calcium2p.io.legacy import read_reference_events


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", choices=("reward", "stimulus"), default="reward")
    parser.add_argument("--data-subdir", default="manuscript/V10")
    parser.add_argument(
        "--floor",
        type=float,
        default=0.0,
        help="minimum exact-match fraction per graded group (default: report only)",
    )
    return parser.parse_args()


def reference_path(source: Path, group: str, side: str, alignment: str) -> Path:
    """Build the reference CSV path for one (group, side)."""
    if alignment == "reward":
        suffix = "all_2s_reward_events" if side == "target" else "all_2s_noreward_events"
    else:
        suffix = "all_500ms_stim_events" if side == "target" else "all_500ms_nostim_events"
    return source / f"{group}_{suffix}.csv"


def main() -> int:
    """Regenerate, grade, and report."""
    args = parse_args()
    source = data_root(interactive=False) / args.data_subdir
    converted = source.parent / (source.name + "_converted")
    tables = {p.stem: pd.read_parquet(p) for p in converted.glob("*.parquet")}

    regenerated = build_event_tables(tables, EventTableParams(), alignment=args.alignment)

    failures = []
    print(f"=== event-table regeneration vs reference ({args.alignment}) ===")
    for group in GROUP_RECIPES:
        for side in ("target", "reference"):
            ref_path = reference_path(source, group, side, args.alignment)
            if not ref_path.exists():
                print(f"  {group}/{side}: no reference file ({ref_path.name})")
                continue
            reference = read_reference_events(ref_path, group)
            report = compare_event_tables(
                regenerated[group][side], reference, group=group, side=side
            )
            print(f"  {report.describe()}")
            n_ref = max(report.n_reference, 1)
            if report.n_exact / n_ref < args.floor:
                failures.append(f"{group}/{side}")

    print("\nnot regenerable from V10 (deltaF traces never present):")
    for group in NOT_REGENERABLE:
        print(f"  {group}")

    if failures:
        print(f"\nbelow floor {args.floor}: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
