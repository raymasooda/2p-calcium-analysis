"""Run a cohort onset-histogram analysis from a YAML config.

Thin CLI over :func:`calcium2p.cohort.run.run_onset_analysis`. All logic lives
in the package.

Usage::

    uv run python scripts/run_onset_analysis.py --config configs/onset_histograms.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from calcium2p.artifacts import ArtifactStore
from calcium2p.cohort.run import run_onset_analysis
from calcium2p.core.cohort_config import load_cohort_config
from calcium2p.core.paths import derivatives_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="cohort-analysis YAML")
    parser.add_argument(
        "--source-dir", type=Path, default=None, help="override the reference-CSV directory"
    )
    parser.add_argument(
        "--converted-dir", type=Path, default=None, help="override the converted-parquet directory"
    )
    return parser.parse_args()


def main() -> int:
    """Run the analysis and print the headline tables."""
    args = parse_args()
    config = load_cohort_config(args.config)
    store = ArtifactStore(derivatives_root(interactive=False))
    out = run_onset_analysis(
        config, store, source_dir=args.source_dir, converted_dir=args.converted_dir
    )
    print(f"run_id: {config.run_id} (legacy_faithful={config.legacy_faithful})")
    print("\nmodulated-ROI proportions:")
    print(out["modulation"].to_string())
    print("\ncomparisons:")
    print(out["stats"].round(4).to_string())
    print(f"\n{len(out['keys'])} artifact keys written under {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
