"""Reproduce a manuscript's comparison figures from its figure manifest.

Thin CLI over :func:`calcium2p.cohort.manifest.run_figure_manifest`. All logic
lives in the package. Defaults to the bundled V10 manifest.

Usage::

    uv run python scripts/run_figure_manifest.py                 # bundled v10
    uv run python scripts/run_figure_manifest.py --manifest path/to/manifest.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from calcium2p.artifacts import ArtifactStore
from calcium2p.cohort.manifest import BUNDLED_MANIFESTS, bundled_manifest, run_figure_manifest
from calcium2p.core.cohort_config import load_figure_manifest
from calcium2p.core.paths import derivatives_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--manifest", type=Path, help="figure-manifest YAML")
    source.add_argument(
        "--bundled",
        default="v10",
        choices=sorted(BUNDLED_MANIFESTS),
        help="name of a manifest shipped with the package (default: v10)",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=None, help="override the reference-CSV directory"
    )
    parser.add_argument(
        "--converted-dir", type=Path, default=None, help="override the converted-parquet directory"
    )
    return parser.parse_args()


def main() -> int:
    """Run every state and print one line per figure."""
    args = parse_args()
    path = args.manifest if args.manifest is not None else bundled_manifest(args.bundled)
    configs = load_figure_manifest(path)
    store = ArtifactStore(derivatives_root(interactive=False))
    stats = run_figure_manifest(
        configs, store, source_dir=args.source_dir, converted_dir=args.converted_dir
    )
    columns = ["state", "figure_file", "test", "n_a", "n_b", "mean_a", "mean_b", "p_value"]
    print(stats[columns].round(4).to_string(index=False))
    print(f"\n{len(stats)} figures from {len(configs)} states; figures under {store.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
