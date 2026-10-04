"""Lick vs Ca2+ latency regression (manuscript Fig 4G) for one manifest state.

Thin CLI over :func:`calcium2p.cohort.latency.run_latency_correlation`. All
logic lives in the package. The passive-axon panel's ROI selection is the
reward-modulated set, identical across the V10 reward states; the default
state is the one the passive rate/latency figures come from.

Usage::

    uv run python scripts/run_latency_correlation.py                      # legacy events
    uv run python scripts/run_latency_correlation.py --bundled v10-passive-methods
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from calcium2p.artifacts import ArtifactStore
from calcium2p.cohort.latency import run_latency_correlation
from calcium2p.cohort.manifest import BUNDLED_MANIFESTS, bundled_manifest
from calcium2p.core.cohort_config import load_figure_manifest
from calcium2p.core.paths import derivatives_root

DEFAULT_STATE = "v10-reward-ev-onset-rate-mannwhitney-modulated"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundled", default="v10", choices=sorted(BUNDLED_MANIFESTS))
    parser.add_argument(
        "--state", default=DEFAULT_STATE, help="state run_id, without the manifest's suffix"
    )
    parser.add_argument("--group", default="passive_axons")
    return parser.parse_args()


def main() -> int:
    """Run the state and print the regression."""
    args = parse_args()
    configs = load_figure_manifest(bundled_manifest(args.bundled))
    matches = [c for c in configs if c.run_id.startswith(args.state)]
    if len(matches) != 1:
        raise SystemExit(f"state {args.state!r} matches {[c.run_id for c in matches]}")
    store = ArtifactStore(derivatives_root(interactive=False))
    out = run_latency_correlation(matches[0], store, group=args.group)
    result = out["result"]
    print(
        f"{matches[0].run_id} {args.group}: r={result.r:.4f} p={result.p:.3g} "
        f"n={result.n_points} trials ({result.n_trials} before the z clip, "
        f"{result.n_sessions} sessions)"
    )
    print("\n".join(f"  {key}" for key in out["keys"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
