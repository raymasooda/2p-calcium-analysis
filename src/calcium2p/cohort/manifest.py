"""Run a figure manifest: every published comparison figure, state by state.

A manuscript's comparison figures were not produced by one notebook run. The
notebooks were re-run with different switches between saves (modulated vs
active ROIs, the statistical test, the passive-dendrite window), so each
figure belongs to a *state*. A figure manifest records that grouping, one
:class:`~calcium2p.core.cohort_config.CohortConfig` per state (see
:func:`~calcium2p.core.cohort_config.expand_figure_manifest`), and this module
runs them all through the artifact store.

Manifests recovered for published manuscripts ship inside the package (see
:data:`BUNDLED_MANIFESTS`), so the recipe survives a fresh clone.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from calcium2p.cohort.run import run_onset_analysis

if TYPE_CHECKING:
    from collections.abc import Iterable

    from calcium2p.artifacts import ArtifactStore
    from calcium2p.core.cohort_config import CohortConfig

#: Manifests shipped with the package: name -> file under ``cohort/manifests/``.
BUNDLED_MANIFESTS: dict[str, str] = {"v10": "v10_figures.yaml"}


def bundled_manifest(name: str = "v10") -> Path:
    """Return the path of a manifest shipped with the package.

    Raises
    ------
    KeyError
        If ``name`` is not in :data:`BUNDLED_MANIFESTS`.
    """
    if name not in BUNDLED_MANIFESTS:
        raise KeyError(f"no bundled manifest {name!r}; available: {sorted(BUNDLED_MANIFESTS)}")
    return Path(str(files("calcium2p.cohort").joinpath("manifests", BUNDLED_MANIFESTS[name])))


def run_figure_manifest(
    configs: Iterable[CohortConfig],
    store: ArtifactStore,
    *,
    source_dir: Path | None = None,
    converted_dir: Path | None = None,
) -> pd.DataFrame:
    """Run every state of a manifest and collect the per-figure statistics.

    Parameters
    ----------
    configs
        One config per notebook state, e.g. from
        :func:`~calcium2p.core.cohort_config.load_figure_manifest`.
    store
        Artifact store receiving every output.
    source_dir, converted_dir
        Passed through to :func:`~calcium2p.cohort.run.run_onset_analysis`.

    Returns
    -------
    pandas.DataFrame
        One row per comparison figure: the state's ``run_id``, the figure's
        original filename (``figure_file``) and store key, and its statistics.
    """
    frames = []
    for config in configs:
        out = run_onset_analysis(config, store, source_dir=source_dir, converted_dir=converted_dir)
        stats = out["stats"].copy()
        stats.insert(0, "state", config.run_id)
        frames.append(stats)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
