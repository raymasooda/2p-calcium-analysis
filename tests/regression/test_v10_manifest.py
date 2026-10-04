"""Regression suite: every reproduced V10 comparison figure, through the package.

Runs the bundled V10 figure manifest (one config per recovered notebook state)
and asserts each figure's statistics equal the values pinned in
``fixtures/v10_figure_stats.json`` -- recorded independently, inside the
verbatim-notebook reproduction, from the objects each original figure was
drawn from.

The dataset location comes from ``DATA_ROOT``; the suite skips when it is not
configured or the V10 inputs are absent (fresh clones, CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from calcium2p.artifacts import ArtifactStore
from calcium2p.cohort.manifest import bundled_manifest, run_figure_manifest
from calcium2p.core.cohort_config import load_figure_manifest
from calcium2p.core.paths import DataRootNotConfiguredError, data_root

if TYPE_CHECKING:
    import pandas as pd

FIXTURE = Path(__file__).parent / "fixtures" / "v10_figure_stats.json"
PINNED: dict[str, dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["figures"]


def _v10_dirs() -> tuple[Path, Path] | None:
    try:
        root = data_root(interactive=False)
    except DataRootNotConfiguredError:
        return None
    source = root / "manuscript" / "V10"
    converted = root / "manuscript" / "V10_converted"
    return (source, converted) if source.is_dir() and converted.is_dir() else None


DIRS = _v10_dirs()

#: Figures whose Mann-Whitney p depends on float rounding: their per-ROI
#: ev_onset means hold values tied on one platform and one ulp (4.4e-16) apart
#: on another (x86-64 vs arm64 summation order), which changes scipy's tie
#: correction. n, means and SEMs still match; p gets this absolute tolerance.
TIE_SENSITIVE_P = {
    "hM4D dendrites pre-CNO cHIT vs. nHIT stim ev_onset.svg": 0.02,
    "hM4D dendrites post-CNO cHIT vs. nHIT stim ev_onset.svg": 0.005,
}

pytestmark = [
    pytest.mark.requires_data,
    pytest.mark.slow,
    pytest.mark.skipif(DIRS is None, reason="V10 dataset not found under DATA_ROOT"),
]


@pytest.fixture(scope="module")
def stats(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Run the whole manifest once, into a throwaway store."""
    assert DIRS is not None
    source, converted = DIRS
    store = ArtifactStore(tmp_path_factory.mktemp("store"))
    configs = load_figure_manifest(bundled_manifest("v10"))
    return run_figure_manifest(configs, store, source_dir=source, converted_dir=converted)


def test_manifest_covers_every_pinned_figure(stats: pd.DataFrame) -> None:
    assert sorted(stats["figure_file"]) == sorted(PINNED)


@pytest.mark.parametrize("figure", sorted(PINNED))
def test_figure_statistics(stats: pd.DataFrame, figure: str) -> None:
    pinned = PINNED[figure]
    row = stats.set_index("figure_file").loc[figure]
    assert row["state"] == pinned["state"]
    if figure in TIE_SENSITIVE_P:
        assert row["p_value"] == pytest.approx(pinned["p"], abs=TIE_SENSITIVE_P[figure])
        assert (row["p_value"] < 0.05) == (pinned["p"] < 0.05)
    else:
        assert row["p_value"] == pytest.approx(pinned["p"], rel=1e-9, abs=1e-15)
    if "n_pairs" in pinned:
        assert int(row["n_a"]) == pinned["n_pairs"]
        return
    assert [int(row["n_a"]), int(row["n_b"])] == pinned["n"]
    assert [row["mean_a"], row["mean_b"]] == pytest.approx(pinned["mean"], rel=1e-9)
    assert [row["sem_a"], row["sem_b"]] == pytest.approx(pinned["sem"], rel=1e-9)


# -- passive groups re-detected by the methods rule ---------------------------

PASSIVE_FIXTURE = Path(__file__).parent / "fixtures" / "v10_passive_methods_stats.json"
PASSIVE_PINNED: dict[str, dict[str, Any]] = json.loads(PASSIVE_FIXTURE.read_text(encoding="utf-8"))[
    "figures"
]


@pytest.fixture(scope="module")
def passive_methods_stats(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """Run the passive-methods manifest once, into a throwaway store."""
    assert DIRS is not None
    source, converted = DIRS
    store = ArtifactStore(tmp_path_factory.mktemp("store-passive"))
    configs = load_figure_manifest(bundled_manifest("v10-passive-methods"))
    return run_figure_manifest(configs, store, source_dir=source, converted_dir=converted)


def test_passive_methods_covers_every_figure(passive_methods_stats: pd.DataFrame) -> None:
    assert sorted(passive_methods_stats["figure_file"]) == sorted(PINNED)
    involves_passive = passive_methods_stats["name"].str.contains("assive", regex=False)
    assert sorted(passive_methods_stats.loc[involves_passive, "figure_file"]) == sorted(
        PASSIVE_PINNED
    )


@pytest.mark.parametrize("figure", sorted(PASSIVE_PINNED))
def test_passive_methods_statistics(passive_methods_stats: pd.DataFrame, figure: str) -> None:
    pinned = PASSIVE_PINNED[figure]
    row = passive_methods_stats.set_index("figure_file").loc[figure]
    assert row["state"] == pinned["state"]
    assert row["p_value"] == pytest.approx(pinned["p"], rel=1e-9, abs=1e-15)
    assert [int(row["n_a"]), int(row["n_b"])] == pinned["n"]
    assert [row["mean_a"], row["mean_b"]] == pytest.approx(pinned["mean"], rel=1e-9)
    assert [row["sem_a"], row["sem_b"]] == pytest.approx(pinned["sem"], rel=1e-9)


def test_passive_methods_leaves_other_figures_unchanged(
    passive_methods_stats: pd.DataFrame, stats: pd.DataFrame
) -> None:
    legacy = stats.set_index("figure_file")
    for row in passive_methods_stats.itertuples():
        if row.figure_file in PASSIVE_PINNED:
            continue
        assert row.p_value == legacy.loc[row.figure_file, "p_value"], row.figure_file
        assert row.mean_a == legacy.loc[row.figure_file, "mean_a"], row.figure_file


# -- Fig 4G: passive-axon lick vs Ca2+ latency regression ---------------------

#: Harness record of the original figure (reward notebook cell 32).
LEGACY_FIG4G = {"r": 0.2615272474698988, "p": 0.00013079040810485187, "n": 209}
#: Same recipe on the methods-rule re-detected passive events.
METHODS_FIG4G = {"r": 0.23407690040157622, "p": 0.0005393073657787408, "n": 215}
FIG4G_STATE = "v10-reward-ev-onset-rate-mannwhitney-modulated"


@pytest.mark.parametrize(
    ("manifest", "expected"),
    [("v10", LEGACY_FIG4G), ("v10-passive-methods", METHODS_FIG4G)],
)
def test_passive_latency_correlation(
    tmp_path: Path, manifest: str, expected: dict[str, float]
) -> None:
    from calcium2p.cohort.latency import run_latency_correlation  # noqa: PLC0415

    assert DIRS is not None
    source, converted = DIRS
    (config,) = [
        c
        for c in load_figure_manifest(bundled_manifest(manifest))
        if c.run_id.startswith(FIG4G_STATE)
    ]
    out = run_latency_correlation(
        config, ArtifactStore(tmp_path), source_dir=source, converted_dir=converted
    )
    result = out["result"]
    assert result.r == pytest.approx(expected["r"], rel=1e-9)
    assert result.p == pytest.approx(expected["p"], rel=1e-9)
    assert result.n_points == expected["n"]
