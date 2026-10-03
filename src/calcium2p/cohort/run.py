"""Orchestrate one cohort onset-histogram analysis run through the store.

Every output flows through :meth:`ArtifactStore.get_or_compute`, so caching,
the materialization policy, and provenance sidecars all apply. Compute
functions are module-level (the store rejects closures) and fall into two
groups:

- **ingest boundary** functions read declared external inputs (the converted
  parquet tables, the reference CSVs); their parameters carry the source paths
  and checksums so the provenance record names exactly what was read;
- **pure** functions transform previously stored artifacts.

``PipelineStage`` subclasses are deliberately not used here: the cohort
analysis is a fan-out over comparisons rather than a linear per-session chain,
and wrapping each step in a class whose only job is ``run()`` is the exact
shape the repo's OOP rule forbids.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from typing import TYPE_CHECKING, Any

import pandas as pd

from calcium2p.cohort.grouping import (
    TrialCounts,
    build_roi_denominators,
    build_trial_counts,
    load_grouped_events,
)
from calcium2p.cohort.metrics import comparison_frame, event_metrics_by_roi, normalize_by_reference
from calcium2p.cohort.modulation import modulated_rois, modulation_index, modulation_proportions
from calcium2p.cohort.stats import run_comparison
from calcium2p.core.cohort_config import CohortConfig, ComparisonSpec
from calcium2p.core.paths import data_root
from calcium2p.io.legacy import read_reference_events, read_reference_frequencies
from calcium2p.provenance.provenance_utils import compute_checksum
from calcium2p.viz.comparisons import comparison_bar, comparison_colors
from calcium2p.viz.onset_histograms import onset_histogram_grid
from calcium2p.viz.render import figure_to_svg

if TYPE_CHECKING:
    from pathlib import Path

    from calcium2p.artifacts import ArtifactStore

#: Converted-table stems the analysis consumes.
CONVERTED_TABLES = (
    "trials_task_axons",
    "trials_no_dreadds",
    "behavior_batch1",
    "behavior_batch3",
    "behavior_task_dendrites",
    "behavior_control_dendrites",
    "traces_batch1",
    "traces_batch3",
    "traces_task_axons",
    "traces_no_dreadds",
)


# -- ingest-boundary compute functions ----------------------------------------


def load_converted_table(*, path: str, sha256: str) -> pd.DataFrame:
    """Load one converted parquet table (ingest boundary).

    The checksum parameter is not used at runtime; it lands in the mutation
    and provenance records so the artifact names the exact bytes it came from.
    """
    del sha256
    return pd.read_parquet(path)


def pool_reference_events(
    *,
    source_dir: str,
    suffix: str,
    active_only: bool,
    exclude: tuple[str, ...],
    legacy_faithful: bool,
) -> pd.DataFrame:
    """Pool one side's reference event CSVs into a single frame.

    Ingest boundary: reads every ``*{suffix}.csv`` under ``source_dir``. The
    per-protocol frames are concatenated with ``protocol`` and ``roi_key``
    columns so the store can hold one artifact; downstream code re-splits.
    """
    from pathlib import Path  # noqa: PLC0415 - keep module import surface tiny

    files: dict[str, pd.DataFrame] = {}
    for path in sorted(Path(source_dir).glob(f"*{suffix}.csv")):
        stem = path.name.replace(f"{suffix}.csv", "")
        title = stem.split("MODIFIED_")[-1] if "MODIFIED" in stem else stem
        files[stem] = read_reference_events(path, title)
    events, _ = load_grouped_events(
        files, active_only=active_only, exclude=exclude, legacy_faithful=legacy_faithful
    )
    combined = []
    for protocol, frame in events.items():
        piece = frame.reset_index(names=[f"roi_level_{i}" for i in range(frame.index.nlevels)])
        # dendrite ROI levels mix floats (57.0) and strings ('42.0_TD3'), which
        # parquet rejects; canonicalize to strings (the frequency tables get
        # the same cast, so key identity stays consistent end to end)
        for column in piece.columns:
            if str(column).startswith("roi_level_"):
                piece[column] = piece[column].astype(str)
        piece.insert(0, "protocol", protocol)
        # protocols differ in index depth; record it so the split can undo
        # the union-concat without inventing phantom levels
        piece.insert(1, "roi_nlevels", frame.index.nlevels)
        combined.append(piece)
    return pd.concat(combined, ignore_index=True)


def load_frequency_tables(*, source_dir: str, pattern: str) -> pd.DataFrame:
    """Pool the frequency CSVs (denominator inputs) into one long frame."""
    from pathlib import Path  # noqa: PLC0415

    pieces = []
    for path in sorted(Path(source_dir).glob(pattern)):
        frame = read_reference_frequencies(path).reset_index(names=["level_0", "level_1"])
        # same string canonicalization as pool_reference_events
        frame["level_0"] = frame["level_0"].astype(str)
        frame["level_1"] = frame["level_1"].astype(str)
        frame.insert(0, "stem", path.stem)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True)


# -- reconstruction helpers ----------------------------------------------------


def _split_events(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Undo :func:`pool_reference_events`'s flattening."""
    out: dict[str, pd.DataFrame] = {}
    all_roi_levels = [c for c in combined.columns if c.startswith("roi_level_")]
    for protocol, piece in combined.groupby("protocol", sort=False):
        nlevels = int(piece["roi_nlevels"].iloc[0])
        own_levels = [f"roi_level_{i}" for i in range(nlevels)]
        spare = [c for c in all_roi_levels if c not in own_levels]
        frame = piece.drop(columns=["protocol", "roi_nlevels", *spare])
        if nlevels == 1:
            frame = frame.set_index(own_levels[0])
            frame.index.name = None
        else:
            frame = frame.set_index(own_levels)
        out[str(protocol)] = frame
    return out


def _split_frequency_tables(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Undo :func:`load_frequency_tables`'s flattening."""
    out: dict[str, pd.DataFrame] = {}
    for stem, piece in combined.groupby("stem", sort=False):
        out[str(stem)] = piece.drop(columns=["stem"]).set_index(["level_0", "level_1"])
    return out


# -- figure naming ---------------------------------------------------------------

_NORMALIZED_PREFIX = "non-consecutive-normalized "


def figure_filename(spec: ComparisonSpec) -> str:
    """Return the notebooks' saved filename for a comparison figure.

    ``f"{title} {epoch} {metric}.svg"``, with the ``non-consecutive-normalized``
    prefix for normalized comparisons -- the name the original artifact carries,
    so package output can be matched to it.
    """
    prefix = _NORMALIZED_PREFIX if spec.normalize is not None else ""
    return f"{prefix}{spec.name} {spec.epoch} {spec.metric}.svg"


def figure_key(run_id: str, spec: ComparisonSpec, mode: str) -> str:
    """Return the artifact-store key of a comparison figure.

    The normalization prefix is part of the key: a normalized comparison
    shares its name and metric with the plain one, and a shared key would make
    the store serve the cached plain figure in place of the normalized one.
    """
    prefix = _NORMALIZED_PREFIX if spec.normalize is not None else ""
    slug = f"{prefix}{spec.name}".replace(" ", "_").replace(".", "").replace("/", "-")
    return f"{run_id}/figures/{slug}/{spec.metric}/{mode}"


# -- the orchestrator ----------------------------------------------------------


def run_onset_analysis(  # noqa: PLR0915 - the linear orchestration reads best unsplit
    config: CohortConfig,
    store: ArtifactStore,
    *,
    source_dir: Path | None = None,
    converted_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the configured analysis end to end, storing every output.

    Parameters
    ----------
    config
        The run configuration.
    store
        Artifact store receiving every output (figures, metric tables,
        reports) under keys prefixed with ``config.run_id``.
    source_dir
        Directory holding the reference CSVs; defaults to
        ``DATA_ROOT/<config.data_subdir>``.
    converted_dir
        Directory holding the converted parquet tables; defaults to
        ``<source_dir>_converted``.

    Returns
    -------
    dict
        Key results: trial counts, modulation table, per-comparison stats,
        and the artifact keys written.
    """
    base = data_root() / config.data_subdir
    source = source_dir if source_dir is not None else base
    converted = (
        converted_dir if converted_dir is not None else base.parent / (base.name + "_converted")
    )
    flag = config.legacy_faithful
    mode = f"legacy-{str(flag).lower()}"
    keys_written: list[str] = []

    tables: dict[str, pd.DataFrame] = {}
    for stem in CONVERTED_TABLES:
        path = converted / f"{stem}.parquet"
        key = f"legacy/{stem}"
        tables[stem] = store.get_or_compute(
            key,
            load_converted_table,
            params={"path": str(path), "sha256": compute_checksum(path) or "missing"},
            note="converted legacy table (ingest boundary)",
        )
        keys_written.append(key)

    if config.alignment == "reward":
        suffixes = ("_all_2s_reward_events", "_all_2s_noreward_events")
        freq_pattern = "*fullreward_event_frequencies.csv"
    else:
        suffixes = ("_all_500ms_stim_events", "_all_500ms_nostim_events")
        freq_pattern = "*500ms_stim_event_frequencies.csv"

    sides: dict[str, dict[str, pd.DataFrame]] = {}
    for side, suffix in zip(("target", "reference"), suffixes, strict=True):
        key = f"{config.run_id}/events/{side}/{mode}"
        combined = store.get_or_compute(
            key,
            pool_reference_events,
            params={
                "source_dir": str(source),
                "suffix": suffix,
                "active_only": config.histogram.active_only,
                "exclude": ("TD1",),
                "legacy_faithful": flag,
            },
            note="pooled reference event tables (ingest boundary)",
        )
        sides[side] = _split_events(combined)
        keys_written.append(key)

    freq_key = f"{config.run_id}/frequencies"
    freq_combined = store.get_or_compute(
        freq_key,
        load_frequency_tables,
        params={"source_dir": str(source), "pattern": freq_pattern},
        note="frequency tables (ingest boundary)",
    )
    keys_written.append(freq_key)

    counts = build_trial_counts(tables, alignment=config.alignment, legacy_faithful=flag)
    denominators = build_roi_denominators(
        _split_frequency_tables(freq_combined), tables, legacy_faithful=flag
    )

    modulation_epoch = config.epoch(config.modulation.epoch)
    modulated: dict[str, list[Any]] = {}
    for protocol in sides["target"]:
        if protocol not in sides["reference"] or protocol not in denominators:
            continue
        lookup = partial(_modulation_lookup, counts, protocol)
        mi = modulation_index(
            sides["target"][protocol],
            sides["reference"][protocol],
            modulation_epoch.window_for(protocol),
            lookup,
        )
        if not mi.empty:
            modulated[protocol] = modulated_rois(mi, config.modulation.threshold)
    proportions = modulation_proportions(
        {p: rois for p, rois in modulated.items() if p in denominators}, denominators
    )
    prop_key = f"{config.run_id}/modulation/{mode}_thr-{config.modulation.threshold}"
    store.get_or_compute(
        prop_key, _return_frame, params={"payload": proportions.to_json()}, kind="metric"
    )
    keys_written.append(prop_key)

    results: list[dict[str, Any]] = []
    for spec in config.comparisons:
        epoch = config.epoch(spec.epoch)
        metrics = event_metrics_by_roi(
            sides["target"],
            sides["reference"],
            epoch,
            counts,
            denominators,
            roi_set=spec.roi_set,
            modulated=modulated if spec.roi_set == "modulated" else None,
        )
        frame = comparison_frame(metrics, spec.metric)
        if spec.normalize is not None:
            frame = normalize_by_reference(frame, spec.a, spec.b, spec.normalize)
        result = run_comparison(frame, spec)
        fig_key = figure_key(config.run_id, spec, mode)
        results.append(
            {
                "name": spec.name,
                "metric": spec.metric,
                "epoch": spec.epoch,
                "test": spec.test,
                **{k: v for k, v in asdict(result).items() if k != "name"},
                "figure_file": figure_filename(spec),
                "figure_key": fig_key,
            }
        )

        edgecolor, facecolor = comparison_colors(spec.name)
        prefix = _NORMALIZED_PREFIX if spec.normalize is not None else ""
        title = f"{prefix}{spec.name} {spec.epoch} ({result.significance_symbol})"
        figure = comparison_bar(
            frame[spec.a],
            frame[spec.b],
            title=title,
            figure=spec.figure,
            edgecolor=edgecolor,
            facecolor=facecolor,
        )
        svg = figure_to_svg(figure)
        store.get_or_compute(
            fig_key, _return_text, params={"payload": svg}, kind="figure", serializer="svg"
        )
        keys_written.append(fig_key)

    stats_frame = pd.DataFrame(results)
    stats_key = f"{config.run_id}/stats/{mode}"
    store.get_or_compute(
        stats_key, _return_frame, params={"payload": stats_frame.to_json()}, kind="metric"
    )
    keys_written.append(stats_key)

    # the notebooks weighted each side by its own active-ROI count; the grid
    # here uses the target side's counts for both -- a documented deviation
    # (the byte-faithful grid is produced by the reproduction harness)
    grid = onset_histogram_grid(
        sides["target"],
        sides["reference"],
        target_trials=counts.target_totals,
        reference_trials=counts.reference_totals,
        roi_counts={p: int(f.index.nunique()) for p, f in sides["target"].items()},
        spec=config.histogram,
        align_at=5.0 if config.alignment == "reward" else 3.0,
    )
    grid_key = f"{config.run_id}/figures/onset_histogram_grid/{mode}"
    store.get_or_compute(
        grid_key,
        _return_text,
        params={"payload": figure_to_svg(grid)},
        kind="figure",
        serializer="svg",
    )
    keys_written.append(grid_key)

    return {
        "trial_counts": counts,
        "modulation": proportions,
        "stats": stats_frame,
        "keys": keys_written,
    }


def _modulation_lookup(counts: TrialCounts, protocol: str, roi_key: Any) -> tuple[int, int]:
    """Per-protocol (target, reference) trial counts, for use with ``partial``."""
    return counts.modulation_denominators(protocol, roi_key)


def _return_frame(*, payload: str) -> pd.DataFrame:
    """Materialize an already-computed frame passed as JSON params.

    The computation happened upstream in :func:`run_onset_analysis`; this
    function exists so the result can flow through the store with the run's
    parameters recorded. Always-materialized kinds (metric/report) never
    replay it.
    """
    import io  # noqa: PLC0415

    return pd.read_json(io.StringIO(payload))


def _return_text(*, payload: str) -> str:
    """Materialize already-rendered SVG markup (see :func:`_return_frame`)."""
    return payload
