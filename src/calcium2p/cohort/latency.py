"""Trial-by-trial lick-latency vs Ca2+ event-latency correlation.

Port of the reward notebook's passive-axon regression (cell 32, the
manuscript's Figure 4G panel). Recipe, recovered and reproduced exactly from
the legacy event tables (r = 0.2615, p = 1.31e-4, n = 209):

1. keep reward-side events with onset strictly inside ``window`` (5--5.5 s)
   from the reward-modulated ROIs, and express onset from reward time;
2. average per (session, trial) over the remaining *events* -- an ROI with two
   events counts twice; trials without an event are absent, not zero;
3. z-score both columns over all pooled trials (``ddof=0``);
4. drop trials with ``|z| >= z_clip`` in either column;
5. ``scipy.stats.linregress(lick, calcium)`` -- r and p equal Pearson's.

Lick latency comes from the behaviour tables (``Rew lick latency``), joined on
(session, trial), so re-detected event tables -- which carry no lick columns
-- go through the same code as the reference CSVs (whose own ``Rew latency``
column agrees with the behaviour tables to 1e-16).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.stats import linregress, zscore

from calcium2p.cohort.event_tables import GROUP_RECIPES
from calcium2p.cohort.run import _return_frame, _return_text, run_onset_analysis
from calcium2p.viz.regressions import latency_regression
from calcium2p.viz.render import figure_to_svg

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from pathlib import Path

    from calcium2p.artifacts import ArtifactStore
    from calcium2p.core.cohort_config import CohortConfig

#: The passive protocol the behaviour join and the event windows assume.
_PASSIVE_PROTOCOL = "NAIVE STIM REW"


@dataclass(frozen=True)
class LatencyCorrelation:
    """Result of one lick vs Ca2+ latency regression.

    Attributes
    ----------
    r, p
        Pearson correlation and its two-sided p-value.
    slope, intercept
        Least-squares line of z-scored Ca2+ latency on z-scored lick latency.
    n_points
        Trials in the regression (after the z clip).
    n_trials
        Trials with a windowed event and a lick latency (before the clip).
    n_no_lick
        Trials dropped before z-scoring because they had no lick latency.
    n_sessions
        Sessions contributing at least one regression point.
    """

    r: float
    p: float
    slope: float
    intercept: float
    n_points: int
    n_trials: int
    n_no_lick: int
    n_sessions: int


def reward_lick_latency(tables: Mapping[str, pd.DataFrame], group: str) -> pd.Series:
    """Per-trial reward lick latency of a passive group, keyed like its events.

    Parameters
    ----------
    tables
        Converted legacy tables (needs the group's ``behavior_*`` tables).
    group
        A passive event-table group (``passive_axons``, ``passive_dendrites``).

    Returns
    -------
    pandas.Series
        ``Rew lick latency`` (s, NaN without a lick) indexed by
        ``(session id, trial)``, the session id spelled as in the event index.

    Raises
    ------
    ValueError
        If the group has no behaviour-joined recipe.
    """
    pieces = []
    for recipe in GROUP_RECIPES[group]:
        if recipe.behavior_table is None:
            continue
        behavior = tables[recipe.behavior_table]
        mask = behavior["protocol"].astype(str) == _PASSIVE_PROTOCOL
        if recipe.compartment is not None:
            mask &= behavior["compartment"].astype(str) == recipe.compartment
        sub = behavior[mask]
        session = sub[list(recipe.group_columns)].astype(str).agg("_".join, axis=1)
        pieces.append(
            pd.Series(
                pd.to_numeric(sub["Rew lick latency"], errors="coerce").to_numpy(np.float64),
                index=pd.MultiIndex.from_arrays(
                    [f"{recipe.id_prefix}_" + session, sub["trial"].astype(int)],
                    names=["session", "trial"],
                ),
                name="lick_latency",
            )
        )
    if not pieces:
        raise ValueError(f"group {group!r} has no behaviour-joined recipe")
    return pd.concat(pieces)


def per_trial_latencies(
    events: pd.DataFrame,
    rois: Sequence[Hashable],
    lick_latency: pd.Series,
    *,
    window: tuple[float, float] = (5.0, 5.5),
) -> pd.DataFrame:
    """Per-trial mean Ca2+ event latency (from ``window[0]``) and lick latency.

    Parameters
    ----------
    events
        Reward-side events of one protocol, indexed by ``(session, ROI)`` with
        ``trial`` and ``ev_onset`` columns.
    rois
        ROI keys to keep (the reward-modulated set).
    lick_latency
        From :func:`reward_lick_latency`.
    window
        Onsets kept strictly inside it; latency is measured from its start.

    Returns
    -------
    pandas.DataFrame
        ``ev_onset`` and ``lick_latency`` indexed by ``(session, trial)``;
        trials without a lick latency are kept here (NaN).
    """
    onset = events["ev_onset"].astype(float)
    kept = events[(onset > window[0]) & (onset < window[1]) & events.index.isin(list(rois))]
    session = pd.Index(kept.index.get_level_values(0).astype(str), name="session")
    trial = pd.Index(kept["trial"].astype(int).to_numpy(), name="trial")
    mean_onset = (kept["ev_onset"].astype(float) - window[0]).groupby([session, trial]).mean()
    return pd.DataFrame(
        {
            "ev_onset": mean_onset,
            "lick_latency": lick_latency.reindex(mean_onset.index),
        }
    )


def latency_correlation(
    per_trial: pd.DataFrame, *, z_clip: float = 3.0
) -> tuple[pd.DataFrame, LatencyCorrelation]:
    """Z-score, clip, and regress Ca2+ latency on lick latency.

    Trials without a lick latency are dropped before z-scoring (the
    notebook's data had none; z-scoring a column holding NaN would void it).

    Returns
    -------
    tuple
        The z-scored points used and the :class:`LatencyCorrelation`.
    """
    with_lick = per_trial.dropna(subset=["lick_latency"])
    z = with_lick[["lick_latency", "ev_onset"]].apply(lambda column: zscore(column, ddof=0))
    z = z[(z > -z_clip) & (z < z_clip)].dropna()
    line = linregress(z["lick_latency"].to_numpy(np.float64), z["ev_onset"].to_numpy(np.float64))
    result = LatencyCorrelation(
        r=float(line.rvalue),
        p=float(line.pvalue),
        slope=float(line.slope),
        intercept=float(line.intercept),
        n_points=int(z.shape[0]),
        n_trials=int(with_lick.shape[0]),
        n_no_lick=int(per_trial.shape[0] - with_lick.shape[0]),
        n_sessions=int(z.index.get_level_values("session").nunique()),
    )
    return z, result


def run_latency_correlation(
    config: CohortConfig,
    store: ArtifactStore,
    *,
    group: str = "passive_axons",
    window: tuple[float, float] = (5.0, 5.5),
    z_clip: float = 3.0,
    source_dir: Path | None = None,
    converted_dir: Path | None = None,
) -> dict[str, Any]:
    """Run one state and store its lick vs Ca2+ latency regression.

    The events and the reward-modulated ROI set are those of
    :func:`~calcium2p.cohort.run.run_onset_analysis` for ``config`` (cached
    in the store), so the regression follows the state's event source --
    reference CSVs or re-detected tables -- and its modulation rule.

    Returns
    -------
    dict
        ``result`` (:class:`LatencyCorrelation`), ``points`` (the z-scored
        trials), and the artifact ``keys`` written.

    Raises
    ------
    KeyError
        If the state has no modulated ROI set for ``group``.
    """
    run = run_onset_analysis(config, store, source_dir=source_dir, converted_dir=converted_dir)
    if group not in run["modulated"]:
        raise KeyError(f"state {config.run_id!r} has no modulated ROI set for {group!r}")
    per_trial = per_trial_latencies(
        run["events"]["target"][group],
        run["modulated"][group],
        reward_lick_latency(run["tables"], group),
        window=window,
    )
    points, result = latency_correlation(per_trial, z_clip=z_clip)

    base = f"{config.run_id}/latency_correlation/{group}/{run['mode']}"
    stats_key = f"{base}/stats"
    store.get_or_compute(
        stats_key,
        _return_frame,
        params={"payload": pd.DataFrame([asdict(result)]).to_json()},
        kind="metric",
    )
    figure, _ = latency_regression(
        points["lick_latency"],
        points["ev_onset"],
        line_color="grey",
        point_edgecolor="darkgrey",
    )
    figure_key = f"{base}/figure"
    store.get_or_compute(
        figure_key,
        _return_text,
        params={"payload": figure_to_svg(figure)},
        kind="figure",
        serializer="svg",
    )
    return {"result": result, "points": points, "keys": [stats_key, figure_key]}
