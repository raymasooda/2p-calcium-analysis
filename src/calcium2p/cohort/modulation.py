"""Reward/stimulus modulation indices and modulated-ROI selection.

Port of the notebooks' modulation-index cells (reward cell 24, stimulus cell
13). For each ROI with at least one event in the modulation window on the
"target" side (rewarded / stimulus trials):

- if the ROI also has events on the reference side, the modulation index is
  ``(target_rate - reference_rate) / (target_rate + reference_rate)``;
- if it has none, the index is the target *rate itself*, capped at 1 -- a
  preserved notebook quirk (the value is a rate, not an index, in that branch).

ROIs whose only events are on the reference side receive no index and can
therefore never be selected as modulated, exactly as in the notebooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Mapping, Sequence

    from calcium2p.core.cohort_config import ThresholdKind

    RoiKey = Hashable


def events_in_window(events: pd.DataFrame, window: tuple[float, float]) -> pd.DataFrame:
    """Rows whose ``ev_onset`` lies strictly inside ``window``.

    Strict inequalities on both sides match the notebooks' filters.
    """
    onset = events["ev_onset"]
    return events[(onset > window[0]) & (onset < window[1])]


def modulation_index(
    target_events: pd.DataFrame,
    reference_events: pd.DataFrame,
    window: tuple[float, float],
    trial_counts: Callable[[Hashable], tuple[int, int]],
) -> pd.Series:
    """Per-ROI modulation index between target and reference event rates.

    Parameters
    ----------
    target_events, reference_events
        Per-event tables (one row per event, ROI-keyed index, ``ev_onset``
        column). Target is the rewarded/stimulus side; reference the
        unrewarded/no-stimulus side.
    window
        ``(start, stop)`` seconds; events strictly inside count.
    trial_counts
        Callable mapping an ROI key to ``(n_target_trials,
        n_reference_trials)`` -- the denominator logic differs per protocol
        and lives in :mod:`calcium2p.cohort.grouping`.

    Returns
    -------
    pandas.Series
        Modulation index per ROI, indexed by the ROI keys of
        ``target_events``.
    """
    windowed_target = events_in_window(target_events, window)
    windowed_reference = events_in_window(reference_events, window)
    # grouping by the index object itself keeps MultiIndex ROI keys whole
    # tuples, exactly as the notebooks' groupby(df.index) did
    target_counts = windowed_target.groupby(windowed_target.index).size()
    reference_counts = windowed_reference.groupby(windowed_reference.index).size()
    # plain dicts keep tuple-valued ROI keys whole for both lookup styles
    reference_map: dict[Hashable, int] = {k: int(v) for k, v in reference_counts.items()}

    indices: dict[Hashable, float] = {}
    for roi, n_events in target_counts.items():
        n_target_trials, n_reference_trials = trial_counts(roi)
        target_rate = float(n_events) / n_target_trials
        if roi not in reference_map:
            # preserved notebook quirk: the rate itself, capped at 1
            indices[roi] = target_rate if target_rate <= 1 else 1.0
        else:
            reference_rate = reference_map[roi] / n_reference_trials
            indices[roi] = (target_rate - reference_rate) / (target_rate + reference_rate)
    return pd.Series(indices, dtype=float)


def modulated_rois(mi: pd.Series, threshold: ThresholdKind) -> list[Hashable]:
    """ROIs whose modulation index exceeds the population threshold.

    Parameters
    ----------
    mi
        Per-ROI modulation indices for one protocol.
    threshold
        ``mean``, ``median``, or ``half_std`` (0.5 x the population standard
        deviation, the notebooks' third option).
    """
    values = mi.to_numpy(dtype=float)
    cutoffs = {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "half_std": 0.5 * float(np.std(values)),
    }
    cutoff = cutoffs[threshold]
    return [roi for roi, value in mi.items() if value > cutoff]


def modulation_proportions(
    modulated: Mapping[str, Sequence[Hashable]],
    denominators: Mapping[str, Sequence[Hashable]],
) -> pd.DataFrame:
    """Tabulate the "N of M protocol (P%) are modulated" summary the notebooks printed.

    Returns
    -------
    pandas.DataFrame
        Indexed by protocol with columns ``n_modulated``, ``n_total`` and
        ``percent`` (rounded to one decimal, matching the printed output).
    """
    rows = {
        protocol: {
            "n_modulated": len(rois),
            "n_total": len(denominators[protocol]),
            "percent": round(len(rois) / len(denominators[protocol]) * 100, 1),
        }
        for protocol, rois in modulated.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")
