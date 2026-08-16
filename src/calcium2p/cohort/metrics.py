"""Per-(protocol, outcome, ROI) event metrics and named comparison frames.

Port of the notebooks' ``generate_event_metrics`` / ``extract_active_ROIs`` /
``extract_inactive_ROIs`` / ``generate_comparison_dfs`` machinery, with the
positional ``view_comparisons`` columns replaced by a ``(protocol, outcome)``
MultiIndex that named comparison specs address directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from calcium2p.cohort.grouping import TrialCounts
    from calcium2p.core.cohort_config import Alignment, EpochSpec, Metric, NormalizeSpec

#: Metric columns carried through, in the notebooks' order (rate appended).
METRIC_COLUMNS = ("peak", "ev_onset", "peak_time", "ev_offset", "ev_duration", "integral", "rate")

#: Outcome whitelists per protocol family (reward alignment); the stimulus
#: notebooks additionally admit FA for dendrites.
_AXON_OUTCOMES = ("CR", "miss", "cHIT", "nHIT", "HIT", "CATCH")
_DENDRITE_OUTCOMES_REWARD = ("CR", "miss", "cHIT", "nHIT", "HIT", "nCATCH", "cCATCH")
_DENDRITE_OUTCOMES_STIMULUS = ("CR", "miss", "cHIT", "nHIT", "HIT", "nCATCH", "cCATCH", "FA")

#: Trial types split out per protocol, in the notebooks' canonical order.
TTYPE_ORDER_REWARD = ("cHIT", "nHIT", "HIT", "Rew", "cRew", "nRew", "CATCH")
TTYPE_ORDER_STIMULUS = ("cHIT", "nHIT", "HIT", "Stim", "cStim", "nStim", "CATCH")


def _prepare_outcomes(events: pd.DataFrame, protocol: str, alignment: Alignment) -> pd.DataFrame:
    """Remap and whitelist outcomes, and append the pooled pseudo-outcomes."""
    frame = events.copy()
    if protocol in ("task_axons", "NO_DREADDs"):
        frame["outcome"] = frame["outcome"].replace({"cCATCH": "CATCH", "nCATCH": "CATCH"})
        frame = frame[frame["outcome"].isin(_AXON_OUTCOMES)]
    elif "passive" in protocol:
        frame["outcome"] = frame["history"]
    else:
        allowed = (
            _DENDRITE_OUTCOMES_REWARD if alignment == "reward" else _DENDRITE_OUTCOMES_STIMULUS
        )
        frame = frame[frame["outcome"].isin(allowed)]

    pooled_marker = "Rew" if alignment == "reward" else "Stim"
    hit_rows = frame[frame["outcome"].str.contains("HIT")].copy()
    hit_rows["outcome"] = "HIT"
    pooled_rows = frame[frame["outcome"].str.contains(pooled_marker)].copy()
    pooled_rows["outcome"] = pooled_marker
    return pd.concat([frame, hit_rows, pooled_rows])


def event_metrics_by_roi(
    target_events: Mapping[str, pd.DataFrame],
    reference_events: Mapping[str, pd.DataFrame],
    epoch: EpochSpec,
    counts: TrialCounts,
    denominators: Mapping[str, Sequence[Hashable]],
    *,
    roi_set: str = "active",
    modulated: Mapping[str, Sequence[Hashable]] | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Windowed per-(outcome, ROI) event metrics for every protocol.

    Parameters
    ----------
    target_events, reference_events
        Pooled per-event tables by protocol (rewarded/unrewarded or
        stim/no-stim sides), from
        :func:`calcium2p.cohort.grouping.load_grouped_events`.
    epoch
        The analysis window (per-protocol overrides honoured).
    counts
        Trial-count lookups providing the rate denominators.
    denominators
        Total-ROI lists per protocol (used for inactive-ROI fill when
        ``roi_set == "all"``).
    roi_set
        ``"active"`` keeps only ROIs with >=1 windowed event (every recorded
        notebook state); ``"all"`` zero/NaN-fills inactive ROIs;
        ``"modulated"`` further restricts to the ``modulated`` lists.
    modulated
        Modulated-ROI lists per protocol, required when
        ``roi_set == "modulated"``.

    Returns
    -------
    dict
        ``protocol -> outcome -> DataFrame`` with a ``(outcome, roi)``
        MultiIndex row per active ROI and :data:`METRIC_COLUMNS` columns.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}
    ttype_order = TTYPE_ORDER_REWARD if counts.alignment == "reward" else TTYPE_ORDER_STIMULUS

    for side in (target_events, reference_events):
        for protocol, raw in side.items():
            window = epoch.window_for(protocol)
            frame = _prepare_outcomes(raw, protocol, counts.alignment)
            frame = frame[(frame["ev_onset"] > window[0]) & (frame["ev_onset"] < window[1])]

            grouped = frame.groupby([frame["outcome"], frame.index])
            metric_means = grouped.mean(numeric_only=True)
            event_counts = grouped.size()

            # dividing on the groupby's own index keeps alignment exact
            denominator = pd.Series(
                [
                    float(counts.rate_denominator(protocol, key[1], str(key[0])))
                    for key in event_counts.index
                ],
                index=event_counts.index,
                dtype=float,
            )
            rate_series = (event_counts / denominator).rename("rate")

            if roi_set == "all":
                rate_series = _fill_inactive(rate_series, denominators[protocol], protocol, counts)

            full = pd.concat([metric_means, rate_series.to_frame("rate")], axis=1)
            full = full[[c for c in METRIC_COLUMNS if c in full.columns]]

            if roi_set == "modulated" and modulated is not None and protocol in modulated:
                keep = set(modulated[protocol])
                full = full[[key[1] in keep for key in full.index]]

            for ttype in ttype_order:
                ttype_rows = full[[key[0] == ttype for key in full.index]]
                if ttype_rows.shape[0] > 0:
                    result.setdefault(protocol, {})[ttype] = ttype_rows
    return result


def _fill_inactive(
    rates: pd.Series,
    all_rois: Sequence[Hashable],
    protocol: str,
    counts: TrialCounts,
) -> pd.Series:
    """Zero/NaN-fill rates for ROIs with no windowed events.

    Port of ``extract_inactive_ROIs``: an inactive ROI gets rate 0 when its
    session has trials of that outcome, else NaN. (No recorded notebook state
    ran this path -- every run used ``active_ROIs=True`` -- so this is a
    corrected-semantics port, kept for completeness.)
    """
    outcomes = {key[0] for key in rates.index}
    filled: dict[Hashable, float] = dict(rates.items())
    for roi in all_rois:
        # task_axons denominators are 1-tuples ('muji_FOV2_66',); their event
        # keys carry the '.0' float suffix the CSVs use
        roi_key: Hashable = (
            f"{roi[0]}.0" if protocol == "task_axons" and isinstance(roi, tuple) else roi
        )
        for outcome in outcomes:
            key = (outcome, roi_key)
            if key in filled:
                continue
            try:
                n_trials = counts.rate_denominator(protocol, roi_key, str(outcome))
            except KeyError:
                n_trials = 0
            filled[key] = 0.0 if n_trials > 0 else np.nan
    return pd.Series(filled, dtype=float)


def comparison_frame(
    event_metrics: Mapping[str, Mapping[str, pd.DataFrame]], metric: Metric
) -> pd.DataFrame:
    """One metric value per ROI, columned by ``(protocol, outcome)``.

    Port of ``generate_comparison_dfs``: within each protocol/outcome the
    per-(outcome, ROI) rows are averaged per ROI (idempotent -- one row per
    ROI exists -- but kept for fidelity), then assembled into a frame whose
    columns named comparison specs address directly.
    """
    pieces: dict[tuple[str, str], pd.Series] = {}
    for protocol, per_ttype in event_metrics.items():
        for ttype, frame in per_ttype.items():
            series = frame[metric]
            rois = [key[1] for key in series.index]
            per_roi = series.groupby(pd.Index(rois)).mean()
            pieces[(protocol, ttype)] = per_roi
    out = pd.concat(pieces, axis=1)
    out.columns = pd.MultiIndex.from_tuples(list(pieces), names=["protocol", "outcome"])
    return out


def normalize_by_reference(
    frame: pd.DataFrame, spec_a: tuple[str, str], spec_b: tuple[str, str], normalize: NormalizeSpec
) -> pd.DataFrame:
    """Divide both comparison sides by their reference outcome's mean.

    Port of the notebooks' "non-consecutive-normalized" construction (reward
    cell 30): side ``a`` divided by ``mean(a_reference)``, side ``b`` by
    ``mean(b_reference)``. Column labels are preserved so the same
    :class:`~calcium2p.core.cohort_config.ComparisonSpec` addresses the
    normalized frame.
    """
    norm_a = frame[spec_a] / frame[normalize.a_reference].mean()
    norm_b = frame[spec_b] / frame[normalize.b_reference].mean()
    out = pd.concat({spec_a: norm_a, spec_b: norm_b}, axis=1)
    out.columns = pd.MultiIndex.from_tuples([spec_a, spec_b], names=["protocol", "outcome"])
    return out
