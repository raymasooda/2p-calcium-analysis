"""Hysteresis-threshold calcium event detection and per-event metrics.

Ports of ``detect_events`` and ``get_ROI_event_metrics`` from the legacy
``process_calcium_traces.py``, minus the removed-in-pandas-2 idioms
(``iteritems``/``append``) and the legacy dead code. Semantics are preserved
exactly; see the per-function notes for the quirks that were kept on purpose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import integrate

if TYPE_CHECKING:
    import numpy.typing as npt

#: Metric columns emitted for every detected event, in output order.
EVENT_METRIC_COLUMNS = ("peak", "ev_onset", "peak_time", "ev_offset", "ev_duration", "integral")


def detect_events(
    trace: pd.Series[float],
    upper: float,
    lower: float,
    min_width: int = 6,
) -> list[pd.Series[float]]:
    """Detect events in one trace with a hysteresis threshold.

    The trace is segmented wherever it crosses ``lower``; a segment counts as
    an event when at least ``min_width`` of its samples exceed ``upper``.

    Parameters
    ----------
    trace
        One trial's dF/F0 trace, indexed by frame number.
    upper
        Upper (event-qualifying) threshold.
    lower
        Lower (segmentation) threshold. The legacy caller always passed
        ``upper == lower``, making the hysteresis degenerate; both parameters
        are kept so the mechanism stays available.
    min_width
        Minimum number of samples above ``upper`` for a segment to count.

    Returns
    -------
    list of pandas.Series
        One Series per event, each a slice of ``trace`` (frame-indexed and
        carrying ``trace.name``).
    """
    edges = np.diff(np.r_[0, (trace > lower).astype(int), 0]).nonzero()[0]
    # positional split at the crossing points -- identical to the legacy
    # np.array_split(trace, edges), without its Series.swapaxes deprecation
    cuts = [0, *edges.tolist(), trace.shape[0]]
    segments = [trace.iloc[cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]
    return [segment for segment in segments if int((segment.to_numpy() > upper).sum()) >= min_width]


def event_metrics_table(
    roi_dff: pd.DataFrame,
    fps: float,
    *,
    threshold_sd: float = 2.0,
    min_samples: int = 6,
    baseline_window_s: tuple[float, float] = (1.0, 2.0),
) -> pd.DataFrame:
    """Detect events in every trial of one ROI and tabulate per-event metrics.

    The detection threshold is ``threshold_sd`` times the standard deviation
    of the baseline window, where that deviation is computed over the
    *flattened* window across all trials (one scalar per ROI, exactly as the
    legacy code did). Detection uses :func:`detect_events` with
    ``upper == lower`` (the legacy degenerate hysteresis).

    Parameters
    ----------
    roi_dff
        Trial-by-frame dF/F0 for one ROI. The row index identifies the trial
        (the legacy frames carried a named MultiIndex; plain indexes work
        too). Columns are frame numbers.
    fps
        Imaging frame rate in Hz, used to convert frame indices to seconds.
    threshold_sd
        Threshold in baseline standard deviations. The legacy analysis used 2.
    min_samples
        Minimum samples above threshold per event. The legacy analysis used 6.
    baseline_window_s
        ``(start, stop)`` of the threshold-defining baseline window in
        seconds. The legacy analysis used the 1--2 s window.

    Returns
    -------
    pandas.DataFrame
        One row per detected event: the trial's index-level values (columns
        named from ``roi_dff.index.names``, falling back to ``level_<i>``)
        followed by :data:`EVENT_METRIC_COLUMNS`. Times are in seconds from
        trial start; ``integral`` is Simpson's rule over the event's dF/F0
        samples at unit (frame) spacing, as in the legacy code. Events whose
        onset is frame 0 are excluded (the legacy ``ev_onset > 0`` filter).
        Empty result frames still carry the full column set.
    """
    frames_by_trial = roi_dff.T
    start = round(baseline_window_s[0] * fps)
    stop = round(baseline_window_s[1] * fps)
    baseline: npt.NDArray[np.float64] = np.asarray(
        frames_by_trial.iloc[start:stop], dtype=np.float64
    )
    threshold = float(baseline.std()) * threshold_sd

    identity_names = [
        name if name is not None else f"level_{i}" for i, name in enumerate(roi_dff.index.names)
    ]

    rows: list[list[object]] = []
    for _, trace in frames_by_trial.items():
        for event in detect_events(trace, threshold, threshold, min_width=min_samples):
            frame_labels = event.index.to_numpy()
            onset = float(frame_labels[0]) / fps
            if onset <= 0:
                continue
            values = event.to_numpy(dtype=np.float64)
            identity = list(event.name) if isinstance(event.name, tuple) else [event.name]
            rows.append(
                [
                    *identity,
                    float(values.max()),
                    onset,
                    float(frame_labels[int(values.argmax())]) / fps,
                    float(frame_labels[-1]) / fps,
                    event.shape[0] / fps,
                    float(integrate.simpson(values)),
                ]
            )

    columns = identity_names + list(EVENT_METRIC_COLUMNS)
    return pd.DataFrame(rows, columns=columns)
