"""dF/F0 computation with median baseline and rolling-median smoothing.

Port of ``calculate_deltaF`` from the legacy ``process_calcium_traces.py``.
The baseline for each trial is the median fluorescence inside a pre-stimulus
window; the per-trial baselines are then smoothed with a 20-sample centred
rolling median across trials (the legacy call ``smooth(y, 20)``, overriding
that helper's ``n=10`` default) before normalising.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Rolling window (in trials) for smoothing the per-trial F0 series. The
#: legacy code hardcoded ``smooth(y, 20)``; kept as a constant, not a
#: parameter, so regenerated outputs stay comparable.
F0_SMOOTHING_WINDOW = 20


def compute_dff(
    roi_traces: pd.DataFrame,
    fps: float,
    baseline_window_s: tuple[float, float] = (2.0, 3.0),
) -> pd.DataFrame:
    """Compute dF/F0 for one ROI's trial-by-frame fluorescence traces.

    Parameters
    ----------
    roi_traces
        Trial-by-frame raw fluorescence; one row is one trial.
    fps
        Imaging frame rate in Hz.
    baseline_window_s
        ``(start, stop)`` of the F0 window in seconds. The legacy analysis
        used the 2--3 s pre-stimulus window.

    Returns
    -------
    pandas.DataFrame
        ``(F - F0) / F0`` with the input's index and columns. ``F0`` is the
        per-trial window median, smoothed across trials with a
        :data:`F0_SMOOTHING_WINDOW`-sample centred rolling median
        (``min_periods=1``).

    Notes
    -----
    The legacy implementation sliced the baseline window as
    ``rows[s - 1 : e - 1]`` where ``s = round(start * fps)`` and
    ``e = round(end * fps)`` -- i.e. shifted one frame *earlier* than the
    nominal window. That off-by-one is preserved deliberately: changing it
    changes every downstream event table, and the port must reproduce the
    manuscript outputs.
    """
    start_s, stop_s = baseline_window_s
    start = round(start_s * fps)
    stop = round(stop_s * fps)

    baselines = [
        float(np.median(row.iloc[start - 1 : stop - 1])) for _, row in roi_traces.iterrows()
    ]
    smoothed = (
        pd.Series(baselines)
        .rolling(window=F0_SMOOTHING_WINDOW, min_periods=1, center=True)
        .median()
    )
    f0 = smoothed.set_axis(roi_traces.index)
    return roi_traces.sub(f0, axis=0).div(f0, axis=0)
