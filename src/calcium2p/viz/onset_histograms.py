"""The 2x5 onset-latency histogram grid (the manuscript's headline figure).

Port of the notebooks' ``plot_histograms``. One deliberate fix: the notebook
always labelled x ticks as ``round(x - 5, 1)`` -- correct for the reward
notebook (alignment at t=5 s) but off by -5 s in the stimulus notebook. The
alignment offset is a parameter here, so the stimulus axis labels come out
right.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from calcium2p.viz.style import (
    HISTOGRAM_GRID_FIGSIZE,
    HISTOGRAM_PANEL_MAP,
    protocol_color,
    title_color,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pandas as pd

    from calcium2p.core.cohort_config import HistogramSpec


def onset_histogram_grid(
    target_events: Mapping[str, pd.DataFrame],
    reference_events: Mapping[str, pd.DataFrame],
    *,
    target_trials: Mapping[str, int],
    reference_trials: Mapping[str, int],
    roi_counts: Mapping[str, int],
    spec: HistogramSpec,
    align_at: float = 5.0,
    panel_map: Mapping[str, tuple[int, int]] | None = None,
) -> Figure:
    """Draw the 2x5 event-latency histogram grid.

    Parameters
    ----------
    target_events, reference_events
        Per-protocol event tables with an ``ev_onset`` column (seconds in
        trial time). Target = rewarded/stimulus side (drawn step-filled in the
        compartment color); reference = unrewarded/no-stimulus side (drawn as
        a black step outline).
    target_trials, reference_trials
        Per-protocol trial counts for the two sides -- the histogram weights.
    roi_counts
        Per-protocol ROI counts; used when ``spec.roi_div`` divides the
        weights a second time.
    spec
        Bin range/count and normalization switches.
    align_at
        Trial time (s) of the aligning event; x tick labels are latencies
        relative to it (reward 5.0, stimulus 3.0).
    panel_map
        Protocol -> (row, col) placement. Defaults to the manuscript layout.

    Returns
    -------
    matplotlib.figure.Figure
        The grid figure. Panels with no protocol assigned stay empty, as in
        the notebooks.
    """
    placement = dict(HISTOGRAM_PANEL_MAP if panel_map is None else panel_map)
    bins = np.linspace(spec.xmin, spec.xmax, spec.nbins + 1)

    fig = Figure(figsize=HISTOGRAM_GRID_FIGSIZE)
    axes = fig.subplots(
        nrows=2,
        ncols=5,
        sharex=True,
        sharey=spec.sharey,
        gridspec_kw={"hspace": 0.25, "wspace": 0.05},
    )

    sides = (
        (target_events, target_trials, True),
        (reference_events, reference_trials, False),
    )
    for events_by_protocol, trials, is_target in sides:
        for protocol, events in events_by_protocol.items():
            if protocol not in placement:
                continue
            row, col = placement[protocol]
            ax = axes[row, col]
            onsets = events.loc[events["ev_onset"] <= spec.xmax, "ev_onset"].to_numpy()

            weight = 1.0 / trials[protocol]
            ylabel = "nEvents / nTrials"
            if spec.roi_div:
                weight /= roi_counts[protocol]
                ylabel = "nEvents / nTrials / nROIs"
            weights = np.full(onsets.shape, weight)

            if is_target:
                ax.hist(
                    onsets,
                    bins=bins,
                    weights=weights,
                    alpha=0.5,
                    color=protocol_color(protocol),
                    histtype="stepfilled",
                )
                ax.set_title(protocol, fontsize=10, color=title_color(protocol))
            else:
                ax.hist(
                    onsets, bins=bins, weights=weights, alpha=0.5, color="black", histtype="step"
                )
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
            if row == 1:
                ax.set_xlabel("Event latency (s)", fontsize=10)
            ax.set_xticks(bins[::2])
            labels = [
                str(round(x - align_at, 1)) if i % 2 == 0 else "" for i, x in enumerate(bins[::2])
            ]
            ax.set_xticklabels(labels, fontsize=8)

    return fig
