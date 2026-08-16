"""Single-ROI trace figures: waterfalls, overlays, and heatmaps.

Generalizations of the notebooks' hardcoded per-ROI cells. Which ROI and which
trials to draw stays the caller's concern; these functions only lay the panels
out. Heatmaps use ``pcolormesh`` rather than seaborn so the package carries no
seaborn dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd


def roi_waterfall(
    trials_df: pd.DataFrame,
    event_trials: Sequence[object],
    *,
    fps: float,
    event_color: str,
    non_event_color: str = "black",
    onset_lines: Sequence[tuple[float, str]] = (),
    xlim_s: tuple[float, float] | None = None,
) -> Figure:
    """One thin panel per trial, event trials on top in color.

    Parameters
    ----------
    trials_df
        Trial-by-frame traces for one ROI (one row per trial).
    event_trials
        Row labels of ``trials_df`` that carry a detected event; they are
        drawn first (top of the waterfall) in ``event_color``, remaining
        trials below in ``non_event_color`` -- the notebooks' ordering.
    fps
        Frame rate, for the vline/xlim second-to-frame conversion.
    event_color, non_event_color
        Trace colors for the two trial classes.
    onset_lines
        ``(time_s, color)`` vertical markers drawn on every panel.
    xlim_s
        Optional ``(start, stop)`` seconds window.

    Returns
    -------
    matplotlib.figure.Figure
    """
    index_list = list(trials_df.index)
    ordered = [t for t in event_trials if t in set(index_list)]
    ordered += [t for t in index_list if t not in set(ordered)]
    n_trials = len(ordered)

    fig = Figure(figsize=(6.34 / 6, 0.25 * n_trials))
    axes = fig.subplots(nrows=max(n_trials, 1), ncols=1, sharey=True, gridspec_kw={"hspace": 0})
    axes_list = np.atleast_1d(axes)
    event_set = {t for t in event_trials if t in set(index_list)}
    values = trials_df.to_numpy(dtype=np.float64)

    for ax, trial in zip(axes_list, ordered, strict=False):
        trace = values[index_list.index(trial)]
        color = event_color if trial in event_set else non_event_color
        ax.plot(trace, color=color, linewidth=0.75)
        for time_s, line_color in onset_lines:
            ax.axvline(time_s * fps, color=line_color, alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(xlim_s[0] * fps, xlim_s[1] * fps)
        ax.set_axis_off()
    return fig


def roi_overlay(
    traces_by_outcome: Mapping[str, pd.DataFrame],
    colors: Mapping[str, str],
    *,
    fps: float,
    mean_linewidth: float = 2.0,
    xlim_s: tuple[float, float] | None = None,
    onset_line_s: float | None = None,
) -> Figure:
    """Per-outcome panels of thin single-trial traces plus a thick mean.

    Parameters
    ----------
    traces_by_outcome
        Outcome label -> trial-by-frame traces to overlay in that panel.
    colors
        Outcome label -> trace color.
    fps
        Frame rate for the second-to-frame conversions.
    mean_linewidth
        Width of the mean trace.
    xlim_s, onset_line_s
        Optional window and navy onset marker, in seconds.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = len(traces_by_outcome)
    fig = Figure(figsize=(2, 2))
    axes = fig.subplots(nrows=1, ncols=max(n_panels, 1), sharey=True, gridspec_kw={"hspace": 0})
    axes_list = np.atleast_1d(axes)

    for ax, (outcome, traces) in zip(axes_list, traces_by_outcome.items(), strict=False):
        values = np.asarray(traces, dtype=np.float64)
        for row in values:
            ax.plot(row, color=colors[outcome], linewidth=0.75, alpha=0.5)
        if values.shape[0]:
            ax.plot(values.mean(axis=0), color=colors[outcome], linewidth=mean_linewidth)
        if onset_line_s is not None:
            ax.axvline(onset_line_s * fps, color="navy", alpha=0.3)
        if xlim_s is not None:
            ax.set_xlim(xlim_s[0] * fps, xlim_s[1] * fps)
        ax.set_axis_off()
    return fig


def roi_heatmap(
    df_by_outcome: Mapping[str, pd.DataFrame],
    cmaps: Mapping[str, str],
    *,
    fps: float,
    onset_line_s: float,
    cbar_ticks: Sequence[float] | None = None,
) -> Figure:
    """Per-outcome trial-by-frame heatmaps with a dashed onset marker.

    Parameters
    ----------
    df_by_outcome
        Outcome label -> trial-by-frame traces (row order = display order,
        the caller's sorting decision).
    cmaps
        Outcome label -> matplotlib colormap name (see
        :data:`calcium2p.viz.style.HEATMAP_CMAPS`).
    fps
        Frame rate; the onset marker is drawn at ``onset_line_s * fps``.
    onset_line_s
        Time of the dashed navy onset line, in seconds.
    cbar_ticks
        Optional explicit colorbar ticks (the notebooks pinned [0, 1, 2]).

    Returns
    -------
    matplotlib.figure.Figure
    """
    n_panels = len(df_by_outcome)
    fig = Figure(figsize=(4, 3.5 / 2))
    axes = fig.subplots(nrows=1, ncols=max(n_panels, 1), gridspec_kw={"wspace": 0.05})
    axes_list = np.atleast_1d(axes)

    for ax, (outcome, frame) in zip(axes_list, df_by_outcome.items(), strict=False):
        values = np.asarray(frame, dtype=np.float64)
        mesh = ax.pcolormesh(values, cmap=cmaps[outcome])
        ax.invert_yaxis()  # first trial on top, matching seaborn.heatmap
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axvline(onset_line_s * fps, color="navy", ls="--", lw=0.75, alpha=0.75)
        ax.set_title(f"{outcome} trials", fontsize=8)
        cbar = fig.colorbar(mesh, ax=ax)
        if cbar_ticks is not None:
            cbar.set_ticks(list(cbar_ticks))
        cbar.ax.tick_params(labelsize=8)
    return fig
