"""Comparison bar+strip figures (port of the notebooks' ``plot_comparisons``).

One deliberate determinism improvement: the notebooks jittered individual
points with seaborn's ``stripplot`` (unseeded global RNG), so no two saves of
the same figure were byte-identical. Points here are jittered with a seeded
``default_rng(0)`` uniform offset instead -- same visual, reproducible output,
and no seaborn dependency in the package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    import pandas as pd

    from calcium2p.core.cohort_config import ComparisonFigure

#: Horizontal extent of the jitter around each group's x position.
_JITTER = 0.18

#: Half-width of the mean bar's SEM whisker caps.
_CAP_HALF_WIDTH = 0.2


def comparison_colors(name: str) -> tuple[str, str]:
    """Edge and face colors for a comparison, from its name.

    Mirrors the notebook convention: axon comparisons dark green; Control
    dendrites dark red with a white (hollow) bar; other dendrites solid dark
    red. Returns ``(edgecolor, facecolor)``.
    """
    if "axons" in name.lower():
        return ("darkgreen", "darkgreen")
    if "Control" in name:
        return ("darkred", "white")
    return ("darkred", "darkred")


def comparison_bar(
    series_a: pd.Series,
    series_b: pd.Series,
    *,
    title: str,
    figure: ComparisonFigure,
    edgecolor: str,
    facecolor: str,
) -> Figure:
    """Draw one two-group mean-bar figure with SEM whiskers and point strips.

    Parameters
    ----------
    series_a, series_b
        Per-ROI metric values for the two groups. NaNs are dropped. The series
        ``name`` attributes label the x ticks (tuples are joined with
        newlines, matching the notebooks).
    title
        Complete, preformatted title (the caller owns the notebooks'
        "epoch label, newline, name epoch (symbol)" convention).
    figure
        Size, y-limits and axis-visibility parameters.
    edgecolor, facecolor
        Bar/point colors, typically from :func:`comparison_colors`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig = Figure(figsize=figure.figsize)
    ax = fig.subplots()
    rng = np.random.default_rng(0)

    for x, series in enumerate((series_a, series_b)):
        values = series.dropna().to_numpy(dtype=np.float64)
        mean = float(values.mean())
        sem = float(values.std(ddof=1) / np.sqrt(values.shape[0])) if values.shape[0] > 1 else 0.0
        bottom, top = mean - sem, mean + sem
        left, right = x - _CAP_HALF_WIDTH, x + _CAP_HALF_WIDTH

        jittered = x + rng.uniform(-_JITTER, _JITTER, size=values.shape[0])
        ax.scatter(
            jittered,
            values,
            facecolors="none",
            edgecolors=edgecolor,
            linewidths=0.75,
            s=16,
            zorder=1,
        )
        ax.bar([x], [mean], color=facecolor, edgecolor=edgecolor, alpha=0.7, zorder=2)
        ax.plot([x, x], [bottom, top], color="black", zorder=3)
        ax.plot([left, right], [top, top], color="black", zorder=4)
        ax.plot([left, right], [bottom, bottom], color="black", zorder=5)

    if figure.ylim is not None:
        ax.set_ylim(*figure.ylim)
    ax.set_xlim(-0.65, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([_tick_label(series_a), _tick_label(series_b)], fontsize=8)
    ax.set_title(title, fontsize=10, y=1.05)
    if figure.remove_axis:
        ax.set_axis_off()
    return fig


def _tick_label(series: pd.Series) -> str:
    """Group label from a series name (tuples newline-joined, as the notebooks did)."""
    name = series.name
    if isinstance(name, tuple):
        return "\n".join(str(part) for part in name)
    return str(name)
