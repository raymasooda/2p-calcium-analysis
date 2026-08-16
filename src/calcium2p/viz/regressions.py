"""Lick-latency vs calcium-latency regression scatter (notebook cells 32/33/18).

The notebooks' p-symbol chain was buggy: it tested ``p > 0.05`` then
``p < 0.05`` first, so the ``**``/``***``/``****`` branches were unreachable
and every significant regression was labelled ``*``. This implementation uses
the correct thresholds; the legend text may therefore show more asterisks than
the original figures for p < 0.01.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure
from scipy.stats import linregress

from calcium2p.viz.style import REGRESSION_FIGSIZE

if TYPE_CHECKING:
    from typing import Any

    import pandas as pd


def _p_symbol(p: float) -> str:
    """Correct significance symbol (see module docstring for the legacy bug)."""
    if p < 0.0001:  # noqa: PLR2004 - published threshold convention
        return "****"
    if p < 0.001:  # noqa: PLR2004
        return "***"
    if p < 0.01:  # noqa: PLR2004
        return "**"
    if p < 0.05:  # noqa: PLR2004
        return "*"
    return "n.s"


def latency_regression(
    x: pd.Series,
    y: pd.Series,
    *,
    xlabel: str = "Z-scored lick latency",
    ylabel: str = "Z-scored Ca2+ latency",
    line_color: str = "black",
    point_edgecolor: str = "black",
) -> tuple[Figure, Any]:
    """Scatter ``y`` against ``x`` with a least-squares line and r/p legend.

    Parameters
    ----------
    x, y
        Z-scored per-trial latencies (already filtered/aligned by the caller).
    xlabel, ylabel
        Axis labels.
    line_color, point_edgecolor
        The passive-axon variant used grey/darkgrey; task axons black/black.

    Returns
    -------
    tuple
        The figure and the ``scipy.stats.linregress`` result (slope,
        intercept, rvalue, pvalue, stderr).
    """
    line = linregress(x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64))
    r = round(float(line.rvalue), 3)
    symbol = _p_symbol(float(line.pvalue))

    fig = Figure(figsize=REGRESSION_FIGSIZE)
    ax = fig.subplots()
    xs = x.to_numpy(dtype=np.float64)
    ax.plot(
        xs,
        line.intercept + line.slope * xs,
        color=line_color,
        linewidth=2,
        label=f"r={r} ({symbol})",
    )
    ax.legend()
    ax.scatter(
        xs,
        y.to_numpy(dtype=np.float64),
        color="white",
        edgecolor=point_edgecolor,
        alpha=0.75,
        linewidth=1,
        s=15,
    )
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xticks(list(range(-3, 4)))
    ax.set_yticks(list(range(-3, 4)))
    ax.tick_params(labelsize=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    return fig, line
