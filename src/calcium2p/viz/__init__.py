"""Plotting.

Note matplotlib >= 3.9 removed ``cm.get_cmap``; use ``matplotlib.colormaps[name]``.
"""

from __future__ import annotations

from calcium2p.viz.comparisons import comparison_bar, comparison_colors
from calcium2p.viz.onset_histograms import onset_histogram_grid
from calcium2p.viz.regressions import latency_regression
from calcium2p.viz.render import figure_to_svg
from calcium2p.viz.traces import roi_heatmap, roi_overlay, roi_waterfall

__all__ = [
    "comparison_bar",
    "comparison_colors",
    "figure_to_svg",
    "latency_regression",
    "onset_histogram_grid",
    "roi_heatmap",
    "roi_overlay",
    "roi_waterfall",
]
