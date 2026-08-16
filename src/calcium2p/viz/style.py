"""Manuscript figure constants: sizes, colors, and legacy trial-type vocabularies.

The sizes and color conventions are the ones hardcoded across the V10
onset-histogram notebooks; the trial-type dictionaries are ported verbatim from
the legacy ``Thalamocortical/pulsar_plots.py`` so pulsar-style figures can be
rebuilt against the same vocabulary.
"""

from __future__ import annotations

from typing import Final

#: Comparison bar+strip figure size, the notebooks' ``(6.34/4, 2)`` inches.
COMPARISON_FIGSIZE: Final[tuple[float, float]] = (6.34 / 4, 2.0)

#: 2x5 onset-latency histogram grid size in inches.
HISTOGRAM_GRID_FIGSIZE: Final[tuple[float, float]] = (10.0, 5.0)

#: Lick-vs-calcium latency regression scatter size in inches.
REGRESSION_FIGSIZE: Final[tuple[float, float]] = (1.394, 1.4741)

#: Compartment colors: axon panels are dark green; dendrite and NO_DREADDs
#: panels dark red (the notebooks treat NO_DREADDs as a dendrite-colored row).
COMPARTMENT_COLORS: Final[dict[str, str]] = {
    "axons": "darkgreen",
    "dendrites": "darkred",
    "NO_DREADDs": "darkred",
}

#: Histogram panel title colors by CNO phase.
TITLE_COLORS: Final[dict[str, str]] = {
    "pre-CNO": "dodgerblue",
    "post-CNO": "goldenrod",
    "default": "black",
}

#: Outcome trace colors used in the single-ROI overlay panels.
OUTCOME_COLORS: Final[dict[str, str]] = {
    "cRew": "darkgreen",
    "nRew": "darkorange",
    "cHIT": "darkgreen",
    "nHIT": "darkorange",
}

#: Heatmap colormaps by outcome, as used in the manuscript heatmap panels.
HEATMAP_CMAPS: Final[dict[str, str]] = {
    "cHIT": "BuGn",
    "nHIT": "Oranges",
    "cRew": "BuPu",
    "nRew": "Reds",
}

#: Trial-type groupings, ported verbatim from ``pulsar_plots.all_trialtypes``.
ALL_TRIALTYPES: Final[dict[str, list[str]]] = {
    "correct responses": ["rewarded hit", "unrewarded hit"],
    "incorrect responses": ["rewarded FA", "unrewarded FA"],
    "unexpected omission": ["unrewarded hit", "unrewarded hit"],
    "expected omission": ["unrewarded FA", "unrewarded FA"],
    "expected reward": ["rewarded hit", "rewarded hit"],
    "unexpected reward": ["rewarded FA", "rewarded FA"],
    "unexpected outcomes": ["unrewarded hit", "rewarded FA"],
    "expected outcomes": ["unrewarded FA", "rewarded hit"],
    "responses": ["rewarded hit", "rewarded FA", "unrewarded hit", "unrewarded FA"],
    "no responses": ["unrewarded miss", "unrewarded CR"],
    "rewarded responses": ["rewarded hit", "rewarded FA"],
    "unrewarded responses": ["unrewarded hit", "unrewarded FA"],
}

#: Trace colors per trial-type grouping, ported verbatim from
#: ``pulsar_plots.trace_colors``.
TRACE_COLORS: Final[dict[str, str]] = {
    "unexpected omission": "magenta",
    "expected omission": "peru",
    "expected reward": "navy",
    "unexpected reward": "lightblue",
    "correct responses": "green",
    "Miss": "grey",
    "incorrect responses": "darkred",
    "Correct Rejections": "darkorange",
    "expected outcomes": "black",
    "unexpected outcomes": "magenta",
    "Go": "navy",
    "NoGo": "grey",
    "rewarded responses": "blue",
    "unrewarded responses": "darkorange",
    "responses": "darkcyan",
    "no responses": "olive",
}

#: Task-phase labels, ported verbatim from ``pulsar_plots.task_phases``.
TASK_PHASES: Final[dict[str, str]] = {
    "Ndet": "Novice association",
    "Edet": "Expert association",
    "Ndis": "Novice pre-reversal",
    "Edis": "Expert pre-reversal",
    "Nrev": "Novice post-reversal",
    "Erev": "Expert post-reversal",
}

#: The notebooks' fixed 2x5 histogram panel layout (protocol -> (row, col)).
HISTOGRAM_PANEL_MAP: Final[dict[str, tuple[int, int]]] = {
    "passive_axons": (0, 0),
    "task_axons": (0, 1),
    "NO_DREADDs": (0, 2),
    "passive_dendrites": (1, 0),
    "task_dendrites (pre-CNO)": (1, 1),
    "task_dendrites (post-CNO)": (1, 2),
    "control_dendrites (pre-CNO)": (1, 3),
    "control_dendrites (post-CNO)": (1, 4),
}


def protocol_color(protocol: str) -> str:
    """Histogram fill color for a protocol panel.

    Dendrite protocols and NO_DREADDs draw dark red; everything else (the axon
    protocols) dark green, matching the notebooks' inline conditional.
    """
    if "dendrites" in protocol or "NO_DREADDs" in protocol:
        return COMPARTMENT_COLORS["dendrites"]
    return COMPARTMENT_COLORS["axons"]


def title_color(protocol: str) -> str:
    """Histogram panel title color for a protocol (CNO phase coded)."""
    if "pre-CNO" in protocol:
        return TITLE_COLORS["pre-CNO"]
    if "post-CNO" in protocol:
        return TITLE_COLORS["post-CNO"]
    return TITLE_COLORS["default"]
