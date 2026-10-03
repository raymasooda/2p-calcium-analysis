"""Figure rendering helpers."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def figure_to_svg(fig: Figure) -> str:
    """Render a figure to SVG markup.

    The SVG ``Date`` metadata is suppressed and matplotlib's ``svg.hashsalt``
    is pinned so identical figures render byte-identically across runs --
    required for artifact-store determinism and reproduction diffs.

    Parameters
    ----------
    fig
        The figure to render.

    Returns
    -------
    str
        Complete SVG document text.
    """
    import matplotlib  # noqa: PLC0415 - defer so importing viz never fixes backend state

    buffer = io.BytesIO()
    with matplotlib.rc_context({"svg.hashsalt": "calcium2p"}):
        fig.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue().decode("utf-8")
