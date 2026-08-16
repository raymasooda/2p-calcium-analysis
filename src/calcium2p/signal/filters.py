"""Zero-phase Butterworth low-pass filtering of trial-by-frame trace frames.

Port of ``filter_traces`` from the legacy ``process_calcium_traces.py``. The
legacy function built a ``(b, a)`` filter and ran ``filtfilt``; it also computed
two ``lfilter`` passes whose results were discarded -- that dead code is not
ported. The repo-standard design is second-order sections (``sosfiltfilt``),
which is numerically stable at higher orders; the ``ba`` design is kept
available because event-table regeneration must be able to reproduce the
original outputs exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy import signal

if TYPE_CHECKING:
    import numpy.typing as npt

FilterDesign = Literal["sos", "ba"]


def lowpass_filter(
    traces: pd.DataFrame,
    order: int = 3,
    critical_freq: float = 0.2,
    *,
    design: FilterDesign = "sos",
) -> pd.DataFrame:
    """Apply a zero-phase Butterworth low-pass filter to each row of a frame.

    Parameters
    ----------
    traces
        Trial-by-frame traces; one row is one trial's fluorescence trace.
        NaN samples are dropped per row before filtering, exactly as the
        legacy implementation did.
    order
        Filter order. The legacy analysis used 3.
    critical_freq
        Normalised critical frequency (fraction of the Nyquist frequency).
        The legacy analysis used 0.2.
    design
        ``"sos"`` (default) uses ``butter(..., output="sos")`` with
        ``sosfiltfilt`` -- the numerically stable form this repo mandates.
        ``"ba"`` uses the legacy ``butter()`` + ``filtfilt`` pair, preserved
        so regenerated outputs can match the originals bit-for-bit.

    Returns
    -------
    pandas.DataFrame
        Filtered traces with the input's row index. Column labels are
        preserved when every row kept its full length; when NaN samples were
        dropped the rows are left-compacted (legacy behaviour) and columns
        fall back to a fresh ``RangeIndex``.

    Raises
    ------
    ValueError
        If ``design`` is not one of ``"sos"`` or ``"ba"``.
    """
    if design not in ("sos", "ba"):
        raise ValueError(f"design must be 'sos' or 'ba', got {design!r}")

    filtered_rows: list[npt.NDArray[np.float64]] = []
    if design == "sos":
        sos = signal.butter(order, critical_freq, output="sos")
        for _, row in traces.iterrows():
            values = row.dropna().to_numpy(dtype=np.float64)
            filtered_rows.append(np.asarray(signal.sosfiltfilt(sos, values), dtype=np.float64))
    else:
        b, a = signal.butter(order, critical_freq)
        for _, row in traces.iterrows():
            values = row.dropna().to_numpy(dtype=np.float64)
            filtered_rows.append(np.asarray(signal.filtfilt(b, a, values), dtype=np.float64))

    filtered = pd.DataFrame(filtered_rows)
    filtered.index = traces.index
    if filtered.shape[1] == traces.shape[1]:
        filtered.columns = traces.columns
    return filtered
