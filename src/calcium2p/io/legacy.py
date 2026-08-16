"""Production-side readers for the converted legacy data and reference CSVs.

Everything here reads pickle-free formats. The pickled ``.npy`` dicts are
converted once by :mod:`calcium2p.io.legacy_bridge` (run under a legacy pandas
environment -- see that module's docstring); this module then reads the
resulting parquet tables, plus the original reference event/frequency CSVs.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from calcium2p.io.legacy_bridge import MANIFEST_NAME

_BRIDGE_COMMAND = (
    "PYTHONPATH=src uv run --no-project --python 3.10 "
    "--with pandas==1.5.3 --with numpy==1.24.4 --with pyarrow==14.0.2 "
    'python -c "from calcium2p.io.legacy_bridge import convert_all; '
    "convert_all('<source_dir>', '<dest_dir>')\""
)


class ConvertedDataMissingError(FileNotFoundError):
    """Raised when the one-time legacy conversion has not been run yet."""


def read_converted(dest_dir: str | Path, name: str) -> pd.DataFrame:
    """Read one converted parquet table (e.g. ``"traces_task_axons"``).

    Parameters
    ----------
    dest_dir
        Directory that :func:`calcium2p.io.legacy_bridge.convert_all` wrote.
    name
        Table name without extension; see the ``legacy_bridge`` module
        docstring for the schema of each.

    Raises
    ------
    ConvertedDataMissingError
        If the table (or the whole conversion) is absent, with the bridge
        command to run.
    """
    path = Path(dest_dir) / f"{name}.parquet"
    if not path.exists():
        raise ConvertedDataMissingError(
            f"{path} not found. Run the one-time legacy conversion first:\n  {_BRIDGE_COMMAND}"
        )
    return pd.read_parquet(path)


def conversion_manifest_path(dest_dir: str | Path) -> Path:
    """Path of the conversion manifest inside a converted-data directory."""
    return Path(dest_dir) / MANIFEST_NAME


def read_reference_events(path: str | Path, group: str) -> pd.DataFrame:
    """Read one original event CSV with the notebooks' per-group index logic.

    The reference CSVs have three distinct layouts, dispatched on the group
    name exactly as the notebooks did:

    - ``'passive' in group``: ``index_col=[0, 12]`` -- (session id, ROI), e.g.
      ``('GCaMP_POm_Batch3_five_FOV1', 'ROI0')``.
    - ``task_axons`` / ``NO_DREADDs``: ``index_col=[1]`` -- the ``ROI`` column,
      e.g. ``'muji_FOV2_1.0'`` (session id + ``_`` + ``str(float(roi))``).
    - dendrites: ``index_col=[0, 10]`` -- (mouse, ROI), ROI like ``'57.0'`` or
      ``'42.0_TD3'`` for the MODIFIED TD3 files.

    Parameters
    ----------
    path
        One ``*_all_2s_*_events.csv`` or ``*_all_500ms_*_events.csv`` file.
    group
        Group name derived from the filename stem (``MODIFIED_`` stripped),
        e.g. ``'passive_axons'``, ``'task_axons'``, ``'task_dendrites_TD0'``.
    """
    if "passive" in group:
        index_col: list[int] = [0, 12]
    elif group in ("task_axons", "NO_DREADDs"):
        index_col = [1]
    else:
        index_col = [0, 10]
    return pd.read_csv(path, index_col=index_col)


def read_reference_frequencies(path: str | Path) -> pd.DataFrame:
    """Read one ``*_event_frequencies.csv`` reference table ((mouse, ROI) index)."""
    return pd.read_csv(path, index_col=[0, 1])


_TRAILING_FLOAT = re.compile(r"^(\d+)\.0$")


def _normalize_token(token: str) -> str:
    """Collapse a ``'66.0'``-style float rendering of an integer to ``'66'``."""
    match = _TRAILING_FLOAT.match(token)
    return match.group(1) if match else token


def canonical_roi_key(raw: str | tuple[object, ...] | int | float) -> str:
    """Normalize an ROI identity into a stable cross-layout string key.

    The reference CSVs render the same ROI in several forms -- task_axons and
    NO_DREADDs append ``str(float(roi))`` (``'muji_FOV2_66.0'``), dendrite
    files use bare ``'57.0'`` or ``'42.0_TD3'``, passive files use
    ``('GCaMP_POm_Batch3_five_FOV1', 'ROI0')`` tuples. This function maps all
    of them onto one form so regenerated and reference tables can be matched:
    integer-valued float tokens lose their ``.0`` and tuple elements are
    joined with ``::``.

    Examples
    --------
    >>> canonical_roi_key("muji_FOV2_66.0")
    'muji_FOV2_66'
    >>> canonical_roi_key(("muji", "42.0_TD3"))
    'muji::42_TD3'
    >>> canonical_roi_key(("GCaMP_POm_Batch3_five_FOV1", "ROI0"))
    'GCaMP_POm_Batch3_five_FOV1::ROI0'
    """
    if isinstance(raw, tuple):
        return "::".join(canonical_roi_key(part) for part in raw)  # type: ignore[arg-type]
    if isinstance(raw, float) and raw.is_integer():
        return str(int(raw))
    text = str(raw)
    parts = [_normalize_token(part) for part in text.split("_")]
    return "_".join(parts)
