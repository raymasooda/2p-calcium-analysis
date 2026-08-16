r"""One-time conversion of the legacy pickled ``.npy`` dicts to parquet.

This module is THE pickle quarantine: :func:`load_pickled_npy` is the only
``np.load(..., allow_pickle=True)`` call site in the package, and nothing
downstream of the conversion ever touches a pickle again.

The four deltaF dicts cannot be unpickled under pandas >= 2.0 at all -- their
leaf DataFrames reference ``pandas.core.indexes.numeric``, deleted in 2.0 -- so
the conversion runs under an ephemeral legacy environment::

    PYTHONPATH=src uv run --no-project --python 3.10 \\
      --with pandas==1.5.3 --with numpy==1.24.4 --with pyarrow==14.0.2 \\
      python -c "from calcium2p.io.legacy_bridge import convert_all; \\
                 convert_all('<source_dir>', '<dest_dir>')"

Accordingly this file must stay importable under BOTH pandas 1.5.3 / Python
3.10 and the repo's pandas 2.2+ / Python 3.12: stdlib + numpy + pandas only,
no other calcium2p imports.

Output schema (all wide trace tables carry metadata columns first, then frame
columns named ``f000``..``fNNN`` renamed *positionally* -- the original frame
labels, 1-based for task_axons and 0-based elsewhere, are not preserved):

``traces_task_axons`` / ``traces_no_dreadds``
    One row per (id, roi, trial) from the ``'ALL'`` branch, with the trial
    MultiIndex levels flattened into columns.
``trials_task_axons`` / ``trials_no_dreadds``
    One row per (id, ttype) recording ``n_trials`` exactly as the notebooks
    counted it: the row count of the *first* ROI's frame in that branch.
``traces_batch1`` / ``traces_batch3``
    One row per (id, protocol, compartment-or-fov, roi, trial).
``behavior_batch1`` / ``behavior_batch3``
    One row per (id, protocol, compartment-or-fov, trial); every raw behaviour
    column is coerced to pandas ``string`` dtype (the raw frames are
    object-typed CSV debris; values are preserved verbatim via ``str()``).
``behavior_task_dendrites`` / ``behavior_control_dendrites``
    Long tables with ``td``/``mouse`` columns; object columns coerced to
    ``string``, numeric/bool kept.

Every table carries an integer ``source_order`` column recording the
dict-encounter order of its (id / fov / compartment) group. The notebooks'
passive ROI-count bug (the ``FOV_dict`` leak) depends on pickle insertion
order, and legacy-faithful mode replays it from this column.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Canonical source filename per dataset name.
SOURCE_FILES: dict[str, str] = {
    "task_axons": "deltaF_dict.npy",
    "no_dreadds": "NO_DREADDs_deltaF_dict.npy",
    "batch1": "RGECO_GCaMP_Batch1_deltaF.npy",
    "batch3": "GCaMP_POm_Batch3_deltaF_dict.npy",
    "task_dendrites": "task_dendrites_cohort_behavior.npy",
    "control_dendrites": "control_dendrites_cohort_behavior.npy",
}

MANIFEST_NAME = "conversion_manifest.json"


def load_pickled_npy(path: str | Path) -> dict[str, Any]:
    """Load one legacy pickled dict.

    .. warning::
        ``allow_pickle=True`` executes arbitrary code embedded in the file on
        load. This function must remain the only such call site in the
        package, must only ever be pointed at the six known manuscript files,
        and runs under the throwaway legacy environment, never the repo one.

    Parameters
    ----------
    path
        A ``.npy`` file produced by ``np.save`` on a nested dict.

    Returns
    -------
    dict
        The unpickled nested dictionary.
    """
    loaded: Any = np.load(Path(path), allow_pickle=True).item()
    if not isinstance(loaded, dict):
        raise TypeError(f"{path}: expected a pickled dict, got {type(loaded).__name__}")
    return loaded


def _frame_columns(n: int) -> list[str]:
    """Positional frame-column names ``f000``..``f{n-1:03d}``."""
    return [f"f{i:03d}" for i in range(n)]


def _stringify_object_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce object-dtype columns to pandas ``string`` dtype via ``str()``.

    Parquet cannot serialize mixed-type object columns; values are preserved
    verbatim as their ``str()`` rendering (lists included), with missing
    values kept as NA.
    """
    out = frame.copy()
    for col in out.columns:
        if out[col].dtype == object:
            mask = out[col].isna()
            out[col] = out[col].map(str).astype("string").mask(mask)
    return out


def flatten_task_axons(raw: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the task_axons deltaF dict (``deltaF_dict.npy``).

    Structure: ``{id: {ttype: {roi(int): DataFrame(trial x frame)}}}`` with
    row MultiIndex ``['Trial number', 'unvalidated type', 'previous type',
    'response latency', 'early lick?']`` and ttypes including ``None``,
    ``'ALL'``, ``'HIT'``, ``'resp HIT'``...

    Returns
    -------
    (traces, trials)
        ``traces``: one row per (id, roi, trial) from the ``'ALL'`` branch.
        ``trials``: per-(id, ttype) trial counts, every non-``None`` branch,
        counted from the first ROI's row count exactly like the notebooks.
    """
    trace_rows: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    for order, (session_id, ttype_dict) in enumerate(raw.items()):
        for ttype, roi_dict in ttype_dict.items():
            if ttype is None or not roi_dict:
                continue
            first_roi = next(iter(roi_dict))
            count_rows.append(
                {
                    "dataset": "task_axons",
                    "id": str(session_id),
                    "ttype": str(ttype),
                    "n_trials": int(roi_dict[first_roi].shape[0]),
                    "source_order": order,
                }
            )
        all_branch = ttype_dict.get("ALL", {})
        for roi, frame in all_branch.items():
            if frame.empty:
                continue
            index = frame.index
            meta = pd.DataFrame(
                {
                    "dataset": "task_axons",
                    "id": str(session_id),
                    "roi": str(roi),
                    "trial": [str(v) for v in index.get_level_values("Trial number")],
                    "unvalidated_type": [
                        str(v) for v in index.get_level_values("unvalidated type")
                    ],
                    "previous_type": [str(v) for v in index.get_level_values("previous type")],
                    "response_latency": [
                        float(v) for v in index.get_level_values("response latency")
                    ],
                    "early_lick": [str(v) for v in index.get_level_values("early lick?")],
                    "source_order": order,
                }
            )
            values = pd.DataFrame(frame.to_numpy(), columns=_frame_columns(frame.shape[1]))
            trace_rows.append(pd.concat([meta, values], axis=1))
    traces = pd.concat(trace_rows, ignore_index=True)
    trials = pd.DataFrame(count_rows)
    return traces, trials


def flatten_no_dreadds(raw: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten the NO_DREADDs deltaF dict (``NO_DREADDs_deltaF_dict.npy``).

    Same nesting as task_axons but the row MultiIndex is
    ``['Trial number', 'trial type']`` and there is no ``None`` branch.
    """
    trace_rows: list[pd.DataFrame] = []
    count_rows: list[dict[str, Any]] = []
    for order, (session_id, ttype_dict) in enumerate(raw.items()):
        for ttype, roi_dict in ttype_dict.items():
            if ttype is None or not roi_dict:
                continue
            first_roi = next(iter(roi_dict))
            count_rows.append(
                {
                    "dataset": "no_dreadds",
                    "id": str(session_id),
                    "ttype": str(ttype),
                    "n_trials": int(roi_dict[first_roi].shape[0]),
                    "source_order": order,
                }
            )
        all_branch = ttype_dict.get("ALL", {})
        for roi, frame in all_branch.items():
            if frame.empty:
                continue
            index = frame.index
            meta = pd.DataFrame(
                {
                    "dataset": "no_dreadds",
                    "id": str(session_id),
                    "roi": str(roi),
                    "trial": [str(v) for v in index.get_level_values("Trial number")],
                    "trial_type": [str(v) for v in index.get_level_values("trial type")],
                    "source_order": order,
                }
            )
            values = pd.DataFrame(frame.to_numpy(), columns=_frame_columns(frame.shape[1]))
            trace_rows.append(pd.concat([meta, values], axis=1))
    traces = pd.concat(trace_rows, ignore_index=True)
    trials = pd.DataFrame(count_rows)
    return traces, trials


def _flatten_series_dataset(
    raw: dict[str, Any], dataset: str, group_field: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten a Batch1/Batch3-shaped dict.

    Structure: ``{id: {protocol: {group: {roi: {trial(int): Series},
    'behavior': {trial(int): DataFrame(1 x k)}}}}}`` where ``group`` is the
    compartment (Batch1) or FOV (Batch3).
    """
    trace_rows: list[pd.DataFrame] = []
    behavior_rows: list[pd.DataFrame] = []
    order = 0
    for session_id, protocol_dict in raw.items():
        for protocol, group_dict in protocol_dict.items():
            for group, entries in group_dict.items():
                behavior = entries.get("behavior", {})
                if behavior:
                    per_trial = []
                    for trial, frame in behavior.items():
                        row = frame.copy()
                        row.insert(0, "behavior_row", [int(i) for i in frame.index])
                        row.index = pd.Index([int(trial)], name="trial")
                        per_trial.append(row)
                    joined = pd.concat(per_trial).reset_index()
                    joined.insert(0, "dataset", dataset)
                    joined.insert(1, "id", str(session_id))
                    joined.insert(2, "protocol", str(protocol))
                    joined.insert(3, group_field, str(group))
                    joined.insert(4, "source_order", order)
                    behavior_rows.append(_stringify_object_columns(joined))
                for roi, trial_dict in entries.items():
                    if roi == "behavior":
                        continue
                    trials_sorted = list(trial_dict.items())
                    values = pd.DataFrame(
                        np.vstack([np.asarray(series) for _, series in trials_sorted]),
                        columns=_frame_columns(len(trials_sorted[0][1])),
                    )
                    meta = pd.DataFrame(
                        {
                            "dataset": dataset,
                            "id": str(session_id),
                            "protocol": str(protocol),
                            group_field: str(group),
                            "roi": str(roi),
                            "trial": [int(t) for t, _ in trials_sorted],
                            "source_order": order,
                        }
                    )
                    trace_rows.append(pd.concat([meta, values], axis=1))
                order += 1
    traces = pd.concat(trace_rows, ignore_index=True)
    behavior_table = (
        pd.concat(behavior_rows, ignore_index=True) if behavior_rows else pd.DataFrame()
    )
    return traces, behavior_table


def flatten_batch1(raw: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten ``RGECO_GCaMP_Batch1_deltaF.npy`` (compartment-grouped)."""
    return _flatten_series_dataset(raw, "batch1", "compartment")


def flatten_batch3(raw: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flatten ``GCaMP_POm_Batch3_deltaF_dict.npy`` (FOV-grouped)."""
    return _flatten_series_dataset(raw, "batch3", "fov")


def flatten_behavior(raw: dict[str, Any], cohort: str) -> pd.DataFrame:
    """Flatten a ``*_cohort_behavior.npy`` dict.

    Structure: ``{td: {mouse: DataFrame}}``. All TD keys are kept (control
    includes a TD4 the notebooks never read); object columns are coerced to
    ``string``.
    """
    rows: list[pd.DataFrame] = []
    for order, (td, mouse_dict) in enumerate(raw.items()):
        for mouse, frame in mouse_dict.items():
            out = frame.reset_index(drop=True)
            out.insert(0, "cohort", cohort)
            out.insert(1, "td", str(td))
            out.insert(2, "mouse", str(mouse))
            out.insert(3, "source_order", order)
            rows.append(_stringify_object_columns(out))
    return pd.concat(rows, ignore_index=True)


def _sha256(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_all(source_dir: str | Path, dest_dir: str | Path) -> dict[str, str]:
    """Convert every legacy pickled dict in ``source_dir`` to parquet.

    Parameters
    ----------
    source_dir
        Directory holding the six files named in :data:`SOURCE_FILES`.
    dest_dir
        Output directory, created if missing. Also receives
        ``conversion_manifest.json`` recording source checksums, row counts,
        and the converting pandas/numpy versions.

    Returns
    -------
    dict
        ``{output table name: written path}``.
    """
    source = Path(source_dir)
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, Any]] = {}

    for dataset in ("task_axons", "no_dreadds", "batch1", "batch3"):
        path = source / SOURCE_FILES[dataset]
        raw = load_pickled_npy(path)
        if dataset == "task_axons":
            traces, extra = flatten_task_axons(raw)
            extra_name = f"trials_{dataset}"
        elif dataset == "no_dreadds":
            traces, extra = flatten_no_dreadds(raw)
            extra_name = f"trials_{dataset}"
        elif dataset == "batch1":
            traces, extra = flatten_batch1(raw)
            extra_name = f"behavior_{dataset}"
        else:
            traces, extra = flatten_batch3(raw)
            extra_name = f"behavior_{dataset}"
        outputs[f"traces_{dataset}"] = traces
        outputs[extra_name] = extra
        sources[dataset] = {"file": path.name, "sha256": _sha256(path)}

    for cohort in ("task_dendrites", "control_dendrites"):
        path = source / SOURCE_FILES[cohort]
        raw = load_pickled_npy(path)
        outputs[f"behavior_{cohort}"] = flatten_behavior(raw, cohort)
        sources[cohort] = {"file": path.name, "sha256": _sha256(path)}

    written: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for name, frame in outputs.items():
        out_path = dest / f"{name}.parquet"
        frame.to_parquet(out_path, index=False)
        written[name] = str(out_path)
        row_counts[name] = int(frame.shape[0])

    manifest = {
        "sources": sources,
        "row_counts": row_counts,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "note": (
            "object columns coerced to string dtype (values via str()); frame "
            "columns renamed positionally to f000..fNNN"
        ),
    }
    (dest / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return written
