"""Regenerate the per-event tables from the converted deltaF traces.

The upstream code that produced the original ``*_events.csv`` tables was never
committed; the recipes here were **recovered** by fitting against those
reference tables (onset-quantization -> per-session fps; threshold-interval
fitting -> the threshold rule; exact re-detection -> label bases and windows):

============  =========================================================
group         recovered recipe
============  =========================================================
task_axons    fps 30.54, 1-based frame labels, detection restricted to
              the validated-outcome trials (cHIT/nHIT/CATCH*/FA),
              threshold = 2 x SD of that stack's flattened 2-3 s
              baseline; validated 14/14 exact on the probe ROI.
NO_DREADDs    same rule with 0-based labels and per-session fps
              (GCaMP_Batch8_muji_FOV3 and GCaMP_Batch8_one_FOV7 ran at
              30.3 Hz, the rest at 30.54 Hz -- fitted from the
              reference onset grids); validated 13/13 exact.
passive       batch1 (RGECO) at 30.3 Hz, 0-based labels, threshold
              from the 0-2 s baseline over every NAIVE-STIM-REW trial;
              batch3 (GCaMP_POm) at 30.54 Hz, 0-based labels -- its
              threshold basis matches no baseline of the stored traces
              and regeneration is PARTIAL by construction (reported
              honestly by ``cohort.validation``).
============  =========================================================

Sides: the reward/stim file holds events from cHIT/nHIT trials (passive:
history contains ``Rew``), the noreward/nostim file events from CATCH*/FA
trials (passive: history contains ``Stim``). Trials outside the validated
set (miss, CR, post-mistake HIT, ...) were never detected on upstream, and
are excluded here the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Final

import numpy as np
import pandas as pd
from scipy import integrate

from calcium2p.signal.events import detect_events

if TYPE_CHECKING:
    from collections.abc import Mapping

    from calcium2p.core.cohort_config import Alignment, EventTableParams

#: Metric columns of a regenerated event row, in reference-file order.
EVENT_COLUMNS: Final = ("peak", "ev_onset", "peak_time", "ev_offset", "ev_duration", "integral")


@dataclass(frozen=True)
class GroupRecipe:
    """One source segment of a regenerated event-table group.

    Attributes
    ----------
    traces_table
        Stem of the converted traces parquet feeding this segment.
    outcome_column
        Trace-metadata column holding the per-trial outcome label; ``None``
        joins the outcome from the behaviour table's ``history`` instead.
    behavior_table
        Behaviour parquet stem for the history join (passive segments).
    label_base
        First frame label (0 or 1); onsets are ``label / fps``.
    fps
        Default sampling rate for the segment.
    fps_overrides
        Per-session sampling rates fitted from the reference onset grids.
    validated_only
        Restrict detection (and the threshold stack) to validated-outcome
        trials, as the task cohorts' branch structure did.
    roi_key_format
        How the reference index spells this segment's ROI keys:
        ``"float_suffix"`` -> ``"{id}_{roi}.0"`` strings;
        ``"tuple"`` -> ``(prefix_id[_fov], roi)`` pairs.
    id_prefix
        Prefix of the tuple key's first level (passive segments).
    group_columns
        Metadata columns identifying one session within the traces table.
    baseline_window_s
        Segment default baseline window (overridable via
        ``EventTableParams.baseline_window_overrides`` keyed by group name).
    """

    traces_table: str
    outcome_column: str | None
    label_base: int
    fps: float
    roi_key_format: str
    group_columns: tuple[str, ...]
    baseline_window_s: tuple[float, float] = (2.0, 3.0)
    behavior_table: str | None = None
    fps_overrides: dict[str, float] = field(default_factory=dict)
    validated_only: bool = True
    id_prefix: str = ""
    compartment: str | None = None
    detection: str = "per_trial"


#: Recovered per-group recipes; passive groups pool two source segments.
GROUP_RECIPES: Final[dict[str, tuple[GroupRecipe, ...]]] = {
    "task_axons": (
        GroupRecipe(
            traces_table="traces_task_axons",
            outcome_column="unvalidated_type",
            label_base=1,
            fps=30.54,
            roi_key_format="float_suffix",
            group_columns=("id",),
        ),
    ),
    "NO_DREADDs": (
        GroupRecipe(
            traces_table="traces_no_dreadds",
            outcome_column="trial_type",
            label_base=0,
            fps=30.54,
            fps_overrides={"GCaMP_Batch8_muji_FOV3": 30.3, "GCaMP_Batch8_one_FOV7": 30.3},
            roi_key_format="float_suffix",
            group_columns=("id",),
            detection="per_branch",
        ),
    ),
    "passive_axons": (
        GroupRecipe(
            traces_table="traces_batch3",
            outcome_column=None,
            behavior_table="behavior_batch3",
            label_base=0,
            fps=30.54,
            roi_key_format="tuple",
            id_prefix="GCaMP_POm_Batch3",
            group_columns=("id", "fov"),
            validated_only=False,
            detection="flattened",
        ),
        GroupRecipe(
            traces_table="traces_batch1",
            outcome_column=None,
            behavior_table="behavior_batch1",
            label_base=0,
            fps=30.3,
            roi_key_format="tuple",
            id_prefix="RGECO_GCaMP_Batch1",
            group_columns=("id",),
            compartment="AXONS",
            baseline_window_s=(0.0, 2.0),
            validated_only=False,
            detection="flattened",
        ),
    ),
    "passive_dendrites": (
        GroupRecipe(
            traces_table="traces_batch1",
            outcome_column=None,
            behavior_table="behavior_batch1",
            label_base=0,
            fps=30.3,
            roi_key_format="tuple",
            id_prefix="RGECO_GCaMP_Batch1",
            group_columns=("id",),
            compartment="DENDRITES",
            baseline_window_s=(0.0, 2.0),
            validated_only=False,
            detection="flattened",
        ),
    ),
}

#: Outcome labels whose trials the task cohorts' validated branches held.
_VALIDATED_EXACT: Final = ("cHIT", "nHIT", "FA")
_VALIDATED_SUBSTRING: Final = "CATCH"

#: Which outcomes land on which side of the reward/noreward (stim/nostim) split.
_TARGET_OUTCOMES: Final = ("cHIT", "nHIT")


#: Length of a positional frame-column name (``f000``).
_FRAME_COLUMN_LENGTH: Final = 4


def _frame_columns(columns: pd.Index) -> list[str]:
    """Select the positional frame columns (``f000``...) of a converted table."""
    return [
        c
        for c in columns
        if len(c) == _FRAME_COLUMN_LENGTH and c.startswith("f") and c[1:].isdigit()
    ]


def _is_validated(labels: pd.Series) -> pd.Series:
    """Mask of trials the upstream validated branches contained."""
    return labels.isin(_VALIDATED_EXACT) | labels.str.contains(_VALIDATED_SUBSTRING, regex=False)


def _emit_stack(
    rows: list[dict[str, object]],
    stack: np.ndarray,
    stack_trials: list[str],
    *,
    roi_name: str,
    labels: pd.Series,
    threshold: float,
    fps: float,
    label_base: int,
    min_samples: int,
    window: tuple[float, float],
) -> None:
    """Detect on one flattened stack (single trial, branch, or all trials).

    Events may span trial boundaries within the stack: an event belongs to
    the trial its onset falls in, and its offset/peak_time/duration/integral
    run over the whole flattened segment (validated exactly against the
    reference tables).
    """
    n_frames = stack.shape[1]
    flat = pd.Series(stack.ravel(), index=np.arange(stack.size))
    for event in detect_events(flat, threshold, threshold, min_width=min_samples):
        global_start = int(event.index[0])
        trial_position = global_start // n_frames
        trial = stack_trials[trial_position]
        trial_origin = trial_position * n_frames
        onset = (global_start - trial_origin + label_base) / fps
        if not (window[0] < onset < window[1]):
            continue
        values = event.to_numpy(dtype=float)
        rows.append(
            {
                "roi": roi_name,
                "trial": trial,
                "outcome": labels.loc[trial],
                "peak": float(values.max()),
                "ev_onset": onset,
                "peak_time": (int(event.index[int(values.argmax())]) - trial_origin + label_base)
                / fps,
                "ev_offset": (int(event.index[-1]) - trial_origin + label_base) / fps,
                "ev_duration": event.shape[0] / fps,
                "integral": float(integrate.simpson(values)),
            }
        )


def _detect_session(
    session: pd.DataFrame,
    outcome_by_trial: pd.Series,
    recipe: GroupRecipe,
    *,
    fps: float,
    baseline_window_s: tuple[float, float],
    params: EventTableParams,
    window: tuple[float, float],
) -> list[dict[str, object]]:
    """Detect and window events for every ROI of one session."""
    frame_cols = _frame_columns(session.columns)
    rows: list[dict[str, object]] = []
    for roi_name, roi_rows in session.groupby("roi", sort=False):
        mat = (
            roi_rows.set_index(roi_rows["trial"].astype(str))[frame_cols]
            .astype(float)
            .dropna(axis=1, how="all")
        )
        labels = outcome_by_trial.reindex(mat.index).astype(str)
        keep = _is_validated(labels) if recipe.validated_only else pd.Series(True, index=mat.index)
        if not bool(keep.any()):
            continue
        arr = mat.to_numpy()
        kept_arr = arr[keep.to_numpy()]
        start = round(baseline_window_s[0] * fps)
        stop = round(baseline_window_s[1] * fps)
        threshold = params.threshold_sd * float(kept_arr[:, start:stop].std())
        emit = partial(
            _emit_stack,
            rows,
            roi_name=str(roi_name),
            labels=labels,
            threshold=threshold,
            fps=fps,
            label_base=recipe.label_base,
            min_samples=params.min_samples,
            window=window,
        )

        kept_trials = [str(t) for i, t in enumerate(mat.index) if bool(keep.iloc[i])]
        if recipe.detection == "per_trial":
            # the upstream detected each trial independently (task_axons)
            for position, trial in enumerate(kept_trials):
                emit(kept_arr[position : position + 1], [trial])
        elif recipe.detection == "per_branch":
            # the upstream detected each validated branch's stacked trials as
            # one flattened trace (NO_DREADDs), so events can span adjacent
            # trials of the same branch
            for branch_mask in (
                labels == "cHIT",
                labels == "nHIT",
                labels.str.contains(_VALIDATED_SUBSTRING, regex=False),
                labels == "FA",
            ):
                mask = (branch_mask & keep).to_numpy()
                branch_trials = [str(t) for t, m in zip(mat.index, mask, strict=True) if m]
                if branch_trials:
                    emit(arr[mask], branch_trials)
        else:
            # the upstream flattened every kept trial into one long trace
            # (passive cohorts), so events can span trial boundaries
            emit(kept_arr, kept_trials)
    return rows


def _session_outcomes(
    session: pd.DataFrame, recipe: GroupRecipe, tables: Mapping[str, pd.DataFrame]
) -> pd.Series:
    """Per-trial outcome labels for one session, keyed by trial string."""
    if recipe.outcome_column is not None:
        return (
            session.drop_duplicates("trial")
            .set_index(session.drop_duplicates("trial")["trial"].astype(str))[recipe.outcome_column]
            .astype(str)
        )
    behavior = tables[recipe.behavior_table or ""]
    mask = pd.Series(True, index=behavior.index)
    for col in recipe.group_columns:
        mask &= behavior[col].astype(str) == str(session[col].iloc[0])
    if recipe.compartment is not None:
        mask &= behavior["compartment"].astype(str) == recipe.compartment
    mask &= behavior["protocol"].astype(str) == "NAIVE STIM REW"
    sub = behavior[mask]
    return sub.set_index(sub["trial"].astype(str))["history"].astype(str)


def _roi_key(recipe: GroupRecipe, session: pd.DataFrame, roi: str) -> object:
    """Spell an ROI key the way the reference index does."""
    if recipe.roi_key_format == "float_suffix":
        return f"{session['id'].iloc[0]}_{roi}.0"
    parts = [recipe.id_prefix] + [str(session[c].iloc[0]) for c in recipe.group_columns]
    return ("_".join(parts), roi)


def build_event_tables(  # noqa: PLR0912 - faithful port of per-family branching
    tables: Mapping[str, pd.DataFrame],
    params: EventTableParams,
    *,
    alignment: Alignment,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Regenerate every group's event tables from the converted traces.

    Parameters
    ----------
    tables
        The converted parquet tables keyed by stem (``traces_task_axons``,
        ``behavior_batch1``, ...).
    params
        Detection and windowing parameters (defaults are the recovered
        upstream values).
    alignment
        ``"reward"`` regenerates the 2 s reward family (window
        ``params.reward_window_s``); ``"stimulus"`` the 500 ms family.

    Returns
    -------
    dict
        ``group -> {"target": events, "reference": events}`` where target is
        the reward/stim side and reference the noreward/nostim side. Each
        frame is indexed by the reference-form ROI key with columns
        ``trial``, ``outcome`` and :data:`EVENT_COLUMNS`.
    """
    window = params.reward_window_s if alignment == "reward" else params.stim_window_s
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for group, recipes in GROUP_RECIPES.items():
        collected: list[pd.DataFrame] = []
        for recipe in recipes:
            if recipe.traces_table not in tables:
                # partial table sets are allowed (tests, single-group runs)
                continue
            traces = tables[recipe.traces_table]
            if "protocol" in traces.columns:
                traces = traces[traces["protocol"].astype(str) == "NAIVE STIM REW"]
            if recipe.compartment is not None:
                traces = traces[traces["compartment"].astype(str) == recipe.compartment]
            baseline = params.baseline_window_overrides.get(group, recipe.baseline_window_s)
            for _, session in traces.groupby(list(recipe.group_columns), sort=False):
                session_id = "_".join(str(session[c].iloc[0]) for c in recipe.group_columns)
                if recipe.id_prefix:
                    session_id = f"{recipe.id_prefix}_{session_id}"
                fps = recipe.fps_overrides.get(session_id, recipe.fps)
                outcomes = _session_outcomes(session, recipe, tables)
                rows = _detect_session(
                    session,
                    outcomes,
                    recipe,
                    fps=fps,
                    baseline_window_s=baseline,
                    params=params,
                    window=window,
                )
                if not rows:
                    continue
                frame = pd.DataFrame(rows)
                keys = [_roi_key(recipe, session, str(r)) for r in frame.pop("roi")]
                if recipe.roi_key_format == "tuple":
                    tuple_keys = [k for k in keys if isinstance(k, tuple)]
                    frame.index = pd.MultiIndex.from_tuples(tuple_keys, names=[None, "ROI"])
                else:
                    frame.index = pd.Index(keys, name="ROI")
                collected.append(frame)
        if not collected:
            continue
        events = pd.concat(collected)
        if alignment == "reward" and any(r.outcome_column is None for r in recipes):
            target_mask = events["outcome"].str.contains("Rew", regex=False)
            reference_mask = events["outcome"].str.contains("Stim", regex=False)
        elif any(r.outcome_column is None for r in recipes):
            target_mask = events["outcome"].str.contains("Stim", regex=False)
            reference_mask = events["outcome"].str.contains("Rew", regex=False)
        else:
            target_mask = events["outcome"].isin(_TARGET_OUTCOMES)
            reference_mask = ~target_mask
        out[group] = {
            "target": events[target_mask],
            "reference": events[reference_mask],
        }
    return out
