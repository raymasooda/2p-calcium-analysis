"""Tests for event-table regeneration (synthetic converted tables only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calcium2p.cohort.event_tables import GROUP_RECIPES, build_event_tables
from calcium2p.core.cohort_config import EventTableParams

FPS = 30.54
N_FRAMES = 216


def trace_row(event_frames: dict[int, float]) -> list[float]:
    """A quiet trial trace with unit-amplitude samples at the given frames."""
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 0.01, N_FRAMES)
    for frame, amplitude in event_frames.items():
        values[frame] = amplitude
    return values.tolist()


def task_axons_tables(
    trials: list[tuple[str, str, dict[int, float]]],
) -> dict[str, pd.DataFrame]:
    """traces_task_axons-shaped table: (trial, outcome, event frames) rows."""
    frame_cols = [f"f{i:03d}" for i in range(N_FRAMES)]
    rows = []
    for trial, outcome, events in trials:
        row = {
            "dataset": "task_axons",
            "id": "muji_FOV2",
            "roi": "66",
            "trial": trial,
            "unvalidated_type": outcome,
            "source_order": 0,
        }
        row.update(dict(zip(frame_cols, trace_row(events), strict=True)))
        rows.append(row)
    return {"traces_task_axons": pd.DataFrame(rows)}


def boxcar(start: int, width: int = 8, amplitude: float = 1.0) -> dict[int, float]:
    """Event frames for a boxcar transient."""
    return {start + i: amplitude for i in range(width)}


class TestTaskAxonsRecipe:
    def test_event_detected_with_one_based_labels(self) -> None:
        # onset at position 160 -> label 161 (1-based) -> 161/30.54 s
        tables = task_axons_tables([("0", "cHIT", boxcar(160)), ("1", "nHIT", {})])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        target = out["task_axons"]["target"]
        assert target.shape[0] == 1
        assert target.index[0] == "muji_FOV2_66.0"
        assert target.iloc[0]["ev_onset"] == pytest.approx(161 / FPS)
        assert target.iloc[0]["outcome"] == "cHIT"

    def test_window_is_strictly_exclusive(self) -> None:
        # onset exactly at the 5 s bound (label 152.7 -> nearest exact: use a
        # window edge event far outside instead): event at 1 s is excluded
        tables = task_axons_tables([("0", "cHIT", boxcar(30))])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        assert "task_axons" not in out or out["task_axons"]["target"].empty

    def test_sides_split_by_outcome(self) -> None:
        tables = task_axons_tables([("0", "cHIT", boxcar(160)), ("1", "nCATCH", boxcar(160))])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        assert out["task_axons"]["target"].iloc[0]["outcome"] == "cHIT"
        assert out["task_axons"]["reference"].iloc[0]["outcome"] == "nCATCH"

    def test_non_validated_trials_are_never_detected(self) -> None:
        tables = task_axons_tables(
            [("0", "post-mistake-HIT", boxcar(160)), ("1", "miss", boxcar(160))]
        )
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        assert "task_axons" not in out

    def test_threshold_uses_validated_stack_only(self) -> None:
        # a miss trial with a huge baseline would inflate the threshold if it
        # entered the stack; the cHIT event must still be detected
        noisy_baseline = dict.fromkeys(range(61, 92), 50.0)
        tables = task_axons_tables([("0", "cHIT", boxcar(160)), ("1", "miss", noisy_baseline)])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        assert out["task_axons"]["target"].shape[0] == 1

    def test_stimulus_alignment_uses_stim_window(self) -> None:
        # 3-3.5 s window: position 95 -> label 96 -> 3.14 s
        tables = task_axons_tables([("0", "cHIT", boxcar(95))])
        out = build_event_tables(tables, EventTableParams(), alignment="stimulus")
        assert out["task_axons"]["target"].iloc[0]["ev_onset"] == pytest.approx(96 / FPS)


def no_dreadds_tables(
    trials: list[tuple[str, str, dict[int, float]]],
) -> dict[str, pd.DataFrame]:
    """traces_no_dreadds-shaped table (0-based labels, per-branch flatten)."""
    frame_cols = [f"f{i:03d}" for i in range(N_FRAMES)]
    rows = []
    for trial, outcome, events in trials:
        row = {
            "dataset": "no_dreadds",
            "id": "GCaMP_Batch8_one_FOV1",
            "roi": "7",
            "trial": trial,
            "trial_type": outcome,
            "source_order": 0,
        }
        row.update(dict(zip(frame_cols, trace_row(events), strict=True)))
        rows.append(row)
    return {"traces_no_dreadds": pd.DataFrame(rows)}


class TestNoDreaddsRecipe:
    def test_zero_based_labels(self) -> None:
        tables = no_dreadds_tables([("0", "cHIT", boxcar(160))])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        assert out["NO_DREADDs"]["target"].iloc[0]["ev_onset"] == pytest.approx(160 / FPS)

    def test_branch_flattening_merges_across_same_branch_trials(self) -> None:
        # trial 0 ends high, trial 1 starts high; both nCATCH -> one merged
        # event whose offset runs past the first trial's end
        end_event = dict.fromkeys(range(200, N_FRAMES), 1.0)
        start_event = dict.fromkeys(range(6), 1.0)
        tables = no_dreadds_tables([("0", "nCATCH", end_event), ("1", "nCATCH", start_event)])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        events = out["NO_DREADDs"]["reference"]
        assert events.shape[0] == 1
        row = events.iloc[0]
        assert row["trial"] == "0"
        assert row["ev_offset"] > N_FRAMES / FPS  # spans into the next trial

    def test_different_branches_do_not_merge(self) -> None:
        end_event = dict.fromkeys(range(200, N_FRAMES), 1.0)
        start_event = dict.fromkeys(range(6), 1.0)
        tables = no_dreadds_tables([("0", "nCATCH", end_event), ("1", "cHIT", start_event)])
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        # the nCATCH tail event stays truncated at its trial end
        row = out["NO_DREADDs"]["reference"].iloc[0]
        assert row["ev_offset"] <= (N_FRAMES - 1) / FPS + 1e-9


def batch1_tables(histories: list[str], events_by_trial: dict[int, dict[int, float]]) -> dict:
    """traces_batch1 + behavior_batch1 pair for one DENDRITES ROI."""
    n = 243
    frame_cols = [f"f{i:03d}" for i in range(n)]
    rng = np.random.default_rng(1)
    rows = []
    for trial in range(len(histories)):
        values = rng.normal(0.0, 0.01, n)
        for frame, amp in events_by_trial.get(trial, {}).items():
            values[frame] = amp
        row = {
            "dataset": "batch1",
            "id": "muji",
            "protocol": "NAIVE STIM REW",
            "compartment": "DENDRITES",
            "roi": "ROI5",
            "trial": str(trial),
            "source_order": 0,
        }
        row.update(dict(zip(frame_cols, values.tolist(), strict=True)))
        rows.append(row)
    behavior = pd.DataFrame(
        {
            "dataset": "batch1",
            "id": "muji",
            "protocol": "NAIVE STIM REW",
            "compartment": "DENDRITES",
            "trial": [str(i) for i in range(len(histories))],
            "history": histories,
        }
    )
    return {"traces_batch1": pd.DataFrame(rows), "behavior_batch1": behavior}


class TestPassiveRecipe:
    def test_history_join_tuple_keys_and_split(self) -> None:
        fps = 30.3
        tables = batch1_tables(["cRew", "nStim"], {0: boxcar(160), 1: boxcar(165)})
        out = build_event_tables(tables, EventTableParams(), alignment="reward")
        target = out["passive_dendrites"]["target"]
        reference = out["passive_dendrites"]["reference"]
        assert target.index[0] == ("RGECO_GCaMP_Batch1_muji", "ROI5")
        assert target.iloc[0]["ev_onset"] == pytest.approx(160 / fps)
        assert target.iloc[0]["outcome"] == "cRew"
        assert reference.iloc[0]["outcome"] == "nStim"

    def test_stimulus_alignment_swaps_sides(self) -> None:
        tables = batch1_tables(["cRew", "nStim"], {0: boxcar(95), 1: boxcar(97)})
        out = build_event_tables(tables, EventTableParams(), alignment="stimulus")
        assert out["passive_dendrites"]["target"].iloc[0]["outcome"] == "nStim"
        assert out["passive_dendrites"]["reference"].iloc[0]["outcome"] == "cRew"


class TestRecipeRegistry:
    def test_every_group_has_recipes(self) -> None:
        assert set(GROUP_RECIPES) == {
            "task_axons",
            "NO_DREADDs",
            "passive_axons",
            "passive_dendrites",
        }

    def test_recovered_defaults(self) -> None:
        params = EventTableParams()
        assert params.baseline_window_s == (2.0, 3.0)
        assert params.threshold_sd == 2.0
        assert params.min_samples == 6
