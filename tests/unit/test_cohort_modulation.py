"""Tests for modulation indices and modulated-ROI selection."""

from __future__ import annotations

import pandas as pd
import pytest

from calcium2p.cohort.modulation import (
    events_in_window,
    modulated_rois,
    modulation_index,
    modulation_proportions,
)


def events_frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    """Per-event frame: (roi key, ev_onset) rows."""
    index = [roi for roi, _ in rows]
    return pd.DataFrame({"ev_onset": [onset for _, onset in rows]}, index=index)


def fixed_trials(roi: object) -> tuple[int, int]:
    """Every ROI has 10 target and 20 reference trials."""
    return (10, 20)


class TestEventsInWindow:
    def test_bounds_are_strict_on_both_sides(self) -> None:
        events = events_frame([("r1", 5.1), ("r2", 5.5), ("r3", 6.1)])
        inside = events_in_window(events, (5.1, 6.1))
        assert inside.index.tolist() == ["r2"]


class TestModulationIndex:
    def test_index_formula_with_both_sides(self) -> None:
        target = events_frame([("r1", 5.5), ("r1", 5.6)])  # 2 events / 10 trials = 0.2
        reference = events_frame([("r1", 5.5)])  # 1 event / 20 trials = 0.05
        mi = modulation_index(target, reference, (5.1, 6.1), fixed_trials)
        assert mi["r1"] == pytest.approx((0.2 - 0.05) / (0.2 + 0.05))

    def test_no_reference_events_gives_rate_capped_at_one(self) -> None:
        # preserved notebook quirk: the value is the rate itself, capped at 1
        target = events_frame([("r1", 5.5), ("r1", 5.6), ("r2", 5.5)])
        reference = events_frame([("other", 5.5)])
        mi = modulation_index(target, reference, (5.1, 6.1), fixed_trials)
        assert mi["r1"] == pytest.approx(0.2)
        assert mi["r2"] == pytest.approx(0.1)

    def test_rate_above_one_saturates(self) -> None:
        target = events_frame([("r1", 5.5)] * 15)  # 15 events / 10 trials = 1.5
        reference = events_frame([("other", 5.5)])
        mi = modulation_index(target, reference, (5.1, 6.1), fixed_trials)
        assert mi["r1"] == 1.0

    def test_reference_only_rois_get_no_index(self) -> None:
        target = events_frame([("r1", 5.5)])
        reference = events_frame([("r2", 5.5)])
        mi = modulation_index(target, reference, (5.1, 6.1), fixed_trials)
        assert "r2" not in mi.index

    def test_multiindex_roi_keys_stay_whole_tuples(self) -> None:
        index = pd.MultiIndex.from_tuples([("mouseA", "roi1"), ("mouseA", "roi1")])
        target = pd.DataFrame({"ev_onset": [5.5, 5.6]}, index=index)
        reference = pd.DataFrame(
            {"ev_onset": [5.5]}, index=pd.MultiIndex.from_tuples([("mouseA", "roi1")])
        )
        mi = modulation_index(target, reference, (5.1, 6.1), fixed_trials)
        assert list(mi.index) == [("mouseA", "roi1")]


class TestModulatedRois:
    def test_median_threshold_is_exclusive(self) -> None:
        mi = pd.Series({"a": 0.1, "b": 0.2, "c": 0.3})
        assert modulated_rois(mi, "median") == ["c"]

    def test_half_std_threshold(self) -> None:
        mi = pd.Series({"a": -1.0, "b": 0.0, "c": 1.0})
        selected = modulated_rois(mi, "half_std")
        # 0.5 * std([-1, 0, 1]) = 0.408...; only c exceeds it
        assert selected == ["c"]


class TestModulationProportions:
    def test_matches_notebook_rounding(self) -> None:
        table = modulation_proportions(
            {"passive_axons": ["r"] * 130}, {"passive_axons": ["r"] * 194}
        )
        row = table.loc["passive_axons"]
        assert row["n_modulated"] == 130
        assert row["n_total"] == 194
        assert row["percent"] == 67.0
