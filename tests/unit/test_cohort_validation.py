"""Tests for the validation layer (constructed mismatches, no real data)."""

from __future__ import annotations

import pandas as pd

from calcium2p.cohort.validation import (
    NOT_REGENERABLE,
    NOTEBOOK_REFERENCE_VALUES,
    check_reference_values,
    compare_event_tables,
)


def events(rows: list[tuple[str, str, float, float, float]]) -> pd.DataFrame:
    """(roi, trial, onset, offset, peak) rows."""
    frame = pd.DataFrame(rows, columns=["roi", "trial", "ev_onset", "ev_offset", "peak"]).set_index(
        "roi"
    )
    frame["integral"] = 1.0
    return frame


class TestCompareEventTables:
    def test_exact_onset_only_missing_and_extra_counted(self) -> None:
        reference = events(
            [
                ("r1", "0", 5.2, 5.5, 1.0),  # exact match
                ("r1", "1", 5.3, 5.9, 1.0),  # onset-only (offset differs)
                ("r2", "0", 5.4, 5.6, 1.0),  # missing
            ]
        )
        regenerated = events(
            [
                ("r1", "0", 5.2, 5.5, 1.0),
                ("r1", "1", 5.3, 6.0, 1.0),
                ("r3", "0", 5.1, 5.2, 1.0),  # extra
            ]
        )
        report = compare_event_tables(regenerated, reference, group="g", side="target")
        assert report.n_exact == 1
        assert report.n_onset_only == 1
        assert report.n_missing == 1
        assert report.n_extra == 1
        assert report.exact_fraction == 1 / 3
        assert "1/3 exact" in report.describe()

    def test_metric_diffs_only_over_exact_matches(self) -> None:
        reference = events([("r1", "0", 5.2, 5.5, 2.0)])
        regenerated = events([("r1", "0", 5.2, 5.5, 2.5)])
        report = compare_event_tables(regenerated, reference, group="g", side="target")
        assert report.max_metric_diff["peak"] == 0.5

    def test_tuple_roi_keys_match(self) -> None:
        index = pd.MultiIndex.from_tuples([("mouseA", "ROI1")])
        reference = pd.DataFrame(
            {"trial": ["0"], "ev_onset": [5.2], "ev_offset": [5.5]}, index=index
        )
        regenerated = pd.DataFrame(
            {"trial": ["0"], "ev_onset": [5.2], "ev_offset": [5.5]}, index=index
        )
        report = compare_event_tables(regenerated, reference, group="g", side="target")
        assert report.n_exact == 1

    def test_empty_reference_is_perfect(self) -> None:
        empty = events([])
        report = compare_event_tables(empty, empty, group="g", side="target")
        assert report.exact_fraction == 1.0


class TestReferenceValues:
    def test_all_twelve_modulated_proportions_pinned(self) -> None:
        reward = NOTEBOOK_REFERENCE_VALUES["reward.modulated_proportions"]["expected"]
        stimulus = NOTEBOOK_REFERENCE_VALUES["stimulus.modulated_proportions"]["expected"]
        assert len(reward) == 8
        assert len(stimulus) == 4
        assert reward["passive_axons"] == (130, 194)
        assert reward["control_dendrites (post-CNO)"] == (91, 186)

    def test_dendrite_groups_marked_not_regenerable(self) -> None:
        assert "task_dendrites_TD1" in NOT_REGENERABLE
        assert "control_dendrites_TD3" in NOT_REGENERABLE

    def test_check_passes_within_tolerance(self) -> None:
        computed = {
            "reward.regression.task_axons_p_value": 0.9108585,  # within rel_tol
        }
        table = check_reference_values(computed)
        row = table[table["name"] == "reward.regression.task_axons_p_value"].iloc[0]
        assert bool(row["passed"])

    def test_check_fails_on_wrong_value_and_missing(self) -> None:
        computed = {"reward.regression.task_axons_p_value": 0.5}
        table = check_reference_values(computed)
        by_name = table.set_index("name")["passed"]
        assert not bool(by_name["reward.regression.task_axons_p_value"])
        assert not bool(by_name["reward.modulated_proportions"])  # missing -> failed

    def test_mapping_check_compares_nested_values(self) -> None:
        computed = {
            "stimulus.modulated_proportions": {
                "task_axons": (72, 353),
                "NO_DREADDs": (25, 112),
                "task_dendrites (pre-CNO)": (36, 331),
                "task_dendrites (post-CNO)": (102, 503),
            }
        }
        table = check_reference_values(computed)
        row = table[table["name"] == "stimulus.modulated_proportions"].iloc[0]
        assert bool(row["passed"])
