"""Tests for per-ROI event metrics and comparison frames."""

from __future__ import annotations

import pandas as pd
import pytest

from calcium2p.cohort.grouping import TrialCounts
from calcium2p.cohort.metrics import (
    comparison_frame,
    event_metrics_by_roi,
    normalize_by_reference,
)
from calcium2p.core.cohort_config import EpochSpec, NormalizeSpec


def make_counts() -> TrialCounts:
    """Flat protocol with a bare HIT key plus outcome variants."""
    return TrialCounts(
        all_trials={
            "task_axons": {"muji_FOV2": {"HIT": 10, "cHIT": 4, "nHIT": 6, "FA": 2, "ALL": 12}}
        },
        target_totals={"task_axons": 10},
        reference_totals={"task_axons": 2},
        alignment="reward",
    )


def make_events() -> pd.DataFrame:
    """Six events for one ROI across outcomes, some outside the window."""
    return pd.DataFrame(
        {
            "outcome": ["cHIT", "cHIT", "nHIT", "nHIT", "FA", "cHIT"],
            "ev_onset": [5.2, 5.4, 5.3, 6.5, 5.2, 5.15],
            "peak": [1.0, 2.0, 3.0, 9.0, 4.0, 5.0],
        },
        index=["muji_FOV2_66.0"] * 6,
    )


EPOCH = EpochSpec(name="early reward", default_window=(5.1, 6.1))


class TestEventMetricsByRoi:
    def test_windowing_rates_and_pooled_hit(self) -> None:
        metrics = event_metrics_by_roi(
            {"task_axons": make_events()},
            {"task_axons": make_events().iloc[:0]},
            EPOCH,
            make_counts(),
            {"task_axons": [("muji_FOV2_66",)]},
            roi_set="active",
        )
        per = metrics["task_axons"]
        # 6.5 falls outside the window; FA is whitelisted but not split out
        chit = per["cHIT"]
        assert chit.shape[0] == 1
        key = chit.index[0]
        assert key[0] == "cHIT"
        assert chit.loc[key, "rate"] == pytest.approx(3 / 4)  # 3 windowed cHIT / 4 cHIT trials
        # pooled HIT row: 4 windowed HIT-ish events / bare HIT key (10 trials)
        hit = per["HIT"]
        assert hit.iloc[0]["rate"] == pytest.approx(4 / 10)
        assert hit.iloc[0]["peak"] == pytest.approx((1.0 + 2.0 + 3.0 + 5.0) / 4)

    def test_modulated_roi_filter(self) -> None:
        metrics = event_metrics_by_roi(
            {"task_axons": make_events()},
            {"task_axons": make_events().iloc[:0]},
            EPOCH,
            make_counts(),
            {"task_axons": []},
            roi_set="modulated",
            modulated={"task_axons": []},
        )
        assert "task_axons" not in metrics  # every ROI filtered out


class TestComparisonFrame:
    def test_columns_are_protocol_outcome_pairs(self) -> None:
        metrics = event_metrics_by_roi(
            {"task_axons": make_events()},
            {"task_axons": make_events().iloc[:0]},
            EPOCH,
            make_counts(),
            {"task_axons": []},
        )
        frame = comparison_frame(metrics, "rate")
        assert ("task_axons", "cHIT") in frame.columns
        assert frame.loc["muji_FOV2_66.0", ("task_axons", "cHIT")] == pytest.approx(3 / 4)


class TestNormalizeByReference:
    def test_divides_each_side_by_reference_mean(self) -> None:
        columns = pd.MultiIndex.from_tuples(
            [("p", "cRew"), ("p", "nRew"), ("q", "cHIT"), ("q", "nHIT")]
        )
        frame = pd.DataFrame(
            [[2.0, 1.0, 6.0, 2.0], [4.0, 3.0, 9.0, 4.0]], columns=columns, index=["r1", "r2"]
        )
        normalized = normalize_by_reference(
            frame,
            ("p", "cRew"),
            ("q", "cHIT"),
            NormalizeSpec(a_reference=("p", "nRew"), b_reference=("q", "nHIT")),
        )
        assert normalized[("p", "cRew")].tolist() == [1.0, 2.0]  # divided by mean(nRew)=2
        assert normalized[("q", "cHIT")].tolist() == [2.0, 3.0]  # divided by mean(nHIT)=3

    def test_offset_measures_from_the_offset(self) -> None:
        # the published normalized onset figure: (x - 5) / (mean(ref) - 5)
        columns = pd.MultiIndex.from_tuples(
            [("p", "cRew"), ("p", "nRew"), ("q", "cHIT"), ("q", "nHIT")]
        )
        frame = pd.DataFrame(
            [[5.2, 5.4, 5.3, 5.5], [5.6, 5.6, 5.9, 5.5]], columns=columns, index=["r1", "r2"]
        )
        normalized = normalize_by_reference(
            frame,
            ("p", "cRew"),
            ("q", "cHIT"),
            NormalizeSpec(a_reference=("p", "nRew"), b_reference=("q", "nHIT"), offset=5.0),
        )
        assert normalized[("p", "cRew")].tolist() == pytest.approx([0.4, 1.2])  # / mean 0.5
        assert normalized[("q", "cHIT")].tolist() == pytest.approx([0.6, 1.8])  # / mean 0.5

    def test_zero_offset_is_the_plain_ratio(self) -> None:
        assert NormalizeSpec(a_reference=("p", "nRew"), b_reference=("q", "nHIT")).offset == 0.0
