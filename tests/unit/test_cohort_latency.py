"""Tests for the lick vs Ca2+ latency regression recipe."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calcium2p.cohort.latency import latency_correlation, per_trial_latencies


def events() -> pd.DataFrame:
    """Two sessions; one ROI fires twice in a trial, one event is out of window."""
    index = pd.MultiIndex.from_tuples(
        [("S1", "ROI0"), ("S1", "ROI0"), ("S1", "ROI1"), ("S1", "ROI2"), ("S2", "ROI0")]
    )
    return pd.DataFrame(
        {"trial": [1, 1, 1, 2, 1], "ev_onset": [5.1, 5.2, 5.4, 5.5, 5.3]}, index=index
    )


def lick() -> pd.Series:
    index = pd.MultiIndex.from_tuples([("S1", 1), ("S1", 2), ("S2", 1)], names=["session", "trial"])
    return pd.Series([0.2, 0.3, np.nan], index=index, name="lick_latency")


class TestPerTrialLatencies:
    def test_event_weighted_mean_from_window_start(self) -> None:
        rois = [("S1", "ROI0"), ("S1", "ROI1"), ("S1", "ROI2"), ("S2", "ROI0")]
        table = per_trial_latencies(events(), rois, lick())
        # S1 trial 1: three events (ROI0 twice) -> mean of 0.1, 0.2, 0.4
        assert table.loc[("S1", 1), "ev_onset"] == pytest.approx((0.1 + 0.2 + 0.4) / 3)
        # 5.5 is outside the strict (5.0, 5.5) window: S1 trial 2 is absent
        assert ("S1", 2) not in table.index
        assert np.isnan(table.loc[("S2", 1), "lick_latency"])

    def test_only_the_given_rois(self) -> None:
        table = per_trial_latencies(events(), [("S1", "ROI1")], lick())
        assert list(table.index) == [("S1", 1)]
        assert table.loc[("S1", 1), "ev_onset"] == pytest.approx(0.4)


class TestLatencyCorrelation:
    def test_drops_missing_licks_then_clips(self) -> None:
        rng = np.random.default_rng(0)
        lick_values = rng.normal(size=60)
        onset_values = 0.5 * lick_values + rng.normal(scale=0.5, size=60)
        onset_values[0] = 50.0  # far outlier: clipped at |z| >= 3
        lick_values[1] = np.nan  # no lick: dropped before z-scoring
        index = pd.MultiIndex.from_arrays([["S"] * 60, range(60)], names=["session", "trial"])
        table = pd.DataFrame({"ev_onset": onset_values, "lick_latency": lick_values}, index=index)
        points, result = latency_correlation(table)
        assert result.n_no_lick == 1
        assert result.n_trials == 59
        assert result.n_points == 58
        assert ("S", 0) not in points.index
        assert result.r > 0.5
        assert result.n_sessions == 1
