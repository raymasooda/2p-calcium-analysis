"""Tests for hysteresis event detection and the per-event metrics table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calcium2p.signal import detect_events, event_metrics_table
from calcium2p.signal.events import EVENT_METRIC_COLUMNS


def boxcar_trace(
    n_frames: int = 100,
    start: int = 50,
    width: int = 10,
    amplitude: float = 5.0,
) -> np.ndarray:
    """Produce a flat trace with one boxcar transient and a known-std baseline.

    Frames 10..19 alternate +/-0.1 so the 1--2 s baseline window at 10 fps has
    a standard deviation of exactly 0.1.
    """
    trace = np.zeros(n_frames)
    trace[10:20] = np.tile([0.1, -0.1], 5)
    trace[start : start + width] = amplitude
    return trace


def single_trial_frame(trace: np.ndarray) -> pd.DataFrame:
    """Wrap one trace as a trial-by-frame DataFrame with a named MultiIndex."""
    index = pd.MultiIndex.from_tuples([("ses1", "roi1", 0)], names=["session", "roi", "trial"])
    return pd.DataFrame([trace], index=index)


class TestDetectEvents:
    def test_boxcar_detected_as_single_event(self) -> None:
        trace = pd.Series(boxcar_trace())
        events = detect_events(trace, upper=0.2, lower=0.2, min_width=6)
        assert len(events) == 1
        assert events[0].index[0] == 50
        assert events[0].index[-1] == 59

    def test_min_width_boundary(self) -> None:
        base = np.zeros(30)
        base[5:10] = 1.0  # 5 samples above threshold
        too_short = pd.Series(base)
        assert detect_events(too_short, 0.5, 0.5, min_width=6) == []

        base = np.zeros(30)
        base[5:11] = 1.0  # exactly 6 samples
        long_enough = pd.Series(base)
        assert len(detect_events(long_enough, 0.5, 0.5, min_width=6)) == 1

    def test_event_at_trace_start_is_still_detected(self) -> None:
        # detect_events itself does not filter onset; that is the metrics
        # table's job (the legacy ev_onset > 0 filter).
        base = np.zeros(30)
        base[0:8] = 1.0
        events = detect_events(pd.Series(base), 0.5, 0.5, min_width=6)
        assert len(events) == 1
        assert events[0].index[0] == 0

    def test_events_carry_trace_name(self) -> None:
        trace = pd.Series(boxcar_trace(), name=("ses1", "roi1", 3))
        events = detect_events(trace, 0.2, 0.2, min_width=6)
        assert events[0].name == ("ses1", "roi1", 3)


class TestEventMetricsTable:
    def test_single_boxcar_metrics(self) -> None:
        frame = single_trial_frame(boxcar_trace())
        out = event_metrics_table(frame, fps=10.0)
        assert out.shape[0] == 1
        row = out.iloc[0]
        assert row["session"] == "ses1"
        assert row["roi"] == "roi1"
        assert row["trial"] == 0
        assert row["peak"] == 5.0
        assert row["ev_onset"] == 5.0
        assert row["peak_time"] == 5.0
        assert row["ev_offset"] == 5.9
        assert row["ev_duration"] == 1.0
        # Simpson's rule over 10 constant samples at unit spacing
        np.testing.assert_allclose(row["integral"], 45.0)

    def test_threshold_uses_flattened_baseline_std(self) -> None:
        # baseline std is 0.1 -> threshold 0.2 at threshold_sd=2. An event of
        # amplitude 0.15 must not be detected; amplitude 0.25 must be.
        quiet = single_trial_frame(boxcar_trace(amplitude=0.15))
        loud = single_trial_frame(boxcar_trace(amplitude=0.25))
        assert event_metrics_table(quiet, fps=10.0).empty
        assert event_metrics_table(loud, fps=10.0).shape[0] == 1

    def test_event_starting_at_frame_zero_is_filtered(self) -> None:
        frame = single_trial_frame(boxcar_trace(start=0))
        out = event_metrics_table(frame, fps=10.0)
        assert out.empty
        assert list(out.columns) == ["session", "roi", "trial", *EVENT_METRIC_COLUMNS]

    def test_multiple_trials_yield_one_row_each(self) -> None:
        index = pd.MultiIndex.from_tuples(
            [("ses1", "roi1", t) for t in range(3)], names=["session", "roi", "trial"]
        )
        frame = pd.DataFrame(
            [boxcar_trace(start=30), boxcar_trace(start=50), boxcar_trace(start=70, width=8)],
            index=index,
        )
        out = event_metrics_table(frame, fps=10.0)
        assert out.shape[0] == 3
        assert sorted(out["trial"]) == [0, 1, 2]
        np.testing.assert_allclose(sorted(out["ev_onset"]), [3.0, 5.0, 7.0])

    def test_min_samples_respected(self) -> None:
        frame = single_trial_frame(boxcar_trace(width=5))
        assert event_metrics_table(frame, fps=10.0, min_samples=6).empty
        assert event_metrics_table(frame, fps=10.0, min_samples=5).shape[0] == 1

    def test_plain_index_gets_level_name_fallback(self) -> None:
        frame = pd.DataFrame([boxcar_trace()], index=[7])
        out = event_metrics_table(frame, fps=10.0)
        assert out.shape[0] == 1
        assert out.iloc[0]["level_0"] == 7
