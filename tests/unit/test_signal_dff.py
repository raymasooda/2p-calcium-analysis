"""Tests for the dF/F0 port, including its preserved legacy quirks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calcium2p.signal import compute_dff


class TestComputeDff:
    def test_constant_trace_gives_zero(self) -> None:
        traces = pd.DataFrame(np.full((5, 120), 7.0))
        out = compute_dff(traces, fps=30.0)
        np.testing.assert_allclose(out.to_numpy(), 0.0)

    def test_baseline_window_offset_quirk_pinned(self) -> None:
        # fps=10, window (2, 3) s -> nominal frames 20..29, but the legacy
        # slice is iloc[19:29]. With an arange trace those medians differ:
        # median(19..28) = 23.5 (quirk) vs median(20..29) = 24.5 (nominal).
        traces = pd.DataFrame([np.arange(100, dtype=float)])
        out = compute_dff(traces, fps=10.0, baseline_window_s=(2.0, 3.0))
        f0 = 23.5
        np.testing.assert_allclose(out.iloc[0, 0], (0.0 - f0) / f0)
        np.testing.assert_allclose(out.iloc[0, 50], (50.0 - f0) / f0)

    def test_rolling_median_smooths_across_trials(self) -> None:
        # Three constant trials with baselines 10, 20, 60. The 20-sample
        # centred rolling window covers all three rows at every position
        # (min_periods=1), so every trial's smoothed F0 is the median, 20.
        traces = pd.DataFrame(
            [np.full(80, 10.0), np.full(80, 20.0), np.full(80, 60.0)],
            index=["t0", "t1", "t2"],
        )
        out = compute_dff(traces, fps=10.0, baseline_window_s=(2.0, 3.0))
        np.testing.assert_allclose(out.loc["t0"].to_numpy(), (10.0 - 20.0) / 20.0)
        np.testing.assert_allclose(out.loc["t1"].to_numpy(), 0.0)
        np.testing.assert_allclose(out.loc["t2"].to_numpy(), (60.0 - 20.0) / 20.0)

    def test_preserves_index_and_columns(self) -> None:
        rng = np.random.default_rng(20260409)
        traces = pd.DataFrame(
            rng.uniform(1.0, 2.0, size=(4, 90)),
            index=pd.MultiIndex.from_product([["ses1"], range(4)], names=["session", "trial"]),
            columns=[f"f{i}" for i in range(90)],
        )
        out = compute_dff(traces, fps=30.0)
        assert out.index.equals(traces.index)
        assert list(out.columns) == list(traces.columns)
