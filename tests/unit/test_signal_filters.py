"""Tests for the zero-phase Butterworth low-pass filter port."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from calcium2p.signal import lowpass_filter


def make_traces(n_trials: int = 4, n_frames: int = 300, seed: int = 20260409) -> pd.DataFrame:
    """Produce smooth sine traces with a little deterministic noise."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 4.0 * np.pi, n_frames)
    rows = [np.sin(t + phase) + rng.normal(scale=0.05, size=n_frames) for phase in range(n_trials)]
    return pd.DataFrame(rows, index=[f"trial{i}" for i in range(n_trials)])


class TestLowpassFilter:
    def test_sos_and_ba_agree_on_smooth_signal(self) -> None:
        traces = make_traces()
        out_sos = lowpass_filter(traces, design="sos")
        out_ba = lowpass_filter(traces, design="ba")
        # Edge padding differs between the designs; the interior must agree.
        interior_sos = out_sos.to_numpy()[:, 20:-20]
        interior_ba = out_ba.to_numpy()[:, 20:-20]
        np.testing.assert_allclose(interior_sos, interior_ba, rtol=1e-4, atol=1e-6)

    def test_ba_matches_legacy_scipy_reference(self) -> None:
        traces = make_traces(n_trials=1)
        out = lowpass_filter(traces, order=3, critical_freq=0.2, design="ba")
        b, a = signal.butter(3, 0.2)
        expected = signal.filtfilt(b, a, traces.to_numpy()[0])
        np.testing.assert_allclose(out.to_numpy()[0], expected)

    def test_attenuates_high_frequency_noise(self) -> None:
        rng = np.random.default_rng(7)
        t = np.linspace(0.0, 4.0 * np.pi, 400)
        clean = np.sin(t)
        noisy = clean + rng.normal(scale=0.3, size=t.size)
        traces = pd.DataFrame([noisy])
        out = lowpass_filter(traces, order=3, critical_freq=0.1).to_numpy()[0]
        assert np.abs(out - clean).mean() < np.abs(noisy - clean).mean()

    def test_preserves_index_and_columns(self) -> None:
        traces = make_traces(n_trials=3, n_frames=64)
        traces.columns = [f"f{i}" for i in range(64)]
        out = lowpass_filter(traces)
        assert list(out.index) == list(traces.index)
        assert list(out.columns) == list(traces.columns)

    def test_nan_rows_are_compacted_like_legacy(self) -> None:
        traces = make_traces(n_trials=2, n_frames=100)
        traces.iloc[1, 90:] = np.nan
        out = lowpass_filter(traces)
        # legacy behaviour: NaN samples are dropped before filtering, so the
        # shorter row is left-compacted and padded with NaN at the end
        assert out.shape == (2, 100)
        assert out.iloc[1, 90:].isna().all()
        assert not out.iloc[0].isna().any()

    def test_rejects_unknown_design(self) -> None:
        with pytest.raises(ValueError, match="design"):
            lowpass_filter(make_traces(), design="fir")  # type: ignore[arg-type]
