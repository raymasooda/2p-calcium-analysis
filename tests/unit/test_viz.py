"""Tests for the viz figure builders (headless, synthetic data)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from calcium2p.core.cohort_config import ComparisonFigure, HistogramSpec
from calcium2p.viz import (
    comparison_bar,
    comparison_colors,
    figure_to_svg,
    latency_regression,
    onset_histogram_grid,
    roi_heatmap,
    roi_overlay,
    roi_waterfall,
)


def make_events(n: int, low: float, high: float, seed: int) -> pd.DataFrame:
    """Per-event frame with ev_onset uniform in [low, high)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"ev_onset": rng.uniform(low, high, size=n)})


def make_series(n: int, seed: int, name: tuple[str, str]) -> pd.Series:
    """Named per-ROI metric series."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(5.5, 0.2, size=n), name=name)


def make_traces(n_trials: int, n_frames: int, seed: int) -> pd.DataFrame:
    """Trial-by-frame trace frame with string trial labels."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.1, size=(n_trials, n_frames)),
        index=[str(i) for i in range(n_trials)],
    )


class TestFigureToSvg:
    def test_returns_svg_document(self) -> None:
        fig = Figure()
        fig.subplots().plot([0, 1], [0, 1])
        svg = figure_to_svg(fig)
        assert svg.startswith("<?xml")
        assert "</svg>" in svg

    def test_identical_figures_render_byte_identically(self) -> None:
        def build() -> str:
            fig = Figure(figsize=(2, 2))
            ax = fig.subplots()
            ax.plot([0, 1, 2], [1, 0, 1], color="darkred")
            return figure_to_svg(fig)

        assert build() == build()


class TestOnsetHistogramGrid:
    def test_grid_shape_and_weight_math(self) -> None:
        spec = HistogramSpec(xmin=5.0, xmax=6.5, nbins=15, roi_div=True)
        target = {"task_axons": make_events(40, 5.0, 6.4, seed=1)}
        reference = {"task_axons": make_events(25, 5.0, 6.4, seed=2)}
        fig = onset_histogram_grid(
            target,
            reference,
            target_trials={"task_axons": 68},
            reference_trials={"task_axons": 30},
            roi_counts={"task_axons": 353},
            spec=spec,
        )
        assert len(fig.axes) == 10
        # panel (0, 1) is task_axons; the stepfilled polygon's peak equals the
        # tallest weighted bin: count * (1 / nTrials / nROIs)
        ax = fig.axes[1]
        polygon_ys = ax.patches[0].get_xy()[:, 1]
        bins = np.linspace(spec.xmin, spec.xmax, spec.nbins + 1)
        onsets = target["task_axons"]["ev_onset"]
        expected, _ = np.histogram(
            onsets[onsets <= spec.xmax],
            bins=bins,
            weights=np.full((onsets <= spec.xmax).sum(), 1 / 68 / 353),
        )
        assert polygon_ys.max() == pytest.approx(expected.max())

    def test_align_at_relabels_ticks(self) -> None:
        # passive_dendrites sits on the bottom row, where shared-x tick labels live
        spec = HistogramSpec(xmin=3.0, xmax=3.5, nbins=15)
        fig = onset_histogram_grid(
            {"passive_dendrites": make_events(5, 3.0, 3.4, seed=3)},
            {},
            target_trials={"passive_dendrites": 10},
            reference_trials={},
            roi_counts={"passive_dendrites": 4},
            spec=spec,
            align_at=3.0,
        )
        labels = [t.get_text() for t in fig.axes[5].get_xticklabels()]
        assert labels
        assert labels[0] == "0.0"


class TestComparisonBar:
    def test_deterministic_across_calls(self) -> None:
        a = make_series(10, seed=1, name=("NO_DREADDs", "cHIT"))
        b = make_series(12, seed=2, name=("NO_DREADDs", "nHIT"))
        fig_params = ComparisonFigure(ylim=(5.0, 6.55))

        def offsets() -> list[float]:
            fig = comparison_bar(
                a,
                b,
                title="early reward\nNO_DREADDs cHIT vs. nHIT early reward (*)",
                figure=fig_params,
                edgecolor="darkred",
                facecolor="darkred",
            )
            scatter = fig.axes[0].collections[0]
            return [float(x) for x, _ in scatter.get_offsets()]

        assert offsets() == offsets()

    def test_layout_and_labels(self) -> None:
        a = make_series(5, seed=3, name=("passive_dendrites", "cRew"))
        b = make_series(5, seed=4, name=("NO_DREADDs", "cHIT"))
        fig = comparison_bar(
            a,
            b,
            title="t",
            figure=ComparisonFigure(),
            edgecolor="darkred",
            facecolor="white",
        )
        ax = fig.axes[0]
        assert ax.get_xlim() == (-0.65, 1.6)
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert labels == ["passive_dendrites\ncRew", "NO_DREADDs\ncHIT"]
        assert fig.get_size_inches() == pytest.approx([6.34 / 4, 2.0])

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Passive axons cRew vs. nRew", ("darkgreen", "darkgreen")),
            ("Control dendrites pre-CNO cHIT vs. nHIT", ("darkred", "white")),
            ("hM4D dendrites post-CNO cHIT vs. nHIT", ("darkred", "darkred")),
        ],
    )
    def test_comparison_colors(self, name: str, expected: tuple[str, str]) -> None:
        assert comparison_colors(name) == expected


class TestLatencyRegression:
    def test_returns_regression_result_and_correct_symbols(self) -> None:
        rng = np.random.default_rng(5)
        x = pd.Series(rng.normal(size=50))
        y = pd.Series(x * 0.9 + rng.normal(scale=0.05, size=50))
        fig, line = latency_regression(x, y)
        assert line.pvalue < 0.0001
        legend_text = fig.axes[0].get_legend().get_texts()[0].get_text()
        assert "(****)" in legend_text
        assert fig.get_size_inches() == pytest.approx([1.394, 1.4741])

    def test_nonsignificant_labelled_ns(self) -> None:
        rng = np.random.default_rng(6)
        x = pd.Series(rng.normal(size=30))
        y = pd.Series(rng.normal(size=30))
        fig, line = latency_regression(x, y)
        assert line.pvalue > 0.05
        legend_text = fig.axes[0].get_legend().get_texts()[0].get_text()
        assert "(n.s)" in legend_text


class TestTraces:
    def test_waterfall_orders_event_trials_first(self) -> None:
        trials = make_traces(6, 50, seed=7)
        fig = roi_waterfall(
            trials,
            ["4", "2"],
            fps=30.3,
            event_color="maroon",
            onset_lines=((1.0, "navy"),),
        )
        assert len(fig.axes) == 6
        first_line = fig.axes[0].lines[0]
        assert first_line.get_color() == "maroon"
        assert fig.axes[2].lines[0].get_color() == "black"

    def test_overlay_draws_mean_thicker(self) -> None:
        traces = {"cRew": make_traces(4, 40, seed=8), "nRew": make_traces(3, 40, seed=9)}
        fig = roi_overlay(
            traces,
            {"cRew": "darkgreen", "nRew": "darkorange"},
            fps=30.0,
            onset_line_s=0.5,
        )
        assert len(fig.axes) == 2
        widths = [line.get_linewidth() for line in fig.axes[0].lines]
        assert max(widths) == 2.0
        assert widths.count(0.75) == 4

    def test_heatmap_panels_and_colorbars(self) -> None:
        frames = {"cHIT": make_traces(5, 60, seed=10), "nHIT": make_traces(7, 60, seed=11)}
        fig = roi_heatmap(
            frames,
            {"cHIT": "BuGn", "nHIT": "Oranges"},
            fps=30.3,
            onset_line_s=1.0,
            cbar_ticks=[0, 1, 2],
        )
        # 2 heatmap axes + 2 colorbar axes
        assert len(fig.axes) == 4
        assert fig.axes[0].get_title() == "cHIT trials"
