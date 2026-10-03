"""Tests for the comparison-figure naming in the cohort orchestrator."""

from __future__ import annotations

from calcium2p.cohort.run import figure_filename, figure_key
from calcium2p.core.cohort_config import ComparisonSpec, NormalizeSpec


def spec(*, normalized: bool) -> ComparisonSpec:
    """A comparison, plain or normalized, otherwise identical."""
    return ComparisonSpec(
        name="Passive dendrites cRew vs. cHIT",
        a=("passive_dendrites", "cRew"),
        b=("NO_DREADDs", "cHIT"),
        metric="rate",
        epoch="early reward",
        normalize=NormalizeSpec(
            a_reference=("passive_dendrites", "nRew"), b_reference=("NO_DREADDs", "nHIT")
        )
        if normalized
        else None,
    )


class TestFigureNaming:
    def test_filename_matches_the_notebook_convention(self) -> None:
        assert figure_filename(spec(normalized=False)) == (
            "Passive dendrites cRew vs. cHIT early reward rate.svg"
        )
        assert figure_filename(spec(normalized=True)) == (
            "non-consecutive-normalized Passive dendrites cRew vs. cHIT early reward rate.svg"
        )

    def test_normalized_and_plain_figures_get_distinct_keys(self) -> None:
        # regression: a shared key made the store serve the cached plain figure
        # in place of the normalized one within the same run
        plain = figure_key("run", spec(normalized=False), "legacy-true")
        normalized = figure_key("run", spec(normalized=True), "legacy-true")
        assert plain != normalized

    def test_plain_key_is_unchanged(self) -> None:
        # existing stores keep their cache hits for plain comparisons
        assert figure_key("run", spec(normalized=False), "legacy-true") == (
            "run/figures/Passive_dendrites_cRew_vs_cHIT/rate/legacy-true"
        )
