"""Tests for declarative two-group comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from calcium2p.cohort.stats import ComparisonResult, run_comparison, run_comparisons
from calcium2p.core.cohort_config import ComparisonSpec


def make_frame() -> pd.DataFrame:
    """Comparison frame with (protocol, outcome) columns and NaN holes."""
    rng = np.random.default_rng(20260816)
    index = [f"roi{i}" for i in range(12)]
    columns = pd.MultiIndex.from_tuples(
        [
            ("NO_DREADDs", "cHIT"),
            ("NO_DREADDs", "nHIT"),
            ("passive_dendrites", "cRew"),
        ]
    )
    frame = pd.DataFrame(rng.normal(5.5, 0.3, size=(12, 3)), index=index, columns=columns)
    frame.iloc[0, 0] = np.nan
    frame.iloc[3, 1] = np.nan
    return frame


def spec(test: str = "student_t", p_threshold: float = 1.0) -> ComparisonSpec:
    """A comparison spec against the synthetic frame."""
    return ComparisonSpec(
        name="NO_DREADDs cHIT vs. nHIT",
        a=("NO_DREADDs", "cHIT"),
        b=("NO_DREADDs", "nHIT"),
        metric="ev_onset",
        epoch="early reward",
        test=test,  # type: ignore[arg-type]
        p_threshold=p_threshold,
    )


class TestRunComparison:
    def test_student_t_matches_scipy_with_equal_var(self) -> None:
        frame = make_frame()
        result = run_comparison(frame, spec("student_t"))
        a = frame[("NO_DREADDs", "cHIT")].dropna()
        b = frame[("NO_DREADDs", "nHIT")].dropna()
        expected = scipy_stats.ttest_ind(a, b, equal_var=True)
        assert result.p_value == pytest.approx(float(expected.pvalue))
        assert result.n_a == 11
        assert result.n_b == 11

    def test_student_and_welch_differ_on_unequal_variances(self) -> None:
        columns = pd.MultiIndex.from_tuples([("p", "a"), ("p", "b")])
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            np.column_stack([rng.normal(0, 0.1, 30), rng.normal(0.5, 3.0, 30)]),
            columns=columns,
        )
        base = ComparisonSpec(name="x", a=("p", "a"), b=("p", "b"), metric="rate", epoch="e")
        student = run_comparison(frame, base)
        welch = run_comparison(
            frame,
            ComparisonSpec(
                name="x", a=("p", "a"), b=("p", "b"), metric="rate", epoch="e", test="welch_t"
            ),
        )
        assert student.p_value != pytest.approx(welch.p_value)

    def test_mannwhitney_matches_scipy(self) -> None:
        frame = make_frame()
        result = run_comparison(frame, spec("mannwhitney"))
        a = frame[("NO_DREADDs", "cHIT")].dropna()
        b = frame[("NO_DREADDs", "nHIT")].dropna()
        expected = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        assert result.p_value == pytest.approx(float(expected.pvalue))

    def test_paired_tests_use_pairwise_complete_rows(self) -> None:
        frame = make_frame()
        result = run_comparison(frame, spec("paired_t"))
        # rows 0 and 3 have a NaN in one of the two groups -> 10 pairs remain
        assert result.n_a == 10
        assert result.n_b == 10

    def test_missing_column_raises_keyerror(self) -> None:
        frame = make_frame()
        bad = ComparisonSpec(
            name="missing", a=("nope", "cHIT"), b=("NO_DREADDs", "nHIT"), metric="rate", epoch="e"
        )
        with pytest.raises(KeyError):
            run_comparison(frame, bad)


class TestSignificanceSymbol:
    @pytest.mark.parametrize(
        ("p", "symbol"),
        [
            (0.00005, "****"),
            (0.0005, "***"),
            (0.005, "**"),
            (0.03, "*"),
            (0.4281, "0.428"),
        ],
    )
    def test_thresholds_match_notebook_convention(self, p: float, symbol: str) -> None:
        result = ComparisonResult(
            name="x",
            statistic=0.0,
            p_value=p,
            n_a=1,
            n_b=1,
            mean_a=0.0,
            mean_b=0.0,
            sem_a=0.0,
            sem_b=0.0,
        )
        assert result.significance_symbol == symbol


class TestRunComparisons:
    def test_threshold_gates_passing_but_not_tidy(self) -> None:
        frame = make_frame()
        specs = [spec("student_t", p_threshold=0.0), spec("student_t", p_threshold=1.0)]
        passing, tidy = run_comparisons(frame, specs)
        assert len(tidy) == 2
        assert len(passing) == 1
        assert tidy["passed"].tolist() == [False, True]
