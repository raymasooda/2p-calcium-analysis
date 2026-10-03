"""Declarative two-group comparisons over per-ROI metric frames.

Replaces the notebooks' ``sig_combinations`` machinery. Each comparison is a
named :class:`~calcium2p.core.cohort_config.ComparisonSpec` addressing its two
groups as ``(protocol, outcome)`` column pairs, so nothing depends on column
position or ``os.listdir`` order.

Test-name semantics (see ``LEGACY_TEST_NAMES`` in the config module): the
notebooks' "parametric" unpaired test was ``scipy.stats.ttest_ind`` with its
default ``equal_var=True`` -- Student's t, not Welch's. ``welch_t`` is offered
for corrected analyses but was never what the manuscript ran. Non-default scipy
arguments are pinned explicitly so p-values stay stable across scipy releases
within the pinned range.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from calcium2p.core.cohort_config import ComparisonSpec, TestName


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of one two-group comparison.

    Attributes
    ----------
    name
        The spec's name (figure title / filename stem).
    statistic, p_value
        Test statistic and two-sided p-value.
    n_a, n_b
        Group sizes after NaN handling.
    mean_a, mean_b, sem_a, sem_b
        Group means and standard errors, matching the notebooks' printed
        ``mean +/- sem`` blocks.
    """

    name: str
    statistic: float
    p_value: float
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    sem_a: float
    sem_b: float

    @property
    def significance_symbol(self) -> str:
        """Asterisk convention used in the notebook figure titles."""
        p = self.p_value
        if p < 0.0001:  # noqa: PLR2004 - the notebook's published thresholds
            return "****"
        if p < 0.001:  # noqa: PLR2004
            return "***"
        if p < 0.01:  # noqa: PLR2004
            return "**"
        if p < 0.05:  # noqa: PLR2004
            return "*"
        return str(round(p, 3))

    def describe(self) -> str:
        """One-line summary in the notebooks' printed format."""
        return (
            f"{self.n_a} ROIs vs. {self.n_b}, p={round(self.p_value, 4)}; "
            f"a: {round(self.mean_a, 3)} +/- {round(self.sem_a, 3)}, "
            f"b: {round(self.mean_b, 3)} +/- {round(self.sem_b, 3)}"
        )


def _run_test(
    test: TestName,
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Dispatch to scipy with explicitly pinned arguments."""
    if test == "student_t":
        result = stats.ttest_ind(a, b, equal_var=True, alternative="two-sided")
    elif test == "welch_t":
        result = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    elif test == "mannwhitney":
        result = stats.mannwhitneyu(a, b, alternative="two-sided", method="auto")
    elif test == "wilcoxon":
        result = stats.wilcoxon(a, b, alternative="two-sided")
    elif test == "paired_t":
        result = stats.ttest_rel(a, b, alternative="two-sided")
    else:  # pragma: no cover - Literal exhausts this
        raise ValueError(f"unknown test {test!r}")
    return float(result.statistic), float(result.pvalue)


def run_comparison(frame: pd.DataFrame, spec: ComparisonSpec) -> ComparisonResult:
    """Run one comparison spec against a ``(protocol, outcome)``-columned frame.

    Parameters
    ----------
    frame
        Rows indexed by ROI key; columns a two-level MultiIndex
        ``(protocol, outcome)`` holding one metric value per ROI.
    spec
        The comparison to run. Unpaired tests drop NaNs per group (the
        notebooks' ``dropna()`` per side); paired tests keep only rows where
        both groups are present (the notebooks' pairwise ``dropna``).

    Returns
    -------
    ComparisonResult

    Raises
    ------
    KeyError
        If either ``(protocol, outcome)`` column is missing from ``frame``.
    """
    series_a = frame[spec.a]
    series_b = frame[spec.b]
    if spec.test in ("wilcoxon", "paired_t"):
        paired = pd.concat([series_a, series_b], axis=1).dropna()
        values_a = paired.iloc[:, 0].to_numpy(dtype=np.float64)
        values_b = paired.iloc[:, 1].to_numpy(dtype=np.float64)
    else:
        values_a = series_a.dropna().to_numpy(dtype=np.float64)
        values_b = series_b.dropna().to_numpy(dtype=np.float64)

    statistic, p_value = _run_test(spec.test, values_a, values_b)
    return ComparisonResult(
        name=spec.name,
        statistic=statistic,
        p_value=p_value,
        n_a=int(values_a.shape[0]),
        n_b=int(values_b.shape[0]),
        mean_a=float(values_a.mean()),
        mean_b=float(values_b.mean()),
        sem_a=float(stats.sem(values_a)),
        sem_b=float(stats.sem(values_b)),
    )


def run_comparisons(
    frame: pd.DataFrame, specs: Sequence[ComparisonSpec]
) -> tuple[list[ComparisonResult], pd.DataFrame]:
    """Run every spec whose p-value passes its threshold.

    Returns
    -------
    tuple
        The passing :class:`ComparisonResult` objects, and a tidy DataFrame of
        every result (one row per spec, passing or not) for reporting.
    """
    results = [run_comparison(frame, spec) for spec in specs]
    tidy = pd.DataFrame(
        {
            "name": [r.name for r in results],
            "statistic": [r.statistic for r in results],
            "p_value": [r.p_value for r in results],
            "n_a": [r.n_a for r in results],
            "n_b": [r.n_b for r in results],
            "mean_a": [r.mean_a for r in results],
            "mean_b": [r.mean_b for r in results],
            "sem_a": [r.sem_a for r in results],
            "sem_b": [r.sem_b for r in results],
            "passed": [
                r.p_value < spec.p_threshold for r, spec in zip(results, specs, strict=True)
            ],
        }
    )
    passing = [r for r, spec in zip(results, specs, strict=True) if r.p_value < spec.p_threshold]
    return passing, tidy
