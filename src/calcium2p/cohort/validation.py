"""Validation against the original notebook outputs and reference tables.

Two layers:

1. :func:`compare_event_tables` grades regenerated event tables against the
   reference CSVs row by row (exact onset+offset matches, onset-only matches,
   missing and extra rows) -- the honest per-group match-rate report the
   regeneration scope calls for. The dendrite groups have no traces in V10
   and are reported as not regenerable.
2. :data:`NOTEBOOK_REFERENCE_VALUES` pins every number recovered from the
   notebooks' recorded outputs; :func:`check_reference_values` grades a
   computed set against them. The CLI and the regression tests consume the
   same pinned values, so there is exactly one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Event-table groups whose deltaF traces are absent from V10: their event
#: CSVs cannot be regenerated from data on this machine.
NOT_REGENERABLE: Final = (
    "task_dendrites_TD0",
    "task_dendrites_TD1",
    "task_dendrites_TD2",
    "task_dendrites_TD3",
    "control_dendrites_TD0",
    "control_dendrites_TD2",
    "control_dendrites_TD3",
)


@dataclass(frozen=True)
class MatchReport:
    """Row-level agreement between a regenerated and a reference event table.

    Attributes
    ----------
    group, side
        Which table this grades (side: ``target`` = reward/stim file,
        ``reference`` = noreward/nostim file).
    n_reference, n_regenerated
        Row counts on each side.
    n_exact
        Reference rows with a regenerated row of the same trial whose onset
        AND offset agree to within ``onset_tolerance``.
    n_onset_only
        Reference rows matched on trial+onset but not offset.
    n_missing
        Reference rows with no regenerated counterpart at all.
    n_extra
        Regenerated rows with no reference counterpart.
    max_metric_diff
        Per-column maximum absolute difference over the exactly-matched rows
        (columns ``peak``/``integral``/``ev_duration``...); empty when no
        exact matches exist.
    """

    group: str
    side: str
    n_reference: int
    n_regenerated: int
    n_exact: int
    n_onset_only: int
    n_missing: int
    n_extra: int
    max_metric_diff: dict[str, float]

    @property
    def exact_fraction(self) -> float:
        """Fraction of reference rows reproduced exactly."""
        return self.n_exact / self.n_reference if self.n_reference else 1.0

    def describe(self) -> str:
        """One-line human summary."""
        return (
            f"{self.group}/{self.side}: {self.n_exact}/{self.n_reference} exact "
            f"({self.exact_fraction:.1%}), {self.n_onset_only} onset-only, "
            f"{self.n_missing} missing, {self.n_extra} extra "
            f"(regenerated {self.n_regenerated})"
        )


def _row_key(index_value: object, trial: object) -> tuple[str, str]:
    """Stable (roi, trial) key across index spellings."""
    roi = (
        "|".join(str(part) for part in index_value)
        if isinstance(index_value, tuple)
        else str(index_value)
    )
    return roi, str(int(float(str(trial))))


def compare_event_tables(
    regenerated: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    group: str,
    side: str,
    onset_tolerance: float = 1e-9,
) -> MatchReport:
    """Grade one regenerated event table against its reference CSV.

    Rows are matched within (ROI, trial): a reference row is *exact* when a
    regenerated row agrees on onset and offset within ``onset_tolerance``,
    *onset-only* when only the onset agrees.
    """
    metric_columns = [
        c for c in ("peak", "peak_time", "ev_duration", "integral") if c in reference.columns
    ]
    regen_by_key: dict[tuple[str, str], list[pd.Series]] = {}
    for idx, row in regenerated.iterrows():
        regen_by_key.setdefault(_row_key(idx, row["trial"]), []).append(row)

    n_exact = n_onset = n_missing = 0
    matched_regen: set[int] = set()
    diffs: dict[str, float] = dict.fromkeys(metric_columns, 0.0)
    for idx, row in reference.iterrows():
        candidates = regen_by_key.get(_row_key(idx, row["trial"]), [])
        exact_hit = None
        onset_hit = None
        for candidate in candidates:
            if abs(float(candidate["ev_onset"]) - float(row["ev_onset"])) <= onset_tolerance:
                onset_hit = candidate
                if abs(float(candidate["ev_offset"]) - float(row["ev_offset"])) <= onset_tolerance:
                    exact_hit = candidate
                    break
        if exact_hit is not None:
            n_exact += 1
            matched_regen.add(id(exact_hit))
            for column in metric_columns:
                if column in exact_hit.index:
                    diffs[column] = max(
                        diffs[column], abs(float(exact_hit[column]) - float(row[column]))
                    )
        elif onset_hit is not None:
            n_onset += 1
            matched_regen.add(id(onset_hit))
        else:
            n_missing += 1

    n_matched = len(matched_regen)
    return MatchReport(
        group=group,
        side=side,
        n_reference=int(reference.shape[0]),
        n_regenerated=int(regenerated.shape[0]),
        n_exact=n_exact,
        n_onset_only=n_onset,
        n_missing=n_missing,
        n_extra=int(regenerated.shape[0]) - n_matched,
        max_metric_diff={k: v for k, v in diffs.items() if n_exact},
    )


#: Every number recovered from the notebooks' recorded outputs. Structure:
#: check name -> {"expected": value, "kind": "int"|"float"|"mapping"}.
#: Consumed by check_reference_values(), the validate CLI, and the
#: regression tests -- one source of truth.
NOTEBOOK_REFERENCE_VALUES: Final[dict[str, dict[str, Any]]] = {
    "reward.rewarded_trials": {
        "kind": "mapping",
        "expected": {
            "task_axons": 68,
            "NO_DREADDs": 102,
            "passive_axons": 293,
            "passive_dendrites": 89,
            "task_dendrites (pre-CNO)": 129,
            "task_dendrites (post-CNO)": 398,
            "control_dendrites (pre-CNO)": 196,
            "control_dendrites (post-CNO)": 392,
        },
    },
    "reward.modulated_proportions": {
        "kind": "mapping",
        "expected": {
            "passive_axons": (130, 194),
            "passive_dendrites": (31, 69),
            "task_axons": (97, 353),
            "NO_DREADDs": (23, 112),
            "task_dendrites (pre-CNO)": (109, 331),
            "task_dendrites (post-CNO)": (168, 503),
            "control_dendrites (pre-CNO)": (41, 191),
            "control_dendrites (post-CNO)": (91, 186),
        },
    },
    "stimulus.modulated_proportions": {
        "kind": "mapping",
        "expected": {
            "task_axons": (72, 353),
            "NO_DREADDs": (25, 112),
            "task_dendrites (pre-CNO)": (36, 331),
            "task_dendrites (post-CNO)": (102, 503),
        },
    },
    "reward.ev_onset.NO_DREADDs_cHIT_vs_nHIT": {
        "kind": "mapping",
        "expected": {
            "n_a": 9,
            "n_b": 42,
            "p": 0.011,
            "mean_a": 5.255,
            "mean_b": 5.429,
            "sem_a": 0.05,
            "sem_b": 0.028,
        },
    },
    "reward.ev_onset.passive_dendrites_cRew_vs_nRew": {
        "kind": "mapping",
        "expected": {"n_a": 28, "n_b": 62, "p": 0.8957, "mean_a": 5.66, "mean_b": 5.653},
    },
    "reward.rate.NO_DREADDs_cHIT_vs_nHIT": {
        "kind": "mapping",
        "expected": {"p": 0.0026, "mean_a": 0.226, "mean_b": 0.161},
    },
    "reward.rate.passive_dendrites_cRew_vs_nRew": {
        "kind": "mapping",
        "expected": {"p": 0.1168, "mean_a": 0.263, "mean_b": 0.201},
    },
    "reward.normalized.passive_dendrites_cRew_vs_NO_DREADDs_cHIT": {
        "kind": "mapping",
        "expected": {"n_a": 28, "n_b": 9, "p": 0.0364, "mean_a": 1.308, "mean_b": 1.403},
    },
    "stimulus.ev_onset.task_dendrites_preCNO_cHIT_vs_nHIT": {
        "kind": "mapping",
        "expected": {
            "n_a": 8,
            "n_b": 23,
            "p": 0.4281,
            "mean_a": 3.384,
            "mean_b": 3.331,
            "sem_a": 0.031,
            "sem_b": 0.024,
        },
    },
    "stimulus.ev_onset.task_dendrites_postCNO_cHIT_vs_nHIT": {
        "kind": "mapping",
        "expected": {
            "n_a": 76,
            "n_b": 97,
            "p": 0.2835,
            "mean_a": 3.306,
            "mean_b": 3.325,
            "sem_a": 0.013,
            "sem_b": 0.009,
        },
    },
    "reward.regression.task_axons_p_value": {
        "kind": "float",
        "expected": 0.9108585107568281,
    },
}


def check_reference_values(computed: Mapping[str, Any], *, rel_tol: float = 1e-3) -> pd.DataFrame:
    """Grade computed values against :data:`NOTEBOOK_REFERENCE_VALUES`.

    Parameters
    ----------
    computed
        Check name -> computed value, in the same shapes as the pinned
        ``expected`` entries. Missing checks are reported, not skipped.
    rel_tol
        Relative tolerance for float comparisons (the notebooks printed
        values rounded to 3-4 digits, so exact equality is not meaningful
        for most checks).

    Returns
    -------
    pandas.DataFrame
        One row per check: ``name``, ``expected``, ``computed``, ``passed``.
    """

    def close(a: float, b: float) -> bool:
        return abs(a - b) <= rel_tol * max(abs(a), abs(b), 1e-12)

    def compare(expected: Any, value: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                return False
            return all(key in value and compare(sub, value[key]) for key, sub in expected.items())
        if isinstance(expected, tuple):
            return tuple(value) == expected
        if isinstance(expected, float):
            return close(expected, float(value))
        return bool(expected == value)

    rows = []
    for name, spec in NOTEBOOK_REFERENCE_VALUES.items():
        value = computed.get(name)
        passed = value is not None and compare(spec["expected"], value)
        rows.append(
            {"name": name, "expected": spec["expected"], "computed": value, "passed": passed}
        )
    return pd.DataFrame(rows)
