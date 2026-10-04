"""Regression suite pinning the recorded manuscript-notebook numbers.

Every value here was recovered from the saved cell outputs of
``reward_onset_histograms.ipynb`` / ``stimulus_onset_histograms.ipynb`` and
byte-exactly reproduced by the legacy-environment reproduction harness. These
tests assert the production package (pandas 2.x, ``legacy_faithful=True``)
computes the same numbers from the machine-local dataset.

Skipped automatically when the dataset is not present (fresh clones, CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from calcium2p.artifacts import ArtifactStore
from calcium2p.cohort.run import run_onset_analysis
from calcium2p.core.cohort_config import CohortConfig, ComparisonSpec, EpochSpec, ModulationSpec
from calcium2p.core.paths import DataRootNotConfiguredError, data_root

if TYPE_CHECKING:
    from typing import Any

try:
    _MANUSCRIPT = data_root(interactive=False) / "manuscript"
except DataRootNotConfiguredError:
    _MANUSCRIPT = Path("/nonexistent")
DATA = _MANUSCRIPT / "V10"
CONVERTED = _MANUSCRIPT / "V10_converted"

pytestmark = [
    pytest.mark.requires_data,
    pytest.mark.slow,
    pytest.mark.skipif(
        not (DATA.is_dir() and CONVERTED.is_dir()),
        reason="machine-local legacy dataset not present",
    ),
]

REWARD_EPOCHS = (
    EpochSpec(
        name="early reward",
        default_window=(5.1, 6.1),
        per_protocol={"NO_DREADDs": (5.05, 5.8), "passive_dendrites": (5.1, 6.2)},
    ),
)

REWARD_COMPARISONS = (
    ComparisonSpec(
        name="NO_DREADDs cHIT vs. nHIT",
        a=("NO_DREADDs", "cHIT"),
        b=("NO_DREADDs", "nHIT"),
        metric="ev_onset",
        epoch="early reward",
        test="student_t",
    ),
    ComparisonSpec(
        name="Passive dendrites cRew vs. nRew",
        a=("passive_dendrites", "cRew"),
        b=("passive_dendrites", "nRew"),
        metric="ev_onset",
        epoch="early reward",
        test="student_t",
    ),
    ComparisonSpec(
        name="NO_DREADDs cHIT vs. nHIT rate",
        a=("NO_DREADDs", "cHIT"),
        b=("NO_DREADDs", "nHIT"),
        metric="rate",
        epoch="early reward",
        test="mannwhitney",
    ),
    ComparisonSpec(
        name="Passive dendrites cRew vs. nRew rate",
        a=("passive_dendrites", "cRew"),
        b=("passive_dendrites", "nRew"),
        metric="rate",
        epoch="early reward",
        test="mannwhitney",
    ),
)

STIMULUS_EPOCHS = (EpochSpec(name="stim", default_window=(3.1, 3.5)),)

STIMULUS_COMPARISONS = (
    ComparisonSpec(
        name="hM4D dendrites pre-CNO cHIT vs. nHIT",
        a=("task_dendrites (pre-CNO)", "cHIT"),
        b=("task_dendrites (pre-CNO)", "nHIT"),
        metric="ev_onset",
        epoch="stim",
        test="mannwhitney",
        roi_set="modulated",
    ),
    ComparisonSpec(
        name="hM4D dendrites post-CNO cHIT vs. nHIT",
        a=("task_dendrites (post-CNO)", "cHIT"),
        b=("task_dendrites (post-CNO)", "nHIT"),
        metric="ev_onset",
        epoch="stim",
        test="mannwhitney",
        roi_set="modulated",
    ),
)

#: Recorded reward-notebook cell outputs (cells 7, 24, 28, 31).
REWARD_TRIAL_TOTALS = {
    "task_axons": 68,
    "NO_DREADDs": 102,
    "passive_axons": 293,
    "passive_dendrites": 89,
    "task_dendrites (pre-CNO)": 129,
    "task_dendrites (post-CNO)": 398,
    "control_dendrites (pre-CNO)": 196,
    "control_dendrites (post-CNO)": 392,
}

REWARD_MODULATED = {
    "passive_axons": (130, 194),
    "passive_dendrites": (31, 69),
    "task_axons": (97, 353),
    "NO_DREADDs": (23, 112),
    "task_dendrites (pre-CNO)": (109, 331),
    "task_dendrites (post-CNO)": (168, 503),
    "control_dendrites (pre-CNO)": (41, 191),
    "control_dendrites (post-CNO)": (91, 186),
}

STIMULUS_MODULATED = {
    "task_axons": (72, 353),
    "NO_DREADDs": (25, 112),
    "task_dendrites (pre-CNO)": (36, 331),
    "task_dendrites (post-CNO)": (102, 503),
}

#: name -> (n_a, n_b, p, mean_a, mean_b) as printed by the notebooks.
REWARD_STATS = {
    "NO_DREADDs cHIT vs. nHIT": (9, 42, 0.011, 5.255, 5.429),
    "Passive dendrites cRew vs. nRew": (28, 62, 0.8957, 5.66, 5.653),
    "NO_DREADDs cHIT vs. nHIT rate": (9, 42, 0.0026, 0.226, 0.161),
    "Passive dendrites cRew vs. nRew rate": (28, 62, 0.1168, 0.263, 0.201),
}

STIMULUS_STATS = {
    "hM4D dendrites pre-CNO cHIT vs. nHIT": (8, 23, 0.4281, 3.384, 3.331),
    "hM4D dendrites post-CNO cHIT vs. nHIT": (76, 97, 0.2835, 3.306, 3.325),
}

#: Stimulus comparisons whose Mann-Whitney p depends on float rounding.
#: Their per-ROI ev_onset means hold values that are tied on one platform and
#: differ by one ulp (4.4e-16) on another (x86-64 vs arm64 summation order),
#: which changes scipy's tie correction. n and means still match exactly; p is
#: checked to this absolute tolerance instead of 4 decimals.
STIMULUS_TIE_SENSITIVE_P = {
    "hM4D dendrites pre-CNO cHIT vs. nHIT": 0.02,
    "hM4D dendrites post-CNO cHIT vs. nHIT": 0.005,
}


def run(alignment: str, tmp_path: Path) -> dict[str, Any]:
    """Run one alignment through a throwaway store."""
    reward = alignment == "reward"
    config = CohortConfig(
        run_id=f"regression-{alignment}",
        alignment=alignment,  # type: ignore[arg-type]
        epochs=REWARD_EPOCHS if reward else STIMULUS_EPOCHS,
        modulation=ModulationSpec(epoch="early reward" if reward else "stim"),
        comparisons=REWARD_COMPARISONS if reward else STIMULUS_COMPARISONS,
    )
    store = ArtifactStore(tmp_path / "derivatives")
    return run_onset_analysis(config, store, source_dir=DATA, converted_dir=CONVERTED)


@pytest.fixture(scope="module")
def reward_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The reward-aligned analysis, run once per module."""
    return run("reward", tmp_path_factory.mktemp("reward"))


@pytest.fixture(scope="module")
def stimulus_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The stimulus-aligned analysis, run once per module."""
    return run("stimulus", tmp_path_factory.mktemp("stimulus"))


class TestRewardAlignment:
    def test_rewarded_trial_totals(self, reward_run: dict[str, Any]) -> None:
        assert reward_run["trial_counts"].target_totals == REWARD_TRIAL_TOTALS

    def test_modulated_proportions(self, reward_run: dict[str, Any]) -> None:
        table = reward_run["modulation"]
        for protocol, (n, total) in REWARD_MODULATED.items():
            assert int(table.loc[protocol, "n_modulated"]) == n, protocol
            assert int(table.loc[protocol, "n_total"]) == total, protocol

    def test_comparison_stats(self, reward_run: dict[str, Any]) -> None:
        stats = reward_run["stats"].set_index("name")
        for name, (n_a, n_b, p, mean_a, mean_b) in REWARD_STATS.items():
            row = stats.loc[name]
            assert (int(row["n_a"]), int(row["n_b"])) == (n_a, n_b), name
            assert round(float(row["p_value"]), 4) == p, name
            assert round(float(row["mean_a"]), 3) == mean_a, name
            assert round(float(row["mean_b"]), 3) == mean_b, name


class TestStimulusAlignment:
    def test_modulated_proportions(self, stimulus_run: dict[str, Any]) -> None:
        table = stimulus_run["modulation"]
        for protocol, (n, total) in STIMULUS_MODULATED.items():
            assert int(table.loc[protocol, "n_modulated"]) == n, protocol
            assert int(table.loc[protocol, "n_total"]) == total, protocol

    def test_comparison_stats(self, stimulus_run: dict[str, Any]) -> None:
        stats = stimulus_run["stats"].set_index("name")
        for name, (n_a, n_b, p, mean_a, mean_b) in STIMULUS_STATS.items():
            row = stats.loc[name]
            assert (int(row["n_a"]), int(row["n_b"])) == (n_a, n_b), name
            if name in STIMULUS_TIE_SENSITIVE_P:
                tolerance = STIMULUS_TIE_SENSITIVE_P[name]
                assert float(row["p_value"]) == pytest.approx(p, abs=tolerance), name
                assert (float(row["p_value"]) < 0.05) == (p < 0.05), name
            else:
                assert round(float(row["p_value"]), 4) == p, name
            assert round(float(row["mean_a"]), 3) == mean_a, name
            assert round(float(row["mean_b"]), 3) == mean_b, name


class TestCorrectedModeDiverges:
    def test_control_post_cno_denominator_fixed(self, tmp_path: Path) -> None:
        """The corrected mode counts MODIFIED_TD3_CONTROL: 186 -> 389 ROIs."""
        config = CohortConfig(
            run_id="regression-corrected",
            alignment="reward",
            legacy_faithful=False,
            epochs=REWARD_EPOCHS,
            modulation=ModulationSpec(epoch="early reward"),
        )
        store = ArtifactStore(tmp_path / "derivatives")
        out = run_onset_analysis(config, store, source_dir=DATA, converted_dir=CONVERTED)
        assert int(out["modulation"].loc["control_dendrites (post-CNO)", "n_total"]) == 389
