"""Tests for the cohort-level analysis configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from calcium2p.core.cohort_config import (
    LEGACY_TEST_NAMES,
    CohortConfig,
    CohortConfigError,
    ComparisonSpec,
    EpochSpec,
    load_cohort_config,
)


def minimal_mapping() -> dict:
    """Smallest mapping that exercises every nested constructor."""
    return {
        "run_id": "onset-hist-test",
        "alignment": "reward",
        "epochs": [
            {"name": "histogram", "default_window": [5.0, 6.5]},
            {
                "name": "early reward",
                "default_window": [5.1, 6.1],
                "per_protocol": {
                    "NO_DREADDs": [5.05, 5.8],
                    "passive_dendrites": [5.1, 6.2],
                },
            },
        ],
        "modulation": {"epoch": "early reward", "threshold": "median"},
        "comparisons": [
            {
                "name": "NO_DREADDs cHIT vs. nHIT",
                "a": ["NO_DREADDs", "cHIT"],
                "b": ["NO_DREADDs", "nHIT"],
                "metric": "ev_onset",
                "epoch": "early reward",
                "test": "student_t",
                "figure": {"figsize": [1.585, 2.0], "ylim": [5.0, 6.55]},
            },
            {
                "name": "non-consecutive-normalized Passive dendrites cRew vs. cHIT",
                "a": ["passive_dendrites", "cRew"],
                "b": ["NO_DREADDs", "cHIT"],
                "metric": "rate",
                "epoch": "early reward",
                "test": "mannwhitney",
                "normalize": {
                    "a_reference": ["passive_dendrites", "nRew"],
                    "b_reference": ["NO_DREADDs", "nHIT"],
                },
                "figure": {"ylim": [0.45, 4.05]},
            },
        ],
    }


class TestFromMapping:
    def test_round_trip_builds_nested_dataclasses(self) -> None:
        config = CohortConfig.from_mapping(minimal_mapping())
        assert config.run_id == "onset-hist-test"
        assert config.epoch("early reward").window_for("NO_DREADDs") == (5.05, 5.8)
        assert config.epoch("early reward").window_for("task_axons") == (5.1, 6.1)
        normalized = config.comparisons[1]
        assert normalized.normalize is not None
        assert normalized.normalize.a_reference == ("passive_dendrites", "nRew")
        assert normalized.figure.ylim == (0.45, 4.05)

    def test_legacy_faithful_defaults_true(self) -> None:
        config = CohortConfig.from_mapping(minimal_mapping())
        assert config.legacy_faithful is True
        assert config.event_source == "reference"

    def test_unknown_key_raises(self) -> None:
        payload = minimal_mapping()
        payload["not_a_field"] = 1
        with pytest.raises(CohortConfigError, match="not_a_field"):
            CohortConfig.from_mapping(payload)

    def test_comparison_referencing_unknown_epoch_raises(self) -> None:
        payload = minimal_mapping()
        payload["comparisons"][0]["epoch"] = "late reward"
        with pytest.raises(CohortConfigError, match="late reward"):
            CohortConfig.from_mapping(payload)

    def test_modulation_referencing_unknown_epoch_raises(self) -> None:
        payload = minimal_mapping()
        payload["modulation"] = {"epoch": "missing"}
        with pytest.raises(CohortConfigError, match="missing"):
            CohortConfig.from_mapping(payload)

    def test_duplicate_epoch_names_raise(self) -> None:
        payload = minimal_mapping()
        payload["epochs"].append({"name": "histogram", "default_window": [1.0, 2.0]})
        with pytest.raises(CohortConfigError, match="duplicate"):
            CohortConfig.from_mapping(payload)

    def test_decreasing_window_raises(self) -> None:
        with pytest.raises(CohortConfigError, match="increasing"):
            EpochSpec(name="bad", default_window=(6.0, 5.0))

    def test_comparison_side_must_be_a_pair(self) -> None:
        with pytest.raises(CohortConfigError, match="protocol, outcome"):
            ComparisonSpec(
                name="bad",
                a=("only-one",),  # type: ignore[arg-type]
                b=("p", "o"),
                metric="rate",
                epoch="e",
            )


class TestLegacyTestNames:
    def test_parametric_unpaired_is_student_not_welch(self) -> None:
        # The notebooks used scipy.stats.ttest_ind with default equal_var=True.
        assert LEGACY_TEST_NAMES[("parametric", "unpaired")] == "student_t"

    def test_all_four_legacy_switch_combinations_are_mapped(self) -> None:
        assert len(LEGACY_TEST_NAMES) == 4


class TestLoadCohortConfig:
    def test_loads_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "run.yaml"
        path.write_text(yaml.safe_dump(minimal_mapping()), encoding="utf-8")
        config = load_cohort_config(path)
        assert config.alignment == "reward"
        assert len(config.comparisons) == 2

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "run.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")
        with pytest.raises(CohortConfigError, match="mapping"):
            load_cohort_config(path)

    def test_example_config_is_valid(self) -> None:
        example = Path(__file__).parents[2] / "configs" / "onset_histograms.example.yaml"
        config = load_cohort_config(example)
        assert config.alignment == "reward"
        assert config.legacy_faithful is True
        assert config.epoch("early reward").window_for("NO_DREADDs") == (5.05, 5.8)
