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
    expand_figure_manifest,
    load_cohort_config,
    load_figure_manifest,
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


def manifest_mapping() -> dict:
    """Two states sharing defaults; the second overrides one default key."""
    return {
        "defaults": {"legacy_faithful": True, "data_subdir": "manuscript/V10"},
        "states": [
            {**minimal_mapping(), "run_id": "state-a"},
            {**minimal_mapping(), "run_id": "state-b", "legacy_faithful": False},
        ],
    }


class TestFigureManifest:
    def test_expands_one_config_per_state_with_defaults(self) -> None:
        configs = expand_figure_manifest(manifest_mapping())
        assert [c.run_id for c in configs] == ["state-a", "state-b"]
        assert all(c.data_subdir == "manuscript/V10" for c in configs)

    def test_state_keys_override_defaults(self) -> None:
        a, b = expand_figure_manifest(manifest_mapping())
        assert a.legacy_faithful is True
        assert b.legacy_faithful is False

    def test_duplicate_run_ids_raise(self) -> None:
        data = manifest_mapping()
        data["states"][1]["run_id"] = "state-a"
        with pytest.raises(CohortConfigError, match="duplicate run_id"):
            expand_figure_manifest(data)

    @pytest.mark.parametrize("states", [None, [], "not-a-list"])
    def test_missing_or_empty_states_raise(self, states: object) -> None:
        with pytest.raises(CohortConfigError, match="states"):
            expand_figure_manifest({"defaults": {}, "states": states})

    def test_invalid_state_names_the_state(self) -> None:
        data = manifest_mapping()
        data["states"][1]["comparisons"][0]["epoch"] = "no such epoch"
        with pytest.raises(CohortConfigError, match=r"states\[1\].*state-b"):
            expand_figure_manifest(data)


class TestBundledV10Manifest:
    """The shipped V10 recipe: shape checks that hold without the dataset."""

    @pytest.fixture(scope="class")
    def configs(self) -> tuple[CohortConfig, ...]:
        from calcium2p.cohort.manifest import bundled_manifest  # noqa: PLC0415

        return load_figure_manifest(bundled_manifest("v10"))

    def test_counts(self, configs: tuple[CohortConfig, ...]) -> None:
        assert len(configs) == 7
        assert sum(len(c.comparisons) for c in configs) == 57

    def test_every_figure_carries_its_original_ylim(
        self, configs: tuple[CohortConfig, ...]
    ) -> None:
        assert all(s.figure.ylim is not None for c in configs for s in c.comparisons)

    def test_legacy_numbers_from_reference_events(self, configs: tuple[CohortConfig, ...]) -> None:
        assert all(c.legacy_faithful and c.event_source == "reference" for c in configs)

    def test_latency_from_reward_normalization_is_recorded(
        self, configs: tuple[CohortConfig, ...]
    ) -> None:
        offsets = {
            (s.name, s.metric): s.normalize.offset
            for c in configs
            for s in c.comparisons
            if s.normalize is not None
        }
        assert offsets[("Passive axons cRew vs. cHIT", "ev_onset")] == 5.0
        assert offsets[("Passive dendrites cRew vs. cHIT", "ev_onset")] == 0.0

    def test_unknown_bundled_name_raises(self) -> None:
        from calcium2p.cohort.manifest import bundled_manifest  # noqa: PLC0415

        with pytest.raises(KeyError, match="no bundled manifest"):
            bundled_manifest("v99")
