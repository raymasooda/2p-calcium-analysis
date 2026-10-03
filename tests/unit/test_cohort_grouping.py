"""Tests for grouping: dual-mode (legacy vs corrected) bug-site behaviour.

Every test builds a tiny synthetic cohort where the legacy and corrected modes
diverge measurably, pinning both behaviours. The real-data equivalents are
pinned by the regression suite (``tests/regression/``).
"""

from __future__ import annotations

import pandas as pd
import pytest

from calcium2p.cohort.grouping import (
    TrialCounts,
    build_roi_denominators,
    build_trial_counts,
    load_grouped_events,
)


def dendrite_behavior(cohort: str, tds: dict[str, list[str]]) -> pd.DataFrame:
    """Behaviour table with one mouse per TD and the given trial types."""
    rows = []
    for td, ttypes in tds.items():
        rows.extend(
            {"cohort": cohort, "td": td, "mouse": "MUJI", "trial type": ttype} for ttype in ttypes
        )
    return pd.DataFrame(rows)


def flat_trials(dataset: str, per_id: dict[str, dict[str, int]]) -> pd.DataFrame:
    """trials_* table in the converted-parquet shape."""
    rows = [
        {"dataset": dataset, "id": identifier, "ttype": ttype, "n_trials": n, "source_order": 0}
        for identifier, counts in per_id.items()
        for ttype, n in counts.items()
    ]
    return pd.DataFrame(rows)


def passive_behavior(dataset: str, group_col: str, histories: list[str]) -> pd.DataFrame:
    """behavior_batch* table with one (id, group) and the given histories."""
    return pd.DataFrame(
        {
            "dataset": dataset,
            "id": "two",
            "protocol": "NAIVE STIM REW",
            group_col: "AXONS" if group_col == "compartment" else "FOV1",
            "source_order": 0,
            "history": histories,
        }
    )


def passive_traces(dataset: str, group_col: str, group: str, rois: list[str]) -> pd.DataFrame:
    """traces_batch* metadata rows (frames omitted; grouping ignores them)."""
    return pd.DataFrame(
        {
            "dataset": dataset,
            "id": "two",
            "protocol": "NAIVE STIM REW",
            group_col: group,
            "roi": rois,
            "trial": range(len(rois)),
            "source_order": 0,
        }
    )


def make_tables() -> dict[str, pd.DataFrame]:
    """Minimal full set of converted tables."""
    return {
        "trials_task_axons": flat_trials(
            "task_axons", {"muji_FOV2": {"HIT": 5, "FA": 1, "CATCH": 2, "ALL": 10, "resp x": 9}}
        ),
        "trials_no_dreadds": flat_trials(
            "no_dreadds", {"GCaMP_Batch8_one_FOV7": {"HIT": 4, "CATCH": 3, "ALL": 9}}
        ),
        "behavior_batch1": passive_behavior("batch1", "compartment", ["cRew", "nRew", "nStim"]),
        "behavior_batch3": passive_behavior("batch3", "fov", ["cRew", "cStim", "nStim"]),
        "behavior_task_dendrites": dendrite_behavior(
            "task_dendrites",
            {"TD0": ["cHIT", "nHIT", "CR"], "TD2": ["cHIT", "FA", "nCATCH"]},
        ),
        "behavior_control_dendrites": dendrite_behavior(
            "control_dendrites", {"TD1": ["cHIT", "miss"], "TD2": ["nHIT", "CATCH"]}
        ),
        "traces_batch1": passive_traces("batch1", "compartment", "AXONS", ["ROI1", "ROI2"]),
        "traces_batch3": passive_traces("batch3", "fov", "FOV1", ["ROI7", "ROI8", "ROI9"]),
        "traces_task_axons": pd.DataFrame(
            {"dataset": "task_axons", "id": "muji_FOV2", "roi": ["1", "2"], "trial": [0, 0]}
        ),
        "traces_no_dreadds": pd.DataFrame(
            {"dataset": "no_dreadds", "id": "GCaMP_Batch8_one_FOV7", "roi": ["3"], "trial": [0]}
        ),
    }


class TestBuildTrialCounts:
    def test_reward_totals(self) -> None:
        counts = build_trial_counts(make_tables(), alignment="reward")
        assert counts.target_totals["task_axons"] == 5
        assert counts.reference_totals["task_axons"] == 3  # FA + CATCH
        # batch1 AXONS has 2 Rew histories, batch3 contributes 1 more
        assert counts.target_totals["passive_axons"] == 3

    def test_resp_and_none_ttypes_are_skipped(self) -> None:
        counts = build_trial_counts(make_tables(), alignment="reward")
        assert "resp x" not in counts.all_trials["task_axons"]["muji_FOV2"]

    def test_dendrite_nostim_bug_reproduced_and_corrected(self) -> None:
        # BUG 4: legacy stimulus no-stim totals equal the stim totals
        legacy = build_trial_counts(make_tables(), alignment="stimulus", legacy_faithful=True)
        fixed = build_trial_counts(make_tables(), alignment="stimulus", legacy_faithful=False)
        label = "task_dendrites (pre-CNO)"
        assert legacy.reference_totals[label] == legacy.target_totals[label] == 2
        assert fixed.reference_totals[label] == 0  # TD0 has no FA/CATCH rows
        post = "task_dendrites (post-CNO)"
        assert fixed.reference_totals[post] == 2  # FA + nCATCH

    def test_control_dendrites_use_td1_as_pre_cno(self) -> None:
        counts = build_trial_counts(make_tables(), alignment="reward")
        assert "control_dendrites (pre-CNO)" in counts.all_trials
        assert counts.all_trials["control_dendrites (pre-CNO)"]["TD1"]["muji"] == {
            "cHIT": 1,
            "miss": 1,
        }


class TestTrialCountLookups:
    def make_counts(self) -> TrialCounts:
        return build_trial_counts(make_tables(), alignment="reward")

    def test_rate_denominator_pools_hit_variants(self) -> None:
        counts = self.make_counts()
        # task dendrites pre-CNO has cHIT + nHIT but no bare HIT key
        n = counts.rate_denominator("task_dendrites (pre-CNO)", ("muji", "5"), "HIT")
        assert n == 2

    def test_rate_denominator_task_axons_strips_float_suffix(self) -> None:
        counts = self.make_counts()
        assert counts.rate_denominator("task_axons", "muji_FOV2_66.0", "HIT") == 5

    def test_cno_td_pick_from_roi_name(self) -> None:
        counts = self.make_counts()
        td2 = counts.protocol_trials("task_dendrites (post-CNO)", ("muji", "42.0"))
        assert td2 == {"cHIT": 1, "FA": 1, "nCATCH": 1}

    def test_modulation_denominators_reward_uses_all_minus_hit(self) -> None:
        counts = self.make_counts()
        target, reference = counts.modulation_denominators("task_axons", "muji_FOV2_66.0")
        assert (target, reference) == (5, 5)  # HIT=5, ALL-HIT=5


def events_file(rois: list[str], onsets: list[float]) -> pd.DataFrame:
    """Per-event frame keyed by ROI index."""
    return pd.DataFrame({"ev_onset": onsets, "outcome": ["cHIT"] * len(onsets)}, index=rois)


class TestLoadGroupedEvents:
    def make_files(self) -> dict[str, pd.DataFrame]:
        return {
            "task_dendrites_TD0": events_file(["r1", "r2"], [5.2, 5.3]),
            "task_dendrites_TD1": events_file(["r2", "r3"], [5.4, 5.5]),
            "MODIFIED_task_dendrites_TD3": events_file(["r4_TD3"], [5.6]),
        }

    def test_legacy_td1_no_op_pools_colliding_rois_only(self) -> None:
        # BUG 1: TD1 events pool into pre-CNO but are filtered against the
        # TD0-derived active set, so only colliding ROI indices survive
        events, active = load_grouped_events(self.make_files(), legacy_faithful=True)
        pre = events["task_dendrites (pre-CNO)"]
        assert sorted(pre.index) == ["r1", "r2", "r2"]  # TD1's r3 dropped, r2 survives
        assert "task_dendrites_TD1" in active  # unmerged legacy key

    def test_corrected_mode_drops_td1_entirely(self) -> None:
        events, active = load_grouped_events(self.make_files(), legacy_faithful=False)
        pre = events["task_dendrites (pre-CNO)"]
        assert sorted(pre.index) == ["r1", "r2"]
        assert "task_dendrites_TD1" not in active

    def test_modified_prefix_is_stripped_and_pooled_to_post(self) -> None:
        events, _ = load_grouped_events(self.make_files(), legacy_faithful=True)
        assert list(events["task_dendrites (post-CNO)"].index) == ["r4_TD3"]


def frequency_table(rois: list[tuple[str, str]]) -> pd.DataFrame:
    """Frequency table with a two-level (mouse, roi) index."""
    index = pd.MultiIndex.from_tuples(rois)
    return pd.DataFrame({"rate": [0.1] * len(rois)}, index=index)


class TestBuildRoiDenominators:
    def make_freq(self) -> dict[str, pd.DataFrame]:
        return {
            "TD0_CONTROL_fullreward_event_frequencies": frequency_table([("m", "1"), ("m", "2")]),
            "TD2_CONTROL_fullreward_event_frequencies": frequency_table([("m", "3")]),
            "MODIFIED_TD3_CONTROL_fullreward_event_frequencies": frequency_table(
                [("m", "4"), ("m", "5")]
            ),
        }

    def test_modified_control_dropped_in_legacy_counted_when_corrected(self) -> None:
        # BUG 2
        tables = make_tables()
        legacy = build_roi_denominators(self.make_freq(), tables, legacy_faithful=True)
        fixed = build_roi_denominators(self.make_freq(), tables, legacy_faithful=False)
        assert len(legacy["control_dendrites (post-CNO)"]) == 1
        assert len(fixed["control_dendrites (post-CNO)"]) == 3

    def test_fov_dict_leak_reproduced_and_corrected(self) -> None:
        # BUG 3: legacy Batch1 lists carry the last Batch3 FOV's ROI names
        tables = make_tables()
        legacy = build_roi_denominators({}, tables, legacy_faithful=True)
        fixed = build_roi_denominators({}, tables, legacy_faithful=False)
        legacy_batch1 = [r for r in legacy["passive_axons"] if r[0].startswith("RGECO")]
        fixed_batch1 = [r for r in fixed["passive_axons"] if r[0].startswith("RGECO")]
        assert [roi for _, roi in legacy_batch1] == ["ROI7", "ROI8", "ROI9"]  # stale batch3 list
        assert [roi for _, roi in fixed_batch1] == ["ROI1", "ROI2"]

    def test_task_axon_entries_are_one_tuples(self) -> None:
        numbers = build_roi_denominators({}, make_tables(), legacy_faithful=True)
        assert numbers["task_axons"] == [("muji_FOV2_1",), ("muji_FOV2_2",)]


class TestBugSitesDivergeMeasurably:
    @pytest.mark.parametrize("legacy", [True, False])
    def test_both_modes_run_end_to_end(self, legacy: bool) -> None:
        tables = make_tables()
        counts = build_trial_counts(tables, alignment="stimulus", legacy_faithful=legacy)
        numbers = build_roi_denominators({}, tables, legacy_faithful=legacy)
        assert counts.target_totals
        assert numbers["passive_axons"]
