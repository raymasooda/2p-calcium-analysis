"""Tests for the legacy ingest layer: bridge flatteners and production readers.

No real manuscript data is touched (fresh-clone discipline): every test builds
tiny synthetic pickled dicts mimicking the structures observed in the real
files, documented per builder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calcium2p.io.legacy import (
    ConvertedDataMissingError,
    canonical_roi_key,
    read_converted,
    read_reference_events,
    read_reference_frequencies,
)
from calcium2p.io.legacy_bridge import (
    SOURCE_FILES,
    convert_all,
    flatten_batch1,
    flatten_batch3,
    flatten_behavior,
    flatten_no_dreadds,
    flatten_task_axons,
    load_pickled_npy,
)

N_FRAMES = 6


def make_trial_frame(n_trials: int, index: pd.MultiIndex) -> pd.DataFrame:
    """A trial x frame block with 1-based integer frame labels (task_axons style)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(size=(n_trials, N_FRAMES)),
        index=index,
        columns=range(1, N_FRAMES + 1),
    )


def task_axons_index(rows: list[tuple[str, str, str, float, str]]) -> pd.MultiIndex:
    """The 5-level task_axons trial index observed in deltaF_dict.npy."""
    return pd.MultiIndex.from_tuples(
        rows,
        names=[
            "Trial number",
            "unvalidated type",
            "previous type",
            "response latency",
            "early lick?",
        ],
    )


def make_task_axons_raw() -> dict[str, object]:
    """Mimic deltaF_dict.npy: {id: {ttype: {roi(int): DataFrame}}} incl. None branch."""
    idx_all = task_axons_index(
        [
            ("0", "cHIT", "Nan", 0.35, "False"),
            ("1", "nHIT", "cHIT", 0.42, "False"),
            ("2", "nCATCH", "nHIT", 0.0, "True"),
        ]
    )
    idx_hit = task_axons_index(
        [("0", "cHIT", "Nan", 0.35, "False"), ("1", "nHIT", "cHIT", 0.42, "False")]
    )
    return {
        "muji_FOV2": {
            None: {1: pd.DataFrame(index=task_axons_index([]), columns=range(4))},
            "ALL": {1: make_trial_frame(3, idx_all), 4: make_trial_frame(3, idx_all)},
            "HIT": {1: make_trial_frame(2, idx_hit), 4: make_trial_frame(2, idx_hit)},
            "resp HIT": {1: make_trial_frame(2, idx_hit)},
        }
    }


def make_no_dreadds_raw() -> dict[str, object]:
    """Mimic NO_DREADDs_deltaF_dict.npy: index is ['Trial number', 'trial type']."""
    idx = pd.MultiIndex.from_tuples(
        [("0", "nHIT"), ("1", "cHIT")], names=["Trial number", "trial type"]
    )
    frame = pd.DataFrame(
        np.arange(2 * N_FRAMES, dtype=float).reshape(2, N_FRAMES),
        index=idx,
        columns=range(N_FRAMES),
    )
    return {"GCaMP_Batch7_one_FOV1": {"ALL": {1: frame}, "HIT": {1: frame.iloc[:1]}}}


def make_behavior_trial(row_label: int, history: str) -> pd.DataFrame:
    """One 1-row behaviour frame like Batch1/Batch3 store per trial."""
    return pd.DataFrame(
        {"Unnamed: 0": ["x"], "history": [history], "Rew licks": [3.0]},
        index=pd.Index([row_label]),
    )


def make_batch1_raw() -> dict[str, object]:
    """Mimic RGECO_GCaMP_Batch1_deltaF.npy: {id: {protocol: {compartment: {...}}}}."""
    trials = {0: pd.Series(np.zeros(N_FRAMES)), 1: pd.Series(np.ones(N_FRAMES))}
    return {
        "muji": {
            "NAIVE STIM REW": {
                "AXONS": {
                    "ROI4": trials,
                    "ROI5": trials,
                    "behavior": {
                        0: make_behavior_trial(4, "cRew"),
                        1: make_behavior_trial(7, "nStim"),
                    },
                }
            }
        }
    }


def make_batch3_raw() -> dict[str, object]:
    """Mimic GCaMP_POm_Batch3_deltaF_dict.npy: {id: {protocol: {fov: {...}}}}."""
    trials = {0: pd.Series(np.zeros(N_FRAMES))}
    return {
        "two": {
            "NAIVE STIM REW": {
                "FOV1": {"ROI0": trials, "behavior": {0: make_behavior_trial(3, "cRew")}},
                "FOV2": {"ROI1": trials, "behavior": {0: make_behavior_trial(5, "nRew")}},
            }
        }
    }


def make_behavior_raw() -> dict[str, object]:
    """Mimic task_dendrites_cohort_behavior.npy: {td: {mouse: DataFrame}}."""
    frame = pd.DataFrame(
        {
            "response latency": [412.0, 250.0],
            "trial type": ["cHIT", "nHIT"],
            "early lick times": [[1.0, 2.0], []],
        }
    )
    return {"TD0": {"MUJI": frame}, "TD2": {"MUJI": frame.iloc[:1]}}


class TestFlattenTaskAxons:
    def test_traces_come_from_all_branch_only(self) -> None:
        traces, _ = flatten_task_axons(make_task_axons_raw())
        assert traces.shape[0] == 6  # 2 rois x 3 trials
        assert set(traces["roi"]) == {"1", "4"}
        assert list(traces.columns[:9]) == [
            "dataset",
            "id",
            "roi",
            "trial",
            "unvalidated_type",
            "previous_type",
            "response_latency",
            "early_lick",
            "source_order",
        ]
        assert traces["unvalidated_type"].tolist()[:3] == ["cHIT", "nHIT", "nCATCH"]
        assert traces["response_latency"].dtype == np.float64

    def test_frame_columns_renamed_positionally(self) -> None:
        traces, _ = flatten_task_axons(make_task_axons_raw())
        assert list(traces.columns[-N_FRAMES:]) == [f"f{i:03d}" for i in range(N_FRAMES)]

    def test_trial_counts_match_notebook_counting(self) -> None:
        _, trials = flatten_task_axons(make_task_axons_raw())
        counts = dict(zip(trials["ttype"], trials["n_trials"], strict=True))
        # first ROI's row count per branch; the None branch is skipped
        assert counts == {"ALL": 3, "HIT": 2, "resp HIT": 2}
        assert (trials["source_order"] == 0).all()


class TestFlattenNoDreadds:
    def test_trial_type_level_flattened(self) -> None:
        traces, trials = flatten_no_dreadds(make_no_dreadds_raw())
        assert traces["trial_type"].tolist() == ["nHIT", "cHIT"]
        counts = dict(zip(trials["ttype"], trials["n_trials"], strict=True))
        assert counts == {"ALL": 2, "HIT": 1}


class TestFlattenSeriesDatasets:
    def test_batch1_traces_and_behavior(self) -> None:
        traces, behavior = flatten_batch1(make_batch1_raw())
        assert traces.shape[0] == 4  # 2 rois x 2 trials
        assert set(traces["compartment"]) == {"AXONS"}
        assert traces["trial"].tolist() == [0, 1, 0, 1]
        assert behavior.shape[0] == 2
        assert behavior["history"].tolist() == ["cRew", "nStim"]
        assert behavior["behavior_row"].tolist() == [4, 7]

    def test_batch3_source_order_increments_per_fov(self) -> None:
        traces, behavior = flatten_batch3(make_batch3_raw())
        fov_orders = dict(zip(traces["fov"], traces["source_order"], strict=False))
        assert fov_orders == {"FOV1": 0, "FOV2": 1}
        assert set(behavior["fov"]) == {"FOV1", "FOV2"}


class TestFlattenBehavior:
    def test_long_table_with_td_and_mouse(self) -> None:
        table = flatten_behavior(make_behavior_raw(), "task_dendrites")
        assert table.shape[0] == 3
        assert set(table["td"]) == {"TD0", "TD2"}
        assert set(table["mouse"]) == {"MUJI"}
        # object columns coerced to string dtype, values preserved via str()
        assert table["early lick times"].dtype == "string"
        assert table["early lick times"].iloc[0] == "[1.0, 2.0]"
        assert table["response latency"].dtype == np.float64


class TestConvertAll:
    def test_end_to_end_roundtrip(self, tmp_path: Path) -> None:
        source = tmp_path / "source"
        dest = tmp_path / "converted"
        source.mkdir()
        raws: dict[str, object] = {
            "task_axons": make_task_axons_raw(),
            "no_dreadds": make_no_dreadds_raw(),
            "batch1": make_batch1_raw(),
            "batch3": make_batch3_raw(),
            "task_dendrites": make_behavior_raw(),
            "control_dendrites": make_behavior_raw(),
        }
        for dataset, raw in raws.items():
            np.save(source / SOURCE_FILES[dataset], raw, allow_pickle=True)

        written = convert_all(source, dest)
        assert set(written) == {
            "traces_task_axons",
            "trials_task_axons",
            "traces_no_dreadds",
            "trials_no_dreadds",
            "traces_batch1",
            "behavior_batch1",
            "traces_batch3",
            "behavior_batch3",
            "behavior_task_dendrites",
            "behavior_control_dendrites",
        }
        roundtrip = read_converted(dest, "traces_task_axons")
        assert roundtrip.shape[0] == 6
        manifest = (dest / "conversion_manifest.json").read_text(encoding="utf-8")
        assert "sha256" in manifest

    def test_missing_conversion_raises_with_bridge_command(self, tmp_path: Path) -> None:
        with pytest.raises(ConvertedDataMissingError, match="legacy_bridge"):
            read_converted(tmp_path, "traces_task_axons")


class TestLoadPickledNpy:
    def test_rejects_non_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "arr.npy"
        np.save(path, np.array(42))
        with pytest.raises(TypeError, match="pickled dict"):
            load_pickled_npy(path)


TASK_AXONS_HEADER = (
    ",ROI,latency,outcome,previous outcome,early lick?,reward latency,trial,"
    "peak,ev_onset,peak_time,ev_offset,ev_duration,integral,truestim,purestim"
)
PASSIVE_HEADER = (
    ",,stimtype,outcome,history,trial,peak,ev_onset,peak_time,ev_offset,"
    "ev_duration,integral,ROI,Rew licks,Rew latency,Stim licks,Stim latency"
)
DENDRITES_HEADER = (
    ",,outcome,trial,peak,ev_onset,peak_time,ev_offset,ev_duration,integral,ROI,latency"
)


class TestReadReferenceEvents:
    def test_task_axons_layout(self, tmp_path: Path) -> None:
        path = tmp_path / "task_axons_all_2s_reward_events.csv"
        path.write_text(
            TASK_AXONS_HEADER
            + "\n1,muji_FOV2_1.0,0.34,cHIT,nHIT,True,0.003,28,"
            + "2.7,5.43,5.56,5.73,0.29,21.8,3.3,3.3\n",
            encoding="utf-8",
        )
        frame = read_reference_events(path, "task_axons")
        assert list(frame.index) == ["muji_FOV2_1.0"]
        assert "ev_onset" in frame.columns

    def test_passive_layout(self, tmp_path: Path) -> None:
        path = tmp_path / "passive_axons_all_2s_reward_events.csv"
        path.write_text(
            PASSIVE_HEADER
            + "\nGCaMP_POm_Batch3_five_FOV1,0,CR,CR,nRew,25,"
            + "5.5,6.4,6.6,7.9,1.5,97.2,ROI0,47.0,0.06,0.0,\n",
            encoding="utf-8",
        )
        frame = read_reference_events(path, "passive_axons")
        assert frame.index[0] == ("GCaMP_POm_Batch3_five_FOV1", "ROI0")

    def test_dendrites_layout(self, tmp_path: Path) -> None:
        path = tmp_path / "task_dendrites_TD0_all_2s_reward_events.csv"
        path.write_text(
            DENDRITES_HEADER + "\nmuji,21,cHIT,22.0,9.1,5.77,6.93,7.75,1.98,439.8,57.0,0.30\n",
            encoding="utf-8",
        )
        frame = read_reference_events(path, "task_dendrites_TD0")
        # pandas type-infers the ROI level: bare '57.0' parses as float 57.0,
        # while '42.0_TD3' would stay a string -- exactly as under pandas 1.5.
        assert frame.index[0] == ("muji", 57.0)

    def test_frequencies_layout(self, tmp_path: Path) -> None:
        path = tmp_path / "TD0_COHORT_fullreward_event_frequencies.csv"
        path.write_text(",,cCATCH,nCATCH,CATCH,cHIT\nmuji,3.0,,0.13,0.09,0.2\n", encoding="utf-8")
        frame = read_reference_frequencies(path)
        assert frame.index[0] == ("muji", 3.0)


class TestCanonicalRoiKey:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("muji_FOV2_66.0", "muji_FOV2_66"),
            ("muji_FOV2_66", "muji_FOV2_66"),
            (("muji", "57.0"), "muji::57"),
            (("muji", "42.0_TD3"), "muji::42_TD3"),
            (("GCaMP_POm_Batch3_five_FOV1", "ROI0"), "GCaMP_POm_Batch3_five_FOV1::ROI0"),
            (66.0, "66"),
            (3, "3"),
        ],
    )
    def test_normalization(self, raw: object, expected: str) -> None:
        assert canonical_roi_key(raw) == expected  # type: ignore[arg-type]
