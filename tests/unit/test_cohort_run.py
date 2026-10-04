"""Tests for the cohort orchestrator: figure naming and mixed event pooling."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from calcium2p.cohort.run import (
    _regenerated_frame,
    figure_filename,
    figure_key,
    pool_mixed_events,
)
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


class TestMixedEventPooling:
    """Regenerated groups stand in for their reference CSVs, others are kept."""

    HEADER = (
        ",,stimtype,outcome,history,trial,peak,ev_onset,peak_time,ev_offset,ev_duration,"
        "integral,ROI,Rew licks,Rew latency,Stim licks,Stim latency\n"
    )

    def write_reference(self, directory: Path, group: str, session: str, onset: float) -> None:
        row = f"{session},0,CR,CR,cRew,3,1.0,{onset},5.6,5.8,0.3,1.0,ROI0,1.0,0.2,0.0,\n"
        path = directory / f"{group}_all_2s_reward_events.csv"
        path.write_text(self.HEADER + row, encoding="utf-8")

    def regenerated(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "group": ["passive_axons", "passive_axons"],
                "side": ["target", "reference"],
                "roi_nlevels": [2, 2],
                "roi_level_0": ["S_new", "S_new"],
                "roi_level_1": ["ROI7", "ROI7"],
                "trial": [4, 5],
                "outcome": ["nRew", "nStim"],
                "history": ["nRew", "nStim"],
                "ev_onset": [5.25, 5.75],
            }
        )

    def test_replaces_only_the_regenerated_group(self, tmp_path: Path) -> None:
        self.write_reference(tmp_path, "passive_axons", "S_old", 5.5)
        self.write_reference(tmp_path, "passive_dendrites", "S_dend", 5.4)
        pooled = pool_mixed_events(
            self.regenerated(),
            source_dir=str(tmp_path),
            suffix="_all_2s_reward_events",
            side="target",
            active_only=True,
            exclude=("TD1",),
            legacy_faithful=True,
        )
        by_protocol = pooled.set_index("protocol")
        axons = by_protocol.loc[["passive_axons"]]
        assert axons["roi_level_0"].tolist() == ["S_new"]
        assert axons["ev_onset"].tolist() == [5.25]
        assert by_protocol.loc[["passive_dendrites"], "roi_level_0"].tolist() == ["S_dend"]

    def test_regenerated_frame_is_indexed_like_the_reference(self) -> None:
        frame = _regenerated_frame(self.regenerated().iloc[:1])
        assert list(frame.index) == [("S_new", "ROI7")]
        assert "group" not in frame.columns
        assert "side" not in frame.columns
