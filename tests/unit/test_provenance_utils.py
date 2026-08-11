"""Tests for the ported provenance module.

Includes explicit regression tests for the two defects corrected during the port
from ``multimodal-toolkit``. Those two tests are the reason this file exists:
without them nothing stops the bugs being reintroduced by a future re-sync with
upstream.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from calcium2p.provenance import provenance_utils as pu


class TestGitState:
    def test_captures_commit_and_branch(self, git_repo: Path) -> None:
        state = pu.get_git_state(git_repo)
        assert len(state["commit_hash"]) == 40
        assert state["commit_hash_short"] == state["commit_hash"][:7]
        assert state["branch"] == "main"
        assert state["is_dirty"] is False
        assert state["commit_message"] == "feat: seed the test repository"

    def test_detects_dirty_tree(self, git_repo: Path) -> None:
        (git_repo / "seed.txt").write_text("modified\n", encoding="utf-8")
        assert pu.get_git_state(git_repo)["is_dirty"] is True

    def test_never_raises_outside_a_repo(self, tmp_path: Path) -> None:
        # Provenance capture must not be the reason a pipeline dies.
        state = pu.get_git_state(tmp_path)
        assert state["commit_hash"] == ""
        assert state["is_dirty"] is False


class TestChecksums:
    def test_round_trips(self, tmp_path: Path) -> None:
        target = tmp_path / "a.bin"
        target.write_bytes(b"calcium")
        digest = pu.compute_checksum(target)
        assert digest is not None
        assert digest.startswith("sha256:")
        assert pu.compute_checksum(target) == digest

    def test_none_for_missing_and_directories(self, tmp_path: Path) -> None:
        assert pu.compute_checksum(tmp_path / "nope.bin") is None
        assert pu.compute_checksum(tmp_path) is None

    def test_none_for_files_over_the_size_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(pu, "LARGE_FILE_BYTES", 4)
        target = tmp_path / "big.bin"
        target.write_bytes(b"more than four bytes")
        # None means "not computed", never "empty".
        assert pu.compute_checksum(target) is None

    def test_file_info_on_missing_file(self, tmp_path: Path) -> None:
        info = pu.file_info(tmp_path / "absent.npy")
        assert info["size_bytes"] is None
        assert info["checksum"] is None


class TestSkipLogic:
    def test_valid_when_git_matches(self, git_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        git = pu.get_git_state(git_repo)
        pu.write_provenance(out, {"code": {"git": git}})
        valid, reason = pu.is_step_valid(out, git)
        assert valid, reason

    def test_invalid_when_commit_differs(self, git_repo: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        git = pu.get_git_state(git_repo)
        pu.write_provenance(out, {"code": {"git": git}})
        moved = {**git, "commit_hash": "0" * 40}
        valid, reason = pu.is_step_valid(out, moved)
        assert not valid
        assert "commit mismatch" in reason

    def test_invalid_when_only_dirty_flag_differs(self, git_repo: Path, tmp_path: Path) -> None:
        # The dirty comparison is what stops an edited-but-uncommitted tree from
        # reusing stale output. Hash equality alone would wrongly skip here.
        out = tmp_path / "out"
        out.mkdir()
        clean = pu.get_git_state(git_repo)
        pu.write_provenance(out, {"code": {"git": clean}})
        valid, reason = pu.is_step_valid(out, {**clean, "is_dirty": True})
        assert not valid
        assert "dirty state mismatch" in reason

    def test_invalid_without_provenance(self, tmp_path: Path) -> None:
        valid, reason = pu.is_step_valid(tmp_path, {"commit_hash": "x", "is_dirty": False})
        assert not valid
        assert "no provenance.json" in reason


class TestRegressions:
    """The two bugs corrected during the port from multimodal-toolkit."""

    def test_auto_tag_agrees_between_the_two_records(self) -> None:
        # Upstream passed `operator` into build_auto_tag's run_id slot, so
        # version_info.yaml's auto_tag silently disagreed with provenance.json's.
        run_id = "sub-bono_ses-20260409_run-01"
        timestamp = "20260409_120000"
        git = {"commit_hash": "a" * 40, "is_dirty": False, "existing_tags": ["t"]}

        version_info = pu.build_version_info_record(
            version="v1",
            mutation_type="event_detection",
            description="d",
            parameters={},
            git_state=git,
            source_data=[],
            output_data=[],
            operator="raymasooda",
            run_id=run_id,
            timestamp=timestamp,
        )

        expected = pu.build_auto_tag("event_detection", run_id, timestamp)
        assert version_info["git"]["auto_tag"] == expected
        assert run_id in version_info["git"]["auto_tag"]
        assert "raymasooda" not in version_info["git"]["auto_tag"]

    def test_zero_duration_is_zero_not_null(self) -> None:
        # Upstream used `round(duration, 2) if duration else None`, so a genuine
        # 0.0-second step serialized as null -- indistinguishable from "unknown".
        moment = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        record = pu.build_provenance_record(
            mutation_type="noop",
            version="v1",
            operator={},
            source_files=[],
            script_path="s.py",
            git_state={},
            pipeline_steps=[],
            environment={},
            output_files={},
            lineage_chain=[],
            start_time=moment,
            end_time=moment,
        )
        assert record["execution_summary"]["total_duration_seconds"] == 0.0

    def test_real_duration_still_recorded(self) -> None:
        start = datetime(2026, 4, 9, 12, 0, 0, tzinfo=timezone.utc)
        record = pu.build_provenance_record(
            mutation_type="noop",
            version="v1",
            operator={},
            source_files=[],
            script_path="s.py",
            git_state={},
            pipeline_steps=[],
            environment={},
            output_files={},
            lineage_chain=[],
            start_time=start,
            end_time=start + timedelta(seconds=1.5),
        )
        assert record["execution_summary"]["total_duration_seconds"] == 1.5

    def test_unknown_duration_is_none(self) -> None:
        record = pu.build_provenance_record(
            mutation_type="noop",
            version="v1",
            operator={},
            source_files=[],
            script_path="s.py",
            git_state={},
            pipeline_steps=[],
            environment={},
            output_files={},
            lineage_chain=[],
        )
        assert record["execution_summary"]["total_duration_seconds"] is None


class TestRecordIO:
    def test_provenance_round_trip(self, tmp_path: Path) -> None:
        record = pu.build_provenance_record(
            mutation_type="event_detection",
            version="v1_thresh-2.5",
            operator={"github_username": "raymasooda"},
            source_files=[pu.file_info(tmp_path)],
            script_path="detect.py",
            git_state=pu.get_git_state(tmp_path),
            pipeline_steps=[],
            environment=pu.get_environment_info([]),
            output_files={},
            lineage_chain=[],
            run_id="run-01",
        )
        pu.write_provenance(tmp_path, record)
        assert pu.load_provenance(tmp_path) == json.loads(
            (tmp_path / "provenance.json").read_text(encoding="utf-8")
        )

    def test_load_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / "provenance.json").write_text("{not json", encoding="utf-8")
        assert pu.load_provenance(tmp_path) is None

    def test_write_survives_non_serializable_values(self, tmp_path: Path) -> None:
        # default=str: losing a field to stringification beats losing the record.
        pu.write_provenance(tmp_path, {"path": Path("/tmp/x"), "when": datetime.now()})
        loaded = pu.load_provenance(tmp_path)
        assert loaded is not None
        assert loaded["path"] == "/tmp/x"

    def test_environment_marks_absent_packages(self) -> None:
        env = pu.get_environment_info(["definitely-not-a-real-package"])
        assert env["key_dependencies"]["definitely-not-a-real-package"] == "not_installed"

    def test_snapshots_written_for_dirty_tree(self, git_repo: Path, tmp_path: Path) -> None:
        (git_repo / "seed.txt").write_text("dirty\n", encoding="utf-8")
        script = git_repo / "run.py"
        script.write_text("print('hi')\n", encoding="utf-8")
        checksums = pu.create_snapshots(tmp_path / "snapshots", script, git_repo)
        assert (tmp_path / "snapshots" / "script_snapshot.py").exists()
        assert checksums["script_snapshot.py"] is not None
        assert "dirty" in (tmp_path / "snapshots" / "git_diff.patch").read_text()
