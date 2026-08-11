"""Typed wrappers over the provenance dict builders.

:mod:`calcium2p.provenance.provenance_utils` deliberately speaks in plain dicts:
it is the on-disk schema, it must stay stdlib-only, and it must import under a
bare interpreter. That makes it invisible to the type checker.

This module adds frozen dataclasses that round-trip to **byte-identical** dicts,
so callers inside the package get static checking and editor completion without
changing anything about the serialized format. ``tests/unit/test_records.py``
asserts the round-trip equivalence.

Use these in new code; use the raw builders when writing a record from a
standalone script that must not import the wider package.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from calcium2p.provenance import provenance_utils as pu

if TYPE_CHECKING:
    # typing.Self is 3.11+. Importing it only under TYPE_CHECKING keeps this
    # module importable under older interpreters, since
    # `from __future__ import annotations` means the annotation is never
    # evaluated at runtime.
    from typing import Self

ArtifactKind = Literal["data", "figure", "metric", "model", "report"]
VersionStatus = Literal["current", "testing", "deprecated"]


@dataclass(frozen=True)
class GitState:
    """A repository's git state at the moment of capture."""

    commit_hash: str = ""
    commit_hash_short: str = ""
    branch: str = ""
    is_dirty: bool = False
    detached_head: bool = False
    remote_url: str = ""
    commit_timestamp: str = ""
    commit_message: str = ""
    commit_author: str = ""
    existing_tags: tuple[str, ...] = ()

    @classmethod
    def capture(cls, repo_path: str | Path = ".") -> Self:
        """Capture the current git state via :func:`~.provenance_utils.get_git_state`."""
        return cls.from_dict(pu.get_git_state(repo_path))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Build from a raw ``code.git`` dict, ignoring extra keys."""
        return cls(
            commit_hash=data.get("commit_hash", ""),
            commit_hash_short=data.get("commit_hash_short", ""),
            branch=data.get("branch", ""),
            is_dirty=bool(data.get("is_dirty", False)),
            detached_head=bool(data.get("detached_head", False)),
            remote_url=data.get("remote_url", ""),
            commit_timestamp=data.get("commit_timestamp", ""),
            commit_message=data.get("commit_message", ""),
            commit_author=data.get("commit_author", ""),
            existing_tags=tuple(data.get("existing_tags", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the exact dict shape :func:`~.provenance_utils.get_git_state` returns."""
        return {
            "commit_hash": self.commit_hash,
            "commit_hash_short": self.commit_hash_short,
            "branch": self.branch,
            "is_dirty": self.is_dirty,
            "detached_head": self.detached_head,
            "remote_url": self.remote_url,
            "commit_timestamp": self.commit_timestamp,
            "commit_message": self.commit_message,
            "commit_author": self.commit_author,
            "existing_tags": list(self.existing_tags),
        }

    @property
    def is_reproducible(self) -> bool:
        """Whether an output built from this state can be reproduced from git alone.

        False when the tree is dirty (uncommitted changes are not recoverable
        from the remote) or when there is no commit at all.
        """
        return bool(self.commit_hash) and not self.is_dirty


@dataclass(frozen=True)
class FileRef:
    """A file identified by path, size and checksum."""

    path: Path
    size_bytes: int | None = None
    checksum: str | None = None
    kind: ArtifactKind | None = None

    @classmethod
    def describe(cls, path: str | Path, kind: ArtifactKind | None = None) -> Self:
        """Stat and checksum ``path`` via :func:`~.provenance_utils.file_info`."""
        info = pu.file_info(path)
        return cls(
            path=Path(info["path"]),
            size_bytes=info["size_bytes"],
            checksum=info["checksum"],
            kind=kind,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict shape :func:`~.provenance_utils.file_info` returns.

        ``kind`` is included only when set, matching the upstream convention of
        splicing a ``type`` key into the file_info dict at the call site.
        """
        out: dict[str, Any] = {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }
        if self.kind is not None:
            out["type"] = self.kind
        return out

    @property
    def exists(self) -> bool:
        """Whether the file is present on disk right now."""
        return self.path.exists()


@dataclass(frozen=True)
class Operator:
    """The person responsible for producing an output."""

    github_username: str = ""
    name: str = ""

    @classmethod
    def from_env(cls, repo_path: str | Path = ".") -> Self:
        """Resolve from ``OPERATOR_*`` env vars, falling back to git config."""

        def _git_config(key: str) -> str:
            try:
                out = subprocess.check_output(
                    ["git", "config", "--get", key],
                    cwd=str(repo_path),
                    stderr=subprocess.DEVNULL,
                )
            except (subprocess.SubprocessError, OSError):
                return ""
            return out.decode(errors="replace").strip()

        return cls(
            github_username=os.environ.get("OPERATOR_GITHUB", "") or _git_config("user.name"),
            name=os.environ.get("OPERATOR_NAME", "") or _git_config("user.name"),
        )

    def to_dict(self) -> dict[str, str]:
        """Serialize to the ``operator`` block."""
        return {"github_username": self.github_username, "name": self.name}


@dataclass(frozen=True)
class PipelineStep:
    """One step within a recorded processing pipeline."""

    step: int
    name: str
    script: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a ``processing_pipeline`` entry."""
        return {
            "step": self.step,
            "name": self.name,
            "script": self.script,
            "parameters": self.parameters,
            "output": self.output,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class ProvenanceRecord:
    """A complete provenance record, buildable and writable in one object.

    Examples
    --------
    >>> record = ProvenanceRecord(                       # doctest: +SKIP
    ...     mutation_type="event_detection",
    ...     version="v1_thresh-2.5",
    ...     run_id="sub-bono_ses-20260409_run-01",
    ...     operator=Operator.from_env(),
    ...     git=GitState.capture(),
    ...     sources=[FileRef.describe(raw_path, kind="data")],
    ...     primary_output=FileRef.describe(out_path, kind="data"),
    ...     script_path=__file__,
    ... )
    >>> record.write(out_dir)                            # doctest: +SKIP
    """

    mutation_type: str
    version: str
    run_id: str
    operator: Operator
    git: GitState
    sources: list[FileRef]
    primary_output: FileRef
    script_path: str
    steps: list[PipelineStep] = field(default_factory=list)
    supplementary_outputs: list[FileRef] = field(default_factory=list)
    lineage_chain: list[str] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    snapshot_info: dict[str, str | None] | None = None
    environment: dict[str, Any] | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Delegate to :func:`~.provenance_utils.build_provenance_record`."""
        lineage = self.lineage_chain or [
            *(str(s.path) for s in self.sources),
            str(self.primary_output.path),
        ]
        warnings = list(self.warnings)
        if self.git.is_dirty and not any(w.get("category") == "git_state" for w in warnings):
            warnings.append(
                {
                    "severity": "warning",
                    "category": "git_state",
                    "message": "Repository has uncommitted changes",
                    "recommendation": "Commit changes before processing",
                }
            )

        return pu.build_provenance_record(
            mutation_type=self.mutation_type,
            version=self.version,
            operator=self.operator.to_dict(),
            source_files=[s.to_dict() for s in self.sources],
            script_path=self.script_path,
            git_state=self.git.to_dict(),
            pipeline_steps=[s.to_dict() for s in self.steps],
            environment=self.environment
            if self.environment is not None
            else pu.get_environment_info(),
            output_files={
                "primary": self.primary_output.to_dict(),
                "supplementary": [o.to_dict() for o in self.supplementary_outputs],
            },
            lineage_chain=lineage,
            warnings=warnings,
            snapshot_info=self.snapshot_info,
            run_id=self.run_id,
            start_time=self.start_time,
            end_time=self.end_time,
        )

    def write(self, output_dir: str | Path) -> Path:
        """Write ``provenance.json`` into ``output_dir``."""
        return pu.write_provenance(output_dir, self.to_dict())


@dataclass(frozen=True)
class VersionInfo:
    """The human-readable companion to a :class:`ProvenanceRecord`."""

    version: str
    mutation_type: str
    description: str
    parameters: dict[str, Any]
    git: GitState
    source_data: list[str]
    output_data: list[str]
    operator: str
    run_id: str = ""
    timestamp: str | None = None
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    status: VersionStatus = "current"

    def to_dict(self) -> dict[str, Any]:
        """Delegate to :func:`~.provenance_utils.build_version_info_record`."""
        return pu.build_version_info_record(
            version=self.version,
            mutation_type=self.mutation_type,
            description=self.description,
            parameters=self.parameters,
            git_state=self.git.to_dict(),
            source_data=self.source_data,
            output_data=self.output_data,
            operator=self.operator,
            run_id=self.run_id,
            timestamp=self.timestamp,
            warnings=self.warnings,
            notes=self.notes,
            status=self.status,
        )

    def write(self, output_dir: str | Path) -> Path:
        """Write ``version_info.yaml`` into ``output_dir``."""
        return pu.write_version_info(output_dir, self.to_dict())
