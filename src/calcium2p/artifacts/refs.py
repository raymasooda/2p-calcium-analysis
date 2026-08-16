"""References to stored artifacts, and records of the mutations that produce them.

Two ways to represent a derived dataset:

:class:`ArtifactRef`
    The data exists on disk. The ref carries its path, size and checksum.
:class:`MutationRecord`
    The data does *not* exist on disk. Instead we record exactly how to rebuild
    it -- which callable, applied to which inputs, with which parameters. Used
    when an artifact is storage-heavy but cheap to recompute.

Both serialize to plain JSON so an agent or a downstream script can read the
lineage without importing this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Self

from calcium2p.provenance import compute_checksum
from calcium2p.provenance.records import ArtifactKind

#: How a stored artifact was serialized. Drives loader selection on read.
SerializerName = Literal["npy", "npz", "parquet", "json", "svg", "csv", "text"]

#: Which representation was chosen for a given artifact.
Representation = Literal["materialized", "mutation_record"]


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A pointer to a materialized artifact on disk.

    Attributes
    ----------
    key
        Stable logical identifier, unique within an :class:`~.store.ArtifactStore`.
        Slashes are permitted and become directories.
    path
        Location of the serialized data.
    serializer
        How it was written; determines how it is read back.
    kind
        Semantic category, shared with the provenance schema.
    size_bytes, checksum
        Populated by :meth:`describe`; ``checksum`` is ``None`` for files above
        :data:`~calcium2p.provenance.LARGE_FILE_BYTES`.
    """

    key: str
    path: Path
    serializer: SerializerName
    kind: ArtifactKind = "data"
    size_bytes: int | None = None
    checksum: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)

    @classmethod
    def describe(
        cls,
        key: str,
        path: str | Path,
        serializer: SerializerName,
        kind: ArtifactKind = "data",
    ) -> Self:
        """Stat and checksum ``path``, returning a fully-populated ref."""
        path = Path(path)
        size = path.stat().st_size if path.exists() else None
        return cls(
            key=key,
            path=path,
            serializer=serializer,
            kind=kind,
            size_bytes=size,
            checksum=compute_checksum(path) if size is not None else None,
        )

    @property
    def exists(self) -> bool:
        """Whether the artifact is present on disk right now."""
        return self.path.exists()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "key": self.key,
            "path": str(self.path),
            "serializer": self.serializer,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output."""
        return cls(
            key=data["key"],
            path=Path(data["path"]),
            serializer=data["serializer"],
            kind=data.get("kind", "data"),
            size_bytes=data.get("size_bytes"),
            checksum=data.get("checksum"),
            created_at=data.get("created_at", _utc_now_iso()),
        )


@dataclass(frozen=True, slots=True)
class MutationRecord:
    """A recipe for rebuilding an artifact that was deliberately not stored.

    This is the "storage-heavy, compute-cheap" branch of the artifact policy:
    rather than writing gigabytes to disk, record which function was applied to
    which inputs to produce what output, and replay it on demand.

    Attributes
    ----------
    key
        Logical identifier of the output this record stands in for.
    function
        Fully-qualified ``module:qualname`` of the callable. Resolved by
        :meth:`~.store.ArtifactStore.replay` at rebuild time.
    parameters
        Keyword arguments passed alongside the inputs. Must be JSON-serializable.
    inputs
        Refs to the upstream artifacts, including their checksums -- so a replay
        can detect that an input changed underneath it.
    output_serializer
        How the rebuilt object would be written, were it materialized.
    compute_seconds
        Measured cost of the original computation. This is the evidence behind
        the policy decision, retained so the decision can be audited later.
    estimated_size_bytes
        Measured size of the output that was declined.
    git_commit
        HEAD at record time. A replay from a different commit may not reproduce.
    """

    key: str
    function: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[ArtifactRef, ...] = ()
    output_serializer: SerializerName = "npy"
    output_kind: ArtifactKind = "data"
    compute_seconds: float | None = None
    estimated_size_bytes: int | None = None
    git_commit: str = ""
    created_at: str = field(default_factory=_utc_now_iso)
    note: str = ""

    @property
    def replay_command(self) -> str:
        """A copy-pasteable one-liner that rebuilds this artifact.

        Human-readable half of the record; :meth:`~.store.ArtifactStore.replay`
        is the machine-readable half.
        """
        return (
            "uv run python -c "
            f'"from calcium2p.artifacts import ArtifactStore; '
            f"ArtifactStore.from_env().resolve('{self.key}')\""
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "key": self.key,
            "function": self.function,
            "parameters": self.parameters,
            "inputs": [i.to_dict() for i in self.inputs],
            "output_serializer": self.output_serializer,
            "output_kind": self.output_kind,
            "compute_seconds": self.compute_seconds,
            "estimated_size_bytes": self.estimated_size_bytes,
            "git_commit": self.git_commit,
            "created_at": self.created_at,
            "note": self.note,
            "replay_command": self.replay_command,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Rebuild from :meth:`to_dict` output.

        ``replay_command`` is derived, not stored state, so it is ignored here.
        """
        return cls(
            key=data["key"],
            function=data["function"],
            parameters=data.get("parameters", {}),
            inputs=tuple(ArtifactRef.from_dict(i) for i in data.get("inputs", ())),
            output_serializer=data.get("output_serializer", "npy"),
            output_kind=data.get("output_kind", "data"),
            compute_seconds=data.get("compute_seconds"),
            estimated_size_bytes=data.get("estimated_size_bytes"),
            git_commit=data.get("git_commit", ""),
            created_at=data.get("created_at", _utc_now_iso()),
            note=data.get("note", ""),
        )
