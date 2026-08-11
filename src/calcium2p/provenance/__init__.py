"""Data provenance: capture, write, and validate the record behind every output.

Two layers, one on-disk schema:

- :mod:`~calcium2p.provenance.provenance_utils` -- stdlib-only dict builders.
  This is the schema. Import it from hooks and standalone scripts.
- :mod:`~calcium2p.provenance.records` -- frozen dataclasses that round-trip to
  identical dicts. Import these from package code so mypy can check them.
"""

from __future__ import annotations

from calcium2p.provenance.provenance_utils import (
    AUTO_TAG_NAMESPACE,
    DEFAULT_KEY_PACKAGES,
    LARGE_FILE_BYTES,
    PROVENANCE_SCHEMA_VERSION,
    build_auto_tag,
    build_provenance_record,
    build_version_info_record,
    compute_checksum,
    create_snapshots,
    file_info,
    get_environment_info,
    get_git_state,
    is_step_valid,
    load_provenance,
    write_provenance,
    write_version_info,
)
from calcium2p.provenance.records import (
    ArtifactKind,
    FileRef,
    GitState,
    Operator,
    PipelineStep,
    ProvenanceRecord,
    VersionInfo,
    VersionStatus,
)

__all__ = [
    "AUTO_TAG_NAMESPACE",
    "DEFAULT_KEY_PACKAGES",
    "LARGE_FILE_BYTES",
    "PROVENANCE_SCHEMA_VERSION",
    "ArtifactKind",
    "FileRef",
    "GitState",
    "Operator",
    "PipelineStep",
    "ProvenanceRecord",
    "VersionInfo",
    "VersionStatus",
    "build_auto_tag",
    "build_provenance_record",
    "build_version_info_record",
    "compute_checksum",
    "create_snapshots",
    "file_info",
    "get_environment_info",
    "get_git_state",
    "is_step_valid",
    "load_provenance",
    "write_provenance",
    "write_version_info",
]
