"""Capture, write, and validate data-provenance records (schema v1.0).

Every pipeline output directory should carry::

    provenance.json    full machine-readable provenance record
    version_info.yaml  human-readable version record
    snapshots/         code snapshot, written when git state is dirty

Skip validation
---------------
A step's output is considered still valid (and therefore skippable) when:

1. ``provenance.json`` exists in the output directory,
2. ``code.git.commit_hash`` matches the current HEAD, and
3. ``code.git.is_dirty`` matches the current dirty state.

Both conditions matter: comparing the dirty flag as well as the hash is what
stops a dirty-tree rerun from being falsely skipped.

Notes
-----
This module is deliberately **stdlib-only**: ``yaml`` and ``psutil`` are soft
imports that degrade gracefully. That lets tooling import it in a fresh checkout
before ``uv sync`` has run, and under whichever interpreter is available -- so
it is also kept syntax-compatible with older Pythons via
``from __future__ import annotations``. Do not add a third-party import here.

Ported from ``multimodal_toolkit.alignment.provenance_utils``. The public API is
intentionally identical so existing call sites transfer unchanged; see
``docs/DECISIONS.md`` for the two bugs corrected during the port.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _HAS_YAML = False

PROVENANCE_SCHEMA_VERSION = "1.0"

#: Files larger than this skip checksum computation. Hashing a multi-GB imaging
#: stack costs minutes and would dominate pipeline runtime; callers must
#: tolerate a ``None`` checksum.
LARGE_FILE_BYTES = 500 * 1024 * 1024  # 500 MB

#: Namespace prefix for auto-generated git tags.
AUTO_TAG_NAMESPACE = "calcium2p"

#: Default packages whose versions are recorded in every environment snapshot.
#: Tuned to the two-photon calcium stack.
DEFAULT_KEY_PACKAGES = (
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "h5py",
    "tifffile",
    "scikit-image",
    "scikit-learn",
    "statsmodels",
    "suite2p",
)


# ---------------------------------------------------------------------------
# Git state
# ---------------------------------------------------------------------------


def get_git_state(repo_path: str | Path = ".") -> dict[str, Any]:
    """Capture the current git state of a repository.

    Every git invocation is failure-tolerant: a missing repo, a missing ``git``
    binary, or any subprocess error yields an empty string rather than raising.
    Provenance capture must never be the reason a pipeline dies.

    Parameters
    ----------
    repo_path
        Directory inside the repository to interrogate.

    Returns
    -------
    dict
        The ``code.git`` block of ``provenance.json``, with keys
        ``commit_hash``, ``commit_hash_short``, ``branch``, ``is_dirty``,
        ``detached_head``, ``remote_url``, ``commit_timestamp``,
        ``commit_message``, ``commit_author`` and ``existing_tags``.
    """

    def _git(*args: str) -> str:
        try:
            out = subprocess.check_output(
                ["git", *args],
                cwd=str(repo_path),
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.SubprocessError, OSError):
            return ""
        return out.decode(errors="replace").strip()

    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty_lines = _git("status", "--porcelain")
    remote = _git("remote", "get-url", "origin")
    commit_ts = _git("log", "-1", "--format=%cI")
    commit_msg = _git("log", "-1", "--format=%s")
    commit_auth = _git("log", "-1", "--format=%an <%ae>")
    tags_str = _git("tag", "--contains", "HEAD") if commit else ""
    tags = [t for t in tags_str.splitlines() if t]

    return {
        "commit_hash": commit,
        "commit_hash_short": commit[:7] if commit else "",
        "branch": branch,
        "is_dirty": bool(dirty_lines),
        "detached_head": branch == "HEAD",
        "remote_url": remote,
        "commit_timestamp": commit_ts,
        "commit_message": commit_msg,
        "commit_author": commit_auth,
        "existing_tags": tags,
    }


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def get_environment_info(key_packages: list[str] | None = None) -> dict[str, Any]:
    """Capture Python runtime and machine details.

    Parameters
    ----------
    key_packages
        Package names to version-check. Defaults to :data:`DEFAULT_KEY_PACKAGES`.
        Packages that are not installed are recorded as ``"not_installed"``
        rather than omitted, so the absence is itself part of the record.

    Returns
    -------
    dict
        The ``environment`` block of ``provenance.json``.
    """
    packages = list(DEFAULT_KEY_PACKAGES) if key_packages is None else key_packages

    key_deps: dict[str, str] = {}
    for pkg in packages:
        try:
            key_deps[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            key_deps[pkg] = "not_installed"

    ram_gb: float | None = None
    with contextlib.suppress(ImportError):
        # Deliberately lazy: psutil is a soft dependency, and this module must
        # import cleanly in environments where it is absent.
        import psutil  # noqa: PLC0415

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "os": f"{platform.system()}-{platform.release()}",
        "cpu_count": os.cpu_count() or 0,
        "ram_gb": ram_gb,
        "key_dependencies": key_deps,
        "virtualenv": os.environ.get("VIRTUAL_ENV", sys.prefix),
        "uv_version": _tool_version("uv", "--version"),
    }


def _tool_version(executable: str, *args: str) -> str:
    """Return the version string of an external tool, or ``""`` if unavailable."""
    try:
        out = subprocess.check_output([executable, *args], stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return ""
    return out.decode(errors="replace").strip()


# ---------------------------------------------------------------------------
# Checksums and file info
# ---------------------------------------------------------------------------


def compute_checksum(path: str | Path, algorithm: str = "sha256") -> str | None:
    """Compute a checksum for a file.

    Parameters
    ----------
    path
        File to hash.
    algorithm
        Any algorithm name accepted by :func:`hashlib.new`.

    Returns
    -------
    str or None
        ``"<algorithm>:<hexdigest>"``, or ``None`` when the path does not
        exist, is a directory (e.g. a zarr store), or exceeds
        :data:`LARGE_FILE_BYTES`. Callers must tolerate ``None``.
    """
    path = Path(path)
    if not path.exists() or path.is_dir():
        return None
    if path.stat().st_size > LARGE_FILE_BYTES:
        return None

    digest = hashlib.new(algorithm)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return f"{algorithm}:{digest.hexdigest()}"


def file_info(path: str | Path) -> dict[str, Any]:
    """Describe a single file for inclusion in a provenance record.

    Returns
    -------
    dict
        Keys ``path``, ``size_bytes`` and ``checksum``. ``size_bytes`` is
        ``None`` when the file is absent.
    """
    path = Path(path)
    size = path.stat().st_size if path.exists() else None
    return {
        "path": str(path),
        "size_bytes": size,
        "checksum": compute_checksum(path) if size is not None else None,
    }


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def create_snapshots(
    snapshots_dir: str | Path,
    script_path: str | Path,
    repo_path: str | Path = ".",
) -> dict[str, str | None]:
    """Write code-reproducibility snapshots.

    Creates ``script_snapshot.py`` (a copy of the invoking script),
    ``git_diff.patch`` (uncommitted changes relative to HEAD) and
    ``requirements_frozen.txt`` inside ``snapshots_dir``.

    Parameters
    ----------
    snapshots_dir
        Destination directory; created if absent.
    script_path
        The script that produced the output being described.
    repo_path
        Repository root, used for the diff.

    Returns
    -------
    dict
        ``{filename: checksum}`` for embedding in ``provenance.json``.
    """
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str | None] = {}

    script_path = Path(script_path)
    if script_path.exists():
        dest = snapshots_dir / "script_snapshot.py"
        # copyfile, not copy2: content only. Preserving metadata raises EPERM on
        # Windows filesystem mounts, which is where much of this data lives.
        shutil.copyfile(script_path, dest)
        checksums["script_snapshot.py"] = compute_checksum(dest)

    diff = _capture(["git", "diff", "HEAD"], cwd=str(repo_path))
    if diff is not None:
        diff_path = snapshots_dir / "git_diff.patch"
        diff_path.write_text(diff, encoding="utf-8")
        checksums["git_diff.patch"] = compute_checksum(diff_path)

    frozen = _freeze_requirements()
    if frozen is not None:
        req_path = snapshots_dir / "requirements_frozen.txt"
        req_path.write_text(frozen, encoding="utf-8")
        checksums["requirements_frozen.txt"] = compute_checksum(req_path)

    return checksums


def _capture(cmd: list[str], cwd: str | None = None) -> str | None:
    """Run ``cmd`` and return decoded stdout, or ``None`` on any failure."""
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL)
    except (subprocess.SubprocessError, OSError):
        return None
    return out.decode(errors="replace")


def _freeze_requirements() -> str | None:
    """Freeze the current environment.

    Prefers ``uv pip freeze`` since this repo is uv-managed, and falls back to
    ``python -m pip freeze`` for environments where uv is unavailable.
    """
    return _capture(["uv", "pip", "freeze"]) or _capture([sys.executable, "-m", "pip", "freeze"])


# ---------------------------------------------------------------------------
# Auto-tag
# ---------------------------------------------------------------------------


def build_auto_tag(
    mutation_type: str,
    run_id: str,
    timestamp: str | None = None,
) -> str:
    """Build a git auto-tag string.

    The convention is ``<namespace>/<mutation_type>/<run_id>/<YYYYMMDD_HHMMSS>``.

    Parameters
    ----------
    mutation_type
        e.g. ``"event_detection"``.
    run_id
        e.g. ``"sub-bono_ses-20260409_run-01"``.
    timestamp
        ``YYYYMMDD_HHMMSS``; defaults to now (UTC).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{AUTO_TAG_NAMESPACE}/{mutation_type}/{run_id}/{timestamp}"


# ---------------------------------------------------------------------------
# Provenance I/O
# ---------------------------------------------------------------------------


def write_provenance(output_dir: str | Path, provenance: dict[str, Any]) -> Path:
    """Write ``provenance.json`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "provenance.json"
    # default=str so a stray Path or datetime cannot crash the write -- losing
    # the record would be far worse than a slightly lossy field.
    target.write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return target


def write_version_info(output_dir: str | Path, version_info: dict[str, Any]) -> Path:
    """Write ``version_info.yaml`` into ``output_dir`` and return its path.

    Falls back to JSON content (under the same ``.yaml`` name) when PyYAML is
    unavailable, so the record is always written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "version_info.yaml"
    if _HAS_YAML:
        content = yaml.dump(
            version_info,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    else:  # pragma: no cover - exercised only in minimal environments
        content = json.dumps(version_info, indent=2, default=str)
    target.write_text(content, encoding="utf-8")
    return target


def load_provenance(output_dir: str | Path) -> dict[str, Any] | None:
    """Read ``provenance.json`` from ``output_dir``.

    Returns
    -------
    dict or None
        ``None`` when the file is missing or unparseable.
    """
    target = Path(output_dir) / "provenance.json"
    if not target.exists():
        return None
    try:
        loaded: Any = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ---------------------------------------------------------------------------
# Skip logic
# ---------------------------------------------------------------------------


def is_step_valid(
    output_dir: str | Path,
    current_git: dict[str, Any],
) -> tuple[bool, str]:
    """Check whether a step's existing output is still valid.

    Parameters
    ----------
    output_dir
        Directory holding the step's output and its ``provenance.json``.
    current_git
        Output of :func:`get_git_state`.

    Returns
    -------
    tuple of (bool, str)
        Validity and a human-readable reason, suitable for logging either way.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False, "output directory does not exist"

    prov = load_provenance(output_dir)
    if prov is None:
        return False, "no provenance.json found"

    try:
        recorded_commit = prov["code"]["git"]["commit_hash"]
        recorded_dirty = prov["code"]["git"]["is_dirty"]
    except (KeyError, TypeError):
        return False, "provenance.json missing git state fields"

    if recorded_commit != current_git["commit_hash"]:
        return False, (
            f"git commit mismatch - recorded: {str(recorded_commit)[:7]}, "
            f"current: {str(current_git['commit_hash'])[:7]}"
        )
    if recorded_dirty != current_git["is_dirty"]:
        return False, (
            f"git dirty state mismatch - recorded: {recorded_dirty}, "
            f"current: {current_git['is_dirty']}"
        )

    return True, "provenance matches current git state"


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------


def build_provenance_record(
    *,
    mutation_type: str,
    version: str,
    operator: dict[str, str],
    source_files: list[dict[str, Any]],
    script_path: str,
    git_state: dict[str, Any],
    pipeline_steps: list[dict[str, Any]],
    environment: dict[str, Any],
    output_files: dict[str, Any],
    lineage_chain: list[str],
    warnings: list[dict[str, Any]] | None = None,
    snapshot_info: dict[str, str | None] | None = None,
    run_id: str = "",
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> dict[str, Any]:
    """Assemble a full ``provenance.json`` record (schema v1.0).

    Parameters
    ----------
    mutation_type
        The transformation performed, e.g. ``"event_detection"``.
    version
        Self-describing parameter string, e.g. ``"v1_fps-30_thresh-2.5"``.
    operator
        ``{"github_username": ..., "name": ...}``.
    source_files
        :func:`file_info` dicts for the inputs.
    script_path
        Script that executed this step.
    git_state
        Output of :func:`get_git_state`.
    pipeline_steps
        Per-step dicts with ``step``, ``name``, ``parameters``, ``output``.
    environment
        Output of :func:`get_environment_info`.
    output_files
        ``{"primary": file_info(...), "supplementary": [...]}``.
    lineage_chain
        Ordered paths from raw input through to this output.
    warnings
        Optional warning dicts.
    snapshot_info
        Optional checksums from :func:`create_snapshots`.
    run_id
        e.g. ``"sub-bono_ses-20260409_run-01"``; drives the auto-tag.
    start_time, end_time
        Execution bounds, used for the duration summary.
    """
    now = datetime.now(timezone.utc)
    auto_tag = (
        build_auto_tag(mutation_type, run_id, now.strftime("%Y%m%d_%H%M%S")) if run_id else ""
    )

    git_block = dict(git_state)
    git_block["auto_tag"] = auto_tag
    git_block["script_path"] = script_path
    git_block["snapshot_required"] = bool(git_state.get("is_dirty")) or not bool(
        git_state.get("existing_tags")
    )
    if snapshot_info:
        git_block["snapshot"] = {
            "required": True,
            "reason": "uncommitted_changes" if git_state.get("is_dirty") else "tag_not_pushed",
            "checksums": snapshot_info,
        }

    duration: float | None = None
    if start_time is not None and end_time is not None:
        duration = (end_time - start_time).total_seconds()

    return {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "mutation_type": mutation_type,
        "version": version,
        "timestamp": now.isoformat(),
        "operator": operator,
        "source_data": {
            "raw_files": source_files,
            "derivative_files": [],
        },
        "code": {
            "script": script_path,
            "git": git_block,
        },
        "processing_pipeline": pipeline_steps,
        "environment": environment,
        "output_files": output_files,
        "warnings": warnings or [],
        "lineage_chain": lineage_chain,
        "execution_summary": {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            # `is not None`, not truthiness: a genuine 0.0-second duration is
            # real data and must not serialize as null.
            "total_duration_seconds": round(duration, 2) if duration is not None else None,
            "state": "completed",
            "exit_code": 0,
        },
    }


def build_version_info_record(
    *,
    version: str,
    mutation_type: str,
    description: str,
    parameters: dict[str, Any],
    git_state: dict[str, Any],
    source_data: list[str],
    output_data: list[str],
    operator: str,
    run_id: str = "",
    timestamp: str | None = None,
    warnings: list[str] | None = None,
    notes: str = "",
    status: str = "current",
) -> dict[str, Any]:
    """Assemble a human-readable ``version_info.yaml`` record.

    Parameters
    ----------
    version
        Self-describing parameter string.
    mutation_type
        The transformation performed.
    description
        Prose explanation of what this version does and why.
    parameters
        The parameters that define this version.
    git_state
        Output of :func:`get_git_state`.
    source_data, output_data
        Input and output paths.
    operator
        GitHub username of the person responsible.
    run_id
        Passed to :func:`build_auto_tag`. Supply the *same* value used for the
        matching :func:`build_provenance_record` call so both records carry an
        identical ``auto_tag``.
    timestamp
        ``YYYYMMDD_HHMMSS`` for the auto-tag; defaults to now (UTC). Pass the
        provenance record's timestamp to make the two tags byte-identical.
    warnings, notes
        Optional free-form annotations.
    status
        One of ``"current"``, ``"testing"`` or ``"deprecated"``.
    """
    return {
        "version": version,
        "mutation_type": mutation_type,
        "status": status,
        "date_created": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "operator": operator,
        "description": description,
        "parameters": parameters,
        "git": {
            "commit": git_state.get("commit_hash", ""),
            "branch": git_state.get("branch", ""),
            "is_dirty": git_state.get("is_dirty", False),
            # The upstream implementation passed `operator` into the run_id slot
            # here, so version_info.yaml's auto_tag silently disagreed with
            # provenance.json's. run_id is now threaded through explicitly.
            "auto_tag": build_auto_tag(mutation_type, run_id, timestamp),
            "tag_pushed": False,
        },
        "source_data": source_data,
        "output_data": output_data,
        "warnings": warnings or [],
        "notes": notes,
    }
