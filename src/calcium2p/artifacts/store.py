"""Content-addressed store for intermediate analysis artifacts.

:class:`ArtifactStore` is the single entry point for "compute this once, then
reuse it". It applies :class:`~.policy.MaterializationPolicy` to decide whether
each result is written to disk or replaced by a
:class:`~.refs.MutationRecord`, and writes a provenance sidecar either way.

Layout on disk, one directory per artifact key::

    <root>/<key>/meta.json         always -- representation, ref/record, verdict
    <root>/<key>/data.<ext>        only when materialized
    <root>/<key>/provenance.json   always

``meta.json`` is the entry point: reading it tells you whether the data is
present or must be replayed, without needing this package.

Examples
--------
>>> store = ArtifactStore(root)                      # doctest: +SKIP
>>> traces = store.get_or_compute(                   # doctest: +SKIP
...     "ses-20260409/dff",
...     compute_dff,
...     inputs=["ses-20260409/filtered"],
...     params={"baseline_window": (0, 90)},
... )
"""

from __future__ import annotations

import importlib
import json
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
import pandas as pd

from calcium2p.artifacts.policy import MaterializationPolicy, PolicyVerdict
from calcium2p.artifacts.refs import ArtifactRef, MutationRecord, SerializerName
from calcium2p.provenance import (
    FileRef,
    GitState,
    Operator,
    ProvenanceRecord,
    get_git_state,
)
from calcium2p.provenance.records import ArtifactKind

META_FILENAME = "meta.json"
ARTIFACT_SCHEMA_VERSION = "1.0"


class ArtifactStoreError(RuntimeError):
    """Raised when an artifact cannot be stored, resolved, or replayed."""


# ---------------------------------------------------------------------------
# Serializers -- a registry, because the behaviour genuinely varies by type
# ---------------------------------------------------------------------------


class Serializer(ABC):
    """Reads and writes one on-disk representation of an in-memory object."""

    name: ClassVar[SerializerName]
    extension: ClassVar[str]

    @abstractmethod
    def save(self, obj: Any, path: Path) -> None:
        """Write ``obj`` to ``path``."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Read back an object previously written by :meth:`save`."""

    @classmethod
    @abstractmethod
    def handles(cls, obj: Any) -> bool:
        """Whether this serializer can store ``obj``."""

    @classmethod
    def estimate_size(cls, obj: Any) -> int | None:  # noqa: ARG003
        """Estimate serialized size in bytes without writing anything.

        Returning an estimate rather than writing-then-measuring is what lets
        the policy decline a multi-gigabyte artifact without first spending the
        I/O to create it. ``None`` means unknown.
        """
        return None


class NumpySerializer(Serializer):
    """Single :class:`numpy.ndarray` as ``.npy``."""

    name = "npy"
    extension = ".npy"

    def save(self, obj: Any, path: Path) -> None:
        """Write the array, without pickle -- object arrays are rejected."""
        np.save(path, obj, allow_pickle=False)

    def load(self, path: Path) -> Any:
        """Read the array back."""
        return np.load(path, allow_pickle=False)

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Return True for non-object ndarrays."""
        return isinstance(obj, np.ndarray) and obj.dtype != np.dtype("O")

    @classmethod
    def estimate_size(cls, obj: Any) -> int | None:
        """Exact for a contiguous array: its buffer size."""
        return int(obj.nbytes) if isinstance(obj, np.ndarray) else None


class NpzSerializer(Serializer):
    """Mapping of name to :class:`numpy.ndarray` as compressed ``.npz``."""

    name = "npz"
    extension = ".npz"

    def save(self, obj: Any, path: Path) -> None:
        """Write all arrays into one compressed archive."""
        np.savez_compressed(path, **obj)

    def load(self, path: Path) -> Any:
        """Read back as a plain dict, so the file handle does not leak."""
        with np.load(path, allow_pickle=False) as handle:
            return {k: handle[k] for k in handle.files}

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Return True for a non-empty dict whose values are all ndarrays."""
        return (
            isinstance(obj, dict)
            and len(obj) > 0
            and all(isinstance(k, str) for k in obj)
            and all(isinstance(v, np.ndarray) for v in obj.values())
        )

    @classmethod
    def estimate_size(cls, obj: Any) -> int | None:
        """Sum of member buffers; an over-estimate, since npz compresses."""
        if not isinstance(obj, dict):
            return None
        return int(sum(v.nbytes for v in obj.values() if isinstance(v, np.ndarray)))


class ParquetSerializer(Serializer):
    """:class:`pandas.DataFrame` as Parquet.

    Chosen over CSV because it round-trips dtypes exactly. Silent dtype loss on
    CSV round-trip was a real defect class in the predecessor repo.
    """

    name = "parquet"
    extension = ".parquet"

    def save(self, obj: Any, path: Path) -> None:
        """Write the frame."""
        obj.to_parquet(path, index=True)

    def load(self, path: Path) -> Any:
        """Read the frame back."""
        return pd.read_parquet(path)

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Return True for DataFrames."""
        return isinstance(obj, pd.DataFrame)

    @classmethod
    def estimate_size(cls, obj: Any) -> int | None:
        """In-memory footprint; an over-estimate, since Parquet compresses."""
        if not isinstance(obj, pd.DataFrame):
            return None
        return int(obj.memory_usage(deep=True).sum())


class JsonSerializer(Serializer):
    """Any JSON-serializable object."""

    name = "json"
    extension = ".json"

    def save(self, obj: Any, path: Path) -> None:
        """Write indented JSON."""
        path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")

    def load(self, path: Path) -> Any:
        """Read the object back."""
        return json.loads(path.read_text(encoding="utf-8"))

    @classmethod
    def handles(cls, obj: Any) -> bool:
        """Return True when ``obj`` actually survives a JSON round-trip."""
        try:
            json.dumps(obj)
        except (TypeError, ValueError):
            return False
        return True


#: Ordered registry. :func:`infer_serializer` takes the first match, so the more
#: specific serializers must come before JsonSerializer's broad check.
SERIALIZERS: tuple[type[Serializer], ...] = (
    NumpySerializer,
    NpzSerializer,
    ParquetSerializer,
    JsonSerializer,
)

_BY_NAME: dict[str, type[Serializer]] = {s.name: s for s in SERIALIZERS}


def infer_serializer(obj: Any) -> type[Serializer]:
    """Select a serializer for ``obj``.

    Raises
    ------
    ArtifactStoreError
        If no registered serializer handles the object.
    """
    for serializer in SERIALIZERS:
        if serializer.handles(obj):
            return serializer
    raise ArtifactStoreError(
        f"No serializer handles {type(obj).__name__!r}. Register one in "
        f"calcium2p.artifacts.store.SERIALIZERS, or pass serializer= explicitly."
    )


def get_serializer(name: SerializerName) -> type[Serializer]:
    """Look up a serializer by name.

    Raises
    ------
    ArtifactStoreError
        If the name is not registered.
    """
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ArtifactStoreError(
            f"Unknown serializer {name!r}; known: {sorted(_BY_NAME)}"
        ) from None


def qualified_name(fn: Callable[..., Any]) -> str:
    """Return ``module:qualname`` for ``fn``.

    Raises
    ------
    ArtifactStoreError
        If ``fn`` is a lambda or a local closure. Such a function cannot be
        re-imported, so a mutation record naming it would be unreplayable -- we
        refuse to write one rather than record a recipe that cannot be followed.
    """
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        raise ArtifactStoreError(f"{fn!r} has no importable name")
    if "<lambda>" in qualname or "<locals>" in qualname:
        raise ArtifactStoreError(
            f"{qualname!r} is a lambda or closure and cannot be re-imported for "
            f"replay. Use a module-level function so the mutation record stays "
            f"replayable."
        )
    return f"{module}:{qualname}"


def resolve_callable(reference: str) -> Callable[..., Any]:
    """Import and return the callable named by ``module:qualname``."""
    module_name, _, qualname = reference.partition(":")
    if not module_name or not qualname:
        raise ArtifactStoreError(f"Malformed function reference {reference!r}")
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ArtifactStoreError(f"Cannot import {module_name!r} for replay") from exc
    for part in qualname.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as exc:
            raise ArtifactStoreError(f"{reference!r} does not resolve to a callable") from exc
    if not callable(obj):
        raise ArtifactStoreError(f"{reference!r} is not callable")
    return cast("Callable[..., Any]", obj)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Compute-once store applying the materialize-vs-record policy.

    Parameters
    ----------
    root
        Directory under which artifact directories are created.
    policy
        Decision rule; defaults to :class:`MaterializationPolicy` defaults.
    repo_path
        Repository used for provenance git capture.
    """

    def __init__(
        self,
        root: str | Path,
        policy: MaterializationPolicy | None = None,
        repo_path: str | Path = ".",
    ) -> None:
        self.root = Path(root)
        self.policy = policy or MaterializationPolicy()
        self.repo_path = Path(repo_path)

    @classmethod
    def from_env(cls, policy: MaterializationPolicy | None = None) -> Self:
        """Build from ``DERIVATIVES_ROOT``, else ``$DATA_ROOT/derivatives``.

        Falls back to ``./artifacts`` so the store is usable with no
        configuration at all.
        """
        root = os.environ.get("DERIVATIVES_ROOT")
        if not root:
            data_root = os.environ.get("DATA_ROOT")
            root = str(Path(data_root) / "derivatives") if data_root else "artifacts"
        return cls(root, policy=policy)

    # -- paths ------------------------------------------------------------

    def artifact_dir(self, key: str) -> Path:
        """Directory holding everything about ``key``."""
        return self.root / key

    def meta_path(self, key: str) -> Path:
        """Path to ``key``'s ``meta.json``."""
        return self.artifact_dir(key) / META_FILENAME

    def exists(self, key: str) -> bool:
        """Whether ``key`` has been computed before, in either representation."""
        return self.meta_path(key).exists()

    # -- reading ----------------------------------------------------------

    def read_meta(self, key: str) -> dict[str, Any] | None:
        """Read ``key``'s metadata, or ``None`` if absent or unparseable."""
        path = self.meta_path(key)
        if not path.exists():
            return None
        try:
            loaded: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def resolve(self, key: str) -> Any:
        """Return ``key``'s data, loading it or replaying its mutation record.

        Raises
        ------
        ArtifactStoreError
            If ``key`` is unknown, or its data is missing and unreplayable.
        """
        meta = self.read_meta(key)
        if meta is None:
            raise ArtifactStoreError(f"Unknown artifact key {key!r} under {self.root}")

        if meta.get("representation") == "materialized":
            ref = ArtifactRef.from_dict(meta["ref"])
            if not ref.exists:
                raise ArtifactStoreError(
                    f"Artifact {key!r} is recorded as materialized but its data "
                    f"file is missing: {ref.path}"
                )
            return get_serializer(ref.serializer)().load(ref.path)

        return self.replay(MutationRecord.from_dict(meta["mutation"]))

    def replay(self, record: MutationRecord) -> Any:
        """Rebuild the artifact described by ``record``.

        Inputs are resolved recursively, so replaying one artifact will replay
        any of its ancestors that were themselves recorded rather than stored.
        """
        fn = resolve_callable(record.function)
        resolved_inputs = [self.resolve(ref.key) for ref in record.inputs]
        return fn(*resolved_inputs, **record.parameters)

    # -- writing ----------------------------------------------------------

    def get_or_compute(
        self,
        key: str,
        fn: Callable[..., Any],
        *,
        inputs: Sequence[str] = (),
        params: Mapping[str, Any] | None = None,
        kind: ArtifactKind = "data",
        serializer: SerializerName | None = None,
        force: bool = False,
        note: str = "",
    ) -> Any:
        """Return ``key``'s data, computing and storing it on first request.

        ``fn`` is invoked as ``fn(*resolved_inputs, **params)``. That signature
        is what makes a mutation record replayable: the same call can be
        reconstructed later from the record alone.

        Parameters
        ----------
        key
            Logical identifier. May contain ``/`` to nest.
        fn
            Module-level callable producing the artifact. Lambdas and closures
            are rejected -- see :func:`qualified_name`.
        inputs
            Keys of upstream artifacts, resolved and passed positionally.
        params
            Extra keyword arguments; must be JSON-serializable to be recorded.
        kind
            Semantic category, consulted by the policy.
        serializer
            Force a serializer instead of inferring one from the result.
        force
            Recompute even when a cached representation exists.
        note
            Free text stored on the mutation record.

        Returns
        -------
        Any
            The computed or cached artifact.
        """
        if self.exists(key) and not force:
            return self.resolve(key)

        function_ref = qualified_name(fn)
        resolved_inputs = [self.resolve(k) for k in inputs]
        input_refs = tuple(self._ref_for(k) for k in inputs)

        started = time.perf_counter()
        result = fn(*resolved_inputs, **(params or {}))
        compute_seconds = time.perf_counter() - started

        serializer_cls = (
            get_serializer(serializer) if serializer is not None else infer_serializer(result)
        )
        size_bytes = serializer_cls.estimate_size(result)

        verdict = self.policy.decide(
            compute_seconds=compute_seconds, size_bytes=size_bytes, kind=kind
        )

        directory = self.artifact_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        git = get_git_state(self.repo_path)

        if verdict.should_materialize:
            self._write_materialized(
                key=key,
                result=result,
                serializer_cls=serializer_cls,
                kind=kind,
                verdict=verdict,
                git=git,
                input_refs=input_refs,
                function_ref=function_ref,
                params=dict(params or {}),
                compute_seconds=compute_seconds,
            )
        else:
            self._write_mutation_record(
                key=key,
                serializer_cls=serializer_cls,
                kind=kind,
                verdict=verdict,
                git=git,
                input_refs=input_refs,
                function_ref=function_ref,
                params=dict(params or {}),
                compute_seconds=compute_seconds,
                size_bytes=size_bytes,
                note=note,
            )

        return result

    # -- internals --------------------------------------------------------

    def _ref_for(self, key: str) -> ArtifactRef:
        """Build a lightweight ref to an existing artifact, for lineage."""
        meta = self.read_meta(key)
        if meta is None:
            raise ArtifactStoreError(f"Unknown input artifact key {key!r}")
        if meta.get("representation") == "materialized":
            return ArtifactRef.from_dict(meta["ref"])
        # A recorded input has no file; carry a path-less ref so the key and
        # serializer still appear in the lineage chain.
        record = MutationRecord.from_dict(meta["mutation"])
        return ArtifactRef(
            key=key,
            path=self.artifact_dir(key),
            serializer=record.output_serializer,
            kind=record.output_kind,
        )

    def _write_materialized(
        self,
        *,
        key: str,
        result: Any,
        serializer_cls: type[Serializer],
        kind: ArtifactKind,
        verdict: PolicyVerdict,
        git: dict[str, Any],
        input_refs: tuple[ArtifactRef, ...],
        function_ref: str,
        params: dict[str, Any],
        compute_seconds: float,
    ) -> None:
        """Serialize the result and write its metadata and provenance."""
        directory = self.artifact_dir(key)
        data_path = directory / f"data{serializer_cls.extension}"
        serializer_cls().save(result, data_path)

        # np.save appends .npy when the path lacks it; re-point at what landed.
        if not data_path.exists():
            candidate = data_path.with_suffix(data_path.suffix + serializer_cls.extension)
            if candidate.exists():
                data_path = candidate

        ref = ArtifactRef.describe(key, data_path, serializer_cls.name, kind)
        self._write_meta(
            key,
            {
                "representation": "materialized",
                "ref": ref.to_dict(),
                "verdict": verdict.to_dict(),
                "function": function_ref,
                "parameters": params,
                "inputs": [i.to_dict() for i in input_refs],
            },
        )
        self._write_provenance(
            key=key,
            git=git,
            input_refs=input_refs,
            output=FileRef.describe(data_path, kind=kind),
            function_ref=function_ref,
            params=params,
            compute_seconds=compute_seconds,
        )

    def _write_mutation_record(
        self,
        *,
        key: str,
        serializer_cls: type[Serializer],
        kind: ArtifactKind,
        verdict: PolicyVerdict,
        git: dict[str, Any],
        input_refs: tuple[ArtifactRef, ...],
        function_ref: str,
        params: dict[str, Any],
        compute_seconds: float,
        size_bytes: int | None,
        note: str,
    ) -> None:
        """Record how to rebuild the result instead of storing it."""
        record = MutationRecord(
            key=key,
            function=function_ref,
            parameters=params,
            inputs=input_refs,
            output_serializer=serializer_cls.name,
            output_kind=kind,
            compute_seconds=compute_seconds,
            estimated_size_bytes=size_bytes,
            git_commit=git.get("commit_hash", ""),
            note=note,
        )
        self._write_meta(
            key,
            {
                "representation": "mutation_record",
                "mutation": record.to_dict(),
                "verdict": verdict.to_dict(),
            },
        )
        self._write_provenance(
            key=key,
            git=git,
            input_refs=input_refs,
            output=FileRef(path=self.meta_path(key), kind=kind),
            function_ref=function_ref,
            params=params,
            compute_seconds=compute_seconds,
        )

    def _write_meta(self, key: str, payload: dict[str, Any]) -> Path:
        """Write ``meta.json`` for ``key``."""
        path = self.meta_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "key": key, **payload}
        path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
        return path

    def _write_provenance(
        self,
        *,
        key: str,
        git: dict[str, Any],
        input_refs: tuple[ArtifactRef, ...],
        output: FileRef,
        function_ref: str,
        params: dict[str, Any],
        compute_seconds: float,
    ) -> Path:
        """Write the provenance sidecar next to the artifact."""
        module_name, _, fn_name = function_ref.partition(":")
        record = ProvenanceRecord(
            mutation_type=fn_name or "unknown",
            version=key,
            run_id=key.replace("/", "_"),
            operator=Operator.from_env(self.repo_path),
            git=GitState.from_dict(git),
            sources=[FileRef(path=r.path, checksum=r.checksum, kind=r.kind) for r in input_refs],
            primary_output=output,
            script_path=module_name,
            steps=[],
        )
        payload = record.to_dict()
        payload["processing_pipeline"] = [
            {
                "step": 1,
                "name": fn_name,
                "script": module_name,
                "parameters": params,
                "output": str(output.path),
                "duration_seconds": round(compute_seconds, 4),
            }
        ]
        directory = self.artifact_dir(key)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provenance.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
