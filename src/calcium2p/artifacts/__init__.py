"""Intermediate artifact storage under an explicit materialize-vs-record policy.

Standing rule: artifacts that are **time-consuming to produce are always
stored**; artifacts that are **storage-heavy but cheap to reproduce** are stored
as a :class:`~calcium2p.artifacts.refs.MutationRecord` -- which function was
applied to which inputs with which parameters -- and rebuilt on demand.

:class:`~calcium2p.artifacts.store.ArtifactStore` applies that rule so callers
never have to.
"""

from __future__ import annotations

from calcium2p.artifacts.policy import MaterializationPolicy, PolicyVerdict, Reason
from calcium2p.artifacts.refs import (
    ArtifactRef,
    MutationRecord,
    Representation,
    SerializerName,
)
from calcium2p.artifacts.store import (
    ARTIFACT_SCHEMA_VERSION,
    SERIALIZERS,
    ArtifactStore,
    ArtifactStoreError,
    JsonSerializer,
    NpzSerializer,
    NumpySerializer,
    ParquetSerializer,
    Serializer,
    get_serializer,
    infer_serializer,
    qualified_name,
    resolve_callable,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SERIALIZERS",
    "ArtifactRef",
    "ArtifactStore",
    "ArtifactStoreError",
    "JsonSerializer",
    "MaterializationPolicy",
    "MutationRecord",
    "NpzSerializer",
    "NumpySerializer",
    "ParquetSerializer",
    "PolicyVerdict",
    "Reason",
    "Representation",
    "Serializer",
    "SerializerName",
    "get_serializer",
    "infer_serializer",
    "qualified_name",
    "resolve_callable",
]
