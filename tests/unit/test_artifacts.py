"""Tests for the materialize-vs-mutation-record artifact policy and store."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from calcium2p.artifacts import (
    ArtifactStore,
    ArtifactStoreError,
    MaterializationPolicy,
    MutationRecord,
    infer_serializer,
    qualified_name,
    resolve_callable,
)
from calcium2p.artifacts.store import (
    JsonSerializer,
    NpzSerializer,
    NumpySerializer,
    ParquetSerializer,
)

_GIB = 1024**3


# Module-level so the store can record a replayable reference to them.
def double(array: np.ndarray) -> np.ndarray:
    """Double every element."""
    return array * 2


def make_array(n: int = 16) -> np.ndarray:
    """Produce a deterministic array."""
    return np.arange(n, dtype=np.float64)


def add_offset(array: np.ndarray, offset: float = 1.0) -> np.ndarray:
    """Add a constant offset."""
    return array + offset


class TestPolicy:
    def test_expensive_is_materialized_even_when_huge(self) -> None:
        # Compute is checked before size: repeating slow work costs more than disk.
        verdict = MaterializationPolicy().decide(
            compute_seconds=300.0, size_bytes=10 * _GIB, kind="data"
        )
        assert verdict.representation == "materialized"
        assert verdict.reason == "expensive_to_recompute"

    def test_huge_but_cheap_is_recorded(self) -> None:
        verdict = MaterializationPolicy().decide(
            compute_seconds=0.5, size_bytes=4 * _GIB, kind="data"
        )
        assert verdict.representation == "mutation_record"
        assert verdict.reason == "storage_heavy_cheap_compute"

    def test_small_and_cheap_is_materialized(self) -> None:
        verdict = MaterializationPolicy().decide(compute_seconds=0.1, size_bytes=1024, kind="data")
        assert verdict.representation == "materialized"
        assert verdict.reason == "cheap_and_small"

    def test_unmeasured_compute_is_assumed_expensive(self) -> None:
        # Failure mode should be a wasted gigabyte, not a lost afternoon.
        verdict = MaterializationPolicy().decide(
            compute_seconds=None, size_bytes=8 * _GIB, kind="data"
        )
        assert verdict.representation == "materialized"

    @pytest.mark.parametrize("kind", ["figure", "metric", "report"])
    def test_small_kinds_always_materialized(self, kind: str) -> None:
        verdict = MaterializationPolicy().decide(
            compute_seconds=0.01, size_bytes=100 * _GIB, kind=kind
        )
        assert verdict.representation == "materialized"
        assert verdict.reason == "kind_always_materialized"

    def test_from_mapping_reads_gib(self) -> None:
        policy = MaterializationPolicy.from_mapping(
            {"compute_seconds_threshold": 5, "size_gib_threshold": 0.5}
        )
        assert policy.compute_seconds_threshold == 5.0
        assert policy.size_bytes_threshold == _GIB // 2

    def test_verdict_explains_itself(self) -> None:
        verdict = MaterializationPolicy().decide(
            compute_seconds=0.5, size_bytes=4 * _GIB, kind="data"
        )
        assert "mutation_record" in verdict.explain()
        assert "4.00GiB" in verdict.explain()


class TestSerializers:
    def test_infers_by_type(self, frame: pd.DataFrame) -> None:
        assert infer_serializer(np.zeros(3)) is NumpySerializer
        assert infer_serializer({"a": np.zeros(3)}) is NpzSerializer
        assert infer_serializer(frame) is ParquetSerializer
        assert infer_serializer({"a": 1}) is JsonSerializer

    def test_parquet_preserves_dtypes(self, frame: pd.DataFrame, tmp_path: Path) -> None:
        # CSV silently loses these; that defect class is why parquet is used.
        target = tmp_path / "f.parquet"
        ParquetSerializer().save(frame, target)
        restored = ParquetSerializer().load(target)
        pd.testing.assert_frame_equal(frame, restored)

    def test_npy_round_trip(self, traces: np.ndarray, tmp_path: Path) -> None:
        target = tmp_path / "a.npy"
        NumpySerializer().save(traces, target)
        np.testing.assert_array_equal(traces, NumpySerializer().load(target))

    def test_size_is_estimated_without_writing(self, traces: np.ndarray) -> None:
        assert NumpySerializer.estimate_size(traces) == traces.nbytes

    def test_unserializable_object_is_rejected(self) -> None:
        with pytest.raises(ArtifactStoreError, match="No serializer"):
            infer_serializer(object())


class TestCallableReferences:
    def test_round_trips(self) -> None:
        assert resolve_callable(qualified_name(double)) is double

    def test_lambda_is_rejected(self) -> None:
        # A record naming a lambda could never be replayed.
        with pytest.raises(ArtifactStoreError, match="lambda or closure"):
            qualified_name(lambda x: x)

    def test_closure_is_rejected(self) -> None:
        def inner() -> None: ...

        with pytest.raises(ArtifactStoreError, match="lambda or closure"):
            qualified_name(inner)

    def test_malformed_reference(self) -> None:
        with pytest.raises(ArtifactStoreError, match="Malformed"):
            resolve_callable("no_colon_here")


class TestStore:
    def test_materializes_and_caches(self, store_root: Path) -> None:
        store = ArtifactStore(store_root)
        first = store.get_or_compute("base", make_array, params={"n": 8})
        assert (store_root / "base" / "meta.json").exists()
        assert (store_root / "base" / "provenance.json").exists()

        meta = store.read_meta("base")
        assert meta is not None
        assert meta["representation"] == "materialized"

        # Second call must load, not recompute.
        np.testing.assert_array_equal(first, store.get_or_compute("base", make_array))

    def test_records_mutation_when_huge_and_cheap(self, store_root: Path) -> None:
        policy = MaterializationPolicy(compute_seconds_threshold=1e9, size_bytes_threshold=1)
        store = ArtifactStore(store_root, policy=policy)
        store.get_or_compute("base", make_array, params={"n": 8})

        meta = store.read_meta("base")
        assert meta is not None
        assert meta["representation"] == "mutation_record"
        # No data file: that is the point.
        assert not list((store_root / "base").glob("data.*"))
        assert meta["mutation"]["function"].endswith(":make_array")
        assert "replay_command" in meta["mutation"]

    def test_replays_a_recorded_artifact(self, store_root: Path) -> None:
        policy = MaterializationPolicy(compute_seconds_threshold=1e9, size_bytes_threshold=1)
        store = ArtifactStore(store_root, policy=policy)
        expected = store.get_or_compute("base", make_array, params={"n": 8})
        np.testing.assert_array_equal(expected, store.resolve("base"))

    def test_replay_chains_through_recorded_inputs(self, store_root: Path) -> None:
        # Resolving a recorded artifact must recursively rebuild its ancestors.
        policy = MaterializationPolicy(compute_seconds_threshold=1e9, size_bytes_threshold=1)
        store = ArtifactStore(store_root, policy=policy)
        store.get_or_compute("base", make_array, params={"n": 8})
        store.get_or_compute("derived", add_offset, inputs=["base"], params={"offset": 5.0})

        np.testing.assert_array_equal(
            store.resolve("derived"), np.arange(8, dtype=np.float64) + 5.0
        )

    def test_lineage_records_input_checksums(self, store_root: Path) -> None:
        store = ArtifactStore(store_root)
        store.get_or_compute("base", make_array, params={"n": 8})
        store.get_or_compute("derived", double, inputs=["base"])

        meta = store.read_meta("derived")
        assert meta is not None
        assert meta["inputs"][0]["key"] == "base"
        assert meta["inputs"][0]["checksum"] is not None

    def test_force_recomputes(self, store_root: Path) -> None:
        store = ArtifactStore(store_root)
        store.get_or_compute("base", make_array, params={"n": 4})
        forced = store.get_or_compute("base", make_array, params={"n": 9}, force=True)
        assert forced.size == 9

    def test_unknown_key_raises(self, store_root: Path) -> None:
        with pytest.raises(ArtifactStoreError, match="Unknown artifact key"):
            ArtifactStore(store_root).resolve("nope")

    def test_missing_data_file_is_detected(self, store_root: Path) -> None:
        store = ArtifactStore(store_root)
        store.get_or_compute("base", make_array, params={"n": 4})
        for stray in (store_root / "base").glob("data.*"):
            stray.unlink()
        with pytest.raises(ArtifactStoreError, match="data file is missing"):
            store.resolve("base")

    def test_dataframe_round_trips_through_store(self, store_root: Path) -> None:
        store = ArtifactStore(store_root)
        store.get_or_compute("f", make_frame, kind="data")
        pd.testing.assert_frame_equal(store.resolve("f"), make_frame())


def make_frame() -> pd.DataFrame:
    """Module-level frame factory, so the store can reference it."""
    return pd.DataFrame({"roi": [0, 1], "amp": [1.5, 2.5]})


class TestMutationRecordSerialization:
    def test_round_trips(self) -> None:
        record = MutationRecord(
            key="k",
            function="mod:fn",
            parameters={"a": 1},
            compute_seconds=0.5,
            estimated_size_bytes=1024,
        )
        restored = MutationRecord.from_dict(record.to_dict())
        assert restored.key == record.key
        assert restored.function == record.function
        assert restored.parameters == record.parameters
        assert restored.compute_seconds == record.compute_seconds
