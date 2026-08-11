"""Pipeline scaffolding: the canonical place where OOP earns its keep here.

A calcium analysis run is a sequence of stages carrying state between them
(config, sampling rate, baseline estimates) and sharing behaviour (artifact
keys, provenance, skip logic). That is state plus polymorphism, so it is a class
hierarchy. Pure numerical transforms called *by* these stages stay plain
functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from calcium2p.artifacts import ArtifactStore
    from calcium2p.core.config import SessionConfig


class PipelineStage(ABC):
    """One step of an analysis pipeline.

    Subclasses implement :meth:`compute` as a pure function of its inputs, and
    inherit caching, provenance, and artifact-policy handling from :meth:`run`.

    Parameters
    ----------
    config
        The session being processed.
    store
        Artifact store used for caching. When ``None``, :meth:`run` computes
        without caching -- useful in tests.
    """

    #: Short identifier used in artifact keys and as ``mutation_type`` in
    #: provenance records. Must be set by every subclass.
    name: str = ""

    #: Keys of the stages this one consumes, resolved from the store and passed
    #: to :meth:`compute` positionally, in order.
    inputs: tuple[str, ...] = ()

    def __init__(self, config: SessionConfig, store: ArtifactStore | None = None) -> None:
        if not self.name:
            raise TypeError(f"{type(self).__name__} must set a class-level `name`")
        self.config = config
        self.store = store

    @abstractmethod
    def compute(self, *inputs: Any) -> Any:
        """Produce this stage's output.

        Must be a pure function of ``inputs`` and :attr:`params` -- no reading
        from or writing to disk. That is what lets the artifact store replay it
        from a mutation record.
        """

    @property
    def params(self) -> dict[str, Any]:
        """Parameters defining this stage's output.

        They land in the artifact key, the mutation record, and the provenance
        record, so anything that changes the result belongs here.
        """
        return {}

    @property
    def version(self) -> str:
        """Self-describing parameter string, e.g. ``v1_fps-30_thresh-2.5``.

        Used as the artifact key suffix and the provenance ``version``, so the
        directory name alone says what produced it.
        """
        parts = "_".join(f"{k}-{v}" for k, v in sorted(self.params.items()))
        return f"v1_{parts}" if parts else "v1"

    @property
    def artifact_key(self) -> str:
        """Store key for this stage's output."""
        return f"{self.config.run_id}/{self.name}/{self.version}"

    def run(self) -> Any:
        """Execute the stage, caching through the artifact store when available.

        Returns
        -------
        Any
            This stage's output, computed or loaded.
        """
        if self.store is None:
            return self.compute(*self._resolve_inputs_uncached())
        return self.store.get_or_compute(
            self.artifact_key,
            type(self).compute_unbound,
            inputs=list(self.inputs),
            params={"stage": type(self).__qualname__, **self.params},
        )

    def _resolve_inputs_uncached(self) -> list[Any]:
        """Resolve inputs without a store. Only valid when there are none."""
        if self.inputs:
            raise RuntimeError(
                f"{type(self).__name__} declares inputs {self.inputs} but has no "
                f"ArtifactStore to resolve them from."
            )
        return []

    @staticmethod
    def compute_unbound(*args: Any, **kwargs: Any) -> Any:
        """Module-level entry point the artifact store can record and replay.

        The store rejects lambdas and closures because a mutation record naming
        one could never be followed; this static method gives every stage an
        importable reference instead.
        """
        raise NotImplementedError(
            "Concrete stages that use ArtifactStore caching must override "
            "compute_unbound with a module-level implementation."
        )


class Pipeline:
    """An ordered sequence of :class:`PipelineStage` objects.

    Parameters
    ----------
    stages
        Executed in order. Each stage's output is available to later stages via
        the artifact store, keyed by :attr:`PipelineStage.artifact_key`.
    """

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self.stages = list(stages)

    def run(self) -> dict[str, Any]:
        """Run every stage, returning ``{stage name: output}``."""
        results: dict[str, Any] = {}
        for stage in self.stages:
            results[stage.name] = stage.run()
        return results

    def __len__(self) -> int:
        """Return the number of stages."""
        return len(self.stages)

    def __repr__(self) -> str:
        """Readable summary naming the stages in order."""
        names = " -> ".join(s.name for s in self.stages)
        return f"Pipeline({names})"
