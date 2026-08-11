"""The rule deciding whether an artifact is stored or merely recorded.

The repo's standing policy, in one sentence: *intermediate artifacts that are
time-consuming to produce are always stored; artifacts that are storage-heavy
but cheap to reproduce are stored as a mutation record instead.*

:class:`MaterializationPolicy` turns that sentence into a function of two
measured quantities -- wall-clock compute time and output size -- so the
decision is reproducible and auditable rather than a judgement call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Self

from calcium2p.artifacts.refs import Representation
from calcium2p.provenance.records import ArtifactKind

#: Human-readable justification attached to every decision.
Reason = Literal[
    "expensive_to_recompute",
    "cheap_and_small",
    "storage_heavy_cheap_compute",
    "kind_always_materialized",
    "kind_never_materialized",
]

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class PolicyVerdict:
    """The outcome of a policy evaluation.

    Carries the reason as well as the decision so it can be logged into the
    mutation record and later explained without re-deriving it.
    """

    representation: Representation
    reason: Reason
    compute_seconds: float | None
    size_bytes: int | None

    @property
    def should_materialize(self) -> bool:
        """Whether the artifact should be written to disk."""
        return self.representation == "materialized"

    def explain(self) -> str:
        """One-line explanation suitable for a log record."""
        secs = "unknown" if self.compute_seconds is None else f"{self.compute_seconds:.1f}s"
        size = "unknown" if self.size_bytes is None else f"{self.size_bytes / _GIB:.2f}GiB"
        return f"{self.representation} ({self.reason}; compute={secs}, size={size})"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "representation": self.representation,
            "reason": self.reason,
            "compute_seconds": self.compute_seconds,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class MaterializationPolicy:
    """Decide how to represent a computed artifact.

    Attributes
    ----------
    compute_seconds_threshold
        At or above this, the artifact is always materialized -- recomputing it
        costs more than the disk it occupies.
    size_bytes_threshold
        At or above this, an artifact is a candidate for a mutation record.
    always_materialize_kinds
        Kinds that are stored regardless of the numbers. Figures and metrics are
        small, and are the things a human actually opens.
    never_materialize_kinds
        Kinds that are always recorded rather than stored, whatever their size.
        Empty by default.

    Notes
    -----
    The two thresholds are deliberately asymmetric. Compute time is checked
    *first*, so an artifact that is both slow and huge is stored: the policy
    optimizes for never repeating expensive work, and treats disk as the cheaper
    resource. Only in the cheap-to-recompute corner does size win.
    """

    compute_seconds_threshold: float = 60.0
    size_bytes_threshold: int = 2 * _GIB
    always_materialize_kinds: frozenset[ArtifactKind] = field(
        default_factory=lambda: frozenset({"figure", "metric", "report"})
    )
    never_materialize_kinds: frozenset[ArtifactKind] = field(default_factory=frozenset)

    def decide(
        self,
        *,
        compute_seconds: float | None,
        size_bytes: int | None,
        kind: ArtifactKind = "data",
    ) -> PolicyVerdict:
        """Evaluate the policy for one artifact.

        Parameters
        ----------
        compute_seconds
            Measured wall-clock time to produce it. ``None`` is treated as
            expensive -- an unmeasured computation is assumed costly, so the
            failure mode is a wasted gigabyte rather than a lost afternoon.
        size_bytes
            Measured serialized size. ``None`` is treated as small.
        kind
            Semantic category, checked against the kind overrides first.

        Returns
        -------
        PolicyVerdict
            The decision and the reason for it.
        """
        if kind in self.never_materialize_kinds:
            return PolicyVerdict(
                "mutation_record", "kind_never_materialized", compute_seconds, size_bytes
            )
        if kind in self.always_materialize_kinds:
            return PolicyVerdict(
                "materialized", "kind_always_materialized", compute_seconds, size_bytes
            )

        # Unmeasured compute is assumed expensive; see the docstring.
        if compute_seconds is None or compute_seconds >= self.compute_seconds_threshold:
            return PolicyVerdict(
                "materialized", "expensive_to_recompute", compute_seconds, size_bytes
            )

        if size_bytes is not None and size_bytes >= self.size_bytes_threshold:
            return PolicyVerdict(
                "mutation_record",
                "storage_heavy_cheap_compute",
                compute_seconds,
                size_bytes,
            )

        return PolicyVerdict("materialized", "cheap_and_small", compute_seconds, size_bytes)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Self:
        """Build from a config mapping (e.g. a parsed YAML block).

        Unknown keys are ignored so a config file can carry commentary fields.
        """
        kwargs: dict[str, Any] = {}
        if "compute_seconds_threshold" in data:
            kwargs["compute_seconds_threshold"] = float(data["compute_seconds_threshold"])
        if "size_gib_threshold" in data:
            kwargs["size_bytes_threshold"] = int(float(data["size_gib_threshold"]) * _GIB)
        elif "size_bytes_threshold" in data:
            kwargs["size_bytes_threshold"] = int(data["size_bytes_threshold"])
        if "always_materialize_kinds" in data:
            kwargs["always_materialize_kinds"] = frozenset(data["always_materialize_kinds"])
        if "never_materialize_kinds" in data:
            kwargs["never_materialize_kinds"] = frozenset(data["never_materialize_kinds"])
        return cls(**kwargs)
