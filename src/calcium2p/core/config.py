"""Session configuration as a frozen dataclass loaded from YAML.

Follows the ``multimodal-toolkit`` idiom: a frozen dataclass plus a module-level
loader, with **no schema library**. Validation is free -- a missing required key
or an unexpected one raises ``TypeError`` from the constructor. ``Literal``
types document the allowed values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from typing import Self

Channel = Literal["green", "red"]


class SessionConfigError(ValueError):
    """Raised when a session YAML cannot be turned into a SessionConfig."""


@dataclass(frozen=True)
class SessionConfig:
    """One imaging session's inputs and acquisition parameters.

    Attributes
    ----------
    subject, study, session
        BIDS-like identifiers; together they locate the session under DATA_ROOT.
    runs
        Explicit run identifiers, or ``"auto"`` to discover them on disk.
    fps
        Imaging frame rate in Hz.
    channels
        Which channels were recorded. The predecessor's two-channel setup was
        thalamic boutons (GCaMP6f, green) and cortical dendrites (jRGECO1a, red).
    baseline_window_s
        ``(start, stop)`` in seconds, used as the F0 window for dF/F0.
    trial_duration_s, stimulus_onset_s, response_window_s
        Behavioural trial structure.
    """

    subject: str
    study: str
    session: str
    runs: list[str] | Literal["auto"] = "auto"
    fps: float = 30.0
    channels: tuple[Channel, ...] = ("green",)
    baseline_window_s: tuple[float, float] = (0.0, 3.0)
    trial_duration_s: float = 12.0
    stimulus_onset_s: float = 3.0
    response_window_s: float = 1.0
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the invariants a type annotation cannot express."""
        if self.fps <= 0:
            raise SessionConfigError(f"fps must be positive, got {self.fps}")
        start, stop = self.baseline_window_s
        if stop <= start:
            raise SessionConfigError(f"baseline_window_s must be increasing, got ({start}, {stop})")
        if not self.channels:
            raise SessionConfigError("at least one channel is required")

    @property
    def run_id(self) -> str:
        """Identifier used in provenance records and artifact keys."""
        return f"{self.subject}_{self.session}"

    @property
    def baseline_samples(self) -> tuple[int, int]:
        """Baseline window converted to sample indices at :attr:`fps`."""
        start, stop = self.baseline_window_s
        return int(start * self.fps), int(stop * self.fps)

    def relative_path(self) -> Path:
        """Session directory relative to DATA_ROOT."""
        return Path(self.study) / self.subject / self.session

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Self:
        """Build from a parsed YAML mapping, coercing sequence fields to tuples."""
        payload = dict(data)
        if "channels" in payload:
            payload["channels"] = tuple(payload["channels"])
        if "baseline_window_s" in payload:
            start, stop = payload["baseline_window_s"]
            payload["baseline_window_s"] = (float(start), float(stop))
        try:
            return cls(**payload)
        except TypeError as exc:
            # The dataclass constructor IS the schema check; surface it usefully.
            raise SessionConfigError(f"invalid session config: {exc}") from exc


def load_session_config(path: str | Path) -> SessionConfig:
    """Read a session YAML into a frozen :class:`SessionConfig`.

    Raises
    ------
    SessionConfigError
        If the file is not a mapping, or does not match the schema.
    """
    import yaml  # noqa: PLC0415

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SessionConfigError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    try:
        return SessionConfig.from_mapping(raw)
    except SessionConfigError as exc:
        raise SessionConfigError(f"{path}: {exc}") from exc
