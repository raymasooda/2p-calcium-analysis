"""Cohort-level analysis configuration as frozen dataclasses loaded from YAML.

The onset-histogram manuscript analysis pools eight protocol groups across six
cohort datasets, so :class:`~calcium2p.core.config.SessionConfig` (one imaging
session) does not fit. This module follows the same idiom: frozen dataclasses
are the schema, ``from_mapping`` is the loader, and ``Literal`` types document
the allowed values.

Two fields deserve special note:

``legacy_faithful``
    The original notebooks contain three counting bugs that are baked into the
    published numbers (a TD1-exclusion no-op, a dropped MODIFIED_TD3_CONTROL
    denominator, and a stale-variable leak building passive ROI counts). The
    default ``True`` reproduces them exactly so outputs validate against the
    original artifacts; ``False`` applies the corrected behaviour. The flag is
    threaded into every affected function and lands in artifact keys, so both
    modes cache side by side.

``comparisons``
    Named :class:`ComparisonSpec` entries replace the notebooks' positional
    ``view_comparisons`` column indices, which depended on ``os.listdir``
    order. Each side of a comparison is an explicit ``(protocol, outcome)``
    pair, resolvable regardless of file-discovery order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from typing import Self

Metric = Literal["peak", "ev_onset", "peak_time", "ev_offset", "ev_duration", "integral", "rate"]
TestName = Literal["student_t", "welch_t", "mannwhitney", "wilcoxon", "paired_t"]
RoiSet = Literal["all", "active", "modulated"]
Alignment = Literal["reward", "stimulus"]
EventSource = Literal["reference", "regenerated"]
ThresholdKind = Literal["mean", "median", "half_std"]

#: How the notebooks' ``ttest``/``pairing`` switches map onto :data:`TestName`.
#: The notebooks' "parametric" unpaired test is ``scipy.stats.ttest_ind`` with
#: its default ``equal_var=True`` -- Student's t, not Welch's.
LEGACY_TEST_NAMES: dict[tuple[str, str], TestName] = {
    ("parametric", "unpaired"): "student_t",
    ("non-parametric", "unpaired"): "mannwhitney",
    ("parametric", "paired"): "paired_t",
    ("non-parametric", "paired"): "wilcoxon",
}


class CohortConfigError(ValueError):
    """Raised when a cohort-analysis YAML cannot be turned into a CohortConfig."""


@dataclass(frozen=True)
class EpochSpec:
    """A named analysis window in trial time, with per-protocol overrides.

    Attributes
    ----------
    name
        Referenced by :attr:`ComparisonSpec.epoch` and
        :attr:`ModulationSpec.epoch`.
    default_window
        ``(start, stop)`` seconds in trial time (reward at t=5 s, stimulus at
        t=3 s in the manuscript data).
    per_protocol
        Protocol-specific window overrides, e.g. the notebooks used
        ``[5.05, 5.8]`` for NO_DREADDs within the ``early reward`` epoch.
    """

    name: str
    default_window: tuple[float, float]
    per_protocol: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that every window is increasing."""
        for label, (start, stop) in {"default": self.default_window, **self.per_protocol}.items():
            if stop <= start:
                raise CohortConfigError(
                    f"epoch {self.name!r}: window for {label!r} must be increasing, "
                    f"got ({start}, {stop})"
                )

    def window_for(self, protocol: str) -> tuple[float, float]:
        """Return the window for ``protocol``, falling back to the default."""
        return self.per_protocol.get(protocol, self.default_window)


@dataclass(frozen=True)
class NormalizeSpec:
    """Normalization of both comparison sides by a reference outcome's mean.

    Reproduces the notebooks' "non-consecutive-normalized" figures: side ``a``
    is divided by the mean of ``a_reference`` and side ``b`` by the mean of
    ``b_reference`` (e.g. cRew / mean(nRew) vs cHIT / mean(nHIT)).
    """

    a_reference: tuple[str, str]
    b_reference: tuple[str, str]
    stat: Literal["mean"] = "mean"


@dataclass(frozen=True)
class ComparisonFigure:
    """Visual parameters of one comparison bar figure."""

    figsize: tuple[float, float] = (1.585, 2.0)
    ylim: tuple[float, float] | None = None
    remove_axis: bool = False


@dataclass(frozen=True)
class ComparisonSpec:
    """One statistical comparison between two (protocol, outcome) groups.

    Attributes
    ----------
    name
        Figure title and output filename stem, e.g.
        ``"NO_DREADDs cHIT vs. nHIT"``.
    a, b
        ``(protocol, outcome)`` pairs, e.g. ``("passive_dendrites", "cRew")``.
    metric
        Which per-ROI event metric is compared.
    epoch
        Name of an :class:`EpochSpec` in the parent config.
    test
        Statistical test; see :data:`LEGACY_TEST_NAMES` for the notebook
        mapping.
    roi_set
        Which ROIs enter the comparison: every ROI, only active ROIs (>=1 event
        in the epoch), or only modulated ROIs.
    normalize
        Optional reference normalization; when set, the saved filename uses
        the notebooks' ``non-consecutive-normalized`` convention.
    p_threshold
        Comparisons with p above this are dropped. The notebooks ran with 1.0
        or 1.5, disabling the gate; kept for completeness.
    """

    name: str
    a: tuple[str, str]
    b: tuple[str, str]
    metric: Metric
    epoch: str
    test: TestName = "student_t"
    roi_set: RoiSet = "active"
    normalize: NormalizeSpec | None = None
    p_threshold: float = 1.0
    figure: ComparisonFigure = field(default_factory=ComparisonFigure)

    def __post_init__(self) -> None:
        """Validate group tuples."""
        for side, pair in (("a", self.a), ("b", self.b)):
            if len(pair) != 2:  # noqa: PLR2004 - a pair is definitionally length 2
                raise CohortConfigError(
                    f"comparison {self.name!r}: side {side!r} must be "
                    f"(protocol, outcome), got {pair!r}"
                )


@dataclass(frozen=True)
class ModulationSpec:
    """How modulated ROIs are selected from per-ROI modulation indices."""

    epoch: str = "early reward"
    threshold: ThresholdKind = "median"


@dataclass(frozen=True)
class EventTableParams:
    """Parameters for regenerating event tables from the deltaF traces.

    The code that produced the original event CSVs was never committed. These
    defaults were *recovered* by fitting against the reference tables (see
    ``calcium2p.cohort.event_tables`` for the per-group recipes and
    ``calcium2p.cohort.validation.compare_event_tables`` for the fit):

    - the detection threshold is ``threshold_sd`` x the standard deviation of
      the flattened baseline window over the ROI's *validated-outcome trial
      stack* (cHIT/nHIT/CATCH*/FA branches for the task cohorts; every
      NAIVE-STIM-REW trial for the passive cohorts), with upper == lower
      (degenerate hysteresis) and ``min_samples`` samples above threshold;
    - the baseline window is 2--3 s for the task cohorts (pre-stimulus) and
      0--2 s for Batch1 passive, both at ``round(t * fps)`` positions with no
      offset; per-group overrides live in ``baseline_window_overrides``;
    - event onsets are kept strictly inside the family window (both bounds
      exclusive), 5--7 s for the 2 s reward family and 3--3.5 s for the 500 ms
      stimulus family.

    The Batch3 passive threshold basis could not be pinned to a single rule
    (its fitted per-session thresholds match no baseline of the stored
    traces); its regeneration is therefore partial and reported as such.
    """

    threshold_sd: float = 2.0
    min_samples: int = 6
    baseline_window_s: tuple[float, float] = (2.0, 3.0)
    reward_window_s: tuple[float, float] = (5.0, 7.0)
    stim_window_s: tuple[float, float] = (3.0, 3.5)
    baseline_window_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class HistogramSpec:
    """Parameters of the 2x5 onset-latency histogram grid."""

    xmin: float = 5.0
    xmax: float = 6.5
    nbins: int = 15
    roi_div: bool = True
    sharey: bool | Literal["row"] = False
    active_only: bool = True


@dataclass(frozen=True)
class CohortConfig:
    """One cohort-level onset-histogram analysis run.

    Attributes
    ----------
    run_id
        Prefix of every artifact key this run writes.
    alignment
        Whether events are aligned to reward (t=5 s) or stimulus (t=3 s).
    legacy_faithful
        Reproduce the three notebook counting bugs (default) or correct them.
        See the module docstring.
    event_source
        ``"reference"`` analyses the original event CSVs; ``"regenerated"``
        analyses tables rebuilt from the deltaF traces via
        :class:`EventTableParams`.
    data_subdir
        Location of the legacy inputs below ``DATA_ROOT``.
    datasets
        Logical dataset name -> filename under :attr:`data_subdir`.
    fps_by_cohort
        Sampling rates per cohort. The notebooks hardcoded both 30.3 and
        30.54 Hz in different cells; this makes the inconsistency explicit
        and auditable.
    """

    run_id: str
    alignment: Alignment
    legacy_faithful: bool = True
    event_source: EventSource = "reference"
    data_subdir: str = "manuscript/V10"
    datasets: dict[str, str] = field(default_factory=dict)
    fps_by_cohort: dict[str, float] = field(default_factory=dict)
    epochs: tuple[EpochSpec, ...] = ()
    modulation: ModulationSpec = field(default_factory=ModulationSpec)
    event_tables: EventTableParams = field(default_factory=EventTableParams)
    comparisons: tuple[ComparisonSpec, ...] = ()
    histogram: HistogramSpec = field(default_factory=HistogramSpec)
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate cross-references a type annotation cannot express."""
        if not self.run_id:
            raise CohortConfigError("run_id must be non-empty")
        names = [epoch.name for epoch in self.epochs]
        if len(names) != len(set(names)):
            raise CohortConfigError(f"duplicate epoch names in {names}")
        known = set(names)
        for comparison in self.comparisons:
            if comparison.epoch not in known:
                raise CohortConfigError(
                    f"comparison {comparison.name!r} references unknown epoch "
                    f"{comparison.epoch!r}; defined epochs: {sorted(known)}"
                )
        if self.epochs and self.modulation.epoch not in known:
            raise CohortConfigError(
                f"modulation references unknown epoch {self.modulation.epoch!r}; "
                f"defined epochs: {sorted(known)}"
            )

    def epoch(self, name: str) -> EpochSpec:
        """Return the epoch spec called ``name``.

        Raises
        ------
        KeyError
            If no epoch with that name is configured.
        """
        for spec in self.epochs:
            if spec.name == name:
                return spec
        raise KeyError(f"no epoch named {name!r}")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Self:
        """Build from a parsed YAML mapping, constructing nested dataclasses."""
        payload = dict(data)
        try:
            if "epochs" in payload:
                payload["epochs"] = tuple(_epoch_from_mapping(e) for e in payload["epochs"])
            if "comparisons" in payload:
                payload["comparisons"] = tuple(
                    _comparison_from_mapping(c) for c in payload["comparisons"]
                )
            if "modulation" in payload:
                payload["modulation"] = ModulationSpec(**payload["modulation"])
            if "event_tables" in payload:
                payload["event_tables"] = EventTableParams(
                    **_coerce_windows(payload["event_tables"])
                )
            if "histogram" in payload:
                payload["histogram"] = HistogramSpec(**payload["histogram"])
            if "fps_by_cohort" in payload:
                payload["fps_by_cohort"] = {
                    k: float(v) for k, v in payload["fps_by_cohort"].items()
                }
            return cls(**payload)
        except TypeError as exc:
            # The dataclass constructor IS the schema check; surface it usefully.
            raise CohortConfigError(f"invalid cohort config: {exc}") from exc


def _pair(value: Any, context: str) -> tuple[float, float]:
    """Coerce a two-element sequence into a float pair."""
    try:
        start, stop = value
    except (TypeError, ValueError) as exc:
        raise CohortConfigError(f"{context}: expected a two-element window, got {value!r}") from exc
    return (float(start), float(stop))


def _coerce_windows(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce every ``*window*`` field of a mapping into float pairs.

    Dict-valued fields (per-group override maps) are coerced element-wise.
    """
    out = dict(data)
    for key, value in out.items():
        if "window" not in key:
            continue
        if isinstance(value, dict):
            out[key] = {group: _pair(window, f"{key}[{group}]") for group, window in value.items()}
        else:
            out[key] = _pair(value, key)
    return out


def _epoch_from_mapping(data: dict[str, Any]) -> EpochSpec:
    """Build an :class:`EpochSpec` from a YAML mapping."""
    payload = dict(data)
    payload["default_window"] = _pair(payload.get("default_window"), "default_window")
    if "per_protocol" in payload:
        payload["per_protocol"] = {
            protocol: _pair(window, f"per_protocol[{protocol}]")
            for protocol, window in payload["per_protocol"].items()
        }
    return EpochSpec(**payload)


def _comparison_from_mapping(data: dict[str, Any]) -> ComparisonSpec:
    """Build a :class:`ComparisonSpec` from a YAML mapping."""
    payload = dict(data)
    for side in ("a", "b"):
        if side in payload:
            payload[side] = tuple(payload[side])
    if payload.get("normalize") is not None:
        norm = dict(payload["normalize"])
        for side in ("a_reference", "b_reference"):
            if side in norm:
                norm[side] = tuple(norm[side])
        payload["normalize"] = NormalizeSpec(**norm)
    if "figure" in payload:
        fig = dict(payload["figure"])
        if "figsize" in fig:
            width, height = fig["figsize"]
            fig["figsize"] = (float(width), float(height))
        if fig.get("ylim") is not None:
            fig["ylim"] = _pair(fig["ylim"], f"{payload.get('name')}: ylim")
        payload["figure"] = ComparisonFigure(**fig)
    return ComparisonSpec(**payload)


def load_cohort_config(path: str | Path) -> CohortConfig:
    """Read a cohort-analysis YAML into a frozen :class:`CohortConfig`.

    Raises
    ------
    CohortConfigError
        If the file is not a mapping, or does not match the schema.
    """
    import yaml  # noqa: PLC0415

    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CohortConfigError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    try:
        return CohortConfig.from_mapping(raw)
    except CohortConfigError as exc:
        raise CohortConfigError(f"{path}: {exc}") from exc
