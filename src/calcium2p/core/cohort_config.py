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
    from collections.abc import Mapping
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

    ``offset`` is subtracted from every value (and so from each reference
    mean) before dividing: ``(x - offset) / (mean(reference) - offset)``. The
    published normalized onset-latency figure used ``offset = 5.0`` -- latency
    measured from reward delivery (t = 5 s) rather than from trial start --
    which no saved notebook state contains; it was recovered from the
    figure's own point values.
    """

    a_reference: tuple[str, str]
    b_reference: tuple[str, str]
    stat: Literal["mean"] = "mean"
    offset: float = 0.0


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

    Passive-group regeneration is capped at roughly a quarter of rows, and the
    cause is now known rather than merely unfitted. The upstream detection
    routine existed in two variants; the one that produced the passive tables
    sliced the wrong axis --

    ``std = np.asarray(df.iloc[:, round(1*fps):round(2*fps)]).std()``

    -- where ``df`` is already transposed, so it takes the deviation over a
    *slice of trials across all frames* rather than over the 1--2 s baseline.
    Because the slice indexes trials, the threshold depends on which trials sat
    in the matrix at processing time: the full multi-protocol stack, of which
    the exported passive traces are only a subset. The threshold is therefore
    unrecoverable in principle from the exported data (it fits to within ~1%,
    an order of magnitude closer than any other candidate rule). Closing it
    bit-exactly needs the passive cohort's own ``_processed_calcium.npy``.
    """

    threshold_sd: float = 2.0
    min_samples: int = 6
    baseline_window_s: tuple[float, float] = (2.0, 3.0)
    reward_window_s: tuple[float, float] = (5.0, 7.0)
    stim_window_s: tuple[float, float] = (3.0, 3.5)
    baseline_window_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        """Build from a parsed mapping, coercing every window into a float pair."""
        return cls(**_coerce_windows(dict(data)))


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
        :class:`EventTableParams` for the groups in
        :attr:`regenerated_groups`, and the reference CSVs for the rest.
    regenerated_groups
        Event-table groups (``passive_axons``, ...) re-detected from the
        traces when ``event_source`` is ``"regenerated"``; must be empty
        otherwise. Groups whose traces are absent cannot be listed.
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
    regenerated_groups: tuple[str, ...] = ()
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
        if (self.event_source == "regenerated") != bool(self.regenerated_groups):
            raise CohortConfigError(
                "regenerated_groups must be non-empty exactly when event_source is "
                f"'regenerated' (got event_source={self.event_source!r}, "
                f"regenerated_groups={list(self.regenerated_groups)})"
            )
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
                payload["event_tables"] = EventTableParams.from_mapping(payload["event_tables"])
            if "histogram" in payload:
                payload["histogram"] = HistogramSpec(**payload["histogram"])
            if "regenerated_groups" in payload:
                payload["regenerated_groups"] = tuple(payload["regenerated_groups"])
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


def expand_figure_manifest(data: dict[str, Any]) -> tuple[CohortConfig, ...]:
    """Expand a figure manifest mapping into one :class:`CohortConfig` per state.

    A manifest records the published figures of a manuscript, grouped by the
    notebook *state* each was saved from (every state is one run: one
    modulation window, one set of switches). ``defaults`` holds the keys every
    state shares (data location, datasets, sampling rates); each entry of
    ``states`` supplies the rest and overrides any default key it repeats.

    Raises
    ------
    CohortConfigError
        If the mapping is malformed, a state is invalid, or run ids collide.
    """
    defaults = data.get("defaults", {})
    states = data.get("states")
    if not isinstance(defaults, dict) or not isinstance(states, list) or not states:
        raise CohortConfigError(
            "figure manifest needs a 'defaults' mapping and a non-empty 'states' list"
        )
    configs = []
    for i, state in enumerate(states):
        if not isinstance(state, dict):
            raise CohortConfigError(f"states[{i}] must be a mapping")
        try:
            configs.append(CohortConfig.from_mapping({**defaults, **state}))
        except CohortConfigError as exc:
            raise CohortConfigError(f"states[{i}] ({state.get('run_id')!r}): {exc}") from exc
    run_ids = [c.run_id for c in configs]
    duplicates = sorted({r for r in run_ids if run_ids.count(r) > 1})
    if duplicates:
        raise CohortConfigError(f"duplicate run_id(s) in figure manifest: {duplicates}")
    return tuple(configs)


#: Keys a derived manifest (one with ``base``) may set.
_DERIVED_MANIFEST_KEYS = frozenset({"base", "defaults", "run_id_suffix"})


def _read_manifest_mapping(path: Path) -> dict[str, Any]:
    """Read a manifest YAML into a mapping, resolving ``base`` inheritance.

    A derived manifest names a ``base`` manifest (relative to its own file)
    and inherits all of its states. Its ``defaults`` override the base's key
    by key, and ``run_id_suffix`` is appended to every inherited run id so the
    two manifests' artifacts never share store keys. A derived manifest cannot
    add or edit states: it is the same set of figures under other defaults.
    """
    import yaml  # noqa: PLC0415

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise CohortConfigError(
            f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}"
        )
    if "base" not in raw:
        return raw
    unknown = sorted(set(raw) - _DERIVED_MANIFEST_KEYS)
    if unknown:
        raise CohortConfigError(
            f"{path}: a derived manifest may only set {sorted(_DERIVED_MANIFEST_KEYS)}; "
            f"got {unknown}"
        )
    suffix = raw.get("run_id_suffix", "")
    if not isinstance(suffix, str) or not suffix:
        raise CohortConfigError(f"{path}: a derived manifest needs a non-empty run_id_suffix")
    base = _read_manifest_mapping(path.parent / str(raw["base"]))
    overrides = raw.get("defaults", {})
    if not isinstance(overrides, dict):
        raise CohortConfigError(f"{path}: 'defaults' must be a mapping")
    states = base.get("states")
    if not isinstance(states, list):
        raise CohortConfigError(f"{path}: base manifest has no 'states' list")
    return {
        "defaults": {**base.get("defaults", {}), **overrides},
        "states": [
            {**state, "run_id": f"{state.get('run_id')}{suffix}"}
            if isinstance(state, dict)
            else state
            for state in states
        ],
    }


def load_figure_manifest(path: str | Path) -> tuple[CohortConfig, ...]:
    """Read a figure-manifest YAML; see :func:`expand_figure_manifest`.

    A manifest may derive from another through ``base`` (see
    :func:`_read_manifest_mapping`).

    Raises
    ------
    CohortConfigError
        If the file is not a mapping, or does not match the schema.
    """
    path = Path(path)
    raw = _read_manifest_mapping(path)
    try:
        return expand_figure_manifest(raw)
    except CohortConfigError as exc:
        raise CohortConfigError(f"{path}: {exc}") from exc


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
