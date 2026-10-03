"""Protocol grouping, trial counts, ROI denominators, and active-ROI sets.

This module hosts every ``legacy_faithful`` branch point. The original
notebooks contain four counting bugs that are baked into the published
numbers; each is reproduced verbatim when ``legacy_faithful=True`` (the
default) and corrected otherwise. The bug sites, each marked ``BUG n`` below:

1. **TD1-exclusion no-op** (reward/stimulus cell 0 vs cell 4): the events
   loader tests ``title not in exclude`` where ``title`` is a full file stem,
   so ``exclude=['TD1']`` never matches and TD1 events are pooled into
   ``task_dendrites (pre-CNO)`` -- while the active-ROI builder keeps TD1
   under its own unmerged key, and the ROI denominators skip TD1 entirely.
2. **MODIFIED_TD3_CONTROL denominator skip** (cell 5): the CONTROL branch
   parses the TD token as ``stem.split('_')[0]``, which is ``"MODIFIED"`` for
   the modified TD3 file, silently dropping 203 ROIs from the control
   post-CNO denominator (91/186 = 48.9% instead of 91/389 = 23.4%).
3. **Stale FOV_dict leak** (cell 5): the Batch1 (passive) ROI lists are built
   from the loop variable left over from the preceding Batch3 section, so
   every Batch1 mouse contributes the ROI names of the *last Batch3 FOV*
   instead of its own.
4. **Dendrite no-stim trial counts** (stimulus cell 4): ``nostim_t`` is
   computed with ``stim_t_list``, so for dendrite protocols the no-stimulus
   trial total equals the stimulus total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from calcium2p.core.cohort_config import Alignment

#: Trial-type keys the notebooks skip when tabulating trial counts.
_SKIPPED_TTYPES = ("resp",)

#: TD tokens belonging to the pre-CNO group; everything else (TD2/TD3) is post.
_PRE_CNO_TDS = ("TD0", "TD1")


@dataclass(frozen=True)
class TrialCounts:
    """Per-protocol trial-count lookups for one alignment.

    Attributes
    ----------
    all_trials
        ``protocol -> id -> ttype -> n`` for flat protocols, and
        ``protocol -> TD -> id -> ttype -> n`` for the CNO (dendrite)
        protocols -- mirroring the notebooks' ``all_trials`` dict shape.
    target_totals, reference_totals
        Pooled totals per protocol: rewarded/unrewarded for the reward
        alignment, stimulus/no-stimulus for the stimulus alignment.
    alignment
        Which alignment these counts were built for.
    """

    all_trials: dict[str, dict[str, Any]]
    target_totals: dict[str, int]
    reference_totals: dict[str, int]
    alignment: Alignment = "reward"

    def protocol_trials(self, protocol: str, roi_key: Hashable) -> dict[str, int]:
        """Per-ttype counts for the mouse/session owning ``roi_key``.

        Ports the notebooks' shared lookup: derive the ID from the ROI key
        (tuples take their first element; strings drop the trailing
        ``_<roi>`` segment when it carries the task-axon ``.0`` suffix), and
        for CNO protocols pick TD3 when the ROI name contains ``TD3``, else
        TD2 (single-TD protocols use their only TD).
        """
        if isinstance(roi_key, tuple):
            identifier = str(roi_key[0])
            roi_name = str(roi_key[1]) if len(roi_key) > 1 else ""
        else:
            identifier = str(roi_key)
            roi_name = str(roi_key)
            if ".0" in identifier:
                identifier = "_".join(identifier.split("_")[:-1])
        if "CNO" in protocol:
            td_level = self.all_trials[protocol]
            if len(td_level) == 1:
                td = next(iter(td_level))
            else:
                td = "TD3" if "TD3" in roi_name else "TD2"
            per_id: dict[str, dict[str, int]] = td_level[td]
        else:
            per_id = self.all_trials[protocol]
        return per_id[identifier]

    def rate_denominator(self, protocol: str, roi_key: Hashable, outcome: str) -> int:
        """Trial count dividing an ROI's event count into a rate.

        Ports ``extract_active_ROIs``: pooled outcomes (``HIT``/``Rew``/
        ``Stim``/``CATCH``) sum every matching ttype when no exact key
        exists; everything else is a direct lookup. (The notebook's CATCH
        branch had a dormant NameError -- ``catch_type`` for ``catch_ttype``
        -- that never fired because no CATCH comparison ran; implemented
        correctly here.)
        """
        counts = self.protocol_trials(protocol, roi_key)
        pooled = {"HIT": "HIT", "Rew": "Rew", "Stim": "Stim", "CATCH": "CATCH"}
        if outcome in pooled and outcome not in counts:
            return sum(n for ttype, n in counts.items() if pooled[outcome] in ttype)
        if outcome == "Rew" and "Rew" not in counts:  # pragma: no cover - covered above
            return sum(n for ttype, n in counts.items() if "Rew" in ttype)
        return counts[outcome]

    def modulation_denominators(self, protocol: str, roi_key: Hashable) -> tuple[int, int]:
        """``(target, reference)`` trial counts for the modulation index.

        Ports the reward cell-24 / stimulus cell-13 lookups. Note the ID
        derivation differs from :meth:`protocol_trials` for string keys (the
        notebooks always drop the last ``_``-segment here); tuple keys use
        their first element as-is.
        """
        if isinstance(roi_key, tuple):
            identifier = str(roi_key[0])
            roi_name = str(roi_key[1]) if len(roi_key) > 1 else ""
        else:
            identifier = "_".join(str(roi_key).split("_")[:-1])
            roi_name = str(roi_key)

        if self.alignment == "reward":
            markers = ["Rew"] if "passive" in protocol else ["HIT"]
        else:
            markers = ["Stim"] if "passive" in protocol else ["HIT", "miss"]

        if "CNO" in protocol:
            td_level = self.all_trials[protocol]
            td = (
                next(iter(td_level))
                if len(td_level) == 1
                else ("TD3" if "TD3" in roi_name else "TD2")
            )
            counts = td_level[td][identifier]
            target_keys = [k for k in counts for m in markers if m in k]
            target = sum(counts[k] for k in target_keys)
            reference = sum(n for k, n in counts.items() if k not in target_keys)
            return target, reference

        counts = self.all_trials[protocol][identifier]
        if "passive" in protocol:
            target = sum(n for k, n in counts.items() if "Rew" in k)
            other = sum(n for k, n in counts.items() if "Rew" not in k)
            if self.alignment == "stimulus":
                target = sum(n for k, n in counts.items() if "Stim" in k)
                other = sum(n for k, n in counts.items() if "Stim" not in k)
            return target, other
        target_keys = [k for k in counts for m in markers if m in k]
        if self.alignment == "reward":
            target = counts["HIT"]
        else:
            target = sum(counts[k] for k in target_keys)
        return target, counts["ALL"] - target


def _cno_label(prefix: str, td: str) -> str:
    """Map a TD token onto the pre/post-CNO protocol label."""
    suffix = " (pre-CNO)" if td in _PRE_CNO_TDS else " (post-CNO)"
    return prefix + suffix


def _tabulate_flat(trials: pd.DataFrame) -> dict[str, dict[str, int]]:
    """``id -> ttype -> n`` from a trials_* table, skipping resp ttypes."""
    out: dict[str, dict[str, int]] = {}
    identifiers = trials["id"].astype(str).tolist()
    ttypes = trials["ttype"].astype(str).tolist()
    numbers = trials["n_trials"].astype(int).tolist()
    for identifier, ttype, n_trials in zip(identifiers, ttypes, numbers, strict=True):
        if ttype == "None" or any(skip in ttype for skip in _SKIPPED_TTYPES):
            continue
        out.setdefault(identifier, {})[ttype] = int(n_trials)
    return out


def build_trial_counts(  # noqa: PLR0912 - a faithful port of branchy notebook counting
    tables: Mapping[str, pd.DataFrame],
    *,
    alignment: Alignment,
    legacy_faithful: bool = True,
) -> TrialCounts:
    """Build every protocol's trial counts from the converted legacy tables.

    Parameters
    ----------
    tables
        The converted parquet tables keyed by their file stems
        (``trials_task_axons``, ``trials_no_dreadds``, ``behavior_batch1``,
        ``behavior_batch3``, ``behavior_task_dendrites``,
        ``behavior_control_dendrites``).
    alignment
        Reward or stimulus; decides which outcomes count as target vs
        reference.
    legacy_faithful
        When ``True``, reproduces **BUG 4** (dendrite no-stim totals equal
        the stim totals under the stimulus alignment).
    """
    all_trials: dict[str, dict[str, Any]] = {}
    target: dict[str, int] = {}
    reference: dict[str, int] = {}

    # -- task axons / NO_DREADDs --------------------------------------------
    for protocol, stem in (
        ("task_axons", "trials_task_axons"),
        ("NO_DREADDs", "trials_no_dreadds"),
    ):
        per_id = _tabulate_flat(tables[stem])
        all_trials[protocol] = per_id
        target[protocol] = sum(counts.get("HIT", 0) for counts in per_id.values())
        reference[protocol] = sum(
            counts.get("FA", 0) + counts.get("CATCH", 0) for counts in per_id.values()
        )

    # -- passive (batch1 + batch3) ------------------------------------------
    for stem, prefix, group_cols in (
        ("behavior_batch1", "RGECO_GCaMP_Batch1", ["id", "compartment"]),
        ("behavior_batch3", "GCaMP_POm_Batch3", ["id", "fov"]),
    ):
        behavior = tables[stem]
        behavior = behavior[behavior["protocol"] == "NAIVE STIM REW"]
        for keys, group in behavior.groupby(group_cols, sort=False):
            identifier, second = str(keys[0]), str(keys[1])
            if stem == "behavior_batch1":
                label = "passive_axons" if second == "AXONS" else "passive_dendrites"
                unique_id = f"{prefix}_{identifier}"
            else:
                label = "passive_axons"
                unique_id = f"{prefix}_{identifier}_{second}"
            history = group["history"].astype(str)
            all_trials.setdefault(label, {})[unique_id] = {
                str(k): int(v) for k, v in history.groupby(history).size().items()
            }
            rew = int(history.str.contains("Rew").sum())
            stim = int(history.str.contains("Stim").sum())
            t, r = (rew, stim) if alignment == "reward" else (stim, rew)
            target[label] = target.get(label, 0) + t
            reference[label] = reference.get(label, 0) + r

    # -- dendrites (task + control) -----------------------------------------
    for stem, prefix, tds in (
        ("behavior_task_dendrites", "task_dendrites", ("TD0", "TD2", "TD3")),
        # control mice recorded TD1 as their first session; the notebooks'
        # inline comment: "control mice are actually using TD1 data"
        ("behavior_control_dendrites", "control_dendrites", ("TD1", "TD2", "TD3")),
    ):
        behavior = tables[stem]
        for td in tds:
            td_df = behavior[behavior["td"] == td]
            if td_df.empty:
                continue
            label = _cno_label(prefix, td)
            ttype = td_df["trial type"].astype(str)
            mouse_series = td_df["mouse"].astype(str)
            for mouse in dict.fromkeys(mouse_series.tolist()):
                mouse_ttypes = ttype[(mouse_series == mouse).to_numpy()]
                all_trials.setdefault(label, {}).setdefault(td, {})[mouse.lower()] = {
                    str(k): int(v) for k, v in mouse_ttypes.value_counts().items()
                }
            if alignment == "reward":
                t = int(ttype.str.contains("HIT").sum())
                r = int((~ttype.str.contains("HIT")).sum())
            else:
                stim_types = [x for x in ttype.unique() if ("miss" in x) or ("HIT" in x)]
                nostim_types = [x for x in ttype.unique() if ("FA" in x) or ("CATCH" in x)]
                t = int(ttype.isin(stim_types).sum())
                if legacy_faithful:
                    # BUG 4: the notebook computed nostim with stim_t_list
                    r = int(ttype.isin(stim_types).sum())
                else:
                    r = int(ttype.isin(nostim_types).sum())
            target[label] = target.get(label, 0) + t
            reference[label] = reference.get(label, 0) + r

    return TrialCounts(
        all_trials=all_trials,
        target_totals=target,
        reference_totals=reference,
        alignment=alignment,
    )


def _strip_modified(stem: str) -> str:
    """Drop a leading ``MODIFIED_`` prefix, as the notebooks did."""
    return stem.rsplit("MODIFIED_", maxsplit=1)[-1] if "MODIFIED" in stem else stem


def _relabel(stem: str) -> str:
    """TD0/TD1 -> pre-CNO, TD2/TD3 -> post-CNO, task vs control by substring."""
    prefix = "task_dendrites" if "task" in stem else "control_dendrites"
    if ("TD0" in stem) or ("TD1" in stem):
        return _cno_label(prefix, "TD0")
    if ("TD2" in stem) or ("TD3" in stem):
        return _cno_label(prefix, "TD2")
    return stem


def load_grouped_events(
    event_files: Mapping[str, pd.DataFrame],
    *,
    active_only: bool = True,
    exclude: Sequence[str] = ("TD1",),
    legacy_faithful: bool = True,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[Hashable]]]:
    """Pool per-file event tables into protocol groups with active-ROI sets.

    Joint port of the notebooks' ``extract_events`` (cell 0) and the
    active-ROI builder (cell 4/2). Files are processed in case-insensitive
    stem order, replicating the NTFS directory order the notebooks' dict-key
    layout depended on.

    Parameters
    ----------
    event_files
        ``file stem -> per-event table`` (from
        :func:`calcium2p.io.legacy.read_reference_events` or the regenerated
        equivalents).
    active_only
        Restrict each group's events to its active-ROI set, as every recorded
        notebook state did.
    exclude
        TD tokens meant to be excluded. **BUG 1**: with
        ``legacy_faithful=True`` the exclusion is a no-op for the pooled
        events (the notebook compared full stems against bare tokens) while
        the active-ROI builder keeps excluded TDs under separate unmerged
        keys -- so TD1 events survive exactly where their ROI indices collide
        with TD0's. With ``legacy_faithful=False`` excluded TDs are dropped
        everywhere.

    Returns
    -------
    tuple
        ``(events by protocol, active ROI keys by title)``.
    """
    ordered = sorted(event_files, key=str.lower)

    active: dict[str, list[Hashable]] = {}
    for stem in ordered:
        title = _strip_modified(stem)
        if legacy_faithful:
            # cell 4: excluded TDs keep their own unmerged key
            if not any(token in title for token in exclude):
                title = _relabel(title)
        else:
            if any(token in title for token in exclude):
                continue
            title = _relabel(title)
        frame = event_files[stem]
        # numeric_only=True is what pandas 1.5 did silently; only the group
        # index matters here
        rois = list(frame.groupby(frame.index).mean(numeric_only=True).index)
        active.setdefault(title, [])
        active[title] = active[title] + rois

    events: dict[str, pd.DataFrame] = {}
    for stem in ordered:
        title = _strip_modified(stem)
        if legacy_faithful:
            # BUG 1: `title not in exclude` never matches a full stem, so the
            # relabel always runs and TD1 pools into pre-CNO
            title = _relabel(title)
        else:
            if any(token in title for token in exclude):
                continue
            title = _relabel(title)
        frame = event_files[stem]
        if active_only:
            frame = frame[frame.index.isin(active[title])]
        events[title] = frame if title not in events else pd.concat([events[title], frame])

    return events, active


def build_roi_denominators(
    frequency_tables: Mapping[str, pd.DataFrame],
    converted: Mapping[str, pd.DataFrame],
    *,
    legacy_faithful: bool = True,
) -> dict[str, list[Hashable]]:
    """Total-ROI lists per protocol (the notebooks' ``ROI_numbers``).

    Parameters
    ----------
    frequency_tables
        ``file stem -> frequency table`` for the alignment's family
        (``*fullreward*`` or ``*500ms_stim*``), read with
        ``index_col=[0, 1]``.
    converted
        The converted legacy tables (traces/behavior parquet).
    legacy_faithful
        Reproduces **BUG 2** (MODIFIED CONTROL file dropped) and **BUG 3**
        (Batch1 ROI lists taken from the stale last Batch3 FOV).
    """
    numbers: dict[str, list[Hashable]] = {}

    def add(label: str, rois: Iterable[Hashable]) -> None:
        numbers.setdefault(label, [])
        numbers[label] = numbers[label] + list(rois)

    for stem in sorted(frequency_tables, key=str.lower):
        frame = frequency_tables[stem]
        if "COHORT" in stem:
            td = stem.split("_")[0] if "MODIFIED" not in stem else stem.split("_")[1]
            prefix = "task_dendrites"
        elif "CONTROL" in stem:
            if legacy_faithful:
                # BUG 2: the notebook's CONTROL branch parsed
                # stem.split('_')[0], turning MODIFIED_TD3_CONTROL into TD
                # "MODIFIED" and silently dropping its 203 ROIs
                td = stem.split("_")[0]
            else:
                td = stem.split("_")[0] if "MODIFIED" not in stem else stem.split("_")[1]
            prefix = "control_dendrites"
        else:
            continue
        if td in ("TD0", "TD2", "TD3"):
            add(_cno_label(prefix, td), [tuple(x) for x in frame.index.unique()])

    # -- batch3 (passive axons) ---------------------------------------------
    batch3 = converted["traces_batch3"]
    batch3 = batch3[batch3["protocol"] == "NAIVE STIM REW"]
    stale_rois: list[str] = []
    for (identifier, fov), group in batch3.groupby(["id", "fov"], sort=False):
        unique_id = f"GCaMP_POm_Batch3_{identifier}_{fov}"
        rois = list(dict.fromkeys(group["roi"].astype(str)))
        add("passive_axons", [(unique_id, roi) for roi in rois])
        stale_rois = rois  # loop-variable leak state, see BUG 3

    # -- batch1 (passive axons + dendrites) ----------------------------------
    batch1 = converted["traces_batch1"]
    batch1 = batch1[batch1["protocol"] == "NAIVE STIM REW"]
    for (identifier, compartment), group in batch1.groupby(["id", "compartment"], sort=False):
        label = "passive_axons" if str(compartment) == "AXONS" else "passive_dendrites"
        unique_id = f"RGECO_GCaMP_Batch1_{identifier}"
        # BUG 3: the notebook reused the stale FOV_dict from the Batch3 loop,
        # so every Batch1 group lists the last Batch3 FOV's ROIs
        rois = stale_rois if legacy_faithful else list(dict.fromkeys(group["roi"].astype(str)))
        add(label, [(unique_id, roi) for roi in rois])

    # -- task axons / NO_DREADDs ---------------------------------------------
    for protocol, stem in (
        ("task_axons", "traces_task_axons"),
        ("NO_DREADDs", "traces_no_dreadds"),
    ):
        traces = converted[stem]
        for identifier, group in traces.groupby("id", sort=False):
            rois = list(dict.fromkeys(group["roi"].astype(str)))
            add(protocol, [(f"{identifier!s}_{roi}",) for roi in rois])

    return numbers
