"""Cohort-level onset-histogram analysis (the productionized manuscript notebooks).

Submodules
----------
grouping
    Protocol relabelling (TD0/TD1 -> pre-CNO, TD2/TD3 -> post-CNO), active-ROI
    sets, ROI denominators, and trial counts -- including the three
    legacy-faithful bug sites.
event_tables
    Regeneration of the per-event tables from the deltaF traces.
modulation
    Reward/stimulus modulation indices and modulated-ROI selection.
metrics
    Per-(protocol, outcome, ROI) event metrics and named comparison frames.
stats
    Declarative two-group comparisons.
validation
    Match-rate reports against the reference tables and the pinned notebook
    numbers.
"""

from __future__ import annotations

__all__: list[str] = []
