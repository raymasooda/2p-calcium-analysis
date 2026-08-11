"""Concrete pipeline stages.

Populated as the analysis is ported. The intended sequence, reconstructed from
the predecessor repo's notebooks:

1. ``load``             suite2p ``F.npy`` / ``Fall.mat`` -> traces
2. ``filter``           zero-phase Butterworth low-pass
3. ``dff``              median-baseline dF/F0 with a rolling-median F0
4. ``events``           hysteresis-threshold event detection
5. ``correlate``        event-based cross-channel ROI correlation
6. ``align_behaviour``  trial/lick/pupil alignment

Each is a :class:`~calcium2p.pipeline.base.PipelineStage`; the numerics they
call live in sibling modules as plain functions.
"""

from __future__ import annotations

__all__: list[str] = []
