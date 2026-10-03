"""Pure signal-processing numerics ported from the legacy Thalamocortical code.

Everything here is a plain function on arrays/frames -- no classes, no I/O, no
module state. The ports preserve the legacy semantics exactly (including two
documented quirks) so that regenerated event tables can be validated against
the original manuscript outputs.
"""

from calcium2p.signal.dff import compute_dff
from calcium2p.signal.events import detect_events, event_metrics_table
from calcium2p.signal.filters import lowpass_filter

__all__ = [
    "compute_dff",
    "detect_events",
    "event_metrics_table",
    "lowpass_filter",
]
