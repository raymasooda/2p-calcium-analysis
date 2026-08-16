"""Loaders for suite2p outputs, TIFF stacks, HDF5 stores, and legacy data.

Deliberately does not eagerly import submodules: anything depending on the
optional ``[suite2p]`` extra must remain importable-on-demand, so the package
as a whole stays usable without that install. The legacy readers below are
plain pandas and safe to re-export; :mod:`calcium2p.io.legacy_bridge` (the
pickle quarantine, run under a legacy environment) is intentionally NOT
re-exported and must be imported explicitly.
"""

from __future__ import annotations

from calcium2p.io.legacy import (
    ConvertedDataMissingError,
    canonical_roi_key,
    conversion_manifest_path,
    read_converted,
    read_reference_events,
    read_reference_frequencies,
)

__all__: list[str] = [
    "ConvertedDataMissingError",
    "canonical_roi_key",
    "conversion_manifest_path",
    "read_converted",
    "read_reference_events",
    "read_reference_frequencies",
]
