"""Two-photon calcium imaging analysis.

Trace processing, event detection, and behaviour alignment for simultaneous
multi-channel two-photon recordings.

Subpackages
-----------
core
    Session configuration and path resolution.
io
    Loaders for suite2p outputs, TIFF stacks, and HDF5 stores.
provenance
    Data-provenance capture: git state, environment, checksums, snapshots.
artifacts
    Materialize-vs-record policy for intermediate data artifacts.
pipeline
    Sequential analysis stages.
viz
    Plotting.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    # Single source of truth: the version declared in pyproject.toml. Reading it
    # back from installed metadata avoids the template repo's habit of hardcoding
    # __version__ here and letting it drift from the packaging metadata.
    __version__ = version("calcium2p")
except PackageNotFoundError:  # pragma: no cover - only when running from a raw checkout
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
