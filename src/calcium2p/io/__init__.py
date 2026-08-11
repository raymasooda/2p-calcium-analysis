"""Loaders for suite2p outputs, TIFF stacks, and HDF5 stores.

Deliberately does not eagerly import submodules: anything depending on the
optional ``[suite2p]`` extra must remain importable-on-demand, so the package as
a whole stays usable without that install.
"""

from __future__ import annotations

__all__: list[str] = []
