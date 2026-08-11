"""Session configuration and machine-local path resolution."""

from __future__ import annotations

from calcium2p.core.config import (
    Channel,
    SessionConfig,
    SessionConfigError,
    load_session_config,
)
from calcium2p.core.paths import (
    DataRootNotConfiguredError,
    data_root,
    derivatives_root,
    repo_root,
)

__all__ = [
    "Channel",
    "DataRootNotConfiguredError",
    "SessionConfig",
    "SessionConfigError",
    "data_root",
    "derivatives_root",
    "load_session_config",
    "repo_root",
]
