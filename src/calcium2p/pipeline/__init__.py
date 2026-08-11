"""Sequential analysis stages.

See :mod:`calcium2p.pipeline.base` for why this layer is object-oriented while
the numerical transforms it calls are plain functions.
"""

from __future__ import annotations

from calcium2p.pipeline.base import Pipeline, PipelineStage

__all__ = ["Pipeline", "PipelineStage"]
