"""Shared fixtures.

Keeps a real git repository available for provenance tests, since git state
capture is central to this package and mocking it would test the mock.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one commit, for provenance tests."""
    # `git init -b` needs git >= 2.28; pointing HEAD by hand works on any version.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "symbolic-ref", "HEAD", "refs/heads/main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
    (tmp_path / "seed.txt").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "feat: seed the test repository"],
        cwd=tmp_path,
        check=True,
    )
    return tmp_path


@pytest.fixture
def traces() -> np.ndarray:
    """A small deterministic (n_samples, n_rois) fluorescence array."""
    rng = np.random.default_rng(20260409)
    return rng.normal(loc=100.0, scale=5.0, size=(600, 12))


@pytest.fixture
def frame() -> pd.DataFrame:
    """A small DataFrame with mixed dtypes, to catch dtype-losing round-trips."""
    return pd.DataFrame(
        {
            "roi": pd.Series([0, 1, 2], dtype="int32"),
            "amplitude": pd.Series([1.5, 2.5, 3.5], dtype="float32"),
            "is_cell": pd.Series([True, False, True], dtype="bool"),
            "label": pd.Series(["a", "b", "c"], dtype="string"),
        }
    )


@pytest.fixture
def store_root(tmp_path: Path) -> Iterator[Path]:
    """An empty ArtifactStore root."""
    root = tmp_path / "derivatives"
    root.mkdir()
    yield root
