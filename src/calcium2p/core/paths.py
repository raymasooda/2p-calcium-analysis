r"""Resolve machine-specific roots without hardcoding them anywhere tracked.

The predecessor repo hardcoded ``r'D:\\three'`` and built paths by string
concatenation in 60+ places, which made it runnable on exactly one machine. Here
every path descends from ``DATA_ROOT``, resolved at runtime.

Resolution order, highest priority first:

1. an explicit argument,
2. the ``DATA_ROOT`` environment variable (``.env`` is loaded automatically),
3. a saved ``configs/paths.yaml`` (gitignored, per-machine),
4. an interactive prompt, offering to save the answer.
"""

from __future__ import annotations

import os
from pathlib import Path

PATHS_CONFIG = Path("configs") / "paths.yaml"


class DataRootNotConfiguredError(RuntimeError):
    """Raised when DATA_ROOT cannot be resolved and prompting is disabled."""


def _load_dotenv(repo_root: Path) -> None:
    """Load ``.env`` if python-dotenv is available.

    Soft import: the CLI should still work in a minimal environment, it just
    requires the variable to be exported by the shell instead.
    """
    try:
        from dotenv import load_dotenv  # noqa: PLC0415
    except ImportError:
        return
    load_dotenv(repo_root / ".env", override=False)


def _read_saved(repo_root: Path) -> dict[str, str]:
    """Read the saved per-machine path config, if present."""
    path = repo_root / PATHS_CONFIG
    if not path.exists():
        return {}
    try:
        import yaml  # noqa: PLC0415

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (ImportError, OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}


def repo_root() -> Path:
    """Locate the repository root from this file's position."""
    return Path(__file__).resolve().parents[3]


def data_root(explicit: str | Path | None = None, *, interactive: bool = True) -> Path:
    """Resolve the local data root.

    Parameters
    ----------
    explicit
        Overrides every other source when given.
    interactive
        When ``True`` and nothing else resolves, prompt and offer to save the
        answer so it is a one-time step. Set ``False`` in scripts and CI, where
        a blocking prompt would hang.

    Returns
    -------
    Path
        The resolved root. It is **not** required to exist -- validate at the
        point of use, where a better error message is possible.

    Raises
    ------
    DataRootNotConfiguredError
        When nothing resolves and ``interactive`` is ``False``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()

    root = repo_root()
    _load_dotenv(root)

    from_env = os.environ.get("DATA_ROOT", "").strip()
    if from_env:
        return Path(from_env).expanduser()

    saved = _read_saved(root).get("data_root", "").strip()
    if saved:
        return Path(saved).expanduser()

    if not interactive:
        raise DataRootNotConfiguredError(
            "DATA_ROOT is not set. Copy .env.example to .env and set DATA_ROOT, "
            "or pass an explicit path."
        )

    answer = input("Enter your local data root (e.g. /data/calcium): ").strip()
    if not answer:
        raise DataRootNotConfiguredError("No data root provided.")
    _offer_to_save(root, answer)
    return Path(answer).expanduser()


def _offer_to_save(root: Path, value: str) -> None:
    """Offer to persist the answer, so this is a one-time step."""
    if input("Save as the default for future runs? [Y/n]: ").strip().lower() == "n":
        return
    try:
        import yaml  # noqa: PLC0415

        target = root / PATHS_CONFIG
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.dump({"data_root": value}), encoding="utf-8")
    except (ImportError, OSError):
        return
    print(f"  Saved -> {PATHS_CONFIG}  (gitignored -- not committed)")


def derivatives_root(explicit: str | Path | None = None, *, interactive: bool = True) -> Path:
    """Resolve where pipeline outputs are written.

    ``DERIVATIVES_ROOT`` if set, else ``<data_root>/derivatives``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    from_env = os.environ.get("DERIVATIVES_ROOT", "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return data_root(interactive=interactive) / "derivatives"
