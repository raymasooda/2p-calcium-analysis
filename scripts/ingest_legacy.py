"""Convert the legacy pickled ``.npy`` dicts to parquet (one-time ingest).

The pickles reference ``pandas.core.indexes.numeric``, deleted in pandas 2.0,
so the conversion must execute under a legacy-matched interpreter. This CLI
verifies an existing conversion by default and, with ``--convert``, provisions
that interpreter through uv and runs
:func:`calcium2p.io.legacy_bridge.convert_all` inside it.

Usage::

    uv run python scripts/ingest_legacy.py                # verify
    uv run python scripts/ingest_legacy.py --convert      # (re)convert
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from calcium2p.core.paths import data_root

LEGACY_PYTHON = "3.10"
LEGACY_PACKAGES = ("pandas==1.5.3", "numpy==1.24.4", "pyarrow==14.0.2")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-subdir", default="manuscript/V10", help="inputs below DATA_ROOT")
    parser.add_argument(
        "--convert", action="store_true", help="run the conversion (default: verify only)"
    )
    return parser.parse_args()


def main() -> int:
    """Verify or run the legacy conversion."""
    args = parse_args()
    source = data_root(interactive=False) / args.data_subdir
    dest = source.parent / (source.name + "_converted")
    manifest = dest / "conversion_manifest.json"

    if args.convert:
        repo_root = Path(__file__).resolve().parents[1]
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            LEGACY_PYTHON,
            *(f"--with={pkg}" for pkg in LEGACY_PACKAGES),
            "python",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from calcium2p.io.legacy_bridge import convert_all; "
            "import json; print(json.dumps(convert_all(sys.argv[2], sys.argv[3])))",
            str(repo_root / "src"),
            str(source),
            str(dest),
        ]
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print("conversion failed", file=sys.stderr)
            return result.returncode

    if not manifest.exists():
        print(f"no conversion manifest at {manifest}; run with --convert", file=sys.stderr)
        return 1
    record = json.loads(manifest.read_text(encoding="utf-8"))
    tables = sorted(p.stem for p in dest.glob("*.parquet"))
    print(f"conversion manifest: {manifest}")
    print(
        f"converted with pandas {record.get('pandas_version', '?')} / "
        f"numpy {record.get('numpy_version', '?')}"
    )
    counts = record.get("row_counts", {})
    for name in tables:
        print(f"  {name}: {counts.get(name, '?')} rows")
    if not tables:
        print(f"no converted tables under {dest}", file=sys.stderr)
        return 1
    print(f"{len(tables)} converted tables present under {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
