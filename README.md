# 2p-calcium-analysis

Two-photon calcium imaging analysis: trace processing, event detection, and
behaviour alignment for simultaneous multi-channel recordings.

Successor to [`Neuronal-Calcium-Dynamics`](https://github.com/raymasooda/Neuronal-Calcium-Dynamics),
rebuilt as an installable, tested, provenance-tracked package.

---

## Prerequisites

| Tool | Why | Install |
|---|---|---|
| [`uv`](https://docs.astral.sh/uv/) | The **only** supported package/environment manager here | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| `git` | Provenance capture reads git state on every pipeline run | system package manager |

Python itself is managed by `uv` — `.python-version` pins **3.12**, and `uv` installs it
if missing. Do not use the system `python3`: on this machine it resolves to 3.8.10 via a
pyenv shim.

## Quick start

```bash
uv sync --extra dev          # create .venv and install (fast: no torch, no suite2p)
cp .env.example .env         # then edit DATA_ROOT to point at your data
uv run pytest                # confirm the install
```

Optional extras, installed only when needed:

```bash
uv sync --extra suite2p      # motion correction / ROI extraction (heavy: torch + cellpose)
uv sync --extra notebooks    # JupyterLab + ipywidgets
```

> **Why `suite2p` is optional:** it pulls `torch`, `cellpose`, and
> `scanimage-tiff-reader` — whose last release (July 2023) ships only a `cp37`
> Linux wheel and therefore builds from source on any modern Python. Keeping it
> out of the core means `uv sync` and CI stay fast and cannot be broken by that
> chain. Trace processing onward needs none of it.

## Reproducibility

| Dependency | Where pinned | How to update |
|---|---|---|
| Python | `.python-version` | edit, then `uv sync` |
| Python packages | `pyproject.toml` + `uv.lock` (committed) | `uv add <pkg>` / `uv lock --upgrade` |
| Environment of a given output | `provenance.json` beside the output | regenerated per run |

Every dependency carries an explicit upper bound with a comment naming the failure it
prevents. Never widen one without testing.

## Repo layout

```
src/calcium2p/      all real logic; scripts and notebooks import from here
├── core/           session config, path resolution
├── io/             suite2p / TIFF / HDF5 loaders
├── provenance/     git + environment + checksum capture, provenance.json
├── artifacts/      materialize-vs-mutation-record policy
├── pipeline/       sequential analysis stages
└── viz/            plotting

scripts/            thin CLI entrypoints (logic lives in src/)
└── scratch/        investigatory scripts; untracked but fully run-logged
notebooks/          exploratory and tutorial notebooks
configs/            machine-specific; only *.example.yaml is tracked
tests/              unit/ and integration/ pytest suites
docs/               dev-facing docs — NOT committed (local-machine state)
```

## Development

```bash
make help        # list targets
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy (strict)
make test        # pytest
make verify      # all of the above
```

`make verify` runs the same gates as CI, so a green run locally means a green
run on the PR. Install the pre-commit hooks to catch them earlier still:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat` `fix` `docs` `style` `refactor` `perf` `test` `build` `ci` `chore`
`revert`) — the CI job parses history for them, and `main` advances only by
merge, never by a direct commit.

## License

MIT
