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
├── core/           session + cohort config, path resolution
├── io/             legacy pickle bridge + reference readers (suite2p/TIFF to come)
├── signal/         filtering, dF/F0, hysteresis event detection (pure functions)
├── cohort/         onset-histogram manuscript analysis (grouping, modulation,
│                   metrics, stats, event-table regeneration, validation, run)
├── provenance/     git + environment + checksum capture, provenance.json
├── artifacts/      materialize-vs-mutation-record policy
├── pipeline/       sequential analysis stages
└── viz/            onset histograms, comparison bars, traces, regressions

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

## Onset-histogram analysis

The productionized reward/stimulus onset-histogram manuscript analysis lives in
`calcium2p.cohort`. One-time ingest converts the legacy pickled `.npy` dicts to
parquet (the pickles reference `pandas.core.indexes.numeric`, deleted in pandas
2.0, so the converter runs under a uv-provisioned pandas-1.5 interpreter):

```bash
uv run python scripts/ingest_legacy.py --convert   # once per machine
uv run python scripts/run_onset_analysis.py --config configs/onset_histograms.yaml
```

Copy `configs/onset_histograms.example.yaml` to get started. Two flags matter:

- `legacy_faithful: true` (default) reproduces the original notebooks'
  counting bugs exactly, so outputs validate against the published artifacts;
  `false` applies the corrected behaviour. Both cache side by side.
- `event_source: reference | regenerated` selects the original event CSVs or
  tables rebuilt from the deltaF traces.

Every output is written through the artifact store with a `provenance.json`
sidecar. The regression suite (`tests/regression/`, marker `requires_data`)
pins the recorded notebook numbers and is skipped where the dataset is absent.

### Reproducing the published comparison figures

The manuscript's comparison figures were not produced by a single notebook
run: the notebooks were re-run with different switches between saves
(modulated vs active ROIs, the statistical test, the passive-dendrite window).
The package ships the recovered recipe as a **figure manifest**,
`src/calcium2p/cohort/manifests/v10_figures.yaml`: one config per notebook
state, 57 figures in 7 states, each with the y-limits of its original.

```bash
uv run python scripts/run_figure_manifest.py      # all 57, into the artifact store
```

Each figure's n, means, SEM and p are pinned in
`tests/regression/test_v10_manifest.py`. Two published figures are not
reproduced and are excluded; the manifest's `notes` names them.

## License

MIT
