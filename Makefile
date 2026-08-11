# calcium2p -- convenience targets over uv.
# Every target goes through `uv run`; never invoke a bare python3 (it may resolve
# to an old interpreter via a pyenv shim).

# Optional local-only targets. The leading dash makes this a no-op when absent,
# so a fresh clone behaves identically.
-include Makefile.local

.PHONY: help setup sync test test-fast lint format typecheck verify clean

help:
	@echo "Targets:"
	@echo "  setup       uv sync with the dev extra, then a smoke import"
	@echo "  sync        uv sync --extra dev"
	@echo "  test        pytest with coverage"
	@echo "  test-fast   pytest, unit tests only, no coverage"
	@echo "  lint        ruff check"
	@echo "  format      ruff format"
	@echo "  typecheck   mypy (strict)"
	@echo "  verify      lint + format check + typecheck + test"
	@echo "  clean       remove caches and build artefacts"

setup: sync
	@uv run python -c "import calcium2p; print('calcium2p', calcium2p.__version__)"
	@echo "Now: cp .env.example .env  and set DATA_ROOT"

sync:
	uv sync --extra dev

test:
	uv run pytest --cov=calcium2p --cov-report=term-missing

test-fast:
	uv run pytest tests/unit -q --no-header

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy src

# Mirrors .github/workflows/ci.yml, so a green `make verify` means a green CI.
verify:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src
	uv run pytest -q

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
