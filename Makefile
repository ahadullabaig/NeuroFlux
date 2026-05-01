.PHONY: install test lint format clean

install:
	uv sync --all-packages

test:
	NUMBA_DISABLE_JIT=1 uv run pytest tests/ packages/ --cov=packages/

lint:
	uv run ruff check .
	uv run mypy .

format:
	uv run ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
