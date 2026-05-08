.PHONY: install test test-fast test-slow lint format clean

install:
	uv sync --all-packages

test:
	NUMBA_DISABLE_JIT=1 uv run pytest tests/ packages/ --cov=packages/ --cov-fail-under=85

test-fast:
	NUMBA_DISABLE_JIT=1 uv run pytest tests/ packages/ -m "not slow" --cov=packages/ --cov-fail-under=85

test-slow:
	NUMBA_DISABLE_JIT=1 uv run pytest tests/ packages/ -m "slow" -v

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
