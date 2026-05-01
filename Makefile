.PHONY: install test lint format clean

install:
	uv sync --all-packages

test:
	pytest tests/ packages/ --cov=packages/

lint:
	ruff check .
	mypy .

format:
	ruff format .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
