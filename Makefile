.PHONY: install install-dev lint format test clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/

clean:
	rm -rf build/ dist/ *.egg-info .ruff_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
