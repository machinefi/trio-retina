# Trio Retina — common dev tasks. Run `make` to see them all.
.DEFAULT_GOAL := help

.PHONY: help install test lint format docs build clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package with dev extras (editable)
	pip install -e '.[dev]'

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check .

format:  ## Auto-format with ruff
	ruff format .

docs:  ## Build the docs site locally (needs the [docs] extra)
	mkdocs serve

build:  ## Build sdist + wheel into dist/
	python -m build

clean:  ## Remove build/test caches and artifacts
	rm -rf dist build site *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
