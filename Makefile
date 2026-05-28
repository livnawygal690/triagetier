.PHONY: help install install-dev format lint type test smoke docker check clean
.DEFAULT_GOAL := help

help:
	@echo "targets: install install-dev format lint type test smoke docker check clean"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

format:
	ruff check --fix .
	isort .
	black .

lint:
	ruff check .
	isort --check-only .
	black --check .

type:
	mypy --strict triagetier

test:
	pytest -q

smoke:
	python -m triagetier.toolchain.run experiment=_smoke

docker:
	docker build -t triagetier .

check: lint type test

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
