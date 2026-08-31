.PHONY: fmt lint test build check

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

test:
	uv run pytest

build:
	uv build

check: lint test build
