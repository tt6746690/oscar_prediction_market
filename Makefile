.PHONY: help install format lint lint-fix typecheck test test-cov dev all

.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync --all-extras

format:  ## Auto-format code with ruff
	uv run ruff format oscar_prediction_market/ tests/

lint:  ## Check code with ruff (no auto-fix)
	uv run ruff check oscar_prediction_market/ tests/

lint-fix:  ## Check and auto-fix linting issues
	uv run ruff check --fix oscar_prediction_market/ tests/

typecheck:  ## Run mypy type checking (excludes one_offs/)
	uv run mypy oscar_prediction_market/ tests/ --exclude oscar_prediction_market/one_offs/

test:  ## Run pytest
	uv run pytest

test-cov:  ## Run pytest with coverage report
	uv run pytest --cov=oscar_prediction_market --cov-report=term-missing

dev: format lint typecheck test  ## Quick development check

all: dev  ## Full validation before commit
