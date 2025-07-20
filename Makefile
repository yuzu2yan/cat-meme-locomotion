.PHONY: help install run extract visualize test lint format clean

help:
	@echo "Cat Meme Locomotion - Available commands:"
	@echo "  make install    - Install dependencies with uv"
	@echo "  make run        - Run the main cat-to-spot replication"
	@echo "  make extract    - Extract motion from GIF only"
	@echo "  make visualize  - Show motion analysis visualization"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean generated files"

install:
	uv pip install -e .
	uv pip install -e ".[dev]"

run:
	uv run cat-unitree

extract:
	uv run extract-motion

test:
	uv run pytest tests/

lint:
	uv run ruff check src/
	uv run mypy src/

format:
	uv run black src/
	uv run ruff check --fix src/

clean:
	rm -rf outputs/
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete