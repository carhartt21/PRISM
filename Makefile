# Makefile for Image Evaluation Pipeline

.PHONY: install test clean lint format help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=. --cov-report=html

lint:  ## Run linting
	flake8 --max-line-length=100 --ignore=E203,W503 .
	mypy --ignore-missing-imports .

format:  ## Format code
	black --line-length=100 .

clean:  ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

example:  ## Run example evaluation
	@echo "Creating example data..."
	@mkdir -p example_data/generated example_data/real
	@python -c "import torch; from PIL import Image; import numpy as np; [Image.fromarray((torch.rand(64,64,3)*255).numpy().astype('uint8')).save(f'example_data/generated/img_{i:03d}.png') for i in range(5)]; [Image.fromarray((torch.rand(64,64,3)*255).numpy().astype('uint8')).save(f'example_data/real/img_{i:03d}.jpg') for i in range(5)]"
	@echo "Running evaluation..."
	python evaluate_generation.py --generated example_data/generated --real example_data/real --metrics ssim psnr --output example_results.json -v
	@echo "Results saved to example_results.json"

clean-example:  ## Clean example data
	rm -rf example_data/ example_results.json
