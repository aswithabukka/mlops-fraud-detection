.PHONY: help setup install test lint format clean docker-build docker-up docker-down generate-data train-local

# Display available commands
help:
	@echo "Available commands:"
	@echo "  make setup            - Initial project setup (create venv, install dependencies)"
	@echo "  make install          - Install project dependencies"
	@echo "  make test             - Run all tests with coverage"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make lint             - Run linters (black, flake8, mypy, isort)"
	@echo "  make format           - Auto-format code with black and isort"
	@echo "  make clean            - Remove build artifacts and cache files"
	@echo "  make docker-build     - Build all Docker images"
	@echo "  make docker-up        - Start all services with docker-compose"
	@echo "  make docker-down      - Stop all Docker services"
	@echo "  make generate-data    - Generate synthetic fraud data"
	@echo "  make train-local      - Train model locally"
	@echo "  make mlflow-ui        - Open MLflow UI"
	@echo "  make airflow-ui       - Open Airflow UI"
	@echo "  make health-check     - Check health of all services"

# Setup project environment
setup:
	@echo "Setting up project environment..."
	python3 -m venv venv
	@echo "Activating virtual environment and installing dependencies..."
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
	@echo "Installing pre-commit hooks..."
	. venv/bin/activate && pre-commit install
	@echo "Setup complete! Activate with: source venv/bin/activate"

# Install dependencies (assumes venv is activated or using poetry)
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Run all tests with coverage
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Run unit tests only
test-unit:
	pytest tests/unit/ -v

# Run integration tests only
test-integration:
	pytest tests/integration/ -v

# Run linters
lint:
	@echo "Running linters..."
	black --check src/ tests/
	isort --check-only src/ tests/
	flake8 src/ tests/
	mypy src/
	bandit -r src/ -ll

# Auto-format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

# Docker commands
docker-build:
	@echo "Building all Docker images..."
	docker-compose build

docker-up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started! Run 'make health-check' to verify"

docker-down:
	@echo "Stopping all services..."
	docker-compose down

docker-logs:
	docker-compose logs -f

# Generate synthetic fraud data
generate-data:
	@echo "Generating synthetic fraud data..."
	@mkdir -p data/raw logs
	PYTHONPATH=. python src/data/generator.py

# Train model locally
train-local:
	@echo "Training model locally..."
	python scripts/train_model.py

# Open MLflow UI
mlflow-ui:
	@echo "Opening MLflow UI at http://localhost:5000"
	@open http://localhost:5000 || xdg-open http://localhost:5000 || echo "Please open http://localhost:5000 in your browser"

# Open Airflow UI
airflow-ui:
	@echo "Opening Airflow UI at http://localhost:8080"
	@open http://localhost:8080 || xdg-open http://localhost:8080 || echo "Please open http://localhost:8080 in your browser"

# Health check for all services
health-check:
	@echo "Checking health of all services..."
	@echo "FastAPI: "
	@curl -s http://localhost:8000/health || echo "FastAPI not running"
	@echo "\nMLflow: "
	@curl -s http://localhost:5000/health || echo "MLflow not running"
	@echo "\nAirflow: "
	@curl -s http://localhost:8080/health || echo "Airflow not running"
	@echo "\nGrafana: "
	@curl -s http://localhost:3000/api/health || echo "Grafana not running"
