# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a production-grade MLOps pipeline for credit card fraud detection, built as an interview preparation project. The system demonstrates end-to-end ML workflow from synthetic data generation through deployment and monitoring.

**Key characteristic**: Extreme class imbalance (0.5-2% fraud rate) requiring specialized handling throughout the pipeline.

## Architecture

The codebase follows a **layered architecture** with clear separation of concerns:

```
Data Layer (src/data/)
  ↓ generates & validates data
Orchestration Layer (airflow/dags/)
  ↓ coordinates workflows
ML Platform Layer (MLflow)
  ↓ tracks experiments & models
Serving Layer (src/serving/)
  ↓ provides predictions
Monitoring Layer (src/monitoring/)
  ↓ detects drift & triggers retraining
```

### Critical Integration Points

1. **Config → Everything**: All components import `config.settings.Settings` for centralized configuration. Configuration is environment-based using Pydantic Settings with `.env` file support.

2. **Data Pipeline Flow**:
   - `FraudDataGenerator` creates synthetic transactions → `DataValidator` validates schema → `FraudPreprocessor` engineers features → Models consume preprocessed data
   - Preprocessor must be saved with models (stored as MLflow artifact) to ensure consistent transformations at inference time

3. **MLflow Integration**:
   - `ModelTrainer` logs everything to MLflow (params, metrics, model, preprocessor)
   - `ModelRegistry` manages model lifecycle (None → Staging → Production)
   - `FastAPI` loads production model by stage alias (not version number)

4. **Airflow DAGs**:
   - `training_pipeline_dag.py`: Weekly model retraining
   - Tasks are loosely coupled - use Airflow's XCom for passing data between tasks sparingly

## Common Commands

### Development Workflow
```bash
# Initial setup
make setup              # Creates venv, installs deps, sets up pre-commit
source venv/bin/activate

# Generate synthetic data (creates data/raw/fraud_YYYYMMDD.csv)
make generate-data

# Train model locally (no Docker needed)
make train-local

# Run tests
make test              # All tests with coverage
make test-unit         # Unit tests only
pytest tests/unit/test_generator.py::TestFraudDataGenerator::test_fraud_rate -v  # Single test

# Code quality
make lint              # Check without changes
make format            # Auto-format with black + isort
```

### Docker Operations
```bash
# Full stack (9 services: postgres, minio, mlflow, airflow x2, fastapi, prometheus, grafana)
make docker-build      # First time or after Dockerfile changes
make docker-up         # Start all services
make docker-down       # Stop all services
make health-check      # Verify all services are healthy

# View logs
make docker-logs       # All services
docker-compose logs -f fastapi  # Single service
```

### Service Access
- MLflow UI: http://localhost:5000 (experiment tracking, model registry)
- Airflow UI: http://localhost:8080 (user: admin, pass: admin)
- FastAPI docs: http://localhost:8000/docs (interactive API testing)
- Grafana: http://localhost:3000 (user: admin, pass: admin)
- MinIO console: http://localhost:9001 (S3-compatible storage)

## Key Design Patterns

### 1. Configuration Management
All configuration is centralized in `config/settings.py` using Pydantic Settings:
```python
from config.settings import settings

# Environment-based: reads from .env or environment variables
mlflow_uri = settings.mlflow_tracking_uri
fraud_threshold = settings.fraud_threshold
```

**Important**: When adding new config, add it to `Settings` class with type hints and defaults. Use `Field()` for validation constraints.

### 2. Base Model Pattern
All ML models inherit from `BaseMLModel` (`src/models/base_model.py`):
- Provides consistent interface: `fit()`, `predict()`, `predict_proba()`, `evaluate()`
- Handles metrics calculation
- Manages MLflow logging

**When adding new models**: Inherit from `BaseMLModel`, implement abstract methods, use `self._log_to_mlflow()` for tracking.

### 3. Storage Abstraction
`src/utils/storage.py` provides unified interface for local and S3 storage:
```python
from src.utils.storage import get_storage

storage = get_storage()  # Returns LocalStorage or S3Storage based on settings
storage.save(data, "path/to/file.csv")  # Works with both backends
```

**Key benefit**: Switch between local (MinIO) and cloud (AWS S3) by changing one config variable.

### 4. Preprocessing Pipeline
`FraudPreprocessor` follows scikit-learn's fit/transform pattern:
- `fit()` on training data to learn encodings/scalings
- `transform()` on test/inference data using learned parameters
- **Critical**: Always save preprocessor with model as MLflow artifact

### 5. Class Imbalance Handling
The dataset has ~1% fraud rate. All models use:
- SMOTE for oversampling minority class (controlled by `handle_imbalance` flag)
- Class weights for cost-sensitive learning
- Threshold optimization for precision-recall tradeoff

**When working with models**: Never remove SMOTE without understanding impact on metrics. Evaluate using AUC-PR (not just AUC-ROC) as it's more informative for imbalanced data.

## Testing Strategy

### Test Structure
```
tests/
├── unit/              # Fast, isolated tests (70% of tests)
│   ├── test_generator.py
│   ├── test_schema.py
│   └── test_preprocessor.py
├── integration/       # Multi-component tests (30% of tests)
│   └── test_training_pipeline.py
└── conftest.py       # Shared fixtures
```

### Running Tests
```bash
# Run all tests with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/test_generator.py -v

# Run single test
pytest tests/unit/test_generator.py::TestFraudDataGenerator::test_fraud_rate -v

# Run with prints visible (debugging)
pytest tests/unit/test_generator.py -v -s
```

### Important Testing Notes
- Mock MLflow in unit tests using `pytest-mock` to avoid dependency on running MLflow server
- Integration tests may require Docker services running
- Test data generation is deterministic (uses seed) for reproducibility

## Working with Docker Services

### Service Dependencies
```
postgres (database)
  ↓ used by
mlflow + airflow (metadata storage)
  ↓ used by
fastapi (loads production model from mlflow)

minio (artifact storage)
  ↓ used by
mlflow (stores model artifacts)
```

### Common Docker Issues

**Problem**: Services fail to start or show "unhealthy"
```bash
# Check service logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>

# Nuclear option: clean restart
make docker-down
docker system prune -a  # Be careful - removes all unused containers/images
make docker-build
make docker-up
```

**Problem**: Port already in use (e.g., 5432, 5000, 8080)
- Edit `docker-compose.yml` to change port mappings (left side of `:`), or
- Stop conflicting service on host

**Problem**: MLflow can't connect to postgres
- Wait 30 seconds for postgres to fully initialize on first startup
- Check `docker-compose logs postgres` for errors

## Data Generation & Validation

### Synthetic Data Generation
```bash
# Generate with defaults (100K samples, 1% fraud rate)
make generate-data

# Generate with custom parameters
PYTHONPATH=. python -c "
from src.data.generator import FraudDataGenerator
gen = FraudDataGenerator(n_samples=50000, fraud_rate=0.02)
df = gen.generate()
gen.save(df, file_format='parquet')
"
```

### Data Validation
Schema validation uses Pandera (`src/data/schema.py`):
- Type checking (datetime, float, int, str)
- Range constraints (amount > 0, hour in 0-23)
- Business logic (fraud rate validation, timestamp ordering)

**When modifying schema**: Update `FraudTransactionSchema` and regenerate data to match.

## MLflow Model Registry Workflow

Models progress through stages:
1. **None**: Newly registered model
2. **Staging**: Model under validation
3. **Production**: Active model serving predictions

**Promotion process**:
```python
from src.models.registry import ModelRegistry

registry = ModelRegistry()
# Register new model
model_version = registry.register_model(model, "fraud_classifier")

# Promote to production (after validation)
registry.promote_model("fraud_classifier", version=model_version, stage="Production")
```

**Critical**: FastAPI loads model by stage alias, so promoting a new model to Production instantly updates the serving layer (no code deploy needed).

## Airflow DAG Development

### Key DAG: training_pipeline_dag.py
- **Schedule**: Weekly (Sundays at 3 AM)
- **Tasks**: generate_training_data → train_models → notify_completion
- **Parallelization**: Multiple models can be trained in parallel using TaskGroups (not implemented yet)

### DAG Development Tips
```bash
# Test DAG imports (catches syntax errors)
python airflow/dags/training_pipeline_dag.py

# Test individual task
airflow tasks test fraud_detection_training train_models 2024-01-01

# Trigger DAG manually via UI or CLI
airflow dags trigger fraud_detection_training
```

## Environment Variables

Key environment variables (set in `.env` or export):
```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@postgres:5432/mlflow

# AWS/MinIO (S3-compatible)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://localhost:9000  # MinIO for local, remove for AWS

# Model serving
FRAUD_THRESHOLD=0.5  # Probability threshold for fraud classification
MODEL_STAGE=Production  # Which MLflow stage to load

# Monitoring
DRIFT_THRESHOLD=0.15  # Alert on moderate drift
DRIFT_SEVERE_THRESHOLD=0.2  # Trigger retraining

# Use .env.example as template
cp .env.example .env
# Edit .env with your values
```

## Important Implementation Details

### 1. PYTHONPATH Requirement
When running scripts directly (not via Makefile), set PYTHONPATH:
```bash
# Required for imports to work
export PYTHONPATH=/Users/aswithabukka/CascadeProjects/MLOps
python src/data/generator.py

# Or use module syntax
python -m src.data.generator
```

### 2. MLflow Artifact Storage
- **Local**: Artifacts stored in `./mlruns/` directory
- **Docker**: Artifacts stored in MinIO (S3-compatible) at `s3://mlflow-artifacts`
- **Production**: Use AWS S3 by setting `MLFLOW_ARTIFACT_ROOT=s3://your-bucket`

### 3. Preprocessor Versioning
The preprocessor **must** be saved with the model:
```python
# In trainer.py
mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    # CRITICAL: Save preprocessor for inference
    artifacts={"preprocessor": preprocessor_path}
)
```

This ensures the same transformations are applied at inference time.

### 4. Fraud Detection Metrics Priority
Due to class imbalance, prioritize these metrics:
1. **AUC-PR** (Precision-Recall curve area) - most important
2. **Recall @ 90% Precision** - business constraint
3. **F1 Score** - balance metric
4. AUC-ROC - useful but can be misleading with imbalance

### 5. Feature Engineering Dependencies
Some engineered features depend on velocity calculations:
- `transactions_last_24h`: Requires grouping by cardholder
- `avg_amount_last_30d`: Requires time-ordered data

**When modifying**: Ensure data is sorted by timestamp before feature engineering.

## Extending the System

### Adding a New Model Algorithm
1. Add algorithm to `FraudClassifier.__init__()` in `src/models/fraud_classifier.py`
2. Update `_initialize_model()` method
3. Add tests in `tests/unit/test_fraud_classifier.py`
4. Update training DAG to include new algorithm

### Adding a New Airflow DAG
1. Create DAG file in `airflow/dags/`
2. Import required modules at top (will be in Airflow's Python path)
3. Use `default_args` from existing DAGs for consistency
4. Test locally before committing: `python airflow/dags/your_dag.py`

### Adding New API Endpoints
1. Add endpoint to `src/serving/api.py`
2. Define request/response models with Pydantic
3. Add endpoint tests in `tests/integration/test_api.py`
4. Update API documentation in comments (auto-generates OpenAPI docs)

## Troubleshooting

### Import Errors
```bash
# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Or use editable install
pip install -e .
```

### MLflow Connection Refused
```bash
# Check if MLflow is running
curl http://localhost:5000/health

# Start MLflow if not running
make docker-up

# Check logs
docker-compose logs mlflow
```

### Airflow DAG Not Appearing
- Check `airflow/dags/` for Python syntax errors
- DAG file must be valid Python (test with `python dag_file.py`)
- Refresh Airflow UI (may take 30-60 seconds to detect new DAGs)
- Check Airflow logs: `docker-compose logs airflow-scheduler`

### Tests Failing with ModuleNotFoundError
```bash
# Install in development mode
pip install -e .

# Or set PYTHONPATH
export PYTHONPATH=$(pwd)
pytest tests/
```

## Project Status

**Completed** (14/17 phases):
- ✅ Data pipeline (generation, validation, preprocessing)
- ✅ ML training (4 algorithms with MLflow tracking)
- ✅ Model registry and versioning
- ✅ FastAPI serving layer
- ✅ Monitoring setup (Prometheus, Grafana)
- ✅ Airflow orchestration (training DAG)
- ✅ Docker Compose (9 services)
- ✅ Unit tests (30+ tests)

**Pending**:
- ⏳ CI/CD (GitHub Actions workflows scaffolded, needs customization)
- ⏳ AWS Terraform (infrastructure code needs completion)
- ⏳ Additional monitoring DAGs (data pipeline, monitoring/drift detection)

## Further Reading

- `README.md` - Project overview and quick start
- `GUIDE.md` - Comprehensive technical guide (5000+ words)
- `BUILD_COMPLETE.md` - Build summary and interview preparation
- `docker-compose.yml` - All service configurations
- `config/settings.py` - All configuration options with defaults
