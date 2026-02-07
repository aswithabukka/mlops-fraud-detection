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

4. **Airflow DAGs** (3 production DAGs):
   - `data_pipeline_dag.py`: Daily data generation and preprocessing (runs at 2 AM)
   - `training_pipeline_dag.py`: Weekly model retraining (runs Sundays at 3 AM)
   - `monitoring_dag.py`: Hourly drift detection with auto-retraining trigger
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

### Production DAGs

**1. data_pipeline_dag.py** (Daily at 2 AM)
- **Schedule**: `0 2 * * *` (daily at 2 AM)
- **Tasks**: setup_directories → generate_data → validate_data → preprocess_data → update_reference_dataset → generate_quality_report
- **Purpose**: Generate 100K synthetic transactions, validate schema, engineer features, update drift detection baseline

**2. training_pipeline_dag.py** (Weekly on Sundays at 3 AM)
- **Schedule**: `0 3 * * 0` (weekly, Sundays at 3 AM)
- **Tasks**: generate_training_data → train_models → evaluate_models → register_best_model → promote_to_production
- **Purpose**: Train multiple models, compare with current production, promote if metrics improved

**3. monitoring_dag.py** (Hourly)
- **Schedule**: `0 * * * *` (hourly)
- **Tasks**: collect_predictions → detect_drift → check_drift_severity → trigger_retraining (conditional)
- **Purpose**: Monitor data drift using EvidentlyAI, automatically trigger retraining when drift > 0.2
- **Key feature**: Event-driven retraining (uses TriggerDagRunOperator)

### DAG Development Tips
```bash
# Test DAG imports (catches syntax errors)
python airflow/dags/data_pipeline_dag.py
python airflow/dags/training_pipeline_dag.py
python airflow/dags/monitoring_dag.py

# Test individual task
airflow tasks test fraud_detection_data generate_data 2024-01-01
airflow tasks test fraud_detection_training train_models 2024-01-01
airflow tasks test fraud_detection_monitoring detect_drift 2024-01-01

# Trigger DAG manually via UI or CLI
airflow dags trigger fraud_detection_data
airflow dags trigger fraud_detection_training
airflow dags trigger fraud_detection_monitoring
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

**✅ ALL PHASES COMPLETE** (17/17 phases):

**Core Pipeline**:
- ✅ Data pipeline (generation, validation, preprocessing)
- ✅ ML training (4 algorithms: XGBoost, LightGBM, RandomForest, LogisticRegression)
- ✅ Model registry and versioning (MLflow with stage promotion)
- ✅ FastAPI serving layer (<100ms p95 latency)
- ✅ Drift detection and monitoring (EvidentlyAI)

**Orchestration**:
- ✅ Airflow orchestration (3 production DAGs):
  - data_pipeline_dag.py (daily data generation)
  - training_pipeline_dag.py (weekly model training)
  - monitoring_dag.py (hourly drift detection with auto-retraining)

**Infrastructure**:
- ✅ Docker Compose (9 services: postgres, minio, mlflow, airflow×2, fastapi, prometheus, grafana)
- ✅ Monitoring stack (Prometheus + Grafana dashboards)
- ✅ AWS Terraform (complete infrastructure: ECS Fargate, RDS, S3, ALB, ECR, CloudWatch)
- ✅ GCP deployment guide (Cloud Run, Cloud SQL, GCS equivalents)
- ✅ Azure deployment guide (Container Apps, PostgreSQL Flexible, Blob Storage)

**Quality & Automation**:
- ✅ CI/CD (GitHub Actions):
  - ci.yml: 7 jobs (lint, test-unit, test-integration, docker-build, security-scan, test-report, badge)
  - cd.yml: 5 jobs (build-push, deploy-staging, approve, deploy-production, tag-release)
- ✅ Testing (60%+ coverage):
  - 30+ unit tests
  - 15+ integration tests
  - Load testing (100 RPS validated)
- ✅ Pre-commit hooks (black, flake8, isort, mypy, bandit)

**Documentation**:
- ✅ Comprehensive documentation (7 files, 15,000+ lines):
  - README.md (quick start)
  - GUIDE.md (technical deep dive, 5000+ words)
  - INTERVIEW_PREP_GUIDE.md (4-week learning path, 50+ Q&A, demo script)
  - DEPLOYMENT_GUIDE.md (AWS deployment, 50 pages)
  - CONCEPTS_AND_TECHNOLOGIES.md (11 technologies explained, AWS+GCP+Azure, 5000+ lines)
  - ENTERPRISE_GRADE_COMPLETE.md (project summary)
  - CLAUDE.md (this file)

## Documentation Library

### Core Documentation (Read in Order)

1. **README.md** (5 min read)
   - Project overview and architecture
   - Quick start guide
   - Tech stack summary
   - Key features and badges

2. **GUIDE.md** (30 min read)
   - Technical deep dive (5000+ words)
   - Implementation details for each component
   - Data generation, model training, API serving
   - Airflow DAGs, monitoring, deployment

3. **CONCEPTS_AND_TECHNOLOGIES.md** (4 hours study)
   - **Most comprehensive resource (5000+ lines)**
   - 11 major technology sections explained:
     - Apache Airflow & DAGs
     - MLflow (experiment tracking, model registry)
     - FastAPI (async API serving)
     - Docker & Docker Compose
     - Terraform (Infrastructure as Code)
     - GitHub Actions (CI/CD)
     - EvidentlyAI (drift detection)
     - AWS Services (ECS, RDS, S3, ALB, ECR, CloudWatch, IAM)
     - GCP Alternative (Cloud Run, Cloud SQL, GCS)
     - Azure Alternative (Container Apps, PostgreSQL Flexible, Blob Storage)
     - Monitoring (Prometheus + Grafana)
     - Production-Grade Architecture
   - Multi-cloud comparison (AWS vs GCP vs Azure)
   - Cost analysis: AWS $304/mo, GCP $209/mo, Azure $231/mo
   - When to choose which cloud
   - 60+ code examples, 40+ comparison tables

4. **INTERVIEW_PREP_GUIDE.md** (2 hours read)
   - 4-week learning path for interview preparation
   - 50+ interview Q&A covering:
     - Architecture and design decisions
     - MLflow and experiment tracking
     - Orchestration and Airflow
     - Monitoring and drift detection
     - Scalability and production practices
     - Business impact ($6.8M savings, 34x ROI)
   - 15-minute demo script
   - Interview readiness checklist

5. **DEPLOYMENT_GUIDE.md** (1 hour read)
   - AWS deployment step-by-step (50 pages)
   - Prerequisites and setup
   - Terraform commands and workflow
   - Docker image building and pushing to ECR
   - GitHub Actions setup and secrets
   - Verification and smoke tests
   - Monitoring and observability
   - Troubleshooting common issues
   - Cost optimization strategies

6. **ENTERPRISE_GRADE_COMPLETE.md** (15 min read)
   - Complete project summary
   - What makes this production-grade
   - All features and components
   - Statistics and metrics
   - Learning resources
   - Next steps for interviews

7. **CLAUDE.md** (10 min read)
   - This file
   - Instructions for AI assistants working with the codebase
   - Development workflow, common commands
   - Architecture patterns, troubleshooting

### Configuration Files

- `docker-compose.yml` - All 9 service configurations
- `config/settings.py` - Centralized configuration with Pydantic
- `.env.example` - Environment variables template
- `Makefile` - Common development commands

### Cloud Deployment

- `deployment/aws/terraform/` - Complete AWS infrastructure (ECS, RDS, S3, ALB)
- `deployment/aws/DEPLOYMENT_GUIDE.md` - AWS deployment instructions
- `CONCEPTS_AND_TECHNOLOGIES.md` sections 8.8-8.10 - GCP and Azure alternatives

### CI/CD

- `.github/workflows/ci.yml` - CI pipeline (lint, test, build, scan)
- `.github/workflows/cd.yml` - CD pipeline (deploy staging/production)
- `.pre-commit-config.yaml` - Pre-commit hooks configuration

### Additional Resources

- `CONTRIBUTING.md` - Contribution guidelines, code of conduct
- `LICENSE` - MIT License
- `.github/BRANCH_PROTECTION_SETUP.md` - GitHub branch protection guide

## Learning Path for New Contributors

**Week 1: Understanding the System**
1. Read README.md (overview)
2. Read GUIDE.md (technical details)
3. Run locally with `make docker-up`
4. Test API with FastAPI docs at http://localhost:8000/docs
5. Explore MLflow UI at http://localhost:5000
6. Trigger Airflow DAGs at http://localhost:8080

**Week 2: Deep Dive on Technologies**
1. Study CONCEPTS_AND_TECHNOLOGIES.md sections 1-5 (core tech)
2. Study CONCEPTS_AND_TECHNOLOGIES.md sections 6-9 (CI/CD, cloud, monitoring)
3. Understand multi-cloud comparison (section 8.10)
4. Review Terraform configuration in `deployment/aws/terraform/`

**Week 3: Interview Preparation**
1. Read INTERVIEW_PREP_GUIDE.md
2. Practice explaining architecture (15-minute demo)
3. Memorize key metrics and trade-offs
4. Review 50+ interview Q&A
5. Practice on whiteboard or with friend

**Week 4: Deployment and Practice**
1. Follow DEPLOYMENT_GUIDE.md to deploy on AWS
2. Set up CI/CD with GitHub Actions
3. Run full E2E tests
4. Practice interview demo 5+ times
5. Review business impact framing ($6.8M ROI)
