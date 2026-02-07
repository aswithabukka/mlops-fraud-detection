# MLOps Concepts & Technologies Guide

**Complete guide to understanding every technology, concept, and design decision in this project.**

This guide explains the "why" and "how" behind every component, making you interview-ready.

---

## 📚 Table of Contents

1. [Apache Airflow & DAGs](#apache-airflow--dags)
2. [MLflow](#mlflow)
3. [FastAPI](#fastapi)
4. [Docker & Docker Compose](#docker--docker-compose)
5. [Terraform](#terraform)
6. [GitHub Actions](#github-actions)
7. [EvidentlyAI](#evidentlyai)
8. [AWS Services](#aws-services)
9. [Monitoring (Prometheus & Grafana)](#monitoring-prometheus--grafana)
10. [Production-Grade Architecture](#production-grade-architecture)
11. [Why This Stack?](#why-this-stack)

---

## 1. Apache Airflow & DAGs

### What is Apache Airflow?

**Apache Airflow** is an open-source **workflow orchestration platform** for authoring, scheduling, and monitoring complex data pipelines.

**In simple terms**: Airflow is like a smart scheduler that runs your ML pipeline tasks in the right order, at the right time, and handles failures gracefully.

### Why Do We Use Airflow?

#### Problems Without Airflow:
```bash
# Manual approach (BAD):
# 9:00 AM - Run data generation script
python generate_data.py

# Wait... is it done? Check manually
# 9:30 AM - Run preprocessing
python preprocess_data.py

# Wait again...
# 10:00 AM - Train model
python train_model.py

# What if preprocessing fails? Manual retry!
# What if I'm on vacation? Pipeline doesn't run!
```

#### With Airflow (GOOD):
- ✅ **Automatic scheduling**: Runs daily/weekly/hourly without manual intervention
- ✅ **Dependency management**: Task B waits for Task A to complete
- ✅ **Error handling**: Automatic retries, failure alerts
- ✅ **Monitoring**: See which tasks succeeded/failed in UI
- ✅ **Scalability**: Distribute tasks across multiple workers
- ✅ **Version control**: DAGs are code, committed to git

### What is a DAG?

**DAG** = **Directed Acyclic Graph**

**Components**:
- **Directed**: Tasks flow in one direction (A → B → C)
- **Acyclic**: No loops (can't go back to previous tasks)
- **Graph**: Network of tasks with dependencies

**Visual Example**:
```
Data Pipeline DAG:
┌─────────────────┐
│  Generate Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate Data  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocess Data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update Reference│
└─────────────────┘

If Validate fails, Preprocess won't run!
```

### Key Airflow Components

#### 1. **DAG** (Directed Acyclic Graph)
The workflow definition - what tasks to run and in what order.

```python
from airflow import DAG

dag = DAG(
    'data_pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['data', 'preprocessing'],
)
```

#### 2. **Operator** (Task)
A unit of work. Different operators for different tasks:
- **PythonOperator**: Run Python function
- **BashOperator**: Run bash command
- **EmailOperator**: Send email
- **Custom Operators**: For specific use cases

```python
generate_task = PythonOperator(
    task_id='generate_data',
    python_callable=generate_data_function,
    dag=dag,
)
```

#### 3. **Task Dependencies**
Define execution order using `>>` or `<<`:

```python
generate >> validate >> preprocess >> update_reference
# Read as: generate THEN validate THEN preprocess THEN update_reference
```

#### 4. **XCom** (Cross-Communication)
Share data between tasks:

```python
# Task A pushes data
context['task_instance'].xcom_push(key='data_path', value='/path/to/data.csv')

# Task B pulls data
data_path = context['task_instance'].xcom_pull(
    task_ids='task_a',
    key='data_path'
)
```

#### 5. **Scheduler**
Background service that:
- Monitors DAGs for tasks ready to run
- Checks schedule intervals
- Submits tasks to executor

#### 6. **Executor**
Determines how tasks are run:
- **SequentialExecutor**: One task at a time (dev only)
- **LocalExecutor**: Multiple tasks locally (our choice)
- **CeleryExecutor**: Distributed across workers (production)
- **KubernetesExecutor**: Each task in separate K8s pod

#### 7. **Webserver**
UI for monitoring DAGs, tasks, logs, and metrics.

### Our 3 DAGs

#### 1. **Data Pipeline DAG** (`data_pipeline_dag.py`)
**Purpose**: Generate and preprocess data daily

**Schedule**: Daily at 2 AM (`0 2 * * *`)

**Tasks**:
1. **setup_directories** - Create data folders
2. **generate_data** - Create 100K transactions
3. **validate_data** - Check schema with Pandera
4. **preprocess_data** - Feature engineering
5. **update_reference_dataset** - Update drift detection baseline
6. **generate_quality_report** - Data quality metrics

**Why this matters**:
- Fresh data every day for training
- Catch data quality issues early
- Maintain reference dataset for drift detection

#### 2. **Training Pipeline DAG** (`training_pipeline_dag.py`)
**Purpose**: Train and register fraud detection models

**Schedule**: Weekly on Sundays at 3 AM (`0 3 * * 0`)

**Tasks**:
1. **generate_training_data** - Load recent data
2. **train_models** - Train ML models with MLflow
3. **notify_completion** - Alert team

**Why this matters**:
- Regular model updates with fresh patterns
- Captures evolving fraud behavior
- Automated model lifecycle

#### 3. **Monitoring Pipeline DAG** (`monitoring_dag.py`)
**Purpose**: Detect drift and trigger retraining

**Schedule**: Hourly (`0 * * * *`)

**Tasks**:
1. **collect_predictions** - Gather recent API predictions
2. **detect_drift** - Check for distribution shifts
3. **check_performance** - Validate model metrics
4. **send_drift_alert** - Notify if drift detected
5. **generate_monitoring_report** - Status summary
6. **check_severe_drift** - Sensor for threshold
7. **trigger_retraining** - Auto-trigger training DAG if drift > 0.2

**Why this matters**:
- **Event-driven retraining** (not just scheduled)
- Fraud patterns change constantly
- System adapts automatically within hours

### When to Use Airflow?

**Use Airflow when**:
- ✅ Multiple tasks with dependencies
- ✅ Need scheduling (daily, weekly, hourly)
- ✅ Long-running workflows (hours)
- ✅ Require retry logic
- ✅ Need monitoring and alerting
- ✅ Want event-driven triggers

**Don't use Airflow for**:
- ❌ Real-time processing (<1 second latency)
- ❌ Single standalone scripts
- ❌ Event streaming (use Kafka instead)
- ❌ Simple cron jobs (use cron)

### Airflow vs Alternatives

| Tool | Best For | Our Choice |
|------|----------|------------|
| **Airflow** | Batch workflows, ML pipelines | ✅ YES |
| **Prefect** | Modern API, easier to use | Alternative |
| **Kubeflow** | Kubernetes-native ML | Too complex for this |
| **Luigi** | Batch jobs, older | Less features |
| **Cron** | Simple scheduled tasks | Too basic |
| **Dagster** | Data engineering, asset-oriented | Alternative |

---

## 2. MLflow

### What is MLflow?

**MLflow** is an open-source platform for managing the **complete ML lifecycle**, including:
1. Experiment tracking
2. Model packaging
3. Model registry
4. Model deployment

**In simple terms**: MLflow is like GitHub for ML models - it versions, tracks, and manages all your experiments and models.

### Why Do We Use MLflow?

#### Problems Without MLflow:
```python
# Without MLflow (CHAOS):
# Experiment 1
model = train_model(max_depth=5, learning_rate=0.1)
# What was the accuracy? Forgot to save!

# Experiment 2 (next day)
model = train_model(max_depth=10, learning_rate=0.01)
# Wait, what parameters did I use yesterday?

# Model saved as model_v1.pkl, model_v2_final.pkl, model_v3_FINAL_FINAL.pkl
# Which one is in production? No idea!
```

#### With MLflow (ORGANIZED):
```python
# With MLflow (CLEAN):
with mlflow.start_run():
    mlflow.log_param("max_depth", 5)
    mlflow.log_param("learning_rate", 0.1)

    model = train_model(...)

    mlflow.log_metric("accuracy", 0.87)
    mlflow.log_metric("auc_roc", 0.92)
    mlflow.sklearn.log_model(model, "model")

# 6 months later: "What model did we use in Q1?"
# Answer: Check MLflow UI - all experiments tracked!
```

### Key MLflow Components

#### 1. **MLflow Tracking**
Records experiments: parameters, metrics, artifacts, code version.

**What gets tracked**:
- **Parameters**: Hyperparameters (max_depth=5, learning_rate=0.1)
- **Metrics**: Results (accuracy=0.87, auc_roc=0.92)
- **Artifacts**: Files (model.pkl, plots, preprocessor)
- **Tags**: Metadata (environment=dev, team=ml-team)
- **Code**: Git commit SHA

**Example**:
```python
import mlflow

mlflow.set_experiment("fraud_detection")

with mlflow.start_run(run_name="xgboost_experiment_1"):
    # Log hyperparameters
    mlflow.log_param("algorithm", "xgboost")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_estimators", 100)

    # Train model
    model = XGBClassifier(max_depth=10, n_estimators=100)
    model.fit(X_train, y_train)

    # Log metrics
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    mlflow.log_metric("auc_roc", auc)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))

    # Log model and artifacts
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("confusion_matrix.png")
```

#### 2. **MLflow Projects**
Package ML code in reusable, reproducible format.

**MLproject file**:
```yaml
name: fraud_detection
conda_env: conda.yaml
entry_points:
  train:
    parameters:
      max_depth: {type: int, default: 10}
    command: "python train.py --max-depth {max_depth}"
```

#### 3. **MLflow Models**
Standard format for packaging models for deployment.

**Benefits**:
- Works with multiple frameworks (sklearn, XGBoost, PyTorch, TensorFlow)
- Includes dependencies and preprocessing
- Can deploy anywhere (REST API, cloud, edge)

```python
# Save model with signature (input/output schema)
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))

mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=X_train[:5],
)
```

#### 4. **MLflow Model Registry**
Centralized model store with versioning and stage management.

**Model Lifecycle**:
```
None → Staging → Production → Archived
```

**Stages**:
- **None**: Just registered, not tested
- **Staging**: Under validation, running tests
- **Production**: Serving live traffic
- **Archived**: Deprecated, kept for audit

**Why this matters**:
- **Atomic updates**: Change Production alias without code changes
- **Rollback**: Instantly switch back to previous version
- **A/B testing**: Serve Staging to 10%, Production to 90%
- **Audit trail**: See who promoted model and when

**Example**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
model_uri = f"runs:/{run_id}/model"
result = client.create_registered_model("fraud_classifier")

model_version = client.create_model_version(
    name="fraud_classifier",
    source=model_uri,
    run_id=run_id
)

# Promote to Production
client.transition_model_version_stage(
    name="fraud_classifier",
    version=model_version.version,
    stage="Production"
)

# In FastAPI, load Production model:
model = mlflow.sklearn.load_model("models:/fraud_classifier/Production")
# Model updates automatically when you promote new version!
```

### How We Use MLflow

#### Experiment Tracking
```python
# In trainer.py
for algorithm in ['logistic', 'rf', 'xgboost', 'lightgbm']:
    with mlflow.start_run(run_name=f"{algorithm}_experiment"):
        mlflow.log_param("algorithm", algorithm)
        mlflow.log_param("handle_imbalance", True)

        model = FraudClassifier(algorithm=algorithm)
        model.fit(X_train, y_train)

        metrics = model.evaluate(X_test, y_test)
        mlflow.log_metrics(metrics)

        mlflow.sklearn.log_model(model.model, "model")

# Result: 4 experiments tracked, easy to compare in MLflow UI
```

#### Model Registry
```python
# Register best model
best_model_uri = get_best_run_id()  # Highest AUC-PR
client.create_model_version(
    name="fraud_classifier",
    source=best_model_uri
)

# Promote after validation
client.transition_model_version_stage(
    name="fraud_classifier",
    version=latest_version,
    stage="Production"
)
```

#### Model Serving
```python
# In FastAPI app.py
from mlflow.sklearn import load_model

# Load Production model on startup
model = load_model("models:/fraud_classifier/Production")

@app.post("/predict")
async def predict(transaction: TransactionRequest):
    # Use loaded model
    prediction = model.predict(transaction.to_df())
    return {"is_fraud": bool(prediction[0])}
```

### When to Use MLflow?

**Use MLflow when**:
- ✅ Running many experiments
- ✅ Need to track hyperparameters and results
- ✅ Multiple people training models
- ✅ Want model versioning
- ✅ Need model governance (who deployed what when)
- ✅ Require reproducibility

**Don't use MLflow for**:
- ❌ Single model, no iteration
- ❌ No experimentation phase
- ❌ Simple rule-based system

### MLflow vs Alternatives

| Tool | Best For | Our Choice |
|------|----------|------------|
| **MLflow** | Open-source, self-hosted, model registry | ✅ YES |
| **Weights & Biases** | Great UI, collaboration, cloud | Vendor lock-in |
| **Neptune** | Metadata tracking, query | Less mature registry |
| **Comet** | Similar to W&B | Paid service |
| **Sacred** | Experiment logging only | No model registry |

---

## 3. FastAPI

### What is FastAPI?

**FastAPI** is a modern, high-performance **web framework** for building APIs with Python 3.7+.

**In simple terms**: FastAPI lets you serve your ML model as a web service that other applications can call over HTTP.

### Why Do We Use FastAPI?

#### Problems Without API:
```python
# Without API (BAD):
# Data scientist has trained model on their laptop
model = joblib.load('model.pkl')

# Mobile app team: "How do we use this model?"
# Answer: "Uh... copy the pickle file and load it in Python?"
# Mobile app: "We code in Swift, not Python!"
# 😞
```

#### With FastAPI (GOOD):
```python
# With FastAPI
@app.post("/predict")
async def predict(transaction: dict):
    prediction = model.predict([transaction])
    return {"is_fraud": bool(prediction[0])}

# Mobile app: Makes HTTP request
# curl -X POST http://api.example.com/predict -d '{"amount": 100}'
# Response: {"is_fraud": false}
# ✅ Any language can use it!
```

### Why FastAPI (not Flask/Django)?

#### Comparison:

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| **Speed** | ⚡ Very Fast | 🐢 Slow | 🐢 Slow |
| **Async** | ✅ Yes | ❌ No | ⚠️ Limited |
| **Type Safety** | ✅ Pydantic | ❌ No | ⚠️ Some |
| **Auto Docs** | ✅ Yes (Swagger) | ❌ Manual | ❌ Manual |
| **Learning Curve** | Easy | Easy | Hard |
| **Use Case** | APIs, ML serving | Websites, APIs | Full web apps |

**Why FastAPI wins for ML**:
1. **Async**: Handle 100s of concurrent requests
2. **Type safety**: Pydantic validates requests automatically
3. **Auto docs**: Swagger UI generated automatically
4. **Performance**: Built on Starlette + Uvicorn (fast!)
5. **Modern**: Uses Python 3.10+ features (type hints)

### Key FastAPI Components

#### 1. **Path Operation (Endpoint)**
A function that handles HTTP requests.

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/predict")  # POST request to /predict
async def predict(transaction: TransactionRequest):
    # Handle request
    return {"is_fraud": False}
```

#### 2. **Pydantic Models (Request/Response Validation)**
Automatic data validation and documentation.

```python
from pydantic import BaseModel, Field

class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str
    hour_of_day: int = Field(..., ge=0, le=23)

    class Config:
        schema_extra = {
            "example": {
                "amount": 100.50,
                "merchant_category": "retail",
                "hour_of_day": 14
            }
        }

# If request has invalid data:
# - amount = -10 → Validation error: must be > 0
# - hour_of_day = 25 → Validation error: must be 0-23
```

#### 3. **Dependency Injection**
Reusable components across endpoints.

```python
from fastapi import Depends

def get_model():
    # Load model once, reuse for all requests
    return mlflow.sklearn.load_model("models:/fraud_classifier/Production")

@app.post("/predict")
async def predict(
    transaction: TransactionRequest,
    model=Depends(get_model)  # Inject dependency
):
    prediction = model.predict(transaction.to_df())
    return {"is_fraud": bool(prediction[0])}
```

#### 4. **Auto-Generated Documentation**
Swagger UI and ReDoc generated automatically.

Access:
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

**Zero code needed** - FastAPI generates docs from:
- Endpoint definitions
- Pydantic models
- Docstrings
- Type hints

#### 5. **Async Support**
Handle multiple requests concurrently.

```python
import asyncio

@app.post("/predict")
async def predict(transaction: TransactionRequest):
    # Can handle 100s of requests concurrently
    # While waiting for database, other requests processed
    result = await database.fetch(transaction.id)
    prediction = model.predict(transaction.to_df())
    return {"is_fraud": bool(prediction[0])}
```

### Our FastAPI Implementation

#### Endpoints:

1. **`GET /`** - Root endpoint, API info
2. **`GET /health`** - Health check (for load balancer)
3. **`POST /predict`** - Single transaction prediction
4. **`POST /predict/batch`** - Batch predictions (100+ transactions)
5. **`GET /metrics`** - Prometheus metrics

**Example**:
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.post("/predict")
async def predict(transaction: TransactionRequest):
    # Validate request (automatic with Pydantic)
    # Preprocess
    features = preprocessor.transform(transaction.to_df())

    # Predict
    probability = model.predict_proba(features)[0, 1]
    is_fraud = probability > FRAUD_THRESHOLD

    # Log prediction
    logger.info(f"Prediction: {is_fraud}, prob={probability:.3f}")

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=probability,
        confidence="high" if abs(probability - 0.5) > 0.3 else "medium"
    )
```

### Production Features

#### 1. **Model Caching**
Load model once on startup, not per request:

```python
model = None

@app.on_event("startup")
async def load_model():
    global model
    model = mlflow.sklearn.load_model("models:/fraud_classifier/Production")
    logger.info("Model loaded successfully")
```

#### 2. **Request Logging**
Track all predictions:

```python
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    return response
```

#### 3. **Error Handling**
Graceful failures:

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(transaction: TransactionRequest):
    try:
        prediction = model.predict(...)
        return {"is_fraud": bool(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")
```

#### 4. **CORS (Cross-Origin Resource Sharing)**
Allow web apps to call API:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### When to Use FastAPI?

**Use FastAPI when**:
- ✅ Building ML model API
- ✅ Need high performance (100+ RPS)
- ✅ Want async support
- ✅ Type safety and validation important
- ✅ Need auto-generated docs

**Don't use FastAPI for**:
- ❌ Full web applications (use Django)
- ❌ Server-side rendering (use Flask/Django)
- ❌ Legacy Python <3.7

---

## 4. Docker & Docker Compose

### What is Docker?

**Docker** is a platform for **containerizing** applications - packaging code and all dependencies into a standardized unit.

**In simple terms**: Docker is like a shipping container for software - same container runs anywhere (laptop, server, cloud).

### Why Do We Use Docker?

#### Problems Without Docker:
```bash
# On your laptop (works):
python app.py
# ✅ "It works on my machine!"

# On production server (fails):
python app.py
# ❌ Error: ModuleNotFoundError: No module named 'sklearn'
# ❌ Error: Python 3.8 required, have 3.7
# ❌ Error: PostgreSQL connection refused
# 😞 "But it works on my machine!"
```

#### With Docker (FIXED):
```dockerfile
FROM python:3.10-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Run app
CMD ["python", "app.py"]
```

```bash
# On your laptop:
docker run my-app
# ✅ Works!

# On production:
docker run my-app
# ✅ Works! Same environment!
```

### Key Docker Concepts

#### 1. **Container**
Running instance of an image. Like a lightweight VM but faster.

- **Isolated**: Has its own filesystem, network, processes
- **Portable**: Runs anywhere Docker runs
- **Ephemeral**: Can be destroyed and recreated instantly

#### 2. **Image**
Template for creating containers. Like a recipe.

Built from **Dockerfile**:
```dockerfile
FROM python:3.10-slim  # Base image

WORKDIR /app  # Set working directory

COPY requirements.txt .  # Copy dependency file
RUN pip install -r requirements.txt  # Install dependencies

COPY . .  # Copy application code

EXPOSE 8000  # Document port

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]  # Start command
```

#### 3. **Dockerfile**
Instructions for building an image.

**Best practices**:
- Use specific base image versions (python:3.10-slim, not python:latest)
- Multi-stage builds for smaller images
- Order layers by change frequency (dependencies before code)
- Use .dockerignore to exclude files

**Example - Multi-stage Build**:
```dockerfile
# Stage 1: Build
FROM python:3.10 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

#### 4. **Volume**
Persistent storage for containers.

```yaml
volumes:
  - ./data:/app/data  # Host path : Container path
  - postgres_data:/var/lib/postgresql/data  # Named volume
```

**Why volumes?**
- Containers are ephemeral (destroyed → recreated)
- Volumes persist data across container restarts
- Share data between containers

### What is Docker Compose?

**Docker Compose** manages **multi-container** applications.

**In simple terms**: Docker Compose orchestrates multiple services (API, database, cache) working together.

### Why Do We Use Docker Compose?

#### Problems Without Compose:
```bash
# Start PostgreSQL
docker run -d --name postgres -e POSTGRES_PASSWORD=pwd postgres

# Start Redis
docker run -d --name redis redis

# Start MLflow
docker run -d --name mlflow --link postgres mlflow-server

# Start Airflow webserver
docker run -d --name airflow-web --link postgres airflow

# Start Airflow scheduler
docker run -d --name airflow-scheduler --link postgres airflow

# Start FastAPI
docker run -d --name api --link mlflow fastapi

# 😫 Managing 6+ containers manually is painful!
# - What if one crashes? Restart manually!
# - How do they find each other? Manual linking!
# - Want to stop everything? 6 separate commands!
```

#### With Docker Compose (EASY):
```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_PASSWORD: mlflow

  mlflow:
    build: ./mlflow
    depends_on:
      - postgres

  airflow-webserver:
    build: ./airflow
    depends_on:
      - postgres

  fastapi:
    build: ./docker
    depends_on:
      - mlflow
```

```bash
# One command to start ALL services:
docker-compose up

# One command to stop ALL:
docker-compose down

# ✅ Easy!
```

### Our Docker Compose Setup

**9 Services**:
1. **postgres** - Database for Airflow & MLflow
2. **minio** - S3-compatible storage (local)
3. **mlflow** - Experiment tracking server
4. **airflow-webserver** - Airflow UI
5. **airflow-scheduler** - Airflow task executor
6. **fastapi** - Prediction API
7. **prometheus** - Metrics collection
8. **grafana** - Metrics visualization
9. **redis** (optional) - Caching

**Key Configuration**:
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "mlflow"]
      interval: 10s
      timeout: 5s
      retries: 5

  mlflow:
    build: ./mlflow
    ports:
      - "5000:5000"
    environment:
      BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
      ARTIFACT_ROOT: s3://mlflow-artifacts
    depends_on:
      postgres:
        condition: service_healthy

  fastapi:
    build:
      context: .
      dockerfile: ./docker/fastapi.Dockerfile
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow

volumes:
  postgres_data:
  minio_data:

networks:
  default:
    name: mlops-network
```

### Docker Benefits for MLOps

1. **Reproducibility**: Same environment everywhere
2. **Isolation**: Services don't interfere
3. **Portability**: Works on laptop, server, cloud
4. **Scalability**: Easy to scale services
5. **Development**: Matches production exactly

### When to Use Docker?

**Use Docker when**:
- ✅ Multiple services/dependencies
- ✅ Need environment consistency
- ✅ Deploying to cloud (ECS, Kubernetes)
- ✅ Team collaboration (same environment for all)

**Don't use Docker for**:
- ❌ Simple scripts (overkill)
- ❌ GUI applications (tricky)
- ❌ Hardware-specific code

---

## 5. Terraform

### What is Terraform?

**Terraform** is an **Infrastructure as Code (IaC)** tool for building, changing, and versioning infrastructure.

**In simple terms**: Terraform lets you define cloud infrastructure (servers, databases, networks) in code files, then automatically creates it.

### Why Do We Use Terraform?

#### Problems Without Terraform (Manual AWS Setup):
```bash
# Manual AWS setup (PAINFUL):
1. Log into AWS Console
2. Click EC2 → Launch Instance → Select AMI → Configure → Launch
3. Click RDS → Create Database → Configure → Create
4. Click S3 → Create Bucket → Set permissions
5. Click VPC → Create VPC → Create Subnets → Create Route Tables
6. ... 50 more clicks ...

# ❌ Takes hours
# ❌ Error-prone (miss a setting?)
# ❌ Not reproducible (how to create staging?)
# ❌ No version control (who changed what?)
# ❌ Can't automate
```

#### With Terraform (AUTOMATED):
```hcl
# main.tf
resource "aws_instance" "api_server" {
  ami           = "ami-12345678"
  instance_type = "t3.small"
}

resource "aws_db_instance" "database" {
  engine         = "postgres"
  instance_class = "db.t3.micro"
}

resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
}
```

```bash
terraform apply
# ✅ Creates all resources in 5 minutes
# ✅ Repeatable (same result every time)
# ✅ Version controlled (git commit)
# ✅ Automated (CI/CD pipeline)
```

### Key Terraform Concepts

#### 1. **Resource**
Infrastructure component to create.

```hcl
resource "aws_instance" "web_server" {
  ami           = "ami-12345678"
  instance_type = "t3.micro"

  tags = {
    Name = "WebServer"
  }
}
```

#### 2. **Provider**
Plugin for cloud platform (AWS, Azure, GCP).

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}
```

#### 3. **Variable**
Input parameters for reusability.

```hcl
variable "environment" {
  type    = string
  default = "production"
}

variable "instance_type" {
  type = string
  default = "t3.small"
}

resource "aws_instance" "server" {
  instance_type = var.instance_type

  tags = {
    Environment = var.environment
  }
}
```

#### 4. **Output**
Values to display after apply.

```hcl
output "api_endpoint" {
  description = "FastAPI endpoint URL"
  value       = aws_lb.main.dns_name
}

output "database_endpoint" {
  description = "RDS endpoint"
  value       = aws_db_instance.main.endpoint
  sensitive   = true
}
```

#### 5. **State**
Terraform tracks what it created in a state file.

**State file**: `terraform.tfstate`
- Maps config to real infrastructure
- Enables updates and deletions
- **Store remotely** (S3) for team collaboration

```hcl
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "prod/terraform.tfstate"
    region = "us-east-1"
    encrypt = true
  }
}
```

#### 6. **Module**
Reusable Terraform configuration.

```hcl
# modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block = var.vpc_cidr
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = var.public_subnet_cidr
}

# Use module in main.tf
module "vpc" {
  source = "./modules/vpc"

  vpc_cidr = "10.0.0.0/16"
  public_subnet_cidr = "10.0.1.0/24"
}
```

### Terraform Workflow

```bash
# 1. Initialize (download providers, setup backend)
terraform init

# 2. Plan (preview changes, no changes made)
terraform plan
# Shows: 23 resources to create

# 3. Apply (create infrastructure)
terraform apply
# Creates: VPC, subnets, EC2, RDS, S3, etc.

# 4. Outputs
terraform output
# api_endpoint = "http://my-alb-1234.us-east-1.elb.amazonaws.com"

# 5. Destroy (delete everything)
terraform destroy
# Deletes all 23 resources
```

### Our Terraform Setup

**What We Create** (20+ AWS resources):
- **VPC** with public/private subnets (2 AZs)
- **ECS Fargate** cluster with 4 services
- **RDS PostgreSQL** (Multi-AZ option)
- **S3 buckets** (data + artifacts)
- **ECR repositories** (4 Docker images)
- **Application Load Balancer**
- **CloudWatch** logs and alarms
- **SNS** topics for alerts
- **IAM** roles and policies
- **Security Groups**

**Structure**:
```
deployment/aws/terraform/
├── main.tf                 # Main configuration
├── variables.tf            # Input variables
├── outputs.tf              # Output values
├── terraform.tfvars        # Variable values
└── modules/
    ├── vpc/               # VPC module
    ├── s3/                # S3 buckets module
    ├── rds/               # Database module
    ├── ecs/               # ECS cluster module
    ├── ecs-service/       # ECS service template
    ├── alb/               # Load balancer module
    ├── security/          # Security groups module
    ├── iam/               # IAM roles module
    └── cloudwatch/        # Monitoring module
```

**Example Module** (VPC):
```hcl
# modules/vpc/main.tf
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-vpc"
  })
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = var.public_subnet_cidrs[count.index]
  availability_zone = var.availability_zones[count.index]

  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}
```

### Terraform Benefits

1. **Version Control**: Infrastructure changes tracked in git
2. **Collaboration**: Team can review infra changes like code
3. **Reproducibility**: Create identical staging/production
4. **Automation**: Deploy via CI/CD
5. **Safety**: Preview changes before applying
6. **Documentation**: Code is documentation

### When to Use Terraform?

**Use Terraform when**:
- ✅ Managing cloud infrastructure
- ✅ Need multiple environments (dev, staging, prod)
- ✅ Team collaboration on infrastructure
- ✅ Want infrastructure version control
- ✅ Complex infrastructure (10+ resources)

**Don't use Terraform for**:
- ❌ Application configuration (use Ansible)
- ❌ Simple single-server setup
- ❌ Real-time infrastructure changes

### Terraform vs Alternatives

| Tool | Type | Best For | Our Choice |
|------|------|----------|------------|
| **Terraform** | Multi-cloud IaC | AWS + others | ✅ YES |
| **CloudFormation** | AWS-native IaC | AWS only | AWS lock-in |
| **Pulumi** | IaC in programming languages | Complex logic | Less mature |
| **Ansible** | Configuration management | App config | Different purpose |
| **AWS CLI** | Command-line | Manual operations | Not IaC |

---

## 6. GitHub Actions - CI/CD Automation

### What is GitHub Actions?

**GitHub Actions** is GitHub's native CI/CD (Continuous Integration/Continuous Deployment) platform that automates software workflows directly from your repository.

**Key Concepts**:
- **Workflow**: Automated process defined in YAML (`.github/workflows/*.yml`)
- **Job**: A set of steps that execute on the same runner
- **Step**: Individual task (run command, use action)
- **Action**: Reusable unit of code (marketplace or custom)
- **Runner**: Server that runs workflows (GitHub-hosted or self-hosted)
- **Event**: Trigger that starts workflow (push, PR, schedule, manual)

### Why Use GitHub Actions?

1. **Native Integration**: Built into GitHub, no external CI tool setup
2. **Free for Public Repos**: Unlimited minutes for open source
3. **Matrix Builds**: Test across multiple Python versions, OS combinations
4. **Secrets Management**: Secure credential storage
5. **Rich Marketplace**: 10,000+ pre-built actions
6. **Fast Feedback**: Results shown directly in PRs

### GitHub Actions in Our Project

We have **2 workflows**:

#### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers**: On every push and pull request

**7 Jobs**:

```yaml
jobs:
  lint:
    # Code quality checks (black, flake8, isort, mypy, bandit)

  test-unit:
    # Run unit tests with coverage (must be >60%)

  test-integration:
    # Run integration tests with Docker services

  docker-build:
    # Build Docker images for all services

  security-scan:
    # Scan Docker images with Trivy for vulnerabilities

  test-report:
    # Generate test coverage report, upload to Codecov

  badge:
    # Update README badges with latest status
```

**Example Job** (test-unit):
```yaml
test-unit:
  runs-on: ubuntu-latest

  services:
    postgres:
      image: postgres:15
      env:
        POSTGRES_PASSWORD: postgres
      options: >-
        --health-cmd pg_isready
        --health-interval 10s

  steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Check coverage threshold
      run: |
        coverage report --fail-under=60

    - name: Upload to Codecov
      uses: codecov/codecov-action@v3
```

#### 2. CD Workflow (`.github/workflows/cd.yml`)

**Triggers**: On push to `main` branch (after CI passes)

**5 Jobs**:

```yaml
jobs:
  build-and-push:
    # Build Docker images, tag with SHA, push to AWS ECR

  deploy-staging:
    # Deploy to staging environment
    # Run smoke tests
    # Wait for health checks

  approve-production:
    # Manual approval gate (protected)
    # Team reviews staging before production

  deploy-production:
    # Blue-green deployment to production ECS
    # Gradual traffic shift (10% → 50% → 100%)
    # Monitor metrics during rollout

  tag-release:
    # Create git tag (v1.0.0, v1.0.1, etc.)
    # Generate release notes
```

**Example: Blue-Green Deployment**
```yaml
deploy-production:
  needs: approve-production
  runs-on: ubuntu-latest

  steps:
    - name: Deploy new version
      run: |
        # Create new task definition
        NEW_TASK_DEF=$(aws ecs register-task-definition \
          --cli-input-json file://task-def.json \
          --query 'taskDefinition.taskDefinitionArn' \
          --output text)

        # Update service (deploy new version alongside old)
        aws ecs update-service \
          --cluster mlops-cluster \
          --service fastapi \
          --task-definition $NEW_TASK_DEF \
          --deployment-configuration "minimumHealthyPercent=100,maximumPercent=200"

    - name: Monitor deployment
      run: |
        # Wait for new tasks to be healthy
        aws ecs wait services-stable \
          --cluster mlops-cluster \
          --services fastapi

        # Check CloudWatch metrics
        python scripts/check_metrics.py --threshold-errors 1%

    - name: Rollback on failure
      if: failure()
      run: |
        # Revert to previous task definition
        aws ecs update-service \
          --cluster mlops-cluster \
          --service fastapi \
          --task-definition $PREVIOUS_TASK_DEF \
          --force-new-deployment
```

### GitHub Actions vs Alternatives

| Tool | Hosting | Best For | Cost |
|------|---------|----------|------|
| **GitHub Actions** | GitHub | GitHub repos | Free (public) / $0.008/min (private) |
| **Jenkins** | Self-hosted | Enterprise, complex pipelines | Infrastructure cost only |
| **GitLab CI** | GitLab | GitLab repos | Free (limited) / $19/user/month |
| **CircleCI** | Cloud | Fast builds, Docker support | Free (limited) / $15/month |
| **Travis CI** | Cloud | Open source projects | Free (public) |

**We chose GitHub Actions** because:
- ✅ Native GitHub integration (no external setup)
- ✅ Free for our public repository
- ✅ Excellent Docker support
- ✅ Easy secrets management
- ✅ Rich marketplace for AWS deployment actions

---

## 7. EvidentlyAI - Data & Model Drift Detection

### What is EvidentlyAI?

**EvidentlyAI** is an open-source Python library for monitoring ML models in production. It detects **data drift** (when input data distribution changes) and **model performance degradation** (when predictions get worse).

**Problem it solves**:
- Real-world data changes over time (concept drift)
- For fraud detection, fraudsters adapt tactics → new fraud patterns emerge
- Models trained on old data become stale → prediction quality drops
- **Need automated detection** to trigger retraining

### Key Concepts

1. **Data Drift**: Input feature distributions change
   - Example: Average transaction amount shifts from $50 to $120
   - Detected using statistical tests (Kolmogorov-Smirnov, chi-squared)

2. **Prediction Drift**: Model output distribution changes
   - Example: Fraud rate predictions shift from 1% to 5%
   - May indicate data drift or model issues

3. **Model Performance Drift**: Accuracy/precision/recall decreases
   - Requires ground truth labels (actual fraud outcomes)
   - Example: Recall drops from 85% to 70%

4. **Reference Data**: Historical data used as baseline for comparison
   - Typically training data or recent production data
   - We update reference data weekly with latest transactions

### EvidentlyAI in Our Project

**File**: `src/monitoring/drift_detector.py`

**Class**: `DriftDetector`

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestShareOfDriftedColumns

class DriftDetector:
    def __init__(self, config: DriftConfig):
        self.drift_threshold = config.drift_threshold  # 0.15
        self.severe_drift_threshold = config.severe_drift_threshold  # 0.2

        # Create report with data drift preset
        self.report = Report(metrics=[
            DataDriftPreset(),  # Detects drift per column
            DataQualityPreset()  # Checks data quality
        ])

    def detect_drift(
        self,
        reference_data: pd.DataFrame,  # Training data
        current_data: pd.DataFrame,    # Recent predictions
        save_report: bool = True
    ) -> Dict:
        """
        Detect drift between reference and current data.

        Returns:
            {
                'drift_detected': True/False,
                'drift_score': 0.23,  # Share of drifted features
                'severe_drift': True/False,
                'drifted_features': ['amount', 'merchant_category'],
                'report_path': 'monitoring/reports/drift_20240107.html'
            }
        """
        # Run drift detection
        self.report.run(
            reference_data=reference_data,
            current_data=current_data
        )

        # Extract results
        results = self.report.as_dict()
        drift_share = results['metrics'][0]['result']['drift_share']
        drifted_columns = [
            col['column_name']
            for col in results['metrics'][0]['result']['drift_by_columns'].values()
            if col['drift_detected']
        ]

        # Determine severity
        drift_detected = drift_share > self.drift_threshold
        severe_drift = drift_share > self.severe_drift_threshold

        # Save HTML report
        if save_report:
            report_path = f"monitoring/reports/drift_{date.today()}.html"
            self.report.save_html(report_path)

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_share,
            'severe_drift': severe_drift,
            'drifted_features': drifted_columns,
            'report_path': report_path
        }
```

### How We Use It

**Airflow DAG**: `monitoring_dag.py` (runs hourly)

```python
@task
def detect_drift():
    """Check for drift in last hour's predictions."""
    # Load reference data (last 30 days)
    reference_df = load_data("s3://mlops-data/reference/fraud_reference.parquet")

    # Load current data (last 1 hour predictions)
    current_df = load_predictions_from_api_logs(hours=1)

    # Detect drift
    detector = DriftDetector()
    drift_result = detector.detect_drift(reference_df, current_df)

    if drift_result['severe_drift']:
        # Alert team
        send_slack_alert(f"🚨 Severe drift detected: {drift_result['drift_score']:.2%}")

        # Trigger retraining DAG
        trigger_dag('training_pipeline')

    return drift_result
```

### Statistical Tests Used

EvidentlyAI automatically selects tests based on data type:

| Data Type | Test | What it Checks |
|-----------|------|----------------|
| **Numerical** | Kolmogorov-Smirnov | Distribution similarity |
| **Categorical** | Chi-squared | Category frequency changes |
| **Binary** | Z-test | Proportion changes |

**Example**: Transaction amount (numerical)
- Reference data: mean=$50, std=$30
- Current data: mean=$75, std=$25
- K-S test p-value = 0.001 (p < 0.05) → **Drift detected** ✅

### Drift Report Example

EvidentlyAI generates interactive HTML reports:

```
Data Drift Report
─────────────────────────────────────
Dataset Drift: DETECTED (4 of 12 features)

Drifted Features:
┌─────────────────────┬────────┬──────────┬─────────┐
│ Feature             │ Type   │ P-Value  │ Drift   │
├─────────────────────┼────────┼──────────┼─────────┤
│ amount              │ num    │ 0.001    │ YES ⚠️  │
│ merchant_category   │ cat    │ 0.023    │ YES ⚠️  │
│ hour_of_day         │ num    │ 0.350    │ NO      │
│ distance_from_home  │ num    │ 0.008    │ YES ⚠️  │
│ transactions_24h    │ num    │ 0.002    │ YES ⚠️  │
└─────────────────────┴────────┴──────────┴─────────┘

Drift Score: 33% (4/12 features)
Recommendation: RETRAIN MODEL
```

### When to Use EvidentlyAI?

**Use EvidentlyAI when**:
- ✅ ML model in production
- ✅ Data distribution can change over time
- ✅ Need automated drift detection
- ✅ Want visual drift reports

**Don't need EvidentlyAI if**:
- ❌ Model not in production yet
- ❌ Static dataset (no new data)
- ❌ Manually reviewing predictions

### EvidentlyAI vs Alternatives

| Tool | Open Source | Best For | Complexity |
|------|-------------|----------|------------|
| **EvidentlyAI** | ✅ Yes | Drift detection, reports | Low |
| **Alibi Detect** | ✅ Yes | Advanced drift methods | Medium |
| **Whylabs** | ❌ Commercial | Enterprise monitoring | Low |
| **Arize AI** | ❌ Commercial | Full observability platform | High |
| **Custom Script** | ✅ DIY | Simple drift checks | Low |

**We chose EvidentlyAI** because:
- ✅ Open source and free
- ✅ Easy to integrate (just 10 lines of code)
- ✅ Beautiful HTML reports for stakeholders
- ✅ Multiple statistical tests out-of-the-box
- ✅ Active community and frequent updates

---

## 8. AWS Services - Cloud Infrastructure

Our MLOps pipeline runs on **AWS (Amazon Web Services)**, using managed services for scalability, reliability, and reduced operational overhead.

### AWS Services We Use

#### 8.1 Amazon ECS (Elastic Container Service) - Container Orchestration

**What**: Fully managed container orchestration service (runs Docker containers at scale)

**Why we use it**:
- ✅ No Kubernetes complexity (ECS is simpler)
- ✅ Deep AWS integration (ALB, CloudWatch, IAM)
- ✅ **Fargate mode**: Serverless compute (no EC2 management)
- ✅ Auto-scaling based on CPU/memory/requests
- ✅ Rolling updates with health checks

**Our ECS Setup**:
```
ECS Cluster: mlops-cluster
├── FastAPI Service (2-10 tasks)
│   ├── Task: 2 vCPU, 4 GB RAM
│   ├── Auto-scaling: target 70% CPU
│   └── ALB: Distributes traffic
├── MLflow Service (1 task)
│   └── Task: 1 vCPU, 2 GB RAM
├── Airflow Webserver (1 task)
└── Airflow Scheduler (1 task)
```

**ECS Key Concepts**:
- **Cluster**: Logical grouping of services
- **Service**: Maintains N running tasks (like ReplicaSet in Kubernetes)
- **Task**: Instance of container(s) running (like Pod in Kubernetes)
- **Task Definition**: Blueprint for task (image, CPU, memory, env vars)

**Example Task Definition** (FastAPI):
```json
{
  "family": "fastapi-fraud-detection",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "containerDefinitions": [
    {
      "name": "fastapi",
      "image": "123456789.dkr.ecr.us-east-1.amazonaws.com/fastapi:latest",
      "portMappings": [{"containerPort": 8000, "protocol": "tcp"}],
      "environment": [
        {"name": "MLFLOW_TRACKING_URI", "value": "http://mlflow:5000"},
        {"name": "MODEL_STAGE", "value": "Production"}
      ],
      "secrets": [
        {"name": "DB_PASSWORD", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/fastapi",
          "awslogs-region": "us-east-1"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

**Why ECS over Kubernetes?**
- Simpler for small teams (no K8s overhead)
- Cheaper (no EKS control plane cost: $0.10/hour = $72/month)
- Faster to deploy (less YAML complexity)
- Good enough for our scale (< 100 containers)

#### 8.2 Amazon RDS (Relational Database Service) - PostgreSQL

**What**: Managed PostgreSQL database (no manual backups, patching, or failover)

**Why we use it**:
- ✅ Automated backups (point-in-time recovery)
- ✅ Multi-AZ for high availability (automatic failover)
- ✅ Automated patching and maintenance
- ✅ Read replicas for scaling reads
- ✅ Encryption at rest and in transit

**Our RDS Setup**:
```
RDS Instance: mlops-postgres
├── Engine: PostgreSQL 15
├── Instance: db.t3.medium (2 vCPU, 4 GB RAM)
├── Storage: 100 GB SSD (auto-scaling to 500 GB)
├── Multi-AZ: Enabled (standby in different AZ)
├── Backups: Daily, 7-day retention
└── Databases:
    ├── mlflow (experiment tracking, model registry)
    └── airflow (DAG metadata, task history)
```

**Cost Optimization**:
- Use `db.t3.medium` (burstable) for dev/staging
- Reserve instance for production (40% savings)
- Stop non-production instances after hours

#### 8.3 Amazon S3 (Simple Storage Service) - Object Storage

**What**: Infinitely scalable object storage (files, images, videos, data)

**Why we use it**:
- ✅ 99.999999999% (11 nines) durability
- ✅ Unlimited storage (pay per GB)
- ✅ Versioning (recover deleted files)
- ✅ Lifecycle policies (auto-archive old data)
- ✅ Fast access from any AWS service

**Our S3 Buckets**:
```
S3 Buckets:
├── mlops-data-{account-id}
│   ├── raw/                  # Raw transaction data
│   ├── processed/            # Preprocessed features
│   ├── reference/            # Drift detection baseline
│   └── predictions/          # API prediction logs
├── mlops-mlflow-artifacts-{account-id}
│   ├── 0/                    # Experiment 0 artifacts
│   │   ├── model/            # Trained model files
│   │   ├── preprocessor/     # Preprocessing pipeline
│   │   └── metrics/          # Evaluation metrics
│   └── models/               # Model registry artifacts
└── mlops-terraform-state-{account-id}
    └── terraform.tfstate     # Infrastructure state
```

**Lifecycle Policy Example**:
```hcl
# Automatically move old data to cheaper storage
lifecycle_rule {
  enabled = true

  transition {
    days          = 90
    storage_class = "STANDARD_IA"  # Infrequent Access (50% cheaper)
  }

  transition {
    days          = 180
    storage_class = "GLACIER"  # Archive (80% cheaper)
  }

  expiration {
    days = 365  # Delete after 1 year
  }
}
```

**Cost**: ~$0.023/GB/month (Standard), ~$0.0125/GB/month (IA), ~$0.004/GB/month (Glacier)

#### 8.4 Application Load Balancer (ALB) - Traffic Distribution

**What**: Layer 7 load balancer (HTTP/HTTPS traffic routing)

**Why we use it**:
- ✅ Distributes traffic across multiple FastAPI tasks
- ✅ SSL/TLS termination (HTTPS encryption)
- ✅ Health checks (removes unhealthy targets)
- ✅ Path-based routing (`/predict` → FastAPI, `/mlflow` → MLflow)
- ✅ Sticky sessions (optional, for stateful apps)

**Our ALB Setup**:
```
ALB: mlops-alb
├── Listener: HTTPS (port 443)
│   ├── SSL Certificate: AWS ACM (free)
│   └── Target Group: fastapi-tg
│       ├── Health Check: GET /health every 30s
│       ├── Targets: 2-10 FastAPI tasks
│       └── Deregistration Delay: 30s
└── Listener: HTTP (port 80)
    └── Redirect to HTTPS (permanent)
```

**Health Check**:
```json
{
  "path": "/health",
  "interval": 30,
  "timeout": 5,
  "healthyThreshold": 2,
  "unhealthyThreshold": 3,
  "matcher": "200"
}
```

If FastAPI task fails health check 3 times, ALB stops sending traffic to it (automatic failover).

#### 8.5 Amazon ECR (Elastic Container Registry) - Docker Image Storage

**What**: Private Docker image registry (like Docker Hub, but private)

**Why we use it**:
- ✅ Private (images not publicly accessible)
- ✅ Image scanning for vulnerabilities (automatic CVE detection)
- ✅ Fast pulls from ECS (same AWS network)
- ✅ Lifecycle policies (delete old images)

**Our ECR Repositories**:
```
ECR Repositories:
├── mlops-fastapi
│   ├── latest (always points to production)
│   ├── v1.0.0
│   └── abc123def (commit SHA tags)
├── mlops-mlflow
├── mlops-airflow
└── mlops-training
```

**Image Scanning**:
```bash
# Push image
docker tag fastapi:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/mlops-fastapi:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/mlops-fastapi:latest

# Automatic scan results
# Critical: 0, High: 1, Medium: 3, Low: 12
# High vulnerability: CVE-2023-1234 in numpy 1.24.0 (upgrade to 1.24.3)
```

#### 8.6 CloudWatch - Logging & Monitoring

**What**: AWS observability service (logs, metrics, alarms)

**Why we use it**:
- ✅ Centralized logs from all ECS tasks
- ✅ Custom metrics (API latency, drift scores)
- ✅ Alarms (notify when errors spike)
- ✅ Dashboards (visualize system health)

**Our CloudWatch Setup**:
```
Log Groups:
├── /ecs/fastapi (API request logs)
├── /ecs/mlflow (MLflow server logs)
├── /ecs/airflow-scheduler (Airflow task logs)
└── /rds/postgresql (Database slow queries)

Alarms:
├── High API Error Rate (> 1%)
├── High API Latency (p95 > 200ms)
├── Low ECS CPU (< 20% → over-provisioned)
└── High Drift Score (> 0.2)
```

**Example Alarm** (API errors):
```hcl
resource "aws_cloudwatch_metric_alarm" "api_errors" {
  alarm_name          = "mlops-api-high-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "5XXError"
  namespace           = "AWS/ApplicationELB"
  period              = 300  # 5 minutes
  statistic           = "Sum"
  threshold           = 10  # More than 10 errors in 5 min

  alarm_actions = [aws_sns_topic.alerts.arn]  # Send to Slack/email
}
```

#### 8.7 IAM (Identity and Access Management) - Security & Permissions

**What**: AWS security service (who can do what)

**Why we use it**:
- ✅ Least privilege (services only get permissions they need)
- ✅ No hardcoded credentials (IAM roles instead)
- ✅ Audit trail (CloudTrail logs all API calls)

**Our IAM Roles**:
```
IAM Roles:
├── ECSTaskExecutionRole
│   └── Pull images from ECR, write logs to CloudWatch
├── FastAPITaskRole
│   └── Read from S3 (data), read/write MLflow (load models)
├── AirflowTaskRole
│   └── Read/write S3, trigger ECS tasks, invoke Lambda
└── TerraformDeployRole
    └── Full access (used by CI/CD to deploy infrastructure)
```

**Example IAM Policy** (FastAPI → S3):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::mlops-data-*/*",
        "arn:aws:s3:::mlops-mlflow-artifacts-*/*"
      ]
    }
  ]
}
```

FastAPI can **read** from S3, but **cannot delete or modify** (least privilege).

### AWS Architecture Diagram

```
                                   ┌─────────────────┐
                                   │   Internet      │
                                   └────────┬────────┘
                                            │
                          ┌─────────────────▼───────────────────┐
                          │  Application Load Balancer (ALB)    │
                          │  - SSL Termination                  │
                          │  - Health Checks                    │
                          └──────────────┬──────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                          │
     ┌────────▼────────┐       ┌────────▼────────┐       ┌────────▼────────┐
     │ FastAPI Task 1  │       │ FastAPI Task 2  │  ...  │ FastAPI Task N  │
     │ (ECS Fargate)   │       │ (ECS Fargate)   │       │ (ECS Fargate)   │
     └────────┬────────┘       └────────┬────────┘       └────────┬────────┘
              │                          │                          │
              └──────────────────────────┼──────────────────────────┘
                                         │
                          ┌──────────────▼────────────┐
                          │     Amazon RDS            │
                          │  (PostgreSQL Multi-AZ)    │
                          │  - MLflow metadata        │
                          │  - Airflow metadata       │
                          └──────────────┬────────────┘
                                         │
                          ┌──────────────▼────────────┐
                          │      Amazon S3            │
                          │  - Data (raw, processed)  │
                          │  - MLflow artifacts       │
                          │  - Prediction logs        │
                          └───────────────────────────┘
```

### AWS Cost Estimate (Monthly)

| Service | Configuration | Cost |
|---------|---------------|------|
| **ECS Fargate** | 4 tasks × 2 vCPU × 4 GB × 24/7 | $120 |
| **RDS PostgreSQL** | db.t3.medium Multi-AZ | $80 |
| **ALB** | 1 ALB | $20 |
| **S3** | 100 GB Standard + 500 GB IA | $5 |
| **ECR** | 50 GB images | $5 |
| **CloudWatch** | Logs (10 GB/month) | $5 |
| **Data Transfer** | 100 GB out to internet | $9 |
| **NAT Gateway** | 2 AZs | $60 |
| **Total** | | **~$304/month** |

**Cost Optimization Tips**:
- Use Fargate Spot (70% cheaper, may be interrupted)
- Stop non-prod environments after hours (50% savings)
- Reserved RDS instance (40% savings for 1-year commit)
- S3 Intelligent-Tiering (auto-moves to cheaper storage)
- Reduce NAT Gateway usage (use VPC endpoints for S3, ECR)

---

## 9. Monitoring Stack - Prometheus & Grafana

Real-time observability is critical for production ML systems. We need to know:
- Is the API responding? (uptime)
- How fast are predictions? (latency)
- Are predictions accurate? (model performance)
- Is data drifting? (drift scores)

We use **Prometheus** (metrics collection) + **Grafana** (visualization).

### 9.1 Prometheus - Metrics Collection

**What**: Open-source monitoring system that scrapes metrics from services and stores time-series data.

**Key Concepts**:
- **Metric**: Measurement over time (e.g., `api_requests_total`)
- **Scrape**: Prometheus pulls metrics from targets every 15s
- **Label**: Dimension to slice metrics (e.g., `endpoint="/predict"`)
- **Time Series**: Sequence of metric values over time

**Metric Types**:
1. **Counter**: Only increases (e.g., total requests, errors)
2. **Gauge**: Can go up/down (e.g., CPU usage, active connections)
3. **Histogram**: Distribution of values (e.g., request latency buckets)
4. **Summary**: Similar to histogram, calculates quantiles

**How Prometheus Works**:
```
┌──────────────┐      HTTP GET /metrics every 15s      ┌──────────────┐
│  Prometheus  │ ────────────────────────────────────> │   FastAPI    │
│   Server     │ <──────────────────────────────────── │  (port 8000) │
└──────────────┘      Returns metrics in text format   └──────────────┘

                      Example metrics:
                      api_requests_total{endpoint="/predict"} 1523
                      api_latency_seconds{endpoint="/predict",quantile="0.95"} 0.089
                      drift_score 0.12
```

**Prometheus Configuration** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s  # How often to scrape targets

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']  # FastAPI service

  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']

  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
```

**Instrumenting FastAPI** (`src/serving/api.py`):
```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
api_requests_total = Counter(
    'api_requests_total',
    'Total API requests',
    ['endpoint', 'method', 'status']
)

api_latency_seconds = Histogram(
    'api_latency_seconds',
    'API request latency',
    ['endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]  # Latency buckets
)

drift_score_gauge = Gauge(
    'drift_score',
    'Current data drift score'
)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Track requests and latency."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # Increment counter
    api_requests_total.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()

    # Record latency
    api_latency_seconds.labels(
        endpoint=request.url.path
    ).observe(duration)

    return response

@app.get("/metrics")
def metrics():
    """Expose metrics for Prometheus to scrape."""
    return Response(content=generate_latest(), media_type="text/plain")
```

**Querying Metrics in Prometheus**:
```promql
# Request rate (requests per second)
rate(api_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, api_latency_seconds_bucket)

# Error rate (percentage)
sum(rate(api_requests_total{status=~"5.."}[5m]))
/
sum(rate(api_requests_total[5m])) * 100

# Average drift score over last hour
avg_over_time(drift_score[1h])
```

### 9.2 Grafana - Visualization & Dashboards

**What**: Open-source platform for visualizing time-series metrics (connects to Prometheus).

**Key Concepts**:
- **Data Source**: Where metrics come from (Prometheus)
- **Dashboard**: Collection of panels
- **Panel**: Individual visualization (graph, gauge, table)
- **Query**: PromQL expression to fetch metrics
- **Alert**: Trigger notification when condition met

**Our Grafana Dashboards**:

#### Dashboard 1: API Performance
```
┌────────────────────────────────────────────────────────────────┐
│  API Performance Dashboard                         Last 6 hours │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Requests per Second                                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  /predict:  ████████████████████  42.3 req/s           │    │
│  │  /health:   ████                   3.1 req/s           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  P95 Latency                                                    │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  89ms  ──────────────▲───────▼────────                 │    │
│  │        target: <100ms │                                 │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Error Rate                                                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  0.3%  ───────────────────────────  (target: <1%)      │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

**Grafana Panel Configuration** (P95 Latency):
```json
{
  "title": "P95 Latency",
  "targets": [
    {
      "expr": "histogram_quantile(0.95, rate(api_latency_seconds_bucket[5m]))",
      "legendFormat": "{{endpoint}}"
    }
  ],
  "thresholds": [
    {"value": 100, "color": "green"},
    {"value": 200, "color": "yellow"},
    {"value": 500, "color": "red"}
  ]
}
```

#### Dashboard 2: Model Monitoring
```
┌────────────────────────────────────────────────────────────────┐
│  Model Monitoring Dashboard                        Last 24 hours │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prediction Distribution                                        │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Fraud:     1.2%  ████                                 │    │
│  │  Legit:    98.8%  ████████████████████████████████     │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Data Drift Score                                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  0.12  ────────────▲──────────  (threshold: 0.15)      │    │
│  │                    │                                    │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Drifted Features (last check)                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  ✅ amount, merchant_category, hour_of_day: OK         │    │
│  │  ⚠️  distance_from_home: DRIFTED (p=0.032)            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Model Performance (when labels available)                     │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Precision:  92.3%  (last 1000 labeled predictions)    │    │
│  │  Recall:     87.1%                                      │    │
│  │  F1:         89.6%                                      │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────┘
```

**Grafana Alerts** (Slack/Email notification):
```yaml
# Alert: High API Latency
alert: HighAPILatency
expr: histogram_quantile(0.95, rate(api_latency_seconds_bucket[5m])) > 0.2
for: 5m
labels:
  severity: warning
annotations:
  summary: "API latency is high (95th percentile > 200ms)"
  description: "Current P95 latency: {{ $value }}s"

# Alert: Severe Drift Detected
alert: SevereDrift
expr: drift_score > 0.2
for: 10m
labels:
  severity: critical
annotations:
  summary: "Severe data drift detected"
  description: "Drift score: {{ $value }} (threshold: 0.2)"
```

### Monitoring Workflow

```
1. FastAPI serves predictions
   └─> Updates Prometheus metrics (requests, latency, predictions)

2. Airflow monitoring DAG runs hourly
   └─> Detects drift using EvidentlyAI
   └─> Updates drift_score metric in Prometheus

3. Prometheus scrapes metrics every 15s
   └─> Stores in time-series database

4. Grafana queries Prometheus
   └─> Displays dashboards in real-time

5. If alert condition met (drift > 0.2)
   └─> Grafana sends alert to Slack
   └─> Airflow triggers retraining DAG
```

### Why Prometheus + Grafana?

| Tool | Alternative | Why Our Choice |
|------|-------------|----------------|
| **Prometheus** | CloudWatch, Datadog | ✅ Open source, pull-based (no agent), powerful queries |
| **Grafana** | Kibana, Datadog | ✅ Open source, beautiful dashboards, multi-datasource |

**Benefits**:
- ✅ Free and open source
- ✅ Works offline (no internet required for local dev)
- ✅ Industry standard (used by Netflix, Uber, SoundCloud)
- ✅ Easy to set up (just 2 Docker containers)

---

## 10. Production-Grade Architecture - What Makes This Professional?

Our MLOps pipeline demonstrates **production-grade engineering** through several key characteristics:

### 10.1 Automation

**Not production-grade**: Manual training, manual deployment, manual monitoring
**Our approach**: Fully automated end-to-end

```
Data Generation (automated daily)
  ↓
Data Validation (automatic schema checks)
  ↓
Model Training (automated weekly)
  ↓
Model Evaluation (automatic promotion if metrics improved)
  ↓
Deployment (automated via CI/CD)
  ↓
Monitoring (automatic drift detection)
  ↓
Retraining (auto-triggered when drift severe)
  ↓
Loop back to Training
```

**Key automation**:
- ✅ Airflow orchestrates all workflows (no manual intervention)
- ✅ GitHub Actions deploys on every commit to main
- ✅ Drift detection triggers retraining automatically
- ✅ Model promotion based on metrics (no manual approval)

### 10.2 Reproducibility

**Not production-grade**: "Works on my machine", can't recreate experiment from 3 months ago
**Our approach**: Everything tracked and versioned

```
Data: Versioned in S3 with timestamps
  ├─ fraud_20240101.parquet
  ├─ fraud_20240102.parquet
  └─ ...

Code: Git version control with semantic versioning
  ├─ v1.0.0 (initial release)
  ├─ v1.1.0 (added drift detection)
  └─ v1.2.0 (improved model)

Models: MLflow tracks everything
  ├─ Hyperparameters (n_estimators=100, max_depth=5)
  ├─ Metrics (AUC=0.92, Precision=0.89)
  ├─ Artifacts (model.pkl, preprocessor.pkl)
  └─ Environment (requirements.txt, Python 3.10)

Infrastructure: Terraform state versioned
  └─ Every change logged in git + Terraform state file
```

**Reproducibility guarantee**:
- Given experiment ID `exp_123`, can reload exact model with exact hyperparameters
- Given git commit `abc123`, can rebuild exact Docker image
- Given date `2024-01-15`, can reload exact data used for training

### 10.3 Scalability

**Not production-grade**: Runs on laptop, breaks under load
**Our approach**: Designed for growth

| Component | Current Scale | Can Scale To | How |
|-----------|---------------|--------------|-----|
| **API** | 10 req/s | 1000+ req/s | ECS auto-scaling (2→20 tasks) |
| **Data** | 100K transactions/day | 100M+/day | S3 (unlimited), partition by date |
| **Training** | 1M samples | 100M+ samples | Distributed training (Dask, Ray) |
| **Database** | 10 GB | 10 TB | RDS read replicas, sharding |
| **Monitoring** | 1 model | 100+ models | Prometheus federation |

**Scalability patterns used**:
- ✅ Stateless API (can horizontally scale infinitely)
- ✅ Load balancer (distributes traffic across replicas)
- ✅ Object storage (S3 scales automatically)
- ✅ Managed services (RDS, ECS handle scaling)

### 10.4 Reliability

**Not production-grade**: Single point of failure, no backups, no recovery plan
**Our approach**: Fault-tolerant by design

```
High Availability:
├── Multi-AZ RDS (automatic failover if primary fails)
├── Multiple FastAPI replicas (if 1 dies, others continue)
├── ALB health checks (removes unhealthy targets automatically)
└── Auto-scaling (replaces failed tasks)

Disaster Recovery:
├── Daily RDS backups (7-day retention, point-in-time recovery)
├── S3 versioning (can recover deleted files)
├── MLflow artifact backups (models never lost)
└── Infrastructure as Code (can rebuild entire stack in 30 min)

Graceful Degradation:
├── If MLflow down → API serves cached model
├── If drift detection fails → use last known drift score
├── If training fails → keep current production model
└── Retry logic on transient failures (network timeouts)
```

**Mean Time To Recovery (MTTR)**:
- Database failure: < 2 minutes (automatic RDS failover)
- API crash: < 30 seconds (ECS restarts task)
- Full infrastructure disaster: < 30 minutes (terraform apply)

### 10.5 Observability

**Not production-grade**: "Check logs manually", no visibility into system health
**Our approach**: Full observability stack

**3 Pillars of Observability**:

1. **Metrics** (Prometheus + Grafana)
   - API latency, request rate, error rate
   - Drift scores, prediction distributions
   - CPU, memory, disk usage

2. **Logs** (CloudWatch + structured logging)
   - Every prediction logged with correlation ID
   - Error stack traces
   - Searchable and filterable

3. **Traces** (optional: AWS X-Ray)
   - Request path: ALB → FastAPI → MLflow → RDS
   - Identify bottlenecks

**Alerts**:
- 🚨 Critical: API error rate > 5% → page on-call
- ⚠️  Warning: Drift score > 0.15 → Slack notification
- ℹ️  Info: New model deployed → email team

### 10.6 Security

**Not production-grade**: Hardcoded credentials, public S3 buckets, no encryption
**Our approach**: Security best practices

```
Authentication & Authorization:
├── IAM roles (no hardcoded AWS keys)
├── Secrets Manager (DB passwords, API keys)
├── VPC with private subnets (databases not internet-accessible)
└── Security groups (firewall rules per service)

Encryption:
├── Data at rest: S3 (AES-256), RDS (encrypted volumes)
├── Data in transit: TLS/SSL (ALB terminates HTTPS)
└── Secrets: AWS Secrets Manager (encrypted)

Vulnerability Management:
├── Docker image scanning (ECR scans on push)
├── Dependency scanning (Dependabot, Trivy)
├── Security linting (Bandit for Python code)
└── Regular updates (automated patching)

Compliance:
├── Audit logs (CloudTrail tracks all AWS API calls)
├── Access logs (ALB logs all requests to S3)
└── Data retention policies (delete old predictions after 90 days)
```

### 10.7 Testing

**Not production-grade**: "I manually tested it once"
**Our approach**: Comprehensive test pyramid

```
Test Coverage: 60%+

Unit Tests (70% of tests):
├── Data generation (schema compliance, fraud rate)
├── Preprocessing (transformations, edge cases)
├── Model training (all algorithms train successfully)
└── API endpoints (valid/invalid inputs)

Integration Tests (25% of tests):
├── Data pipeline (generate → validate → preprocess)
├── Training pipeline (train → register → promote)
├── API + MLflow (load production model, make prediction)
└── Drift detection (reference vs current data)

End-to-End Tests (5% of tests):
├── Full Docker stack (all 9 services)
├── Airflow DAG execution (data → training → monitoring)
└── API load testing (100 RPS for 5 minutes)
```

**Testing in CI/CD**:
- Every PR: Unit tests, linting, type checking
- Every merge: Integration tests, Docker builds, security scans
- Before deploy: E2E tests in staging, smoke tests

### 10.8 Documentation

**Not production-grade**: README with 3 bullet points, no comments
**Our approach**: Comprehensive documentation

```
Documentation (5 core files):
├── README.md (quick start, architecture overview)
├── GUIDE.md (comprehensive technical guide, 50+ pages)
├── INTERVIEW_PREP_GUIDE.md (interview Q&A, demo script)
├── DEPLOYMENT_GUIDE.md (AWS deployment step-by-step)
└── CLAUDE.md (instructions for AI assistants, development guide)

Code Documentation:
├── Docstrings for all public functions/classes
├── Type hints (mypy strict mode)
├── Inline comments for complex logic
└── Architecture diagrams (ASCII + Mermaid)

API Documentation:
└── Auto-generated OpenAPI docs (FastAPI /docs endpoint)
```

---

## 11. Why This Tech Stack? - Justification & Alternatives

### Design Decisions & Trade-offs

#### Why Airflow over alternatives?

**Alternatives considered**: Prefect, Dagster, Luigi, Argo Workflows

| Tool | Pros | Cons | Decision |
|------|------|------|----------|
| **Apache Airflow** | ✅ Industry standard<br>✅ Mature (10+ years)<br>✅ Rich UI<br>✅ Large community | ❌ Heavy (requires postgres)<br>❌ Steep learning curve | ✅ **CHOSEN** - Most used in industry, best for interviews |
| Prefect | ✅ Modern Python API<br>✅ Easier than Airflow | ❌ Smaller community<br>❌ Less established | ❌ Too new |
| Dagster | ✅ Software-defined assets<br>✅ Better for data engineers | ❌ Niche (less known)<br>❌ Different paradigm | ❌ Less interview value |
| Luigi (Spotify) | ✅ Simple | ❌ Unmaintained<br>❌ No UI | ❌ Outdated |
| Argo Workflows | ✅ Kubernetes-native | ❌ Requires K8s<br>❌ More DevOps-focused | ❌ Overkill for our scale |

**Decision**: Airflow - best recognition in job market, most interview questions about Airflow

#### Why MLflow over alternatives?

**Alternatives considered**: Weights & Biases, Neptune.ai, Comet.ml, Kubeflow

| Tool | Pros | Cons | Decision |
|------|------|------|----------|
| **MLflow** | ✅ Open source<br>✅ Complete lifecycle<br>✅ Easy to deploy<br>✅ Industry standard | ❌ Basic UI<br>❌ No built-in collaboration | ✅ **CHOSEN** - Best for self-hosted, free |
| Weights & Biases | ✅ Beautiful UI<br>✅ Great collaboration | ❌ Cloud-only<br>❌ Expensive ($50+/user/month) | ❌ Cost prohibitive |
| Neptune.ai | ✅ Good UI<br>✅ Experiment comparison | ❌ Cloud-only<br>❌ Less popular | ❌ Vendor lock-in |
| Kubeflow | ✅ Complete ML platform | ❌ Requires Kubernetes<br>❌ Very complex | ❌ Overkill |

**Decision**: MLflow - free, self-hosted, most popular open-source option

#### Why FastAPI over Flask/Django?

**Alternatives considered**: Flask, Django, Tornado, Falcon

| Framework | Pros | Cons | Decision |
|-----------|------|------|----------|
| **FastAPI** | ✅ Async (fast)<br>✅ Type validation (Pydantic)<br>✅ Auto docs<br>✅ Modern (2018) | ❌ Newer (less Stack Overflow) | ✅ **CHOSEN** - Best for ML serving, fastest |
| Flask | ✅ Simple<br>✅ Huge community | ❌ Synchronous (slow)<br>❌ No type validation<br>❌ Manual docs | ❌ Too slow for production ML |
| Django | ✅ Batteries included | ❌ Heavy (ORM, admin)<br>❌ Overkill for API | ❌ Too complex for API-only |
| Falcon | ✅ Minimalist<br>✅ Fast | ❌ Low-level<br>❌ Small community | ❌ Too bare-bones |

**Decision**: FastAPI - async performance + type safety + auto docs = perfect for ML APIs

#### Why ECS over Kubernetes?

**Alternatives considered**: Kubernetes (EKS), ECS, Fargate, Lambda

| Platform | Pros | Cons | Decision |
|----------|------|------|----------|
| **ECS Fargate** | ✅ Simple (no K8s complexity)<br>✅ Managed (no nodes)<br>✅ AWS-native | ❌ AWS lock-in<br>❌ Less powerful than K8s | ✅ **CHOSEN** - Right complexity for our scale |
| Kubernetes (EKS) | ✅ Industry standard<br>✅ Powerful<br>✅ Portable | ❌ Complex (steep learning curve)<br>❌ Expensive ($72/month EKS control plane) | ❌ Overkill for < 100 containers |
| AWS Lambda | ✅ Serverless<br>✅ Pay per request | ❌ Cold starts (latency spikes)<br>❌ 15 min timeout | ❌ Not good for ML (large models) |
| Self-managed EC2 | ✅ Full control | ❌ Manual scaling<br>❌ Manual patching<br>❌ Manual failover | ❌ Too much operational burden |

**Decision**: ECS Fargate - sweet spot between simplicity and power

#### Why Terraform over CloudFormation?

**Alternatives considered**: Terraform, CloudFormation, Pulumi, AWS CDK

| IaC Tool | Pros | Cons | Decision |
|----------|------|------|----------|
| **Terraform** | ✅ Multi-cloud<br>✅ HCL (declarative)<br>✅ Huge community<br>✅ Mature (10+ years) | ❌ State file management | ✅ **CHOSEN** - Industry standard, not AWS-locked |
| CloudFormation | ✅ AWS-native<br>✅ No state file | ❌ AWS-only<br>❌ Verbose YAML<br>❌ Slow | ❌ AWS lock-in |
| Pulumi | ✅ Real programming languages<br>✅ Modern | ❌ Less mature<br>❌ Smaller community | ❌ Too new |
| AWS CDK | ✅ AWS-native<br>✅ TypeScript/Python | ❌ AWS-only<br>❌ Compiles to CloudFormation | ❌ AWS lock-in |

**Decision**: Terraform - portable, mature, best for learning/interviews

### Why This Architecture Pattern?

**Pattern**: Layered architecture with event-driven retraining

**Alternatives considered**:
1. **Monolithic ML app** (FastAPI serves and trains)
   - ❌ Can't scale independently
   - ❌ Training blocks serving

2. **Microservices with synchronous calls** (API calls training service)
   - ❌ Tight coupling
   - ❌ Training latency affects API

3. **Serverless (Lambda + Step Functions)** (event-driven)
   - ❌ Cold starts (bad for ML models)
   - ❌ Limited execution time (15 min)

4. **Our choice: Layered + Event-driven** ✅
   - ✅ Each layer scales independently
   - ✅ Async retraining (doesn't block API)
   - ✅ Clear separation of concerns
   - ✅ Easy to test each layer

### Production-Grade Checklist

Compare typical ML projects vs our project:

| Requirement | Typical ML Project | Our Project |
|-------------|-------------------|-------------|
| **Versioned data** | ❌ CSV on laptop | ✅ S3 with timestamps |
| **Versioned models** | ❌ `model_final_v3_FINAL.pkl` | ✅ MLflow registry with stages |
| **Automated training** | ❌ Manually run notebook | ✅ Airflow DAG (weekly) |
| **Drift detection** | ❌ Hope data doesn't change | ✅ EvidentlyAI (hourly) |
| **Automated retraining** | ❌ Train when someone notices issue | ✅ Auto-triggered on severe drift |
| **CI/CD** | ❌ SSH into server, copy files | ✅ GitHub Actions (automated deploy) |
| **Infrastructure as Code** | ❌ ClickOps in AWS console | ✅ Terraform (version controlled) |
| **Monitoring** | ❌ Check logs manually | ✅ Prometheus + Grafana dashboards |
| **Alerts** | ❌ User complains → investigate | ✅ Proactive Slack alerts on drift |
| **Testing** | ❌ "It works on my machine" | ✅ 60%+ coverage, CI runs tests |
| **Scalability** | ❌ Single server | ✅ Auto-scaling (2-10 replicas) |
| **High Availability** | ❌ If server down → downtime | ✅ Multi-AZ, auto-restart, load balancer |
| **Disaster Recovery** | ❌ Hope nothing breaks | ✅ Daily backups, IaC (rebuild in 30 min) |
| **Documentation** | ❌ Minimal README | ✅ 5 docs (README, GUIDE, INTERVIEW, DEPLOY, CONCEPTS) |

**Score**: Typical project 1/14 ❌ | Our project: 14/14 ✅

---

## Final Thoughts: Learning Beyond This Project

This stack demonstrates production MLOps, but remember:

### When to Use This Stack

✅ **Use this architecture when**:
- Deploying ML models to production
- Need automated retraining
- Multiple models to manage
- Team of 2-10 engineers
- Budget of $300-500/month for AWS

❌ **Don't use this stack if**:
- Research project (just use Jupyter notebooks)
- One-off analysis (too much overhead)
- Real-time streaming (use Kafka + Flink instead)
- Massive scale (100M+ requests/day → need Kubernetes)

### Next Steps for Learning

**To go deeper**:
1. **Distributed Training**: Dask, Ray, Horovod for 100M+ samples
2. **Feature Store**: Feast, Tecton for feature engineering at scale
3. **Model Serving Optimization**: ONNX, TorchServe, TensorFlow Serving for 10x faster inference
4. **Advanced Monitoring**: Explainability (SHAP, LIME), fairness metrics
5. **Kubernetes**: EKS for 100+ containers, service mesh (Istio)
6. **Stream Processing**: Kafka + Flink for real-time fraud detection

**Interview Preparation**:
- Read `INTERVIEW_PREP_GUIDE.md` for 50+ Q&A
- Practice explaining each component (use this guide as reference)
- Demo the system live (15-minute walkthrough)
- Be ready to discuss trade-offs and alternatives

---

## Summary: Key Takeaways

| Technology | What It Does | Why We Use It | Alternative |
|------------|--------------|---------------|-------------|
| **Airflow** | Workflow orchestration | Automate pipelines (data, training, monitoring) | Prefect, Dagster |
| **MLflow** | ML lifecycle management | Track experiments, version models, registry | W&B, Neptune |
| **FastAPI** | Async web framework | Serve predictions (<100ms latency) | Flask (slower) |
| **Docker** | Containerization | Package services, consistent environments | None (industry standard) |
| **Terraform** | Infrastructure as Code | Define AWS resources in code, version control | CloudFormation (AWS-only) |
| **GitHub Actions** | CI/CD automation | Test on every PR, deploy on merge | Jenkins, GitLab CI |
| **EvidentlyAI** | Drift detection | Detect data/model drift, trigger retraining | Alibi Detect, Whylabs |
| **ECS Fargate** | Container orchestration | Run Docker containers at scale, serverless | Kubernetes (more complex) |
| **RDS PostgreSQL** | Managed database | Store metadata (MLflow, Airflow), backups | Self-managed Postgres |
| **S3** | Object storage | Store data, models, artifacts (unlimited scale) | None (AWS native) |
| **ALB** | Load balancer | Distribute traffic, SSL termination, health checks | NGINX (self-managed) |
| **Prometheus** | Metrics collection | Scrape metrics from services (requests, latency) | CloudWatch (AWS-only) |
| **Grafana** | Visualization | Dashboards for metrics, alerts | Kibana, Datadog |

---

**Congratulations!** 🎉

You now understand every component of a production-grade MLOps system. This knowledge positions you to:
- Build scalable ML systems from scratch
- Discuss MLOps architecture in interviews with confidence
- Make informed technology choices for your projects
- Recognize when to use (or not use) specific tools

**Next**: Practice explaining this architecture in your own words. The best way to learn is to teach!

---

*End of Guide*

**Total Guide Stats**:
- 11 major technology sections
- 50+ code examples
- 20+ architecture diagrams (ASCII)
- 30+ comparison tables
- 100+ key concepts explained

**Estimated Reading Time**: 3-4 hours (comprehensive study)

**Recommended Use**: Keep this guide open during interviews as reference for deep technical discussions!
