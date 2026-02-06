"""Model training orchestrator with MLflow integration."""
from typing import Dict, Optional, Tuple
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
import optuna

from src.models.fraud_classifier import FraudClassifier
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class ModelTrainer:
    """Orchestrate model training with MLflow tracking."""

    def __init__(self, experiment_name: str = "fraud_detection"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def train(
        self,
        X_train, y_train,
        X_val, y_val,
        algorithm: str = 'xgboost',
        **params
    ) -> Tuple[FraudClassifier, Dict]:
        """Train single model with MLflow logging."""

        with mlflow.start_run(run_name=f"{algorithm}_train"):
            # Log parameters
            mlflow.log_params({'algorithm': algorithm, **params})

            # Train model
            model = FraudClassifier(algorithm=algorithm, **params)
            model.fit(X_train, y_train)

            # Evaluate
            metrics = model.evaluate(X_val, y_val)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(model.model, "model")

            logger.info(f"✅ Trained {algorithm}: AUC={metrics['auc_roc']:.4f}")

            return model, metrics
