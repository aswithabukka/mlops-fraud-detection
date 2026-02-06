"""MLflow model registry interface."""
import mlflow
from mlflow.tracking import MlflowClient
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Interface for MLflow Model Registry operations."""

    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()

    def register_model(self, run_id: str, model_name: str) -> str:
        """Register model to registry."""
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Registered {model_name} version {result.version}")
        return result.version

    def promote_to_production(self, model_name: str, version: str):
        """Promote model to Production stage."""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Promoted {model_name} v{version} to Production")

    def load_production_model(self, model_name: str):
        """Load current production model."""
        model_uri = f"models:/{model_name}/Production"
        return mlflow.pyfunc.load_model(model_uri)
