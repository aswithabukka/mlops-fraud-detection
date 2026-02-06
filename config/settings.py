"""
Configuration management using Pydantic Settings.
Supports environment-based configuration with type validation.
"""
from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"

    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_artifact_root: str = "./mlruns"
    mlflow_backend_store_uri: str = "postgresql://mlflow:mlflow@localhost:5432/mlflow"
    model_name: str = "fraud_classifier"
    model_stage: Literal["Staging", "Production"] = "Production"

    # Database Configuration
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "mlflow"
    db_password: str = "mlflow"
    db_name: str = "mlflow"

    # MinIO/S3 Configuration
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"
    aws_endpoint_url: Optional[str] = "http://localhost:9000"
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "mlops-fraud-detection"

    # Airflow Configuration
    airflow_home: str = "./airflow"
    airflow_uid: int = 50000

    # FastAPI Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    api_log_level: Literal["debug", "info", "warning", "error"] = "info"

    # Model Configuration
    fraud_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, gt=0)
    max_features: int = Field(default=50, gt=0)

    # Monitoring Configuration
    drift_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    drift_severe_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    performance_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    monitoring_interval: int = Field(default=3600, gt=0)  # seconds

    # Alerting Configuration
    enable_slack_alerts: bool = False
    slack_webhook_url: Optional[str] = None
    enable_email_alerts: bool = False
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    alert_email: Optional[str] = None

    # Data Generation Configuration
    data_size: int = Field(default=100000, gt=0)
    fraud_rate: float = Field(default=0.01, ge=0.0, le=1.0)

    # Logging Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: str = "./logs/app.log"

    # Prometheus Configuration
    prometheus_port: int = 9090

    # Grafana Configuration
    grafana_port: int = 3000
    grafana_admin_user: str = "admin"
    grafana_admin_password: str = "admin"

    @field_validator("drift_severe_threshold")
    @classmethod
    def validate_drift_severe_threshold(cls, v: float, info) -> float:
        """Ensure severe drift threshold is greater than regular drift threshold."""
        drift_threshold = info.data.get("drift_threshold", 0.15)
        if v <= drift_threshold:
            raise ValueError(
                f"drift_severe_threshold ({v}) must be greater than "
                f"drift_threshold ({drift_threshold})"
            )
        return v

    @property
    def database_url(self) -> str:
        """Construct database URL for SQLAlchemy."""
        return (
            f"postgresql://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get application settings instance.
    Useful for dependency injection in FastAPI.
    """
    return settings


# Example usage for logging
def configure_logging() -> dict:
    """
    Generate logging configuration based on settings.
    Returns a dict compatible with logging.config.dictConfig.
    """
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            },
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": settings.log_format,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": settings.log_format,
                "filename": settings.log_file,
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "root": {
            "level": settings.log_level,
            "handlers": ["console", "file"],
        },
        "loggers": {
            "uvicorn": {"level": "INFO", "handlers": ["console"], "propagate": False},
            "mlflow": {"level": "WARNING", "handlers": ["console"], "propagate": False},
            "airflow": {"level": "INFO", "handlers": ["console"], "propagate": False},
        },
    }


if __name__ == "__main__":
    # Print current configuration for debugging
    print("Current Configuration:")
    print(f"Environment: {settings.environment}")
    print(f"MLflow Tracking URI: {settings.mlflow_tracking_uri}")
    print(f"Database URL: {settings.database_url}")
    print(f"API Port: {settings.api_port}")
    print(f"Fraud Threshold: {settings.fraud_threshold}")
    print(f"Drift Threshold: {settings.drift_threshold}")
    print(f"Data Size: {settings.data_size}")
    print(f"Fraud Rate: {settings.fraud_rate * 100}%")
