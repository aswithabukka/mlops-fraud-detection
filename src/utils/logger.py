"""
Structured logging utility for the MLOps pipeline.
Provides consistent logging across all components.
"""
import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(
    config_path: Optional[Path] = None,
    default_level: int = logging.INFO,
) -> None:
    """
    Setup logging configuration.

    Args:
        config_path: Path to logging configuration YAML file
        default_level: Default logging level if config file not found
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "logging_config.yaml"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Name of the logger (usually __name__ of the module)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting data generation")
    """
    return logging.getLogger(name)


# Setup logging on module import
setup_logging()


# Example usage
if __name__ == "__main__":
    # Test logging at different levels
    logger = get_logger(__name__)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    # Test logging with extra context
    logger.info(
        "Processing transaction",
        extra={
            "transaction_id": "txn_12345",
            "amount": 150.50,
            "is_fraud": False,
        },
    )
