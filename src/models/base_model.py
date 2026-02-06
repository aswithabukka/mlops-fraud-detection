"""
Base model interface for fraud detection models.

Provides abstract base class ensuring consistent model interface.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseMLModel(ABC):
    """
    Abstract base class for ML models.

    Ensures consistent interface across all model implementations.
    """

    def __init__(self, name: str = "BaseModel"):
        """
        Initialize base model.

        Args:
            name: Model name
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.threshold = 0.5

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseMLModel':
        """
        Train the model.

        Args:
            X: Training features
            y: Training target
            **kwargs: Additional training parameters

        Returns:
            Self (fitted model)
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        pass

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Features
            threshold: Classification threshold (default: self.threshold)

        Returns:
            Array of predicted labels (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        threshold = threshold or self.threshold
        probas = self.predict_proba(X)

        # Use probability of fraud class (class 1)
        return (probas[:, 1] >= threshold).astype(int)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold

        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")

        # Get predictions
        y_pred = self.predict(X, threshold=threshold)
        y_proba = self.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'auc_roc': roc_auc_score(y, y_proba),
            'average_precision': average_precision_score(y, y_proba),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'threshold': threshold or self.threshold,
        }

        logger.info(f"Model evaluation: AUC-ROC={metrics['auc_roc']:.4f}, "
                   f"Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}")

        return metrics

    def get_confusion_matrix(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Get confusion matrix.

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold

        Returns:
            Confusion matrix as 2x2 array
        """
        y_pred = self.predict(X, threshold=threshold)
        return confusion_matrix(y, y_pred)

    def get_classification_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        threshold: Optional[float] = None
    ) -> str:
        """
        Get classification report.

        Args:
            X: Features
            y: True labels
            threshold: Classification threshold

        Returns:
            Classification report string
        """
        y_pred = self.predict(X, threshold=threshold)
        return classification_report(y, y_pred, target_names=['Normal', 'Fraud'])

    def save(self, filepath: Path) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Saved model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'BaseMLModel':
        """
        Load model from disk.

        Args:
            filepath: Path to model file

        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded model from {filepath}")
        return model

    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        if self.model is None:
            return {}
        return self.model.get_params() if hasattr(self.model, 'get_params') else {}

    def set_params(self, **params) -> 'BaseMLModel':
        """Set model hyperparameters."""
        if self.model is not None and hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        return self

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} ({fitted_str})"
