"""
Fraud detection classifier implementations.

Supports multiple algorithms:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
"""
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

from src.models.base_model import BaseMLModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudClassifier(BaseMLModel):
    """
    Fraud detection classifier with multiple algorithm support.

    Args:
        algorithm: Model algorithm ('logistic', 'random_forest', 'xgboost', 'lightgbm')
        handle_imbalance: Whether to use SMOTE for class imbalance
        **model_params: Algorithm-specific hyperparameters
    """

    def __init__(
        self,
        algorithm: str = 'xgboost',
        handle_imbalance: bool = True,
        **model_params
    ):
        """Initialize fraud classifier."""
        super().__init__(name=f"FraudClassifier_{algorithm}")

        self.algorithm = algorithm
        self.handle_imbalance = handle_imbalance
        self.model_params = model_params
        self.smote = None

        # Initialize model based on algorithm
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize model based on algorithm choice."""
        # Default parameters optimized for fraud detection
        if self.algorithm == 'logistic':
            default_params = {
                'max_iter': 1000,
                'class_weight': 'balanced',
                'random_state': 42,
            }
            default_params.update(self.model_params)
            return LogisticRegression(**default_params)

        elif self.algorithm == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.model_params)
            return RandomForestClassifier(**default_params)

        elif self.algorithm == 'xgboost':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': 10,  # Handle imbalance
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'aucpr',
            }
            default_params.update(self.model_params)
            return XGBClassifier(**default_params)

        elif self.algorithm == 'lightgbm':
            default_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1,
            }
            default_params.update(self.model_params)
            return LGBMClassifier(**default_params)

        else:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Choose from: logistic, random_forest, xgboost, lightgbm"
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_smote: Optional[bool] = None,
        **kwargs
    ) -> 'FraudClassifier':
        """
        Train the fraud classifier.

        Args:
            X: Training features
            y: Training labels
            use_smote: Whether to apply SMOTE (overrides self.handle_imbalance)
            **kwargs: Additional training parameters

        Returns:
            Self (fitted model)
        """
        logger.info(f"Training {self.name}...")
        logger.info(f"Training set: {len(X)} samples, {y.mean()*100:.2f}% fraud")

        # Store feature names
        self.feature_names = list(X.columns)

        # Handle class imbalance with SMOTE if requested
        X_train, y_train = X, y
        if use_smote is None:
            use_smote = self.handle_imbalance

        if use_smote:
            logger.info("Applying SMOTE for class imbalance...")
            self.smote = SMOTE(random_state=42)
            X_train, y_train = self.smote.fit_resample(X, y)
            logger.info(f"After SMOTE: {len(X_train)} samples, "
                       f"{y_train.mean()*100:.2f}% fraud")

        # Train model
        self.model.fit(X_train, y_train, **kwargs)
        self.is_fitted = True

        logger.info(f"✅ {self.name} trained successfully")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features

        Returns:
            Array of probabilities for each class
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Return only top N features (None = all)

        Returns:
            DataFrame with features and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance = np.abs(self.model.coef_[0])
        else:
            logger.warning(f"{self.algorithm} does not support feature importance")
            return pd.DataFrame()

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_n is not None:
            importance_df = importance_df.head(top_n)

        return importance_df

    def optimize_threshold(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'f1'
    ) -> float:
        """
        Optimize classification threshold.

        Args:
            X: Validation features
            y: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Optimal threshold
        """
        from sklearn.metrics import precision_recall_curve

        # Get probabilities
        y_proba = self.predict_proba(X)[:, 1]

        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y, y_proba)

        # Calculate F1 scores
        with np.errstate(divide='ignore', invalid='ignore'):
            if metric == 'f1':
                f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
                best_idx = np.nanargmax(f1_scores)
            elif metric == 'precision':
                best_idx = np.nanargmax(precisions)
            elif metric == 'recall':
                best_idx = np.nanargmax(recalls)
            else:
                raise ValueError(f"Unknown metric: {metric}")

        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        self.threshold = optimal_threshold

        logger.info(f"Optimal threshold: {optimal_threshold:.3f} "
                   f"(optimizing for {metric})")

        return optimal_threshold


def create_fraud_classifier(
    algorithm: str = 'xgboost',
    **params
) -> FraudClassifier:
    """
    Factory function to create fraud classifier.

    Args:
        algorithm: Model algorithm
        **params: Model parameters

    Returns:
        FraudClassifier instance
    """
    return FraudClassifier(algorithm=algorithm, **params)


if __name__ == "__main__":
    """Test fraud classifier."""
    from src.data.generator import FraudDataGenerator
    from src.data.preprocessor import prepare_train_test_split, FraudPreprocessor

    # Generate sample data
    logger.info("Generating sample data...")
    generator = FraudDataGenerator(n_samples=1000, fraud_rate=0.1, seed=42)
    df = generator.generate()

    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df, test_size=0.2)

    # Preprocess
    preprocessor = FraudPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Test each algorithm
    for algo in ['logistic', 'random_forest', 'xgboost', 'lightgbm']:
        print(f"\n{'='*60}")
        print(f"Testing {algo.upper()}")
        print('='*60)

        # Create and train model
        model = FraudClassifier(algorithm=algo)
        model.fit(X_train_processed, y_train)

        # Evaluate
        metrics = model.evaluate(X_test_processed, y_test)
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Feature importance
        if algo in ['random_forest', 'xgboost', 'lightgbm']:
            importance = model.get_feature_importance(top_n=5)
            print(f"\nTop 5 Features:")
            print(importance)

    print("\n✅ All classifiers tested successfully!")
