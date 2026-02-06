"""
Data preprocessing pipeline for fraud detection.

Handles:
- Feature engineering (interaction features, aggregations)
- Encoding categorical variables
- Scaling numerical features
- Train-test splitting with stratification
- Scikit-learn compatible fit/transform interface
"""
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FraudPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline for fraud transaction data.

    Features:
    - Scikit-learn compatible (fit/transform)
    - Feature engineering (derived features, interactions)
    - Categorical encoding
    - Numerical scaling
    - Serializable (pickle)

    Example:
        >>> preprocessor = FraudPreprocessor()
        >>> X_train_transformed = preprocessor.fit_transform(X_train, y_train)
        >>> X_test_transformed = preprocessor.transform(X_test)
    """

    def __init__(
        self,
        scaling_method: str = "standard",
        create_interactions: bool = True,
        drop_columns: Optional[List[str]] = None,
    ):
        """
        Initialize preprocessor.

        Args:
            scaling_method: Scaling method ('standard', 'robust', or 'none')
            create_interactions: Whether to create interaction features
            drop_columns: Columns to drop before processing
        """
        self.scaling_method = scaling_method
        self.create_interactions = create_interactions
        self.drop_columns = drop_columns or []

        # Initialize transformers (fit during fit())
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = []
        self.numerical_features = []

        # Fitted flag
        self.is_fitted = False

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """Identify categorical and numerical features."""
        # Categorical features
        self.categorical_features = [
            col for col in df.select_dtypes(include=['object', 'category']).columns
            if col not in self.drop_columns
        ]

        # Numerical features (excluding target and IDs)
        exclude_cols = ['is_fraud', 'transaction_id', 'timestamp',
                       'cardholder_id', 'cardholder_name'] + self.drop_columns
        self.numerical_features = [
            col for col in df.select_dtypes(include=['int64', 'float64']).columns
            if col not in exclude_cols
        ]

        logger.info(f"Identified {len(self.categorical_features)} categorical features")
        logger.info(f"Identified {len(self.numerical_features)} numerical features")

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional engineered features
        """
        df = df.copy()

        # Amount-based features
        df['amount_squared'] = df['amount'] ** 2
        df['amount_cubed'] = df['amount'] ** 3
        df['amount_sqrt'] = np.sqrt(df['amount'])

        # Velocity ratios
        df['velocity_ratio'] = (
            df['transactions_last_1h'] /
            (df['transactions_last_24h'] + 1)  # Add 1 to avoid division by zero
        )

        df['amount_velocity_ratio'] = (
            df['amount'] /
            (df['total_amount_last_24h'] + 1)
        )

        # Time-based features
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype(int)
        df['is_business_hours'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 17)).astype(int)

        # Location features
        df['is_domestic'] = (df['merchant_country'] == 'USA').astype(int)
        df['is_foreign'] = (~df['is_domestic'].astype(bool)).astype(int)

        # Interaction features (if enabled)
        if self.create_interactions:
            # Amount x Time interactions
            df['amount_x_hour'] = df['amount'] * df['hour_of_day']
            df['amount_x_is_weekend'] = df['amount'] * df['is_weekend'].astype(int)
            df['amount_x_is_online'] = df['amount'] * df['is_online'].astype(int)

            # Velocity x Amount interactions
            df['velocity_x_amount'] = df['transactions_last_24h'] * df['amount']

            # Distance x Amount
            df['distance_x_amount'] = df['distance_from_home'] * df['amount']

        logger.info(f"Engineered {len(df.columns) - len(self.feature_names) if self.feature_names else 0} new features")

        return df

    def _encode_categorical(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fit encoders)

        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()

        for col in self.categorical_features:
            if col not in df.columns:
                continue

            if is_training:
                # Fit label encoder
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Transform using fitted encoder
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].astype(str).map(
                    lambda x: self.label_encoders[col].transform([x])[0]
                    if x in self.label_encoders[col].classes_
                    else -1  # Unseen category
                )

            # Drop original column
            df = df.drop(columns=[col])

        logger.info(f"Encoded {len(self.categorical_features)} categorical features")

        return df

    def _scale_numerical(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data (fit scaler)

        Returns:
            DataFrame with scaled numerical features
        """
        if self.scaling_method == "none":
            return df

        df = df.copy()

        # Get numerical columns present in df
        numerical_cols = [col for col in self.numerical_features if col in df.columns]

        if not numerical_cols:
            return df

        if is_training:
            # Initialize and fit scaler
            if self.scaling_method == "standard":
                self.scaler = StandardScaler()
            elif self.scaling_method == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")

            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            logger.info(f"Fitted {self.scaling_method} scaler on {len(numerical_cols)} features")
        else:
            # Transform using fitted scaler
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            logger.info(f"Scaled {len(numerical_cols)} numerical features")

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FraudPreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            X: Training features
            y: Training target (optional, not used but included for sklearn compatibility)

        Returns:
            Self (fitted preprocessor)
        """
        logger.info("Fitting preprocessor...")

        # Store original feature names
        self.feature_names = X.columns.tolist()

        # Identify feature types
        self._identify_feature_types(X)

        # Engineer features (to identify all features before encoding)
        X_engineered = self._engineer_features(X)

        # Update feature types after engineering
        self._identify_feature_types(X_engineered)

        # Fit encoders
        X_encoded = self._encode_categorical(X_engineered, is_training=True)

        # Fit scaler
        _ = self._scale_numerical(X_encoded, is_training=True)

        self.is_fitted = True
        logger.info("✅ Preprocessor fitted successfully")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            X: Features to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        logger.info(f"Transforming {len(X)} samples...")

        # Engineer features
        X_transformed = self._engineer_features(X)

        # Encode categorical
        X_transformed = self._encode_categorical(X_transformed, is_training=False)

        # Scale numerical
        X_transformed = self._scale_numerical(X_transformed, is_training=False)

        logger.info(f"✅ Transformed to {X_transformed.shape[1]} features")

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            X: Training features
            y: Training target (optional)

        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y).transform(X)

    def save(self, filepath: Path) -> None:
        """
        Save fitted preprocessor to disk.

        Args:
            filepath: Path to save preprocessor
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Saved preprocessor to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'FraudPreprocessor':
        """
        Load fitted preprocessor from disk.

        Args:
            filepath: Path to preprocessor file

        Returns:
            Loaded preprocessor
        """
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)

        logger.info(f"Loaded preprocessor from {filepath}")
        return preprocessor

    def get_feature_names(self) -> List[str]:
        """Get names of transformed features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first")

        # This would need to be tracked during transform
        # For now, return empty list (can be enhanced)
        return []


def prepare_train_test_split(
    df: pd.DataFrame,
    target_col: str = 'is_fraud',
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame
        target_col: Name of target column
        test_size: Fraction of data for test set
        random_state: Random seed
        stratify: Whether to stratify split by target

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data: {len(df)} samples, test_size={test_size}")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None,
    )

    logger.info(f"Train set: {len(X_train)} samples ({y_train.mean()*100:.2f}% fraud)")
    logger.info(f"Test set: {len(X_test)} samples ({y_test.mean()*100:.2f}% fraud)")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """Test preprocessor with sample data."""
    from src.data.generator import FraudDataGenerator

    # Generate sample data
    logger.info("Generating sample data for testing...")
    generator = FraudDataGenerator(n_samples=1000, fraud_rate=0.01, seed=42)
    df = generator.generate()

    # Prepare train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    print(f"\nOriginal data shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)[:10]}...")

    # Initialize and fit preprocessor
    preprocessor = FraudPreprocessor(
        scaling_method="standard",
        create_interactions=True,
    )

    # Fit on training data
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)

    print(f"\nTransformed training data shape: {X_train_transformed.shape}")
    print(f"Transformed test data shape: {X_test_transformed.shape}")
    print(f"\nSample transformed features:")
    print(X_train_transformed.head())

    # Test save/load
    preprocessor.save(Path("models/preprocessor.pkl"))
    loaded_preprocessor = FraudPreprocessor.load(Path("models/preprocessor.pkl"))

    print("\n✅ Preprocessor test completed successfully!")
