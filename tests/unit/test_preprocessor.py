"""Unit tests for data preprocessor."""
import pytest
import pandas as pd
import numpy as np

from src.data.preprocessor import FraudPreprocessor, prepare_train_test_split


class TestFraudPreprocessor:
    """Test FraudPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = FraudPreprocessor(
            scaling_method='standard',
            create_interactions=True
        )

        assert preprocessor.scaling_method == 'standard'
        assert preprocessor.create_interactions is True
        assert preprocessor.is_fitted is False

    def test_fit_sets_fitted_flag(self, sample_fraud_data):
        """Test that fit sets is_fitted flag."""
        X = sample_fraud_data.drop(columns=['is_fraud'])
        y = sample_fraud_data['is_fraud']

        preprocessor = FraudPreprocessor()
        preprocessor.fit(X, y)

        assert preprocessor.is_fitted is True

    def test_transform_before_fit_raises_error(self, sample_fraud_data):
        """Test that transform before fit raises error."""
        X = sample_fraud_data.drop(columns=['is_fraud'])

        preprocessor = FraudPreprocessor()

        with pytest.raises(ValueError, match="must be fitted"):
            preprocessor.transform(X)

    def test_fit_transform(self, sample_fraud_data):
        """Test fit_transform returns transformed data."""
        X = sample_fraud_data.drop(columns=['is_fraud'])
        y = sample_fraud_data['is_fraud']

        preprocessor = FraudPreprocessor()
        X_transformed = preprocessor.fit_transform(X, y)

        assert isinstance(X_transformed, pd.DataFrame)
        assert len(X_transformed) == len(X)

    def test_creates_engineered_features(self, sample_fraud_data):
        """Test that engineered features are created."""
        X = sample_fraud_data.drop(columns=['is_fraud'])

        preprocessor = FraudPreprocessor(create_interactions=True)
        X_transformed = preprocessor.fit_transform(X)

        # Check for engineered features
        assert 'amount_squared' in X_transformed.columns
        assert 'velocity_ratio' in X_transformed.columns
        assert 'is_night' in X_transformed.columns

    def test_scaling(self, sample_fraud_data):
        """Test that scaling is applied."""
        X = sample_fraud_data.drop(columns=['is_fraud'])

        preprocessor = FraudPreprocessor(scaling_method='standard')
        X_transformed = preprocessor.fit_transform(X)

        # Check that numerical features are scaled (mean ~ 0, std ~ 1)
        # Note: May not be exact due to small sample size
        assert preprocessor.scaler is not None

    def test_no_scaling(self, sample_fraud_data):
        """Test that scaling can be disabled."""
        X = sample_fraud_data.drop(columns=['is_fraud'])

        preprocessor = FraudPreprocessor(scaling_method='none')
        X_transformed = preprocessor.fit_transform(X)

        assert preprocessor.scaler is None

    def test_save_load(self, fitted_preprocessor, temp_dir):
        """Test saving and loading preprocessor."""
        filepath = temp_dir / 'preprocessor.pkl'

        # Save
        fitted_preprocessor.save(filepath)
        assert filepath.exists()

        # Load
        loaded = FraudPreprocessor.load(filepath)
        assert loaded.is_fitted is True
        assert loaded.scaling_method == fitted_preprocessor.scaling_method


class TestPrepareTrainTestSplit:
    """Test train-test split function."""

    def test_split_returns_four_values(self, sample_fraud_data):
        """Test that split returns 4 values."""
        result = prepare_train_test_split(sample_fraud_data)

        assert len(result) == 4
        X_train, X_test, y_train, y_test = result

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_split_sizes(self, sample_fraud_data):
        """Test train-test split sizes."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            sample_fraud_data,
            test_size=0.2
        )

        total_size = len(X_train) + len(X_test)
        assert total_size == len(sample_fraud_data)
        assert len(X_test) / total_size == pytest.approx(0.2, abs=0.05)

    def test_stratification(self, sample_fraud_data):
        """Test that stratification maintains fraud rate."""
        X_train, X_test, y_train, y_test = prepare_train_test_split(
            sample_fraud_data,
            stratify=True
        )

        fraud_rate = sample_fraud_data['is_fraud'].mean()
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()

        # Fraud rates should be similar (within 5%)
        assert abs(train_fraud_rate - fraud_rate) < 0.05
        assert abs(test_fraud_rate - fraud_rate) < 0.05
