"""
Pytest configuration and shared fixtures.

Provides reusable fixtures for testing across the project.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.data.generator import FraudDataGenerator
from src.data.preprocessor import FraudPreprocessor


@pytest.fixture
def sample_size():
    """Small sample size for fast tests."""
    return 100


@pytest.fixture
def fraud_rate():
    """Test fraud rate."""
    return 0.1  # 10% for easier testing


@pytest.fixture
def sample_fraud_data(sample_size, fraud_rate):
    """Generate small sample of fraud data for testing."""
    generator = FraudDataGenerator(
        n_samples=sample_size,
        fraud_rate=fraud_rate,
        n_cardholders=sample_size // 5,
        seed=42
    )
    return generator.generate()


@pytest.fixture
def sample_transactions():
    """Sample transaction records for unit tests."""
    return pd.DataFrame({
        'transaction_id': ['TXN00000001', 'TXN00000002', 'TXN00000003'],
        'timestamp': [datetime.now()] * 3,
        'cardholder_id': ['CH000001', 'CH000002', 'CH000003'],
        'cardholder_name': ['John Doe', 'Jane Smith', 'Bob Johnson'],
        'amount': [100.50, 250.75, 50.25],
        'merchant_name': ['Acme Corp', 'Beta Inc', 'Gamma LLC'],
        'merchant_category': ['grocery', 'restaurant', 'gas_station'],
        'merchant_city': ['New York', 'Los Angeles', 'Chicago'],
        'merchant_country': ['USA', 'USA', 'USA'],
        'card_last4': ['1234', '5678', '9012'],
        'device_fingerprint': ['a' * 16, 'b' * 16, 'c' * 16],
        'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'user_agent': ['Mozilla/5.0'] * 3,
        'is_online': [True, False, True],
        'hour_of_day': [14, 19, 8],
        'day_of_week': [0, 5, 3],
        'is_weekend': [False, True, False],
        'transactions_last_24h': [2, 1, 0],
        'total_amount_last_24h': [150.0, 250.75, 0.0],
        'transactions_last_1h': [0, 0, 0],
        'amount_log': [np.log1p(100.50), np.log1p(250.75), np.log1p(50.25)],
        'distance_from_home': [0.0, 500.0, 10.0],
        'is_fraud': [0, 0, 1],
    })


@pytest.fixture
def fitted_preprocessor(sample_fraud_data):
    """Fitted preprocessor for testing."""
    X = sample_fraud_data.drop(columns=['is_fraud'])
    y = sample_fraud_data['is_fraud']

    preprocessor = FraudPreprocessor(scaling_method='standard')
    preprocessor.fit(X, y)

    return preprocessor


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup after test
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def temp_file(temp_dir):
    """Temporary file path for testing."""
    return temp_dir / "test_file.csv"
