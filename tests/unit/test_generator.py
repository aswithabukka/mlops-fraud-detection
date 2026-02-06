"""Unit tests for fraud data generator."""
import pytest
import pandas as pd
from datetime import datetime

from src.data.generator import FraudDataGenerator, FraudPattern


class TestFraudDataGenerator:
    """Test FraudDataGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = FraudDataGenerator(
            n_samples=1000,
            fraud_rate=0.01,
            n_cardholders=100,
            seed=42
        )

        assert generator.n_samples == 1000
        assert generator.fraud_rate == 0.01
        assert generator.n_cardholders == 100
        assert len(generator.cardholders) == 100

    def test_generate_returns_dataframe(self, sample_size, fraud_rate):
        """Test that generate returns a DataFrame."""
        generator = FraudDataGenerator(
            n_samples=sample_size,
            fraud_rate=fraud_rate,
            seed=42
        )
        df = generator.generate()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == sample_size

    def test_generate_has_required_columns(self, sample_fraud_data):
        """Test that generated data has all required columns."""
        required_columns = [
            'transaction_id', 'timestamp', 'cardholder_id',
            'amount', 'merchant_category', 'is_fraud'
        ]

        for col in required_columns:
            assert col in sample_fraud_data.columns

    def test_fraud_rate(self, sample_size, fraud_rate):
        """Test that fraud rate is approximately correct."""
        generator = FraudDataGenerator(
            n_samples=sample_size,
            fraud_rate=fraud_rate,
            seed=42
        )
        df = generator.generate()

        actual_fraud_rate = df['is_fraud'].mean()
        # Allow 5% tolerance
        assert abs(actual_fraud_rate - fraud_rate) < 0.05

    def test_transaction_id_format(self, sample_fraud_data):
        """Test transaction ID format."""
        pattern = r'^TXN\d{8}$'
        assert sample_fraud_data['transaction_id'].str.match(pattern).all()

    def test_transaction_id_uniqueness(self, sample_fraud_data):
        """Test that transaction IDs are unique."""
        assert sample_fraud_data['transaction_id'].is_unique

    def test_amount_positive(self, sample_fraud_data):
        """Test that all amounts are positive."""
        assert (sample_fraud_data['amount'] > 0).all()

    def test_merchant_categories(self, sample_fraud_data):
        """Test that merchant categories are valid."""
        valid_categories = [
            'grocery', 'gas_station', 'restaurant', 'online_retail',
            'entertainment', 'travel', 'healthcare', 'utilities'
        ]

        assert sample_fraud_data['merchant_category'].isin(valid_categories).all()

    def test_velocity_features_non_negative(self, sample_fraud_data):
        """Test that velocity features are non-negative."""
        assert (sample_fraud_data['transactions_last_24h'] >= 0).all()
        assert (sample_fraud_data['transactions_last_1h'] >= 0).all()
        assert (sample_fraud_data['total_amount_last_24h'] >= 0).all()

    def test_timestamp_type(self, sample_fraud_data):
        """Test that timestamp is datetime."""
        assert pd.api.types.is_datetime64_any_dtype(sample_fraud_data['timestamp'])

    def test_save_csv(self, temp_dir):
        """Test saving data to CSV."""
        generator = FraudDataGenerator(n_samples=10, seed=42)
        df = generator.generate()

        filepath = generator.save(df, output_dir=temp_dir, file_format='csv')

        assert filepath.exists()
        assert filepath.suffix == '.csv'

        # Load and verify
        loaded_df = pd.read_csv(filepath)
        assert len(loaded_df) == 10

    def test_save_parquet(self, temp_dir):
        """Test saving data to Parquet."""
        generator = FraudDataGenerator(n_samples=10, seed=42)
        df = generator.generate()

        filepath = generator.save(df, output_dir=temp_dir, file_format='parquet')

        assert filepath.exists()
        assert filepath.suffix == '.parquet'

        # Load and verify
        loaded_df = pd.read_parquet(filepath)
        assert len(loaded_df) == 10

    def test_reproducibility(self):
        """Test that same seed produces same data."""
        gen1 = FraudDataGenerator(n_samples=100, seed=42)
        gen2 = FraudDataGenerator(n_samples=100, seed=42)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Compare transaction IDs (should be identical)
        assert (df1['transaction_id'] == df2['transaction_id']).all()
        assert (df1['amount'] == df2['amount']).all()


class TestFraudPattern:
    """Test FraudPattern configuration."""

    def test_fraud_pattern_defaults(self):
        """Test FraudPattern default values."""
        pattern = FraudPattern()

        assert pattern.unusual_amount_multiplier == 5.0
        assert pattern.unusual_time_prob == 0.7
        assert pattern.foreign_transaction_prob == 0.6
        assert pattern.velocity_multiplier == 10.0
