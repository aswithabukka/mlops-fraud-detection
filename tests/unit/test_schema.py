"""Unit tests for data schemas."""
import pytest
import pandas as pd
import pandera as pa

from src.data.schema import (
    FraudTransactionSchema,
    validate_data,
    get_raw_data_schema,
)


class TestFraudTransactionSchema:
    """Test FraudTransactionSchema validation."""

    def test_valid_data(self, sample_transactions):
        """Test validation with valid data."""
        schema = FraudTransactionSchema.to_schema()
        validated_df = schema.validate(sample_transactions)

        assert isinstance(validated_df, pd.DataFrame)
        assert len(validated_df) == len(sample_transactions)

    def test_invalid_transaction_id_format(self, sample_transactions):
        """Test that invalid transaction ID format fails."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'transaction_id'] = 'INVALID'

        schema = FraudTransactionSchema.to_schema()

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_df)

    def test_negative_amount(self, sample_transactions):
        """Test that negative amounts fail validation."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'amount'] = -100.0

        schema = FraudTransactionSchema.to_schema()

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_df)

    def test_invalid_merchant_category(self, sample_transactions):
        """Test that invalid merchant category fails."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'merchant_category'] = 'invalid_category'

        schema = FraudTransactionSchema.to_schema()

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_df)

    def test_hour_out_of_range(self, sample_transactions):
        """Test that hour out of range fails."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'hour_of_day'] = 25

        schema = FraudTransactionSchema.to_schema()

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_df)

    def test_fraud_label_values(self, sample_transactions):
        """Test that is_fraud must be 0 or 1."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'is_fraud'] = 2

        schema = FraudTransactionSchema.to_schema()

        with pytest.raises(pa.errors.SchemaError):
            schema.validate(invalid_df)


class TestValidateData:
    """Test validate_data function."""

    def test_validate_valid_data(self, sample_fraud_data):
        """Test validation with valid generated data."""
        is_valid, error = validate_data(sample_fraud_data, schema_type='raw')

        assert is_valid is True
        assert error is None

    def test_validate_returns_error_message(self, sample_transactions):
        """Test that validation returns error message on failure."""
        invalid_df = sample_transactions.copy()
        invalid_df.loc[0, 'amount'] = -50.0

        is_valid, error = validate_data(invalid_df, schema_type='raw')

        assert is_valid is False
        assert error is not None
        assert isinstance(error, str)


class TestGetSchema:
    """Test schema getter functions."""

    def test_get_raw_data_schema(self):
        """Test getting raw data schema."""
        schema = get_raw_data_schema()

        assert isinstance(schema, pa.DataFrameSchema)
        assert 'transaction_id' in schema.columns
        assert 'is_fraud' in schema.columns
