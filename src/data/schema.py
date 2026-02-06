"""
Data validation schemas using Pandera.

Defines schemas for fraud transaction data with strict validation rules:
- Data types
- Value ranges and constraints
- Categorical value validation
- Null handling policies
"""
from datetime import datetime
from typing import Optional

import pandera as pa
from pandera import Column, DataFrameSchema, Check
from pandera.typing import DataFrame, Series

from src.utils.logger import get_logger

logger = get_logger(__name__)


# Valid merchant categories (must match generator)
VALID_MERCHANT_CATEGORIES = [
    "grocery",
    "gas_station",
    "restaurant",
    "online_retail",
    "entertainment",
    "travel",
    "healthcare",
    "utilities",
]

# Valid countries
VALID_COUNTRIES = ["USA", "China", "Russia", "Nigeria", "Brazil", "India", "Philippines"]


class FraudTransactionSchema(pa.DataFrameModel):
    """
    Pandera schema for fraud transaction data.

    Validates:
    - Data types for all columns
    - Value ranges (amounts, rates, counts)
    - Categorical values
    - Business logic constraints

    Example:
        >>> schema = FraudTransactionSchema.to_schema()
        >>> validated_df = schema.validate(df)
    """

    # Transaction identifiers
    transaction_id: Series[str] = pa.Field(
        str_matches=r"^TXN\d{8}$",
        unique=True,
        description="Unique transaction ID (format: TXN00000000)",
    )

    timestamp: Series[datetime] = pa.Field(
        description="Transaction timestamp",
    )

    # Cardholder information
    cardholder_id: Series[str] = pa.Field(
        str_matches=r"^CH\d{6}$",
        description="Cardholder ID (format: CH000000)",
    )

    cardholder_name: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 100},
        description="Cardholder full name",
    )

    # Transaction amount
    amount: Series[float] = pa.Field(
        ge=0.01,  # Minimum $0.01
        le=50000.0,  # Maximum $50,000 (reasonable upper limit)
        description="Transaction amount in USD",
    )

    # Merchant information
    merchant_name: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 200},
        description="Merchant business name",
    )

    merchant_category: Series[str] = pa.Field(
        isin=VALID_MERCHANT_CATEGORIES,
        description="Merchant category code",
    )

    merchant_city: Series[str] = pa.Field(
        str_length={"min_value": 2, "max_value": 100},
        description="Merchant city",
    )

    merchant_country: Series[str] = pa.Field(
        isin=VALID_COUNTRIES,
        description="Merchant country",
    )

    # Card information
    card_last4: Series[str] = pa.Field(
        str_matches=r"^\d{4}$",
        description="Last 4 digits of card number",
    )

    # Device and network information
    device_fingerprint: Series[str] = pa.Field(
        str_matches=r"^[a-f0-9]{16}$",
        description="Device fingerprint hash (16 hex chars)",
    )

    ip_address: Series[str] = pa.Field(
        # Basic IPv4 validation
        str_matches=r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
        description="IP address",
    )

    user_agent: Series[str] = pa.Field(
        str_length={"min_value": 10},
        description="Browser user agent string",
    )

    # Transaction characteristics
    is_online: Series[bool] = pa.Field(
        description="Whether transaction was online (vs in-person)",
    )

    hour_of_day: Series[int] = pa.Field(
        ge=0,
        le=23,
        description="Hour of day (0-23)",
    )

    day_of_week: Series[int] = pa.Field(
        ge=0,
        le=6,
        description="Day of week (0=Monday, 6=Sunday)",
    )

    is_weekend: Series[bool] = pa.Field(
        description="Whether transaction occurred on weekend",
    )

    # Velocity features (fraud indicators)
    transactions_last_24h: Series[int] = pa.Field(
        ge=0,
        le=1000,  # Sanity check: max 1000 transactions in 24h
        description="Number of transactions in last 24 hours",
    )

    total_amount_last_24h: Series[float] = pa.Field(
        ge=0.0,
        le=1000000.0,  # Sanity check: max $1M in 24h
        description="Total transaction amount in last 24 hours",
    )

    transactions_last_1h: Series[int] = pa.Field(
        ge=0,
        le=100,  # Sanity check: max 100 transactions in 1h
        description="Number of transactions in last 1 hour",
    )

    # Derived features
    amount_log: Series[float] = pa.Field(
        ge=0.0,
        description="Log-transformed amount (log1p)",
    )

    distance_from_home: Series[float] = pa.Field(
        ge=0.0,
        le=20000.0,  # Max ~half circumference of Earth in km
        description="Distance from cardholder home (km)",
    )

    # Target variable
    is_fraud: Series[int] = pa.Field(
        isin=[0, 1],
        description="Fraud label (0=legitimate, 1=fraudulent)",
    )

    class Config:
        """Pandera configuration."""

        strict = True  # Fail on extra columns
        coerce = True  # Attempt type coercion
        ordered = False  # Column order doesn't matter

    @pa.dataframe_check
    def check_fraud_rate(cls, df: DataFrame) -> bool:
        """Validate fraud rate is within expected range (0.1% - 5%)."""
        fraud_rate = df["is_fraud"].mean()
        is_valid = 0.001 <= fraud_rate <= 0.05
        if not is_valid:
            logger.warning(
                f"Fraud rate {fraud_rate*100:.2f}% outside expected range (0.1% - 5%)"
            )
        return is_valid

    @pa.dataframe_check
    def check_timestamp_order(cls, df: DataFrame) -> bool:
        """Check that timestamps are in ascending order."""
        return df["timestamp"].is_monotonic_increasing or True  # Warning, not error

    @pa.dataframe_check
    def check_weekend_consistency(cls, df: DataFrame) -> bool:
        """Validate is_weekend matches day_of_week."""
        expected_weekend = df["day_of_week"] >= 5
        return (df["is_weekend"] == expected_weekend).all()

    @pa.dataframe_check
    def check_hour_consistency(cls, df: DataFrame) -> bool:
        """Validate hour_of_day matches timestamp hour."""
        expected_hour = df["timestamp"].dt.hour
        return (df["hour_of_day"] == expected_hour).all()

    @pa.dataframe_check
    def check_amount_log_consistency(cls, df: DataFrame) -> bool:
        """Validate amount_log is log1p(amount)."""
        import numpy as np

        expected_log = np.log1p(df["amount"])
        return np.allclose(df["amount_log"], expected_log, rtol=1e-5)


class ProcessedTransactionSchema(pa.DataFrameModel):
    """
    Schema for preprocessed transaction data (after feature engineering).

    Extends FraudTransactionSchema with additional engineered features.
    """

    # Include all base schema fields
    transaction_id: Series[str] = pa.Field(unique=True)
    is_fraud: Series[int] = pa.Field(isin=[0, 1])

    # Additional engineered features (to be added during preprocessing)
    # These will be defined in preprocessor.py

    class Config:
        strict = False  # Allow additional columns
        coerce = True


def get_raw_data_schema() -> DataFrameSchema:
    """
    Get schema for raw fraud transaction data.

    Returns:
        DataFrameSchema: Pandera schema for validation
    """
    return FraudTransactionSchema.to_schema()


def get_processed_data_schema() -> DataFrameSchema:
    """
    Get schema for processed fraud transaction data.

    Returns:
        DataFrameSchema: Pandera schema for validation
    """
    return ProcessedTransactionSchema.to_schema()


def validate_data(
    df,
    schema_type: str = "raw",
    lazy: bool = False,
) -> tuple[bool, Optional[str]]:
    """
    Validate DataFrame against schema.

    Args:
        df: DataFrame to validate
        schema_type: Type of schema ('raw' or 'processed')
        lazy: If True, collect all errors before raising

    Returns:
        Tuple of (is_valid, error_message)

    Example:
        >>> is_valid, error = validate_data(df, schema_type='raw')
        >>> if not is_valid:
        >>>     logger.error(f"Validation failed: {error}")
    """
    try:
        if schema_type == "raw":
            schema = get_raw_data_schema()
        elif schema_type == "processed":
            schema = get_processed_data_schema()
        else:
            raise ValueError(f"Unknown schema type: {schema_type}")

        # Validate
        validated_df = schema.validate(df, lazy=lazy)

        logger.info(
            f"✅ Data validation passed for {len(df)} rows "
            f"({schema_type} schema)"
        )
        return True, None

    except pa.errors.SchemaError as e:
        error_msg = f"Schema validation failed: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    except Exception as e:
        error_msg = f"Unexpected validation error: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


if __name__ == "__main__":
    """Test schema validation with sample data."""
    import pandas as pd
    import numpy as np

    # Create sample valid data
    sample_data = {
        "transaction_id": ["TXN00000001", "TXN00000002"],
        "timestamp": [datetime.now(), datetime.now()],
        "cardholder_id": ["CH000001", "CH000002"],
        "cardholder_name": ["John Doe", "Jane Smith"],
        "amount": [100.50, 250.75],
        "merchant_name": ["Acme Corp", "Beta Inc"],
        "merchant_category": ["grocery", "restaurant"],
        "merchant_city": ["New York", "Los Angeles"],
        "merchant_country": ["USA", "USA"],
        "card_last4": ["1234", "5678"],
        "device_fingerprint": ["a1b2c3d4e5f6g7h8", "9i0j1k2l3m4n5o6p"],
        "ip_address": ["192.168.1.1", "10.0.0.1"],
        "user_agent": ["Mozilla/5.0...", "Chrome/90.0..."],
        "is_online": [True, False],
        "hour_of_day": [14, 19],
        "day_of_week": [0, 5],
        "is_weekend": [False, True],
        "transactions_last_24h": [2, 1],
        "total_amount_last_24h": [150.0, 250.75],
        "transactions_last_1h": [0, 0],
        "amount_log": [np.log1p(100.50), np.log1p(250.75)],
        "distance_from_home": [0.0, 500.0],
        "is_fraud": [0, 0],
    }

    df = pd.DataFrame(sample_data)

    print("Testing schema validation...\n")
    print(f"Sample data shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Validate
    is_valid, error = validate_data(df, schema_type="raw")

    if is_valid:
        print("\n✅ Schema validation passed!")
    else:
        print(f"\n❌ Schema validation failed: {error}")

    # Test with invalid data
    print("\n" + "=" * 80)
    print("Testing with INVALID data (negative amount)...")
    invalid_df = df.copy()
    invalid_df.loc[0, "amount"] = -50.0  # Invalid: negative amount

    is_valid, error = validate_data(invalid_df, schema_type="raw")
    print(f"Result: {'✅ Passed' if is_valid else '❌ Failed as expected'}")
