"""
Synthetic fraud transaction data generator.

Creates realistic credit card transaction data with fraud patterns including:
- Normal spending behavior
- Fraudulent transactions (0.5-2% fraud rate)
- Fraud indicators: velocity, location anomalies, unusual amounts
- Transaction features: amount, merchant, location, time, device
"""
import hashlib
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FraudPattern:
    """Configuration for fraud pattern generation."""

    unusual_amount_multiplier: float = 5.0  # Fraud amounts are 5x higher
    unusual_time_prob: float = 0.7  # 70% of fraud happens at unusual times
    foreign_transaction_prob: float = 0.6  # 60% of fraud is foreign
    velocity_multiplier: float = 10.0  # Fraud has 10x more transactions per day


class FraudDataGenerator:
    """
    Generate synthetic credit card fraud transaction data.

    Features:
    - Realistic transaction patterns for normal and fraudulent behavior
    - Highly imbalanced dataset (configurable fraud rate 0.5-2%)
    - Transaction metadata: amount, merchant, location, time, device
    - Fraud indicators: velocity, location anomalies, unusual amounts

    Example:
        >>> generator = FraudDataGenerator(n_samples=100000, fraud_rate=0.01)
        >>> df = generator.generate()
        >>> df.to_csv("fraud_transactions.csv", index=False)
    """

    def __init__(
        self,
        n_samples: int = 100000,
        fraud_rate: float = 0.01,
        n_cardholders: int = 10000,
        seed: Optional[int] = 42,
    ):
        """
        Initialize fraud data generator.

        Args:
            n_samples: Total number of transactions to generate
            fraud_rate: Percentage of fraudulent transactions (0.0-1.0)
            n_cardholders: Number of unique cardholders
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.n_cardholders = n_cardholders
        self.seed = seed

        # Initialize random generators
        np.random.seed(seed)
        random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)

        # Initialize fraud pattern configuration
        self.fraud_pattern = FraudPattern()

        # Merchant categories with spending patterns
        self.merchant_categories = {
            "grocery": {"avg_amount": 75, "std": 30, "weight": 0.25},
            "gas_station": {"avg_amount": 50, "std": 20, "weight": 0.15},
            "restaurant": {"avg_amount": 45, "std": 25, "weight": 0.20},
            "online_retail": {"avg_amount": 120, "std": 80, "weight": 0.15},
            "entertainment": {"avg_amount": 60, "std": 40, "weight": 0.10},
            "travel": {"avg_amount": 350, "std": 200, "weight": 0.05},
            "healthcare": {"avg_amount": 150, "std": 100, "weight": 0.05},
            "utilities": {"avg_amount": 100, "std": 30, "weight": 0.05},
        }

        # Generate cardholders
        self.cardholders = self._generate_cardholders()

        logger.info(
            f"Initialized FraudDataGenerator: "
            f"n_samples={n_samples}, fraud_rate={fraud_rate*100:.2f}%, "
            f"n_cardholders={n_cardholders}"
        )

    def _generate_cardholders(self) -> List[Dict]:
        """Generate cardholder profiles."""
        cardholders = []
        for i in range(self.n_cardholders):
            cardholder = {
                "cardholder_id": f"CH{i:06d}",
                "name": self.faker.name(),
                "email": self.faker.email(),
                "phone": self.faker.phone_number(),
                "home_location": self.faker.city(),
                "home_country": "USA",
                "avg_transaction_per_day": np.random.gamma(2, 1.5),  # Average ~3 per day
                "avg_amount": np.random.gamma(5, 20),  # Average ~$100
            }
            cardholders.append(cardholder)
        return cardholders

    def _generate_device_fingerprint(self) -> str:
        """Generate a device fingerprint hash."""
        device_info = f"{self.faker.user_agent()}_{random.randint(1000, 9999)}"
        return hashlib.md5(device_info.encode()).hexdigest()[:16]

    def _select_merchant_category(self) -> str:
        """Select merchant category based on weights."""
        categories = list(self.merchant_categories.keys())
        weights = [self.merchant_categories[cat]["weight"] for cat in categories]
        return np.random.choice(categories, p=weights)

    def _generate_transaction_amount(
        self, category: str, is_fraud: bool = False
    ) -> float:
        """Generate transaction amount based on category and fraud status."""
        cat_info = self.merchant_categories[category]
        base_amount = np.random.normal(cat_info["avg_amount"], cat_info["std"])

        if is_fraud:
            # Fraudulent transactions tend to be larger
            base_amount *= self.fraud_pattern.unusual_amount_multiplier

        # Ensure amount is positive and round to 2 decimals
        amount = max(1.0, base_amount)
        return round(amount, 2)

    def _generate_timestamp(
        self, start_date: datetime, is_fraud: bool = False
    ) -> datetime:
        """Generate transaction timestamp."""
        # Random time within 90 days
        random_seconds = random.randint(0, 90 * 24 * 60 * 60)
        timestamp = start_date + timedelta(seconds=random_seconds)

        if is_fraud and random.random() < self.fraud_pattern.unusual_time_prob:
            # Fraudulent transactions often happen at unusual times (late night)
            timestamp = timestamp.replace(
                hour=random.randint(23, 23) if random.random() > 0.5 else random.randint(2, 4),
                minute=random.randint(0, 59),
            )

        return timestamp

    def _generate_location(
        self, cardholder: Dict, is_fraud: bool = False
    ) -> Tuple[str, str]:
        """Generate transaction location (city, country)."""
        if is_fraud and random.random() < self.fraud_pattern.foreign_transaction_prob:
            # Fraudulent transactions more likely to be foreign
            city = self.faker.city()
            country = random.choice(
                ["China", "Russia", "Nigeria", "Brazil", "India", "Philippines"]
            )
        else:
            # Normal transactions near cardholder's home
            if random.random() < 0.85:  # 85% near home
                city = cardholder["home_location"]
                country = cardholder["home_country"]
            else:
                city = self.faker.city()
                country = "USA"

        return city, country

    def _calculate_velocity_features(
        self, cardholder_id: str, timestamp: datetime, transactions: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate velocity features (transaction frequency indicators).

        Args:
            cardholder_id: Cardholder ID
            timestamp: Current transaction timestamp
            transactions: List of previous transactions

        Returns:
            Dictionary with velocity features
        """
        # Filter transactions for this cardholder in last 24 hours
        recent_transactions = [
            t
            for t in transactions
            if t["cardholder_id"] == cardholder_id
            and (timestamp - t["timestamp"]).total_seconds() < 86400
        ]

        # Calculate features
        velocity_24h = len(recent_transactions)
        total_amount_24h = sum(t["amount"] for t in recent_transactions)

        # Transactions in last hour
        last_hour = [
            t
            for t in recent_transactions
            if (timestamp - t["timestamp"]).total_seconds() < 3600
        ]
        velocity_1h = len(last_hour)

        return {
            "transactions_last_24h": velocity_24h,
            "total_amount_last_24h": round(total_amount_24h, 2),
            "transactions_last_1h": velocity_1h,
        }

    def generate(self) -> pd.DataFrame:
        """
        Generate synthetic fraud transaction dataset.

        Returns:
            pd.DataFrame: Transaction data with fraud labels
        """
        logger.info(f"Generating {self.n_samples} transactions...")

        # Determine fraud/non-fraud split
        n_fraud = int(self.n_samples * self.fraud_rate)
        n_normal = self.n_samples - n_fraud

        logger.info(f"Normal transactions: {n_normal}, Fraudulent: {n_fraud}")

        transactions = []
        start_date = datetime.now() - timedelta(days=90)

        # Track transactions for velocity features
        temp_transactions = []

        # Generate transactions
        for i in range(self.n_samples):
            is_fraud = i >= n_normal  # Last n_fraud transactions are fraudulent

            # Select cardholder
            if is_fraud:
                # Fraudulent transactions can target any cardholder
                cardholder = random.choice(self.cardholders)
            else:
                # Normal transactions follow cardholder patterns
                cardholder = random.choice(self.cardholders)

            # Generate transaction features
            category = self._select_merchant_category()
            amount = self._generate_transaction_amount(category, is_fraud)
            timestamp = self._generate_timestamp(start_date, is_fraud)
            city, country = self._generate_location(cardholder, is_fraud)

            # Calculate velocity features
            velocity_features = self._calculate_velocity_features(
                cardholder["cardholder_id"], timestamp, temp_transactions
            )

            # Generate transaction
            transaction = {
                "transaction_id": f"TXN{i:08d}",
                "timestamp": timestamp,
                "cardholder_id": cardholder["cardholder_id"],
                "cardholder_name": cardholder["name"],
                "amount": amount,
                "merchant_name": self.faker.company(),
                "merchant_category": category,
                "merchant_city": city,
                "merchant_country": country,
                "card_last4": f"{random.randint(1000, 9999)}",
                "device_fingerprint": self._generate_device_fingerprint(),
                "ip_address": self.faker.ipv4(),
                "user_agent": self.faker.user_agent(),
                "is_online": random.choice([True, False]),
                "hour_of_day": timestamp.hour,
                "day_of_week": timestamp.weekday(),
                "is_weekend": timestamp.weekday() >= 5,
                **velocity_features,
                "is_fraud": 1 if is_fraud else 0,
            }

            transactions.append(transaction)
            temp_transactions.append(
                {
                    "cardholder_id": cardholder["cardholder_id"],
                    "timestamp": timestamp,
                    "amount": amount,
                }
            )

            # Log progress
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1}/{self.n_samples} transactions...")

        # Convert to DataFrame
        df = pd.DataFrame(transactions)

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Add derived features
        df["amount_log"] = np.log1p(df["amount"])
        df["distance_from_home"] = df.apply(
            lambda row: (
                0.0
                if row["merchant_country"] == "USA"
                and row["merchant_city"]
                == self._get_cardholder_home(row["cardholder_id"])
                else random.uniform(50, 5000)
            ),
            axis=1,
        )

        logger.info(f"Generated dataset shape: {df.shape}")
        logger.info(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
        logger.info(f"Features: {list(df.columns)}")

        return df

    def _get_cardholder_home(self, cardholder_id: str) -> str:
        """Get cardholder's home location."""
        for ch in self.cardholders:
            if ch["cardholder_id"] == cardholder_id:
                return ch["home_location"]
        return "Unknown"

    def save(
        self,
        df: pd.DataFrame,
        output_dir: Path = Path("data/raw"),
        file_format: str = "csv",
    ) -> Path:
        """
        Save generated data to file.

        Args:
            df: DataFrame to save
            output_dir: Output directory
            file_format: File format ('csv' or 'parquet')

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"fraud_{timestamp}.{file_format}"
        filepath = output_dir / filename

        # Save file
        if file_format == "csv":
            df.to_csv(filepath, index=False)
        elif file_format == "parquet":
            df.to_parquet(filepath, index=False, engine="pyarrow")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        logger.info(f"Saved {len(df)} transactions to {filepath}")
        return filepath


def main():
    """Main function to generate and save fraud data."""
    # Use settings from config
    generator = FraudDataGenerator(
        n_samples=settings.data_size,
        fraud_rate=settings.fraud_rate,
        n_cardholders=settings.data_size // 10,  # 10 transactions per cardholder avg
    )

    # Generate data
    df = generator.generate()

    # Display summary statistics
    print("\n" + "=" * 80)
    print("FRAUD TRANSACTION DATA SUMMARY")
    print("=" * 80)
    print(f"\nTotal Transactions: {len(df):,}")
    print(f"Fraud Transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"Normal Transactions: {(1-df['is_fraud']).sum():,}")
    print(f"\nDate Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nAmount Statistics:")
    print(df.groupby("is_fraud")["amount"].describe())
    print(f"\nMerchant Category Distribution:")
    print(df["merchant_category"].value_counts())
    print(f"\nTop 5 Features:")
    print(df.head())

    # Save both CSV and Parquet
    csv_path = generator.save(df, file_format="csv")
    parquet_path = generator.save(df, file_format="parquet")

    print(f"\n✅ Data saved to:")
    print(f"   - {csv_path}")
    print(f"   - {parquet_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
