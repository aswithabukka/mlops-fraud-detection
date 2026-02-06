"""
Data validation utilities.

Provides high-level validation functions for fraud transaction data
including schema validation, statistical checks, and data quality reports.
"""
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from src.data.schema import validate_data, get_raw_data_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Comprehensive data validator for fraud transaction data.

    Performs:
    - Schema validation
    - Statistical validation
    - Data quality checks
    - Distribution analysis
    """

    def __init__(self, schema_type: str = "raw"):
        """
        Initialize validator.

        Args:
            schema_type: Type of schema to validate against ('raw' or 'processed')
        """
        self.schema_type = schema_type
        self.validation_results = []

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Validate DataFrame against Pandera schema.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        logger.info("Running schema validation...")
        is_valid, error = validate_data(df, schema_type=self.schema_type, lazy=False)

        self.validation_results.append(
            {
                "check": "Schema Validation",
                "passed": is_valid,
                "message": error if not is_valid else "All schema checks passed",
            }
        )

        return is_valid, error

    def validate_nulls(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Check for null values in required columns.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of columns with nulls)
        """
        logger.info("Checking for null values...")

        null_cols = df.columns[df.isnull().any()].tolist()
        null_counts = df[null_cols].isnull().sum().to_dict() if null_cols else {}

        is_valid = len(null_cols) == 0

        message = (
            "No null values found"
            if is_valid
            else f"Found nulls in columns: {null_counts}"
        )

        self.validation_results.append(
            {
                "check": "Null Value Check",
                "passed": is_valid,
                "message": message,
            }
        )

        return is_valid, null_cols

    def validate_duplicates(
        self, df: pd.DataFrame, subset: Optional[List[str]] = None
    ) -> Tuple[bool, int]:
        """
        Check for duplicate rows.

        Args:
            df: DataFrame to validate
            subset: Columns to check for duplicates (default: transaction_id)

        Returns:
            Tuple of (is_valid, number of duplicates)
        """
        logger.info("Checking for duplicate rows...")

        if subset is None:
            subset = ["transaction_id"]

        n_duplicates = df.duplicated(subset=subset).sum()
        is_valid = n_duplicates == 0

        message = (
            "No duplicates found"
            if is_valid
            else f"Found {n_duplicates} duplicate rows on {subset}"
        )

        self.validation_results.append(
            {
                "check": "Duplicate Check",
                "passed": is_valid,
                "message": message,
            }
        )

        return is_valid, n_duplicates

    def validate_distributions(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate statistical distributions.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, distribution statistics)
        """
        logger.info("Validating statistical distributions...")

        stats = {}
        warnings = []

        # Check fraud rate
        fraud_rate = df["is_fraud"].mean()
        stats["fraud_rate"] = fraud_rate
        if not (0.001 <= fraud_rate <= 0.05):
            warnings.append(
                f"Fraud rate {fraud_rate*100:.2f}% outside expected range (0.1%-5%)"
            )

        # Check amount distribution
        amount_stats = df["amount"].describe()
        stats["amount_mean"] = amount_stats["mean"]
        stats["amount_median"] = amount_stats["50%"]
        stats["amount_std"] = amount_stats["std"]

        # Check for outliers (amounts beyond 3 std devs)
        amount_outliers = (
            (df["amount"] > amount_stats["mean"] + 3 * amount_stats["std"])
            | (df["amount"] < amount_stats["mean"] - 3 * amount_stats["std"])
        ).sum()

        stats["amount_outliers"] = amount_outliers
        if amount_outliers > len(df) * 0.05:  # More than 5% outliers
            warnings.append(
                f"High number of amount outliers: {amount_outliers} "
                f"({amount_outliers/len(df)*100:.1f}%)"
            )

        # Check velocity features
        velocity_24h_mean = df["transactions_last_24h"].mean()
        stats["velocity_24h_mean"] = velocity_24h_mean
        if velocity_24h_mean > 50:  # Sanity check
            warnings.append(
                f"Unusually high average velocity: {velocity_24h_mean:.1f} txns/24h"
            )

        # Check category distribution
        category_dist = df["merchant_category"].value_counts(normalize=True)
        stats["category_distribution"] = category_dist.to_dict()

        # Check if any category is overrepresented (>50%)
        if category_dist.max() > 0.5:
            warnings.append(
                f"Category '{category_dist.idxmax()}' is overrepresented "
                f"({category_dist.max()*100:.1f}%)"
            )

        is_valid = len(warnings) == 0
        message = (
            "All distributions look healthy"
            if is_valid
            else f"Distribution warnings: {'; '.join(warnings)}"
        )

        self.validation_results.append(
            {
                "check": "Distribution Validation",
                "passed": is_valid,
                "message": message,
            }
        )

        return is_valid, stats

    def validate_all(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Run all validation checks.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (all_valid, results_dict)
        """
        logger.info(f"Running comprehensive validation on {len(df)} rows...")

        self.validation_results = []

        # Run all checks
        schema_valid, schema_error = self.validate_schema(df)
        nulls_valid, null_cols = self.validate_nulls(df)
        dups_valid, n_dups = self.validate_duplicates(df)
        dist_valid, dist_stats = self.validate_distributions(df)

        # Aggregate results
        all_valid = all(r["passed"] for r in self.validation_results)

        results = {
            "overall_valid": all_valid,
            "checks": self.validation_results,
            "statistics": dist_stats,
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "null_columns": len(null_cols),
                "duplicates": n_dups,
                "fraud_rate": dist_stats.get("fraud_rate", 0),
            },
        }

        return all_valid, results

    def print_report(self, results: Dict) -> None:
        """
        Print validation report.

        Args:
            results: Results dictionary from validate_all()
        """
        print("\n" + "=" * 80)
        print("DATA VALIDATION REPORT")
        print("=" * 80)

        # Summary
        summary = results["summary"]
        print(f"\n📊 Summary:")
        print(f"   Total Rows: {summary['total_rows']:,}")
        print(f"   Total Columns: {summary['total_columns']}")
        print(f"   Fraud Rate: {summary['fraud_rate']*100:.2f}%")
        print(f"   Null Columns: {summary['null_columns']}")
        print(f"   Duplicates: {summary['duplicates']}")

        # Validation checks
        print(f"\n✅ Validation Checks:")
        for check in results["checks"]:
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            print(f"   {status} - {check['check']}")
            if not check["passed"]:
                print(f"      └─ {check['message']}")

        # Overall result
        print(f"\n{'='*80}")
        if results["overall_valid"]:
            print("✅ ALL VALIDATION CHECKS PASSED")
        else:
            print("❌ SOME VALIDATION CHECKS FAILED")
            print("   Review the errors above and fix data quality issues")
        print("=" * 80 + "\n")


def validate_file(file_path: Path, schema_type: str = "raw") -> bool:
    """
    Validate data from a file.

    Args:
        file_path: Path to data file (CSV or Parquet)
        schema_type: Schema type to validate against

    Returns:
        True if validation passes, False otherwise
    """
    logger.info(f"Loading data from {file_path}")

    # Load data
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
    elif file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Validate
    validator = DataValidator(schema_type=schema_type)
    all_valid, results = validator.validate_all(df)

    # Print report
    validator.print_report(results)

    return all_valid


def main():
    """Command-line interface for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate fraud transaction data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate raw data
  python -m src.data.validator data/raw/fraud_20240101.csv

  # Validate processed data
  python -m src.data.validator data/processed/fraud_20240101.parquet --schema processed
        """,
    )

    parser.add_argument(
        "file_path",
        type=str,
        help="Path to data file (CSV or Parquet)",
    )

    parser.add_argument(
        "--schema",
        type=str,
        choices=["raw", "processed"],
        default="raw",
        help="Schema type to validate against",
    )

    args = parser.parse_args()

    # Validate
    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return 1

    is_valid = validate_file(file_path, schema_type=args.schema)

    # Exit with appropriate code
    return 0 if is_valid else 1


if __name__ == "__main__":
    exit(main())
