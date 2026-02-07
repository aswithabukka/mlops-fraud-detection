"""
Data drift detection using EvidentlyAI.

Detects distribution shifts between reference (training) data and current (production) data.
Generates drift reports and triggers alerts when thresholds exceeded.
"""
from typing import Dict, List, Optional, Union
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.metrics import DataDriftTable
except ImportError:
    logging.warning("EvidentlyAI not installed. Run: pip install evidently")


class DriftDetector:
    """
    Detect data drift using statistical tests.

    Uses EvidentlyAI to compare reference (training) data with current (production) data
    and calculate drift scores for each feature.
    """

    def __init__(
        self,
        drift_threshold: float = 0.15,
        severe_threshold: float = 0.2,
    ):
        """
        Initialize drift detector.

        Args:
            drift_threshold: Threshold for moderate drift warning (0-1)
            severe_threshold: Threshold for severe drift requiring action (0-1)
        """
        self.drift_threshold = drift_threshold
        self.severe_threshold = severe_threshold
        self.logger = logging.getLogger(__name__)

    def detect_drift(
        self,
        reference_data: Union[pd.DataFrame, str],
        current_data: Union[pd.DataFrame, str],
        save_report: bool = True,
        report_dir: str = "./monitoring/reports",
    ) -> Dict:
        """
        Detect drift between reference and current data.

        Args:
            reference_data: Reference dataset (training data) as DataFrame or path
            current_data: Current dataset (recent predictions) as DataFrame or path
            save_report: Whether to save HTML drift report
            report_dir: Directory to save reports

        Returns:
            Dictionary with drift results:
                - drift_detected: bool
                - drift_score: float (0-1)
                - severe_drift: bool
                - drifted_features: list of feature names
                - report_path: path to HTML report (if saved)
        """
        self.logger.info("Starting drift detection...")

        # Load data
        df_reference = self._load_data(reference_data)
        df_current = self._load_data(current_data)

        self.logger.info(f"Reference data: {df_reference.shape}")
        self.logger.info(f"Current data: {df_current.shape}")

        # Ensure same columns
        common_cols = list(set(df_reference.columns) & set(df_current.columns))

        if not common_cols:
            raise ValueError("No common columns between reference and current data")

        df_reference = df_reference[common_cols]
        df_current = df_current[common_cols]

        # Create drift report
        try:
            report = Report(metrics=[
                DataDriftPreset(),
            ])

            report.run(
                reference_data=df_reference,
                current_data=df_current,
            )

            # Extract drift metrics
            drift_results = self._extract_drift_metrics(report)

            # Determine drift severity
            drift_score = drift_results['drift_score']
            drift_detected = drift_score >= self.drift_threshold
            severe_drift = drift_score >= self.severe_threshold

            self.logger.info(f"Drift score: {drift_score:.4f}")
            self.logger.info(f"Drift detected: {drift_detected}")
            self.logger.info(f"Severe drift: {severe_drift}")

            # Save report
            report_path = None
            if save_report:
                report_path = self._save_report(report, report_dir)
                self.logger.info(f"Drift report saved to: {report_path}")

            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'severe_drift': severe_drift,
                'drifted_features': drift_results['drifted_features'],
                'n_features': drift_results['n_features'],
                'n_drifted': drift_results['n_drifted'],
                'report_path': report_path,
                'timestamp': datetime.now().isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Drift detection failed: {str(e)}")
            raise

    def _load_data(self, data: Union[pd.DataFrame, str]) -> pd.DataFrame:
        """Load data from DataFrame or file path."""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, str):
            path = Path(data)
            if path.suffix == '.csv':
                return pd.read_csv(path)
            elif path.suffix == '.parquet':
                return pd.read_parquet(path)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

    def _extract_drift_metrics(self, report: Report) -> Dict:
        """Extract drift metrics from EvidentlyAI report."""
        try:
            # Get report as dictionary
            report_dict = report.as_dict()

            # Navigate to drift metrics
            metrics = report_dict.get('metrics', [])

            # Find DataDriftTable metric
            drift_table = None
            for metric in metrics:
                if metric.get('metric') == 'DataDriftTable':
                    drift_table = metric.get('result', {})
                    break

            if not drift_table:
                # Fallback: calculate basic drift
                return self._calculate_basic_drift(report_dict)

            # Extract drift information
            n_features = drift_table.get('number_of_columns', 0)
            n_drifted = drift_table.get('number_of_drifted_columns', 0)
            drift_share = drift_table.get('share_of_drifted_columns', 0.0)

            # Get drifted feature names
            drifted_features = []
            drift_by_columns = drift_table.get('drift_by_columns', {})

            for col_name, col_drift in drift_by_columns.items():
                if col_drift.get('drift_detected', False):
                    drifted_features.append(col_name)

            return {
                'drift_score': drift_share,
                'n_features': n_features,
                'n_drifted': n_drifted,
                'drifted_features': drifted_features,
            }

        except Exception as e:
            self.logger.warning(f"Could not extract drift metrics: {str(e)}")
            return {
                'drift_score': 0.0,
                'n_features': 0,
                'n_drifted': 0,
                'drifted_features': [],
            }

    def _calculate_basic_drift(self, report_dict: Dict) -> Dict:
        """Calculate basic drift metrics as fallback."""
        # Simple fallback if detailed metrics not available
        return {
            'drift_score': 0.0,
            'n_features': 0,
            'n_drifted': 0,
            'drifted_features': [],
        }

    def _save_report(self, report: Report, report_dir: str) -> str:
        """Save drift report as HTML."""
        # Create report directory
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'drift_report_{timestamp}.html'
        full_path = report_path / filename

        # Save report
        report.save_html(str(full_path))

        return str(full_path)

    def calculate_feature_drift(
        self,
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        feature_name: str,
    ) -> Dict:
        """
        Calculate drift for a single feature.

        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            feature_name: Name of feature to analyze

        Returns:
            Dictionary with feature drift metrics
        """
        from scipy.stats import ks_2samp, chi2_contingency

        if feature_name not in reference_data.columns or feature_name not in current_data.columns:
            raise ValueError(f"Feature {feature_name} not found in data")

        ref_values = reference_data[feature_name].dropna()
        curr_values = current_data[feature_name].dropna()

        # Determine if categorical or numerical
        if pd.api.types.is_numeric_dtype(ref_values):
            # Numerical: Use Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(ref_values, curr_values)

            drift_detected = p_value < 0.05  # 95% confidence

            return {
                'feature': feature_name,
                'type': 'numerical',
                'test': 'kolmogorov_smirnov',
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': drift_detected,
                'ref_mean': ref_values.mean(),
                'curr_mean': curr_values.mean(),
                'ref_std': ref_values.std(),
                'curr_std': curr_values.std(),
            }
        else:
            # Categorical: Use Chi-square test
            # Create contingency table
            ref_counts = ref_values.value_counts()
            curr_counts = curr_values.value_counts()

            # Align categories
            all_categories = set(ref_counts.index) | set(curr_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_aligned = [curr_counts.get(cat, 0) for cat in all_categories]

            contingency = [ref_aligned, curr_aligned]

            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                drift_detected = p_value < 0.05

                return {
                    'feature': feature_name,
                    'type': 'categorical',
                    'test': 'chi_square',
                    'statistic': chi2,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'ref_categories': len(ref_counts),
                    'curr_categories': len(curr_counts),
                }
            except Exception as e:
                self.logger.warning(f"Chi-square test failed for {feature_name}: {str(e)}")
                return {
                    'feature': feature_name,
                    'type': 'categorical',
                    'test': 'chi_square',
                    'drift_detected': False,
                    'error': str(e),
                }


class PerformanceTracker:
    """
    Track model performance over time.

    Monitors metrics like AUC-ROC, precision, recall and alerts on degradation.
    """

    def __init__(self, performance_threshold: float = 0.75):
        """
        Initialize performance tracker.

        Args:
            performance_threshold: Minimum acceptable AUC-ROC
        """
        self.performance_threshold = performance_threshold
        self.logger = logging.getLogger(__name__)

    def track_performance(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_prob: Optional[pd.Series] = None,
    ) -> Dict:
        """
        Calculate and track performance metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for AUC-ROC)

        Returns:
            Dictionary with performance metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
            confusion_matrix,
        )

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'timestamp': datetime.now().isoformat(),
        }

        # Calculate AUC if probabilities provided
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {str(e)}")
                metrics['auc_roc'] = None

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

        # Check for degradation
        auc = metrics.get('auc_roc')
        if auc is not None:
            metrics['performance_degradation'] = auc < self.performance_threshold
        else:
            metrics['performance_degradation'] = False

        self.logger.info(f"Performance metrics: {metrics}")

        return metrics


# Example usage
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/Users/aswithabukka/CascadeProjects/MLOps')

    from src.data.generator import FraudDataGenerator

    # Generate reference and current data
    gen_ref = FraudDataGenerator(n_samples=5000, fraud_rate=0.01, seed=42)
    df_reference = gen_ref.generate()

    # Current data with drift (different fraud patterns)
    gen_curr = FraudDataGenerator(n_samples=1000, fraud_rate=0.015, seed=43)
    df_current = gen_curr.generate()

    # Remove target column for drift detection
    X_reference = df_reference.drop(columns=['is_fraud', 'timestamp'])
    X_current = df_current.drop(columns=['is_fraud', 'timestamp'])

    # Detect drift
    detector = DriftDetector(drift_threshold=0.15, severe_threshold=0.2)

    drift_results = detector.detect_drift(
        reference_data=X_reference,
        current_data=X_current,
        save_report=True,
        report_dir='./monitoring/reports'
    )

    print("\n=== Drift Detection Results ===")
    print(f"Drift Detected: {drift_results['drift_detected']}")
    print(f"Drift Score: {drift_results['drift_score']:.4f}")
    print(f"Severe Drift: {drift_results['severe_drift']}")
    print(f"Drifted Features ({drift_results['n_drifted']}/{drift_results['n_features']}):")
    for feature in drift_results['drifted_features']:
        print(f"  - {feature}")

    if drift_results['report_path']:
        print(f"\nDrift report saved to: {drift_results['report_path']}")

    # Example: Track performance
    tracker = PerformanceTracker(performance_threshold=0.75)

    y_true = df_current['is_fraud']
    # Simulate predictions (in reality, these would come from model)
    y_pred = (df_current['amount'] > 200).astype(int)  # Simple rule
    y_prob = df_current['amount'] / df_current['amount'].max()  # Simulate probabilities

    performance = tracker.track_performance(y_true, y_pred, y_prob)

    print("\n=== Performance Metrics ===")
    for metric, value in performance.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
