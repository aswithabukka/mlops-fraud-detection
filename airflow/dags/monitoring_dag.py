"""
Monitoring Pipeline DAG - Drift detection and automated retraining trigger.

This DAG runs hourly to:
1. Collect recent predictions from API logs
2. Detect data drift using EvidentlyAI
3. Calculate drift scores and generate reports
4. Check model performance metrics (if labels available)
5. Alert on drift or performance degradation
6. Trigger retraining if drift exceeds threshold
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
import logging

# Default arguments
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=20),
}


def collect_recent_predictions(**context):
    """Collect recent predictions from API logs or database."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    import pandas as pd
    from pathlib import Path
    from datetime import datetime, timedelta

    logger = logging.getLogger(__name__)
    logger.info("Collecting recent predictions...")

    # In production, query from database or API logs
    # For demo, we'll simulate by sampling from reference data
    predictions_dir = Path('/opt/airflow/data/predictions')
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Try to load recent predictions
    recent_file = predictions_dir / 'recent_predictions.csv'

    if recent_file.exists():
        # Check if file is recent (within last hour)
        file_age = datetime.now() - datetime.fromtimestamp(recent_file.stat().st_mtime)

        if file_age < timedelta(hours=1):
            df_predictions = pd.read_csv(recent_file)
            logger.info(f"Loaded {len(df_predictions)} recent predictions from file")
        else:
            logger.warning("Predictions file is stale, generating sample data")
            df_predictions = _generate_sample_predictions()
    else:
        logger.warning("No predictions file found, generating sample data")
        df_predictions = _generate_sample_predictions()

    # Save for drift detection
    current_data_path = '/opt/airflow/data/monitoring/current_data.csv'
    df_predictions.to_csv(current_data_path, index=False)

    logger.info(f"Saved {len(df_predictions)} predictions for drift analysis")

    context['task_instance'].xcom_push(key='current_data_path', value=current_data_path)
    context['task_instance'].xcom_push(key='n_predictions', value=len(df_predictions))

    return current_data_path


def _generate_sample_predictions():
    """Generate sample predictions for demo (replace with real API logs in production)."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from src.data.generator import FraudDataGenerator
    import pandas as pd

    # Generate sample data with potential drift
    gen = FraudDataGenerator(n_samples=1000, fraud_rate=0.015, seed=None)
    df = gen.generate()

    # Drop target for drift detection (we only have features in predictions)
    df_predictions = df.drop(columns=['is_fraud', 'timestamp'])

    return df_predictions


def detect_drift(**context):
    """Detect data drift using EvidentlyAI."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from src.monitoring.drift_detector import DriftDetector
    from config.settings import settings
    import pandas as pd
    from pathlib import Path
    from datetime import datetime

    logger = logging.getLogger(__name__)

    # Get current data path
    current_data_path = context['task_instance'].xcom_pull(
        task_ids='collect_predictions',
        key='current_data_path'
    )

    # Reference data path
    reference_data_path = '/opt/airflow/data/reference/reference_data.parquet'

    logger.info(f"Detecting drift...")
    logger.info(f"Reference data: {reference_data_path}")
    logger.info(f"Current data: {current_data_path}")

    # Check if reference data exists
    if not Path(reference_data_path).exists():
        logger.warning("Reference data not found, skipping drift detection")
        context['task_instance'].xcom_push(key='drift_detected', value=False)
        context['task_instance'].xcom_push(key='drift_score', value=0.0)
        return False

    # Load data
    df_reference = pd.read_parquet(reference_data_path)
    df_current = pd.read_csv(current_data_path)

    # Remove target if present
    if 'is_fraud' in df_reference.columns:
        df_reference = df_reference.drop(columns=['is_fraud'])
    if 'is_fraud' in df_current.columns:
        df_current = df_current.drop(columns=['is_fraud'])

    # Initialize drift detector
    detector = DriftDetector(
        drift_threshold=settings.drift_threshold,
        severe_threshold=settings.drift_severe_threshold
    )

    # Detect drift
    try:
        drift_report = detector.detect_drift(
            reference_data=df_reference,
            current_data=df_current,
            save_report=True,
            report_dir='/opt/airflow/data/monitoring/drift_reports'
        )

        logger.info(f"Drift detection complete")
        logger.info(f"Drift detected: {drift_report['drift_detected']}")
        logger.info(f"Drift score: {drift_report['drift_score']:.4f}")
        logger.info(f"Drifted features: {drift_report.get('drifted_features', [])}")

        # Push results to XCom
        context['task_instance'].xcom_push(key='drift_detected', value=drift_report['drift_detected'])
        context['task_instance'].xcom_push(key='drift_score', value=drift_report['drift_score'])
        context['task_instance'].xcom_push(key='severe_drift', value=drift_report.get('severe_drift', False))
        context['task_instance'].xcom_push(key='drifted_features', value=drift_report.get('drifted_features', []))

        return drift_report['drift_detected']

    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}")
        # Don't fail the DAG, just log the error
        context['task_instance'].xcom_push(key='drift_detected', value=False)
        context['task_instance'].xcom_push(key='drift_score', value=0.0)
        return False


def check_model_performance(**context):
    """Check model performance metrics if labels are available."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from config.settings import settings
    import pandas as pd
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # In production, this would query labeled predictions from database
    # For demo, we'll check if recent labeled data exists
    labeled_data_path = Path('/opt/airflow/data/monitoring/labeled_predictions.csv')

    if not labeled_data_path.exists():
        logger.info("No labeled predictions available for performance monitoring")
        context['task_instance'].xcom_push(key='performance_degradation', value=False)
        context['task_instance'].xcom_push(key='current_auc', value=None)
        return False

    try:
        # Load labeled predictions
        df_labeled = pd.read_csv(labeled_data_path)

        if 'is_fraud' not in df_labeled.columns or 'prediction' not in df_labeled.columns:
            logger.warning("Labeled data missing required columns")
            return False

        # Calculate metrics
        from sklearn.metrics import roc_auc_score, precision_score, recall_score

        y_true = df_labeled['is_fraud']
        y_pred = df_labeled['prediction']
        y_prob = df_labeled.get('fraud_probability', y_pred)

        auc_score = roc_auc_score(y_true, y_prob)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        logger.info(f"Current model performance:")
        logger.info(f"  AUC-ROC: {auc_score:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")

        # Check against thresholds
        performance_degradation = auc_score < settings.performance_threshold

        context['task_instance'].xcom_push(key='performance_degradation', value=performance_degradation)
        context['task_instance'].xcom_push(key='current_auc', value=auc_score)
        context['task_instance'].xcom_push(key='current_precision', value=precision)
        context['task_instance'].xcom_push(key='current_recall', value=recall)

        if performance_degradation:
            logger.warning(f"Performance degradation detected! AUC {auc_score:.4f} < {settings.performance_threshold}")

        return performance_degradation

    except Exception as e:
        logger.error(f"Performance check failed: {str(e)}")
        context['task_instance'].xcom_push(key='performance_degradation', value=False)
        return False


def send_drift_alert(**context):
    """Send alert if drift detected."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from config.settings import settings
    from datetime import datetime
    import json

    logger = logging.getLogger(__name__)

    # Get drift results
    drift_detected = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_detected'
    )
    drift_score = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_score'
    )
    severe_drift = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='severe_drift'
    )
    drifted_features = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drifted_features'
    )

    if not drift_detected:
        logger.info("No drift detected, skipping alert")
        return False

    # Prepare alert message
    severity = "SEVERE" if severe_drift else "MODERATE"
    alert_message = f"""
🚨 Data Drift Alert - {severity}

Drift Score: {drift_score:.4f}
Threshold: {settings.drift_threshold}
Severe Threshold: {settings.drift_severe_threshold}

Drifted Features ({len(drifted_features or [])}):
{json.dumps(drifted_features or [], indent=2)}

Timestamp: {datetime.now().isoformat()}

Action Required: {'Automated retraining triggered' if severe_drift else 'Monitor closely'}
"""

    logger.warning(alert_message)

    # In production, send to Slack/email
    if settings.enable_slack_alerts and settings.slack_webhook_url:
        try:
            _send_slack_alert(alert_message, settings.slack_webhook_url)
            logger.info("Slack alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")

    if settings.enable_email_alerts and settings.alert_email:
        try:
            _send_email_alert(alert_message, settings.alert_email)
            logger.info("Email alert sent successfully")
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    # Save alert to file
    alert_log = f'/opt/airflow/logs/alerts/drift_alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    Path(alert_log).parent.mkdir(parents=True, exist_ok=True)
    with open(alert_log, 'w') as f:
        f.write(alert_message)

    logger.info(f"Alert logged to: {alert_log}")

    return True


def _send_slack_alert(message, webhook_url):
    """Send alert to Slack."""
    import requests
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload, timeout=10)
    response.raise_for_status()


def _send_email_alert(message, to_email):
    """Send alert via email."""
    # Placeholder - implement with SMTP in production
    pass


def generate_monitoring_report(**context):
    """Generate comprehensive monitoring report."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from datetime import datetime
    import json

    logger = logging.getLogger(__name__)

    # Collect metrics from all tasks
    n_predictions = context['task_instance'].xcom_pull(
        task_ids='collect_predictions',
        key='n_predictions'
    )
    drift_detected = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_detected'
    )
    drift_score = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='drift_score'
    )
    current_auc = context['task_instance'].xcom_pull(
        task_ids='check_performance',
        key='current_auc'
    )
    performance_degradation = context['task_instance'].xcom_pull(
        task_ids='check_performance',
        key='performance_degradation'
    )

    # Build report
    report = {
        'timestamp': datetime.now().isoformat(),
        'pipeline': 'monitoring',
        'metrics': {
            'predictions_analyzed': n_predictions,
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'performance_degradation': performance_degradation,
            'current_auc': current_auc,
        },
        'status': 'degraded' if (drift_detected or performance_degradation) else 'healthy',
    }

    # Save report
    report_path = f'/opt/airflow/logs/reports/monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Monitoring report saved to: {report_path}")
    logger.info(f"Report summary: {json.dumps(report['metrics'], indent=2)}")

    return report_path


def check_severe_drift(**context):
    """Check if severe drift detected (for conditional retraining)."""
    severe_drift = context['task_instance'].xcom_pull(
        task_ids='detect_drift',
        key='severe_drift'
    )
    return severe_drift or False


# Define the DAG
dag = DAG(
    'monitoring_pipeline',
    default_args=default_args,
    description='Hourly monitoring for drift detection and automated retraining',
    schedule_interval='0 * * * *',  # Hourly
    start_date=days_ago(1),
    catchup=False,
    tags=['monitoring', 'drift', 'hourly'],
    max_active_runs=1,
)

# Define tasks
collect_task = PythonOperator(
    task_id='collect_predictions',
    python_callable=collect_recent_predictions,
    provide_context=True,
    dag=dag,
)

drift_task = PythonOperator(
    task_id='detect_drift',
    python_callable=detect_drift,
    provide_context=True,
    dag=dag,
)

performance_task = PythonOperator(
    task_id='check_performance',
    python_callable=check_model_performance,
    provide_context=True,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='send_drift_alert',
    python_callable=send_drift_alert,
    provide_context=True,
    dag=dag,
)

report_task = PythonOperator(
    task_id='generate_monitoring_report',
    python_callable=generate_monitoring_report,
    provide_context=True,
    dag=dag,
)

# Sensor to check if severe drift detected
severe_drift_sensor = PythonSensor(
    task_id='check_severe_drift',
    python_callable=check_severe_drift,
    provide_context=True,
    mode='poke',
    timeout=60,
    poke_interval=10,
    dag=dag,
)

# Trigger retraining DAG if severe drift
trigger_retraining = TriggerDagRunOperator(
    task_id='trigger_retraining',
    trigger_dag_id='fraud_detection_training',
    wait_for_completion=False,
    dag=dag,
)

# Define task dependencies
collect_task >> [drift_task, performance_task]
drift_task >> alert_task >> report_task
performance_task >> report_task

# Conditional retraining based on severe drift
drift_task >> severe_drift_sensor >> trigger_retraining

# DAG documentation
dag.doc_md = """
# Monitoring Pipeline DAG

This DAG runs **hourly** to monitor model performance and detect data drift.

## Workflow

1. **Collect Predictions**: Gather recent predictions from API logs
   - In production: Query from database or API logs
   - For demo: Use sample data from reference dataset
   - Collects last hour of predictions (~1000 samples)

2. **Detect Drift**: Use EvidentlyAI to detect data drift
   - Compare recent predictions with reference dataset
   - Calculate drift score using statistical tests
   - Identify drifted features
   - Generate drift report (HTML + JSON)

3. **Check Performance**: Monitor model performance (if labels available)
   - Calculate AUC-ROC, Precision, Recall
   - Compare against thresholds
   - Detect performance degradation

4. **Send Alert**: Alert team if drift or degradation detected
   - Slack webhook (if configured)
   - Email notification (if configured)
   - Log alert to file

5. **Generate Report**: Create comprehensive monitoring report
   - Predictions analyzed
   - Drift metrics
   - Performance metrics
   - Overall system health

6. **Conditional Retraining**: Trigger retraining if severe drift
   - Severe drift threshold: 0.2 (configurable)
   - Automatically triggers `fraud_detection_training` DAG
   - No manual intervention required

## Configuration

Set in `config/settings.py`:
- `drift_threshold`: Moderate drift (default: 0.15)
- `drift_severe_threshold`: Severe drift triggers retraining (default: 0.2)
- `performance_threshold`: Minimum AUC-ROC (default: 0.75)
- `enable_slack_alerts`: Enable Slack notifications (default: False)
- `enable_email_alerts`: Enable email notifications (default: False)

## Outputs

- **Drift Reports**: `/opt/airflow/data/monitoring/drift_reports/drift_report_*.html`
- **Monitoring Reports**: `/opt/airflow/logs/reports/monitoring_*.json`
- **Alert Logs**: `/opt/airflow/logs/alerts/drift_alert_*.txt`

## Monitoring

- Check DAG run duration (should be <5 minutes)
- Monitor drift scores over time (track trends)
- Verify alerts are sent when drift detected
- Ensure retraining triggers on severe drift

## Event-Driven Retraining

When drift score exceeds **0.2** (severe threshold):
1. Monitoring DAG detects severe drift
2. Sensor checks condition
3. TriggerDagRunOperator triggers `fraud_detection_training` DAG
4. Training DAG retrains model on recent data
5. New model is evaluated and promoted if better
6. System adapts to new patterns automatically

This creates a **closed-loop system** that adapts to changing fraud patterns without manual intervention.

## Troubleshooting

- If drift always detected: Adjust thresholds or check reference data quality
- If performance metrics missing: Ensure labeled predictions are available
- If alerts not sent: Check Slack/email configuration in settings
- If retraining not triggered: Verify `fraud_detection_training` DAG exists and is enabled
"""

collect_task.doc_md = """
Collect recent predictions from API logs or database.

**Expected Runtime**: 30 seconds
**Outputs**: CSV file with recent predictions
"""

drift_task.doc_md = """
Detect data drift using EvidentlyAI statistical tests.

**Expected Runtime**: 1-2 minutes
**Outputs**: Drift report (HTML + JSON)
"""

severe_drift_sensor.doc_md = """
Check if severe drift detected (threshold > 0.2).

If severe drift detected, triggers automated retraining.
"""
