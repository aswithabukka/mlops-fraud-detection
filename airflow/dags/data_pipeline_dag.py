"""
Data Pipeline DAG - Daily data generation, validation, and preprocessing.

This DAG runs daily to:
1. Generate synthetic fraud transaction data
2. Validate data schema and quality
3. Preprocess and engineer features
4. Store processed data for training
5. Update reference dataset for drift detection
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import logging

# Default arguments for the DAG
default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=30),
}


def generate_data(**context):
    """Generate synthetic fraud transaction data."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from src.data.generator import FraudDataGenerator
    from config.settings import settings
    import pandas as pd
    from datetime import datetime

    logger = logging.getLogger(__name__)
    logger.info("Starting data generation...")

    # Generate data
    generator = FraudDataGenerator(
        n_samples=settings.data_size,
        fraud_rate=settings.fraud_rate,
        seed=None  # Use different data each day
    )

    df = generator.generate()

    # Save to raw data directory
    date_str = datetime.now().strftime('%Y%m%d')
    output_path = f'/opt/airflow/data/raw/fraud_{date_str}.csv'
    df.to_csv(output_path, index=False)

    logger.info(f"Generated {len(df)} transactions")
    logger.info(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    logger.info(f"Saved to: {output_path}")

    # Push metadata to XCom for downstream tasks
    context['task_instance'].xcom_push(key='data_path', value=output_path)
    context['task_instance'].xcom_push(key='n_samples', value=len(df))
    context['task_instance'].xcom_push(key='fraud_rate', value=float(df['is_fraud'].mean()))

    return output_path


def validate_data(**context):
    """Validate data schema and quality."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from src.data.validator import DataValidator
    import pandas as pd

    logger = logging.getLogger(__name__)

    # Get data path from previous task
    data_path = context['task_instance'].xcom_pull(
        task_ids='generate_data',
        key='data_path'
    )

    logger.info(f"Validating data from: {data_path}")

    # Load and validate
    df = pd.read_csv(data_path)
    validator = DataValidator()

    try:
        is_valid, errors = validator.validate(df)

        if not is_valid:
            logger.error(f"Validation failed with errors: {errors}")
            raise ValueError(f"Data validation failed: {errors}")

        logger.info("Data validation passed!")

        # Run quality checks
        quality_report = validator.generate_quality_report(df)
        logger.info(f"Quality report: {quality_report}")

        context['task_instance'].xcom_push(key='validation_status', value='passed')
        context['task_instance'].xcom_push(key='quality_report', value=quality_report)

        return True

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        raise


def preprocess_data(**context):
    """Preprocess data and engineer features."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    from src.data.preprocessor import FraudPreprocessor
    import pandas as pd
    from datetime import datetime
    import pickle

    logger = logging.getLogger(__name__)

    # Get data path from XCom
    data_path = context['task_instance'].xcom_pull(
        task_ids='generate_data',
        key='data_path'
    )

    logger.info(f"Preprocessing data from: {data_path}")

    # Load data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=['is_fraud', 'timestamp'])
    y = df['is_fraud']

    # Fit preprocessor on all data (for inference pipeline)
    preprocessor = FraudPreprocessor()
    X_processed = preprocessor.fit_transform(X, y)

    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Processed features: {X_processed.shape[1]}")

    # Save processed data
    date_str = datetime.now().strftime('%Y%m%d')

    # Save as parquet for efficiency
    processed_df = pd.DataFrame(
        X_processed,
        columns=preprocessor.get_feature_names_out()
    )
    processed_df['is_fraud'] = y.values

    output_path = f'/opt/airflow/data/processed/fraud_{date_str}.parquet'
    processed_df.to_parquet(output_path, index=False)

    # Save preprocessor for inference
    preprocessor_path = f'/opt/airflow/data/preprocessors/preprocessor_{date_str}.pkl'
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)

    logger.info(f"Saved processed data to: {output_path}")
    logger.info(f"Saved preprocessor to: {preprocessor_path}")

    context['task_instance'].xcom_push(key='processed_data_path', value=output_path)
    context['task_instance'].xcom_push(key='preprocessor_path', value=preprocessor_path)

    return output_path


def update_reference_dataset(**context):
    """Update reference dataset for drift detection."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    import pandas as pd
    import shutil
    from pathlib import Path

    logger = logging.getLogger(__name__)

    # Get processed data path
    processed_data_path = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='processed_data_path'
    )

    logger.info(f"Updating reference dataset from: {processed_data_path}")

    # Copy processed data to reference directory
    reference_dir = Path('/opt/airflow/data/reference')
    reference_dir.mkdir(parents=True, exist_ok=True)

    reference_path = reference_dir / 'reference_data.parquet'

    # Load new data
    df_new = pd.read_parquet(processed_data_path)

    # If reference exists, combine with new data (keep last 30 days)
    if reference_path.exists():
        df_existing = pd.read_parquet(reference_path)

        # Combine and keep last 100K samples (rolling window)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_reference = df_combined.tail(100000)

        logger.info(f"Combined {len(df_existing)} existing + {len(df_new)} new samples")
        logger.info(f"Keeping last {len(df_reference)} samples as reference")
    else:
        df_reference = df_new
        logger.info(f"Creating new reference dataset with {len(df_reference)} samples")

    # Save reference dataset
    df_reference.to_parquet(reference_path, index=False)

    logger.info(f"Reference dataset updated: {reference_path}")
    logger.info(f"Reference fraud rate: {df_reference['is_fraud'].mean():.2%}")

    context['task_instance'].xcom_push(key='reference_path', value=str(reference_path))

    return str(reference_path)


def generate_data_quality_report(**context):
    """Generate comprehensive data quality report."""
    import sys
    sys.path.insert(0, '/opt/airflow/dags')

    import pandas as pd
    from datetime import datetime
    import json

    logger = logging.getLogger(__name__)

    # Get metadata from XCom
    n_samples = context['task_instance'].xcom_pull(
        task_ids='generate_data',
        key='n_samples'
    )
    fraud_rate = context['task_instance'].xcom_pull(
        task_ids='generate_data',
        key='fraud_rate'
    )
    quality_report = context['task_instance'].xcom_pull(
        task_ids='validate_data',
        key='quality_report'
    )

    # Build comprehensive report
    report = {
        'date': datetime.now().isoformat(),
        'pipeline': 'data_pipeline',
        'metrics': {
            'total_samples': n_samples,
            'fraud_rate': fraud_rate,
            'validation_status': 'passed',
        },
        'quality_checks': quality_report,
        'status': 'success'
    }

    # Save report
    date_str = datetime.now().strftime('%Y%m%d')
    report_path = f'/opt/airflow/logs/reports/data_quality_{date_str}.json'

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Data quality report saved to: {report_path}")
    logger.info(f"Report summary: {json.dumps(report['metrics'], indent=2)}")

    return report_path


# Define the DAG
dag = DAG(
    'data_pipeline',
    default_args=default_args,
    description='Daily data generation, validation, and preprocessing pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['data', 'preprocessing', 'daily'],
    max_active_runs=1,
)

# Define tasks
generate_task = PythonOperator(
    task_id='generate_data',
    python_callable=generate_data,
    provide_context=True,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag,
)

update_reference_task = PythonOperator(
    task_id='update_reference_dataset',
    python_callable=update_reference_dataset,
    provide_context=True,
    dag=dag,
)

quality_report_task = PythonOperator(
    task_id='generate_quality_report',
    python_callable=generate_data_quality_report,
    provide_context=True,
    dag=dag,
)

# Create directories (run once at DAG start)
setup_directories = BashOperator(
    task_id='setup_directories',
    bash_command='''
    mkdir -p /opt/airflow/data/raw
    mkdir -p /opt/airflow/data/processed
    mkdir -p /opt/airflow/data/preprocessors
    mkdir -p /opt/airflow/data/reference
    mkdir -p /opt/airflow/logs/reports
    echo "Directories created successfully"
    ''',
    dag=dag,
)

# Define task dependencies
setup_directories >> generate_task >> validate_task >> preprocess_task >> update_reference_task >> quality_report_task

# Task documentation
dag.doc_md = """
# Data Pipeline DAG

This DAG runs **daily at 2 AM** to generate, validate, and preprocess fraud transaction data.

## Workflow

1. **Setup Directories**: Create necessary directories for data storage
2. **Generate Data**: Create synthetic fraud transactions using FraudDataGenerator
   - Configurable sample size (default: 100K)
   - Configurable fraud rate (default: 1%)
   - Realistic fraud patterns (5x amounts, late-night, foreign)

3. **Validate Data**: Run schema and quality validation using Pandera
   - Type checking (datetime, float, int, str)
   - Range constraints (amount > 0, hour 0-23)
   - Business logic (fraud rate within bounds)
   - Generate quality report

4. **Preprocess Data**: Feature engineering and transformation
   - Fit FraudPreprocessor on data
   - Engineer features (interactions, ratios, polynomials)
   - Scale and encode features
   - Save processed data as Parquet
   - Save preprocessor for inference

5. **Update Reference Dataset**: Maintain rolling window of reference data
   - Combine with existing reference (last 100K samples)
   - Used for drift detection in monitoring pipeline
   - Updated daily to reflect recent patterns

6. **Generate Quality Report**: Create comprehensive data quality report
   - Sample counts, fraud rate
   - Validation status
   - Quality metrics
   - Saved as JSON for tracking

## Configuration

Set in `config/settings.py`:
- `data_size`: Number of samples to generate (default: 100000)
- `fraud_rate`: Percentage of fraudulent transactions (default: 0.01)

## Outputs

- **Raw Data**: `/opt/airflow/data/raw/fraud_YYYYMMDD.csv`
- **Processed Data**: `/opt/airflow/data/processed/fraud_YYYYMMDD.parquet`
- **Preprocessor**: `/opt/airflow/data/preprocessors/preprocessor_YYYYMMDD.pkl`
- **Reference Data**: `/opt/airflow/data/reference/reference_data.parquet`
- **Quality Report**: `/opt/airflow/logs/reports/data_quality_YYYYMMDD.json`

## Monitoring

- Check DAG run duration (should be <10 minutes)
- Monitor fraud rate (should be 0.5-2%)
- Verify validation always passes
- Track reference dataset size (should be ~100K)

## Troubleshooting

- If validation fails: Check data generator for issues
- If preprocessing fails: Check for new categories or missing values
- If DAG times out: Reduce `data_size` setting
"""

generate_task.doc_md = """
Generate synthetic fraud transaction data using FraudDataGenerator.

**Expected Runtime**: 2-3 minutes for 100K samples
**Outputs**: CSV file in `/opt/airflow/data/raw/`
"""

validate_task.doc_md = """
Validate data schema and quality using Pandera.

**Expected Runtime**: 30-60 seconds
**Checks**: Schema compliance, data quality, business rules
"""

preprocess_task.doc_md = """
Preprocess data and engineer features.

**Expected Runtime**: 2-4 minutes for 100K samples
**Outputs**: Processed Parquet file and pickled preprocessor
"""
