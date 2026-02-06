"""
Airflow DAG for model training pipeline.

Runs weekly to retrain fraud detection models.
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'fraud_detection_training',
    default_args=default_args,
    description='Train fraud detection models weekly',
    schedule_interval='0 3 * * 0',  # Weekly on Sunday at 3 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'fraud', 'training'],
)


def generate_training_data():
    """Generate synthetic training data."""
    from src.data.generator import FraudDataGenerator
    generator = FraudDataGenerator(n_samples=100000, fraud_rate=0.01)
    df = generator.generate()
    generator.save(df, file_format='parquet')
    print(f"✅ Generated {len(df)} training samples")


def train_models():
    """Train multiple fraud detection models."""
    from src.data.generator import FraudDataGenerator
    from src.data.preprocessor import FraudPreprocessor, prepare_train_test_split
    from src.models.trainer import ModelTrainer

    # Load data
    generator = FraudDataGenerator(n_samples=10000, fraud_rate=0.01)
    df = generator.generate()

    # Split data
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)

    # Preprocess
    preprocessor = FraudPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train models
    trainer = ModelTrainer()
    for algo in ['xgboost', 'lightgbm']:
        model, metrics = trainer.train(
            X_train_processed, y_train,
            X_test_processed, y_test,
            algorithm=algo
        )
        print(f"✅ Trained {algo}: AUC={metrics['auc_roc']:.4f}")


# Define tasks
generate_data_task = PythonOperator(
    task_id='generate_training_data',
    python_callable=generate_training_data,
    dag=dag,
)

train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

notify_task = BashOperator(
    task_id='notify_completion',
    bash_command='echo "Training pipeline completed successfully"',
    dag=dag,
)

# Define task dependencies
generate_data_task >> train_models_task >> notify_task
