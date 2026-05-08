from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import subprocess
import sys
sys.path.insert(0, '/Users/kanish/mlops-platform')
# Default arguments
default_args = {
    'owner': 'kanishka',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline with drift detection',
    schedule='0 0 * * 0',  # Every Sunday midnight
    catchup=False,
    tags=['mlops', 'drift', 'retraining']
)

# Task 1: Validate data
def validate_data():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/feature_store/validate_data.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception("Data validation failed!")
    return "Data validation passed!"

# Task 2: Check drift
def check_drift():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/src/detect_drift.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    return "Drift check complete!"

# Task 3: Retrain model
def retrain_model():
    result = subprocess.run(
        ['dvc', 'repro'],
        capture_output=True, text=True,
        cwd='/Users/kanish/mlops-platform'
    )
    print(result.stdout)
    return "Model retrained!"

# Task 4: Validate model
def validate_model():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/src/validate_model.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception("Model validation failed! AUC below threshold!")
    return "Model validation passed!"

# Task 5: Monitor predictions
def monitor_predictions():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/src/monitor_predictions.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    return "Prediction monitoring complete!"

# Define tasks
t1_validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

t2_check_drift = PythonOperator(
    task_id='check_drift',
    python_callable=check_drift,
    dag=dag
)

t3_retrain_model = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

t4_validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag
)

t5_monitor_predictions = PythonOperator(
    task_id='monitor_predictions',
    python_callable=monitor_predictions,
    dag=dag
)

# Define order
t1_validate_data >> t2_check_drift >> t3_retrain_model >> t4_validate_model >> t5_monitor_predictions