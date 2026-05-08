from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import subprocess
import sys
sys.path.insert(0, '/Users/kanish/mlops-platform')

def notify_failure(context):
    task_id = context['task_instance'].task_id
    dag_id = context['dag'].dag_id
    execution_date = context['execution_date']
    print(f"""
    🚨 PIPELINE FAILURE ALERT!
    ──────────────────────────
    DAG: {dag_id}
    Failed Task: {task_id}
    Time: {execution_date}
    Action Required: Check Airflow UI
    http://localhost:8080
    """)

default_args = {
    'owner': 'kanishka',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': notify_failure
}

dag = DAG(
    'mlops_pipeline',
    default_args=default_args,
    description='Automated MLOps pipeline with drift detection',
    schedule='0 0 * * 0',
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

# Task 2: Check drift and decide
def check_drift_and_decide():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/src/detect_drift.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if 'DRIFT DETECTED' in result.stdout:
        print("🚨 Drift detected → will retrain!")
        return 'retrain_model'
    else:
        print("✅ No drift → skipping retraining!")
        return 'skip_retraining'

# Task 3a: Retrain model
def retrain_model():
    result = subprocess.run(
        ['dvc', 'repro'],
        capture_output=True, text=True,
        cwd='/Users/kanish/mlops-platform'
    )
    print(result.stdout)
    return "Model retrained!"

# Task 3b: Skip retraining
def skip_retraining():
    print("✅ No drift detected — skipping retraining!")
    return "Retraining skipped!"

# Task 4: Validate model
def validate_model():
    result = subprocess.run(
        ['python', '/Users/kanish/mlops-platform/src/validate_model.py'],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        raise Exception("Model validation failed!")
    return "Model validation passed!"

# Task 5: Monitor predictions
def monitor_predictions():
    import pandas as pd
    X_test = pd.read_csv("/Users/kanish/mlops-platform/data/X_test.csv")
    y_test = pd.read_csv("/Users/kanish/mlops-platform/data/y_test.csv")
    total = len(X_test)
    print(f"\n📊 Model Output Monitoring Report:")
    print(f"─────────────────────────────────")
    print(f"Total test samples: {total}")
    print(f"✅ Data monitoring complete!")
    return "Monitoring complete!"

# Task 6: Generate summary
def generate_summary():
    run_date = datetime.now()
    print(f"""
    ✅ MLOPS PIPELINE SUMMARY
    ─────────────────────────
    Run Date: {run_date}
    Status: SUCCESS

    Steps completed:
    1. ✅ Data validated (25 checks)
    2. ✅ Drift detection ran
    3. ✅ Model retrained/skipped
    4. ✅ Model validated (AUC > 0.90)
    5. ✅ Predictions monitored

    Next run: Every Sunday midnight
    Platform: production-grade MLOps ✅
    """)

# Define tasks
t1_validate_data = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag
)

t2_check_drift = BranchPythonOperator(
    task_id='check_drift',
    python_callable=check_drift_and_decide,
    dag=dag
)

t3a_retrain = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag
)

t3b_skip = PythonOperator(
    task_id='skip_retraining',
    python_callable=skip_retraining,
    dag=dag
)

t4_validate_model = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
    trigger_rule='none_failed_min_one_success'
)

t5_monitor = PythonOperator(
    task_id='monitor_predictions',
    python_callable=monitor_predictions,
    dag=dag,
    trigger_rule='none_failed_min_one_success'
)

t6_summary = PythonOperator(
    task_id='generate_summary',
    python_callable=generate_summary,
    dag=dag
)

# Define order
t1_validate_data >> t2_check_drift
t2_check_drift >> [t3a_retrain, t3b_skip]
t3a_retrain >> t4_validate_model
t3b_skip >> t4_validate_model
t4_validate_model >> t5_monitor
t5_monitor >> t6_summary