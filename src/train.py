import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Find root folder absolutely
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load params
with open(os.path.join(BASE, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

n_estimators      = params["model"]["n_estimators"]
max_depth         = params["model"]["max_depth"]
min_samples_split = params["model"]["min_samples_split"]
random_state      = params["model"]["random_state"]

# Load processed data using absolute paths
X_train = pd.read_csv(os.path.join(BASE, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(BASE, "data", "y_test.csv")).squeeze()

print(f"Training with {len(X_train)} rows...")

# Set MLflow to save in the right place
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("week1-dvc-pipeline")

with mlflow.start_run(run_name="dvc-pipeline-run"):
    mlflow.log_params(params["model"])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
    mlflow.sklearn.log_model(model, "model")

    print(f"Done! acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")