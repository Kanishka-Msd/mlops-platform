import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Find root folder absolutely
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load params
with open(os.path.join(BASE, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

# Load processed data
X_train = pd.read_csv(os.path.join(BASE, "data", "X_train.csv"))
X_test  = pd.read_csv(os.path.join(BASE, "data", "X_test.csv"))
y_train = pd.read_csv(os.path.join(BASE, "data", "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(BASE, "data", "y_test.csv")).squeeze()

print(f"Training with {len(X_train)} rows...")

# Point MLflow to root mlflow.db
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("week1-model-comparison")

def evaluate(model, run_name, log_params):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model")
        print(f"{run_name:30s} | acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
        return auc

# --- Train Random Forest ---
rf_params = params["model"]["random_forest"]
rf = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    min_samples_split=rf_params["min_samples_split"],
    random_state=rf_params["random_state"]
)
rf_auc = evaluate(rf, "random-forest", rf_params)

# --- Train XGBoost ---
xgb_params = params["model"]["xgboost"]
xgb = XGBClassifier(
    n_estimators=xgb_params["n_estimators"],
    max_depth=xgb_params["max_depth"],
    learning_rate=xgb_params["learning_rate"],
    subsample=xgb_params["subsample"],
    random_state=xgb_params["random_state"],
    eval_metric="logloss",
    verbosity=0
)
xgb_auc = evaluate(xgb, "xgboost", xgb_params)

# --- Declare winner ---
print("\n" + "="*50)
if xgb_auc > rf_auc:
    print(f"WINNER: XGBoost (AUC {xgb_auc:.3f} vs {rf_auc:.3f})")
else:
    print(f"WINNER: Random Forest (AUC {rf_auc:.3f} vs {xgb_auc:.3f})")
print("="*50)