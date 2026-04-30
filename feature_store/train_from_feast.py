import pandas as pd
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading data from feature store...")
X = pd.read_csv(os.path.join(BASE, "feature_store", "data", "fs_X.csv"))
y = pd.read_csv(os.path.join(BASE, "feature_store", "data", "fs_y.csv")).squeeze()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("week2-feast-training")

with mlflow.start_run(run_name="xgboost-from-feast"):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)

    mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
    mlflow.log_param("data_source", "feast_feature_store")
    mlflow.sklearn.log_model(model, "model")

    print(f"\nResults from Feast data:")
    print(f"  accuracy : {acc:.3f}")
    print(f"  f1       : {f1:.3f}")
    print(f"  roc_auc  : {auc:.3f}")

    print(f"\nOriginal pipeline results:")
    print(f"  accuracy : 0.874")
    print(f"  f1       : 0.720")
    print(f"  roc_auc  : 0.927")

    diff = auc - 0.927
    print(f"\nAUC difference: {diff:+.3f}")
    if abs(diff) < 0.005:
        print("Feature store data produces equivalent model quality")
    else:
        print("Difference detected — investigate data pipeline")
