import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from preprocess import load_data
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

X_train, X_test, y_train, y_test = load_data()

mlflow.set_experiment("week1-adult-income")

def run_experiment(model, params: dict, run_name: str):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds)
        auc   = roc_auc_score(y_test, proba)
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, "model")
        print(f"{run_name:35s} | acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")

# --- 10 experiments ---
run_experiment(RandomForestClassifier(n_estimators=50,  max_depth=5,  random_state=42), {"n_estimators":50,  "max_depth":5,  "model":"rf"}, "rf-50trees-depth5")
run_experiment(RandomForestClassifier(n_estimators=100, max_depth=5,  random_state=42), {"n_estimators":100, "max_depth":5,  "model":"rf"}, "rf-100trees-depth5")
run_experiment(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), {"n_estimators":100, "max_depth":10, "model":"rf"}, "rf-100trees-depth10")
run_experiment(RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), {"n_estimators":200, "max_depth":10, "model":"rf"}, "rf-200trees-depth10")
run_experiment(RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42), {"n_estimators":200, "max_depth":15, "model":"rf"}, "rf-200trees-depth15")
run_experiment(LogisticRegression(C=0.01, max_iter=1000), {"C":0.01, "model":"lr"}, "lr-C0.01")
run_experiment(LogisticRegression(C=0.1,  max_iter=1000), {"C":0.1,  "model":"lr"}, "lr-C0.1")
run_experiment(LogisticRegression(C=1.0,  max_iter=1000), {"C":1.0,  "model":"lr"}, "lr-C1.0")
run_experiment(LogisticRegression(C=10.0, max_iter=1000), {"C":10.0, "model":"lr"}, "lr-C10.0")
run_experiment(RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, random_state=42), {"n_estimators":300,"max_depth":20,"min_samples_split":5,"model":"rf"}, "rf-best-attempt")