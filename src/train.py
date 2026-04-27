import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
import os
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
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

FEATURE_NAMES = ["age","workclass","fnlwgt","education","education-num",
                 "marital-status","occupation","relationship","race","sex",
                 "capital-gain","capital-loss","hours-per-week","native-country"]

print(f"Training with {len(X_train)} rows...")

# Point MLflow to root mlflow.db
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("week1-final")

def plot_feature_importance(importances, feature_names, run_name):
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Create the chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_features[::-1], sorted_importances[::-1])
    
    # Color the top 3 features differently
    for i, bar in enumerate(bars):
        if i >= len(bars) - 3:
            bar.set_color("#1D9E75")
        else:
            bar.set_color("#B5D4F4")

    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance — {run_name}")
    plt.tight_layout()

    # Save chart to file
    path = os.path.join(BASE, f"data/{run_name}_feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path

def evaluate(model, run_name, log_params, feature_names):
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(log_params)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)
        mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
        
        # Save model
        mlflow.sklearn.log_model(model, "model")
        
        # Feature importance
        importances = model.feature_importances_
        chart_path = plot_feature_importance(importances, feature_names, run_name)
        mlflow.log_artifact(chart_path)
        
        # Print top 3 most important features
        indices = np.argsort(importances)[::-1]
        print(f"\n{run_name} results:")
        print(f"  acc={acc:.3f} | f1={f1:.3f} | auc={auc:.3f}")
        print(f"  Top 3 features:")
        for i in range(3):
            print(f"    {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        return auc

# --- Train Random Forest ---
rf_params = params["model"]["random_forest"]
rf = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    min_samples_split=rf_params["min_samples_split"],
    random_state=rf_params["random_state"]
)
rf_auc = evaluate(rf, "random-forest", rf_params, FEATURE_NAMES)

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
xgb_auc = evaluate(xgb, "xgboost", xgb_params, FEATURE_NAMES)

# --- Save best model to disk ---
print("\n" + "="*50)
if xgb_auc > rf_auc:
    winner = xgb
    winner_name = "XGBoost"
    winner_auc = xgb_auc
else:
    winner = rf
    winner_name = "Random Forest"
    winner_auc = rf_auc

print(f"WINNER: {winner_name} (AUC {winner_auc:.3f})")

# Save winner as pickle for later use
import pickle
model_path = os.path.join(BASE, "data", "best_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(winner, f)
print(f"Best model saved to: data/best_model.pkl")
print("="*50)