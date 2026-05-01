import json
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# Load model and test data
model = pickle.load(open("data/best_model.pkl", "rb"))
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Calculate metrics
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

metrics = {
    "roc_auc": round(roc_auc_score(y_test, y_prob), 3),
    "f1_score": round(f1_score(y_test, y_pred), 3),
    "accuracy": round(accuracy_score(y_test, y_pred), 3)
}

# Save metrics report
with open("metrics_report.md", "w") as f:
    f.write("## 🤖 Model Metrics Report\n\n")
    f.write("| Metric | Score |\n")
    f.write("|--------|-------|\n")
    f.write(f"| ROC-AUC | {metrics['roc_auc']} |\n")
    f.write(f"| F1 Score | {metrics['f1_score']} |\n")
    f.write(f"| Accuracy | {metrics['accuracy']} |\n")

print("Metrics saved to metrics_report.md")