import json
import pickle
import pandas as pd
import sys
from sklearn.metrics import roc_auc_score

# Thresholds
MIN_AUC = 0.99

# Load model and test data
model = pickle.load(open("data/best_model.pkl", "rb"))
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Calculate AUC
y_prob = model.predict_proba(X_test)[:, 1]
auc = round(roc_auc_score(y_test, y_prob), 3)

print(f"Model AUC: {auc}")
print(f"Minimum required AUC: {MIN_AUC}")

# Validation gate
if auc < MIN_AUC:
    print(f"❌ FAILED: AUC {auc} is below threshold {MIN_AUC}")
    print("Pipeline blocked — model not good enough!")
    sys.exit(1)
else:
    print(f"✅ PASSED: AUC {auc} meets threshold {MIN_AUC}")
    print("Model approved — safe to deploy!")