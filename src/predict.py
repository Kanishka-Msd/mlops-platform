import pickle
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the saved best model
with open(os.path.join(BASE, "data", "best_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Define 5 very different people
people = [
    {
        "name": "Person 1 — Young student, part time job",
        "features": [20, 4, 180000, 9, 10, 4, 3, 3, 4, 1, 0, 0, 20, 39]
    },
    {
        "name": "Person 2 — Middle aged, married, good education",
        "features": [35, 4, 200000, 9, 13, 2, 3, 0, 4, 1, 5000, 0, 45, 39]
    },
    {
        "name": "Person 3 — Senior executive, lots of capital",
        "features": [52, 6, 300000, 9, 16, 2, 4, 0, 4, 1, 99999, 0, 60, 39]
    },
    {
        "name": "Person 4 — Single mother, part time, no capital",
        "features": [34, 4, 150000, 7, 9, 4, 1, 5, 4, 0, 0, 0, 25, 39]
    },
    {
        "name": "Person 5 — Retired, older, no work income",
        "features": [68, 0, 100000, 9, 12, 2, 0, 0, 4, 1, 0, 2000, 5, 39]
    }
]

cols = ["age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country"]

print("="*55)
print("YOUR XGBOOST MODEL — LIVE PREDICTIONS")
print("="*55)

for p in people:
    df = pd.DataFrame([p["features"]], columns=cols)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    result = "OVER $50K" if prediction == 1 else "UNDER $50K"
    confidence = probability if prediction == 1 else 1 - probability
    print(f"\n{p['name']}")
    print(f"  Prediction : {result}")
    print(f"  Confidence : {confidence:.1%}")

print("\n" + "="*55)