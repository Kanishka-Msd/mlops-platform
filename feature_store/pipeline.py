import os
import sys
import time
import subprocess
import pandas as pd
import pickle
from feast import FeatureStore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(BASE, "feature_store")

print("="*55)
print("WEEK 2 UNIFIED PIPELINE")
print("="*55)

# ── Stage 1: Data Validation ─────────────────────────────
print("\n[Stage 1] Running data validation...")
result = subprocess.run(
    [sys.executable, os.path.join(REPO, "validate_data.py")],
    capture_output=True, text=True
)
if result.returncode != 0:
    print("VALIDATION FAILED — stopping pipeline")
    print(result.stdout)
    sys.exit(1)
print("Validation passed — data is clean")

# ── Stage 2: Feature Retrieval ───────────────────────────
print("\n[Stage 2] Retrieving features from Feast offline store...")
store = FeatureStore(repo_path=REPO)

labels = pd.read_parquet(
    os.path.join(BASE, "feature_store", "data", "labels.parquet")
)

training_df = store.get_historical_features(
    entity_df=labels,
    features=[
        "personal_features:age",
        "personal_features:race",
        "personal_features:sex",
        "personal_features:native-country",
        "work_features:workclass",
        "work_features:education",
        "work_features:education-num",
        "work_features:occupation",
        "work_features:hours-per-week",
        "financial_features:fnlwgt",
        "financial_features:capital-gain",
        "financial_features:capital-loss",
        "financial_features:marital-status",
        "financial_features:relationship",
    ]
).to_df()

# Clean retrieved data
training_df = training_df[training_df["age"] != "age"].copy()
for col in training_df.columns:
    if col not in ["event_timestamp"]:
        training_df[col] = pd.to_numeric(training_df[col], errors="coerce")
training_df.dropna(inplace=True)

X = training_df.drop(["income","person_id","event_timestamp"], axis=1)
y = training_df["income"]
print(f"Retrieved {len(X)} rows with {len(X.columns)} features")

# ── Stage 3: Model Training ──────────────────────────────
print("\n[Stage 3] Training XGBoost on Feast features...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE, 'mlflow.db')}")
mlflow.set_experiment("week2-unified-pipeline")

with mlflow.start_run(run_name="week2-final"):
    model = XGBClassifier(
        n_estimators=200, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        random_state=42, eval_metric="logloss",
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
    mlflow.log_param("pipeline", "unified_week2")
    mlflow.sklearn.log_model(model, "model")

    print(f"  accuracy : {acc:.3f}")
    print(f"  f1       : {f1:.3f}")
    print(f"  roc_auc  : {auc:.3f}")

# Save model
model_path = os.path.join(BASE, "data", "best_model_feast.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"  Model saved to data/best_model_feast.pkl")

# ── Stage 4: Real-time Prediction from Redis ─────────────
print("\n[Stage 4] Real-time prediction using Redis features...")

# Simulate 3 new people arriving at your API
new_people = [
    {"person_id": 500,  "description": "Person 500 from dataset"},
    {"person_id": 1500, "description": "Person 1500 from dataset"},
    {"person_id": 5000, "description": "Person 5000 from dataset"},
]

for person in new_people:
    start = time.time()

    # Get features from Redis online store
    online_features = store.get_online_features(
        features=[
            "personal_features:age",
            "personal_features:sex",
            "personal_features:race",
            "personal_features:native-country",
            "work_features:workclass",
            "work_features:education",
            "work_features:education-num",
            "work_features:occupation",
            "work_features:hours-per-week",
            "financial_features:fnlwgt",
            "financial_features:capital-gain",
            "financial_features:capital-loss",
            "financial_features:marital-status",
            "financial_features:relationship",
        ],
        entity_rows=[{"person_id": person["person_id"]}]
    ).to_dict()

    # Build feature row for model
    feature_row = pd.DataFrame([{
        "age":            online_features["age"][0],
        "race":           online_features["race"][0],
        "sex":            online_features["sex"][0],
        "native-country": online_features["native-country"][0],
        "workclass":      online_features["workclass"][0],
        "education":      online_features["education"][0],
        "education-num":  online_features["education-num"][0],
        "occupation":     online_features["occupation"][0],
        "hours-per-week": online_features["hours-per-week"][0],
        "fnlwgt":         online_features["fnlwgt"][0],
        "capital-gain":   online_features["capital-gain"][0],
        "capital-loss":   online_features["capital-loss"][0],
        "marital-status": online_features["marital-status"][0],
        "relationship":   online_features["relationship"][0],
    }])

    # Predict
    prediction = model.predict(feature_row)[0]
    confidence = model.predict_proba(feature_row)[0][1]
    elapsed = (time.time() - start) * 1000

    result = "OVER $50K" if prediction == 1 else "UNDER $50K"
    conf = confidence if prediction == 1 else 1 - confidence

    print(f"\n  {person['description']}")
    print(f"  Features from Redis : {elapsed:.1f}ms")
    print(f"  Prediction          : {result}")
    print(f"  Confidence          : {conf:.1%}")

print("\n" + "="*55)
print("WEEK 2 PIPELINE COMPLETE")
print("  Stage 1: Data validation    PASSED")
print("  Stage 2: Feast offline      PASSED")
print("  Stage 3: XGBoost training   PASSED")
print("  Stage 4: Redis serving      PASSED")
print("="*55)
