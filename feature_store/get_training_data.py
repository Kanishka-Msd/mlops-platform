import pandas as pd
import os
from datetime import datetime, timezone
from feast import FeatureStore

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PATH = os.path.join(BASE, "feature_store")

print("Connecting to feature store...")
store = FeatureStore(repo_path=REPO_PATH)

# Load the labels file — this tells us WHICH people and WHEN
# Feast will only retrieve features that existed at event_timestamp
labels = pd.read_parquet(
    os.path.join(BASE, "feature_store", "data", "labels.parquet")
)

print(f"Requesting features for {len(labels)} people...")
print(f"Sample timestamps: {labels['event_timestamp'].iloc[0]}")

# This is the magic of Feast
# We ask: for each person_id at their specific timestamp
# give me all features from all 3 feature views
# Feast automatically joins them together correctly
training_df = store.get_historical_features(
    entity_df=labels,
    features=[
        # personal features
        "personal_features:age",
        "personal_features:race",
        "personal_features:sex",
        "personal_features:native-country",
        # work features
        "work_features:workclass",
        "work_features:education",
        "work_features:education-num",
        "work_features:occupation",
        "work_features:hours-per-week",
        # financial features
        "financial_features:fnlwgt",
        "financial_features:capital-gain",
        "financial_features:capital-loss",
        "financial_features:marital-status",
        "financial_features:relationship",
    ]
).to_df()

print(f"\nRetrieved training dataset:")
print(f"  Rows:     {len(training_df)}")
print(f"  Columns:  {list(training_df.columns)}")
print(f"\nFirst 3 rows:")
print(training_df.head(3).to_string())

# Save the training dataset
out_path = os.path.join(BASE, "feature_store", "data", "training_dataset.parquet")
training_df.to_parquet(out_path, index=False)
print(f"\nSaved training dataset to: feature_store/data/training_dataset.parquet")
