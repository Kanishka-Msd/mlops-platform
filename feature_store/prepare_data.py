import pandas as pd
import os
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Loading raw data...")
cols = ["age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"]

df = pd.read_csv(
    os.path.join(BASE, "data", "adult.csv"),
    names=cols, na_values=" ?", skipinitialspace=True
)
df.dropna(inplace=True)

# Add a unique person ID — Feast needs an entity identifier
df["person_id"] = range(1, len(df) + 1)

# Add a timestamp — Feast needs to know WHEN each row was recorded
# This enables point-in-time correct feature retrieval
df["event_timestamp"] = datetime.now(tz=timezone.utc)

# Separate features into logical groups
# Group 1: Personal features
personal_features = df[[
    "person_id", "event_timestamp",
    "age", "race", "sex", "native-country"
]].copy()

# Group 2: Education and work features  
work_features = df[[
    "person_id", "event_timestamp",
    "workclass", "education", "education-num",
    "occupation", "hours-per-week"
]].copy()

# Group 3: Financial features
financial_features = df[[
    "person_id", "event_timestamp",
    "fnlwgt", "capital-gain", "capital-loss",
    "marital-status", "relationship"
]].copy()

# Group 4: Target label (income)
labels = df[[
    "person_id", "event_timestamp", "income"
]].copy()

# Save each group as a Parquet file
# Parquet is like CSV but much faster and compressed
os.makedirs(os.path.join(BASE, "feature_store", "data"), exist_ok=True)

personal_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "personal_features.parquet"),
    index=False
)
work_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "work_features.parquet"),
    index=False
)
financial_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "financial_features.parquet"),
    index=False
)
labels.to_parquet(
    os.path.join(BASE, "feature_store", "data", "labels.parquet"),
    index=False
)

print(f"Saved {len(df)} people split into 3 feature groups:")
print(f"  personal_features.parquet  — age, race, sex, country")
print(f"  work_features.parquet      — job, education, hours")
print(f"  financial_features.parquet — income signals, capital")
print(f"  labels.parquet             — income target (0 or 1)")
print("Done!")