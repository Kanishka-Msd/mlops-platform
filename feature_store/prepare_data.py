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

# Remove header row if it snuck in as data
df = df[df["age"] != "age"].copy()
df = df[df["income"] != "income"].copy()

# Convert numeric columns properly
numeric_cols = ["age","fnlwgt","education-num","capital-gain",
                "capital-loss","hours-per-week"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df.dropna(inplace=True)

# Encode text columns to numbers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
text_cols = ["workclass","education","marital-status","occupation",
             "relationship","race","sex","native-country","income"]
for col in text_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Add person_id and timestamp
df["person_id"] = range(1, len(df) + 1)
df["event_timestamp"] = datetime.now(tz=timezone.utc)

# Split into feature groups
personal_features = df[[
    "person_id","event_timestamp",
    "age","race","sex","native-country"
]].copy()

work_features = df[[
    "person_id","event_timestamp",
    "workclass","education","education-num",
    "occupation","hours-per-week"
]].copy()

financial_features = df[[
    "person_id","event_timestamp",
    "fnlwgt","capital-gain","capital-loss",
    "marital-status","relationship"
]].copy()

labels = df[[
    "person_id","event_timestamp","income"
]].copy()

# Save as Parquet
os.makedirs(os.path.join(BASE, "feature_store", "data"), exist_ok=True)

personal_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "personal_features.parquet"),
    index=False)
work_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "work_features.parquet"),
    index=False)
financial_features.to_parquet(
    os.path.join(BASE, "feature_store", "data", "financial_features.parquet"),
    index=False)
labels.to_parquet(
    os.path.join(BASE, "feature_store", "data", "labels.parquet"),
    index=False)

print(f"Saved {len(df)} people split into 3 feature groups:")
print(f"  personal_features.parquet  — age, race, sex, country")
print(f"  work_features.parquet      — job, education, hours")
print(f"  financial_features.parquet — income signals, capital")
print(f"  labels.parquet             — income target")
print("Done!")
