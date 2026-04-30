import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE, "feature_store", "data", "training_dataset.parquet")

print("Loading Feast training dataset...")
df = pd.read_parquet(path)
print(f"Before cleaning: {len(df)} rows")

# Remove the header row that snuck in as data
df = df[df["age"] != "age"].copy()
df = df[df["income"] != "income"].copy()
print(f"After removing header row: {len(df)} rows")

# Encode all text columns to numbers
# ML models only understand numbers
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    if col not in ["event_timestamp"]:
        df[col] = le.fit_transform(df[col].astype(str))

# Convert all columns to numeric
for col in df.columns:
    if col != "event_timestamp":
        df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(inplace=True)
print(f"After encoding: {len(df)} rows, {len(df.columns)} columns")

# Split into features and label
X = df.drop(["income","person_id","event_timestamp"], axis=1)
y = df["income"]

print(f"\nFeature matrix shape: {X.shape}")
print(f"Label distribution:")
print(f"  UNDER 50K: {(y==0).sum()} people ({(y==0).mean():.1%})")
print(f"  OVER 50K:  {(y==1).sum()} people ({(y==1).mean():.1%})")

# Save clean split files
X.to_csv(os.path.join(BASE, "feature_store", "data", "fs_X.csv"), index=False)
y.to_csv(os.path.join(BASE, "feature_store", "data", "fs_y.csv"), index=False)
print(f"\nSaved fs_X.csv and fs_y.csv")
print("Feature store training data is ready!")
