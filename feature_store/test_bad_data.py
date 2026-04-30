import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE, "feature_store", "data", "personal_features.parquet")

print("Injecting bad data into personal_features.parquet...")

# Load good data
df = pd.read_parquet(path)
print(f"Before: {df.shape} — {df.isnull().sum().sum()} nulls")

# Save original for restore
df.to_parquet(path + ".backup", index=False)

# Inject nulls into age column
df.loc[0:500, "age"] = None

# Drop a column
df = df.drop("race", axis=1)

print(f"After:  {df.shape} — {df.isnull().sum().sum()} nulls injected")
print("Saved corrupted data. Now run validate_data.py to catch it.")

df.to_parquet(path, index=False)
