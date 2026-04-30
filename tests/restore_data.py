import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(BASE, "feature_store", "data", "personal_features.parquet")

backup = path + ".backup"
if os.path.exists(backup):
    df = pd.read_parquet(backup)
    df.to_parquet(path, index=False)
    os.remove(backup)
    print("Good data restored successfully!")
else:
    print("No backup found!")
