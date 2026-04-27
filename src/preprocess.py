import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# This finds the root folder (mlops-platform/) no matter where script runs from
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load params.yaml from root folder
with open(os.path.join(BASE, "params.yaml"), "r") as f:
    params = yaml.safe_load(f)

# Build absolute paths to all files
raw_path  = os.path.join(BASE, "data", "adult.csv")
test_size    = params["data"]["test_size"]
random_state = params["data"]["random_state"]

print(f"Loading data from: {raw_path}")

# Load raw data
cols = ["age","workclass","fnlwgt","education","education-num",
        "marital-status","occupation","relationship","race","sex",
        "capital-gain","capital-loss","hours-per-week","native-country","income"]

df = pd.read_csv(raw_path, names=cols, na_values=" ?", skipinitialspace=True)
df.dropna(inplace=True)

# Encode text columns to numbers
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Split into train and test
X = df.drop("income", axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# Save processed data using absolute paths
X_train.to_csv(os.path.join(BASE, "data", "X_train.csv"), index=False)
X_test.to_csv(os.path.join(BASE, "data", "X_test.csv"),  index=False)
y_train.to_csv(os.path.join(BASE, "data", "y_train.csv"), index=False)
y_test.to_csv(os.path.join(BASE, "data", "y_test.csv"),  index=False)

print(f"Done. Train: {len(X_train)} rows | Test: {len(X_test)} rows")