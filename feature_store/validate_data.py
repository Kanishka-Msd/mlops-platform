import pandas as pd
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*55)
print("DATA VALIDATION — Great Expectations")
print("="*55)

# Load all 3 feature group Parquet files
print("\nLoading feature groups...")
personal = pd.read_parquet(os.path.join(BASE, "feature_store", "data", "personal_features.parquet"))
work     = pd.read_parquet(os.path.join(BASE, "feature_store", "data", "work_features.parquet"))
financial= pd.read_parquet(os.path.join(BASE, "feature_store", "data", "financial_features.parquet"))
labels   = pd.read_parquet(os.path.join(BASE, "feature_store", "data", "labels.parquet"))

print(f"  personal:  {personal.shape}")
print(f"  work:      {work.shape}")
print(f"  financial: {financial.shape}")
print(f"  labels:    {labels.shape}")

# Track results
passed = 0
failed = 0
failures = []

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name} {detail}")
        failed += 1
        failures.append(name)

print("\n--- Row count checks ---")
check("personal has >30000 rows",    len(personal) > 30000)
check("work has >30000 rows",        len(work) > 30000)
check("financial has >30000 rows",   len(financial) > 30000)
check("all groups have same rows",   len(personal) == len(work) == len(financial))

print("\n--- Column existence checks ---")
check("personal has age column",          "age" in personal.columns)
check("personal has sex column",          "sex" in personal.columns)
check("personal has race column",         "race" in personal.columns)
check("work has education-num column",    "education-num" in work.columns)
check("work has hours-per-week column",   "hours-per-week" in work.columns)
check("financial has capital-gain column","capital-gain" in financial.columns)
check("financial has relationship column","relationship" in financial.columns)
check("labels has income column",         "income" in labels.columns)
check("labels has person_id column",      "person_id" in labels.columns)

print("\n--- Null value checks ---")
check("personal has no nulls",    personal.isnull().sum().sum() == 0,
      f"({personal.isnull().sum().sum()} nulls found)")
check("work has no nulls",        work.isnull().sum().sum() == 0,
      f"({work.isnull().sum().sum()} nulls found)")
check("financial has no nulls",   financial.isnull().sum().sum() == 0,
      f"({financial.isnull().sum().sum()} nulls found)")
check("labels has no nulls",      labels.isnull().sum().sum() == 0,
      f"({labels.isnull().sum().sum()} nulls found)")

print("\n--- Value range checks ---")
check("age has valid timestamp col",  "event_timestamp" in personal.columns)
check("labels has income values",     labels["income"].nunique() > 0)
check("person_id is unique",          labels["person_id"].nunique() == len(labels),
      f"({labels['person_id'].nunique()} unique vs {len(labels)} rows)")

print("\n--- Data type checks ---")
check("event_timestamp exists in personal",  "event_timestamp" in personal.columns)
check("event_timestamp exists in work",      "event_timestamp" in work.columns)
check("event_timestamp exists in financial", "event_timestamp" in financial.columns)

print("\n--- Distribution checks ---")
label_counts = labels["income"].value_counts()
total = len(labels)
most_common_pct = label_counts.iloc[0] / total
check("income not 100% one class",    most_common_pct < 0.99,
      f"(dominant class is {most_common_pct:.1%})")
check("income has at least 2 values", labels["income"].nunique() >= 2)

print("\n" + "="*55)
print(f"RESULTS: {passed} passed | {failed} failed")
print("="*55)

if failed > 0:
    print(f"\nFAILED CHECKS:")
    for f in failures:
        print(f"  - {f}")
    print("\nData quality issues detected.")
    print("Fix the data before training.")
    sys.exit(1)
else:
    print("\nAll checks passed.")
    print("Data is clean and ready for training.")
    sys.exit(0)
