import os
import time
import pandas as pd
from feast import FeatureStore

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
store = FeatureStore(repo_path=os.path.join(BASE, "feature_store"))

# Test 3 different people
test_people = [
    {"person_id": 1},
    {"person_id": 100},
    {"person_id": 1000},
]

print("="*55)
print("REDIS ONLINE STORE — REAL-TIME FEATURE SERVING")
print("="*55)

for person in test_people:
    start = time.time()

    features = store.get_online_features(
        features=[
            "personal_features:age",
            "personal_features:sex",
            "work_features:education-num",
            "work_features:hours-per-week",
            "financial_features:capital-gain",
            "financial_features:relationship",
        ],
        entity_rows=[person]
    ).to_dict()

    elapsed = (time.time() - start) * 1000

    print(f"\nPerson {person['person_id']} — retrieved in {elapsed:.1f}ms")
    print(f"  age          : {features['age'][0]}")
    print(f"  sex          : {features['sex'][0]}")
    print(f"  education-num: {features['education-num'][0]}")
    print(f"  hours/week   : {features['hours-per-week'][0]}")
    print(f"  capital-gain : {features['capital-gain'][0]}")
    print(f"  relationship : {features['relationship'][0]}")

print("\n" + "="*55)
print("All features served from Redis in memory")
print("="*55)
