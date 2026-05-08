import pandas as pd
import numpy as np
import pickle

# Load model
print("Loading model...")
model = pickle.load(open("data/best_model.pkl", "rb"))

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").values.ravel()

# Reorder columns to match model's expected order
correct_order = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
]
X_test = X_test[correct_order]

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate output metrics
total = len(predictions)
over_50k = sum(predictions == 1)
under_50k = sum(predictions == 0)
over_50k_pct = over_50k / total
under_50k_pct = under_50k / total

confidence_scores = model.predict_proba(X_test).max(axis=1)
avg_confidence = confidence_scores.mean()
low_confidence = sum(confidence_scores < 0.6) / total

print("\n📊 Model Output Monitoring Report:")
print("─" * 45)
print(f"Total predictions:     {total}")
print(f"Over $50K:             {over_50k} ({over_50k_pct:.1%})")
print(f"Under $50K:            {under_50k} ({under_50k_pct:.1%})")
print(f"Avg confidence:        {avg_confidence:.3f}")
print(f"Low confidence (<60%): {low_confidence:.1%}")
print("─" * 45)

# Check for output drift
print("\n🔍 Output Health Checks:")

# Check 1: prediction distribution
if over_50k_pct < 0.10 or over_50k_pct > 0.50:
    print("🚨 Prediction distribution unusual!")
    print(f"   Expected ~24% over_50k, got {over_50k_pct:.1%}")
else:
    print(f"✅ Prediction distribution normal ({over_50k_pct:.1%} over_50k)")

# Check 2: confidence scores
if avg_confidence < 0.70:
    print("🚨 Model confidence too low!")
    print(f"   Average confidence: {avg_confidence:.3f}")
else:
    print(f"✅ Model confidence healthy ({avg_confidence:.3f})")

# Check 3: low confidence predictions
if low_confidence > 0.20:
    print(f"🚨 Too many low confidence predictions ({low_confidence:.1%})")
else:
    print(f"✅ Low confidence predictions acceptable ({low_confidence:.1%})")