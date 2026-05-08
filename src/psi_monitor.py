import pandas as pd
import numpy as np

def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index"""
    
    # Create bins from reference data
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1
    )
    
    # Calculate distributions
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    
    # Convert to percentages
    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)
    
    # Avoid division by zero
    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)
    
    # Calculate PSI
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return psi

# Load data
print("Loading data...")
reference_data = pd.read_csv("data/X_train.csv")
production_data = reference_data.copy()

# Simulate drift
production_data['age'] = production_data['age'] + np.random.normal(5, 2, len(production_data))
production_data['hours-per-week'] = production_data['hours-per-week'] - np.random.normal(3, 1, len(production_data))

# Calculate PSI for numeric columns
numeric_cols = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']

print("\n📊 PSI Report:")
print("─" * 40)

for col in numeric_cols:
    psi = calculate_psi(reference_data[col], production_data[col])
    
    if psi < 0.1:
        status = "✅ Stable"
    elif psi < 0.2:
        status = "⚠️ Monitor"
    else:
        status = "🚨 Retrain!"
    
    print(f"{col:20} PSI: {psi:.3f}  {status}")

print("─" * 40)
print("\nPSI Thresholds:")
print("< 0.1  = Stable ✅")
print("0.1-0.2 = Monitor ⚠️")
print("> 0.2  = Retrain 🚨")