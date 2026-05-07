import pandas as pd
import numpy as np
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset

# Load training data (reference)
print("Loading training data...")
reference_data = pd.read_csv("data/X_train.csv")

# Simulate production data with drift
print("Simulating production data with drift...")
production_data = reference_data.copy()

# Introduce artificial drift
production_data['age'] = production_data['age'] + np.random.normal(5, 2, len(production_data))
production_data['education-num'] = production_data['education-num'] + np.random.normal(1, 0.5, len(production_data))
production_data['hours-per-week'] = production_data['hours-per-week'] - np.random.normal(3, 1, len(production_data))

print("Generating drift report...")

report = Report([DataDriftPreset()])

results = report.run(
    reference_data=Dataset.from_pandas(reference_data),
    current_data=Dataset.from_pandas(production_data)
)

results.save_html("data/drift_report.html")
print("✅ Drift report saved to data/drift_report.html")