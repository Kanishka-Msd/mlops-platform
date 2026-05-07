import pandas as pd
import numpy as np
from evidently import Dataset, Report
from evidently.presets import DataDriftPreset
import subprocess

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

report = Report([DataDriftPreset(drift_share=0.2)])

results = report.run(
    reference_data=Dataset.from_pandas(reference_data),
    current_data=Dataset.from_pandas(production_data)
)

results.save_html("data/drift_report.html")
print("✅ Drift report saved to data/drift_report.html")

# Get drift results
results_dict = results.dump_dict()
first_key = list(results_dict['metric_results'].keys())[0]
first_metric = results_dict['metric_results'][first_key]

drifted_count = first_metric['count']['value']
drift_share = first_metric['share']['value']

print(f"Drifted columns: {int(drifted_count)}/14")
print(f"Drift share: {drift_share:.1%}")

# Trigger retraining if drift detected
if drift_share > 0.2:
    print("🚨 DRIFT DETECTED — Model needs retraining!")
    result = subprocess.run(["dvc", "repro"], capture_output=True, text=True)
    print(result.stdout)
    print("✅ Retraining triggered!")
else:
    print("✅ No drift — model is healthy!")