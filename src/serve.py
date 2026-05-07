import bentoml
import pandas as pd
import pickle
import time
import os
from prometheus_client import Counter, Histogram
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# Load your best model
model_path = os.path.join(os.path.dirname(__file__), "best_model_feast.pkl")
model = pickle.load(open(model_path, "rb"))

# Prometheus metrics
prediction_counter = Counter(
    'income_predictions_total',
    'Total predictions made',
    ['prediction', 'version']
)

confidence_histogram = Histogram(
    'income_prediction_confidence',
    'Model confidence scores',
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
)

request_latency = Histogram(
    'income_request_latency_seconds',
    'Request latency in seconds'
)

class InputData(BaseModel):
    age: int
    workclass: str
    fnlwgt: float
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str

@bentoml.service()
class IncomeClassifier:

    @bentoml.api()
    def predict(self, input_data: InputData) -> dict:
        start_time = time.time()

        # Build dataframe with correct column ORDER
        df = pd.DataFrame([{
            "age": input_data.age,
            "race": input_data.race,
            "sex": input_data.sex,
            "native-country": input_data.native_country,
            "workclass": input_data.workclass,
            "education": input_data.education,
            "education-num": input_data.education_num,
            "occupation": input_data.occupation,
            "hours-per-week": input_data.hours_per_week,
            "fnlwgt": input_data.fnlwgt,
            "capital-gain": input_data.capital_gain,
            "capital-loss": input_data.capital_loss,
            "marital-status": input_data.marital_status,
            "relationship": input_data.relationship
        }])

        # Encode categorical columns
        categorical_cols = [
            'workclass', 'education', 'marital-status',
            'occupation', 'relationship', 'race',
            'sex', 'native-country'
        ]
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0].max()
        result = "over_50k" if prediction == 1 else "under_50k"

        # Track ML metrics
        prediction_counter.labels(
            prediction=result,
            version="2.0.3"
        ).inc()
        confidence_histogram.observe(float(probability))
        request_latency.observe(time.time() - start_time)

        return {
            "prediction": result,
            "probability": round(float(probability), 3)
        }

    @bentoml.api()
    def health(self) -> dict:
        return {
            "status": "healthy",
            "version": "2.0.3",
            "model": "XGBoost AUC 0.930"
        }