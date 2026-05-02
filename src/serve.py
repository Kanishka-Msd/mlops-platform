import bentoml
import pandas as pd
import pickle
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder

# Load your best model
model = pickle.load(open("data/best_model_feast.pkl", "rb"))

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

        return {
            "prediction": "over_50k" if prediction == 1 else "under_50k",
            "probability": round(float(probability), 3)
        }
        
        @bentoml.api()
        def health(self) -> dict:
            return {
                "status": "healthy",
                "version": "1.0.3",
                "model": "XGBoost AUC 0.930"
            }