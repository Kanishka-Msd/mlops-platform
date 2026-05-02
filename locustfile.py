from locust import HttpUser, task, between

class IncomeAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_income(self):
        self.client.post("/predict", json={
            "input_data": {
                "age": 35,
                "workclass": "Private",
                "fnlwgt": 200000,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Prof-specialty",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        })