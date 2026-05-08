# MLOps Platform 🚀

A production-grade MLOps platform built from scratch covering the complete ML lifecycle.

## 🌐 Live Demo
- **API:** https://income-classifier-1.onrender.com
- **GitHub:** https://github.com/Kanishka-Msd/mlops-platform
- **DagsHub:** https://dagshub.com/Kanishka-Msd/mlops-platform

## 🎯 What this project does
Trains a machine learning model to predict whether a person earns over $50K/year using US census data — with full experiment tracking, data versioning, feature store, CI/CD, monitoring, drift detection, and automated orchestration.

## 🏆 Results
- **Best Model:** XGBoost
- **ROC-AUC:** 0.930 (Week 2 Feast pipeline)
- **F1 Score:** 0.719
- **Accuracy:** 87.4%
- **Serving latency:** <25ms

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Experiment Tracking | MLflow |
| Data Versioning | DVC + DagsHub |
| Feature Store | Feast + Redis |
| CI/CD | GitHub Actions + CML |
| Model Serving | BentoML + Docker |
| Orchestration | Kubernetes |
| Monitoring | Prometheus + Grafana |
| Drift Detection | EvidentlyAI |
| Pipeline Automation | Apache Airflow |

## 📁 Project Structure

mlops-platform/
├── data/                    # datasets (DVC tracked)
├── src/
│   ├── preprocess.py        # data cleaning
│   ├── train.py             # model training + MLflow
│   ├── evaluate.py          # metrics evaluation
│   ├── validate_model.py    # model quality gate
│   ├── serve.py             # BentoML REST API
│   ├── detect_drift.py      # EvidentlyAI drift detection
│   ├── psi_monitor.py       # PSI score monitoring
│   └── monitor_predictions.py # output monitoring
├── feature_store/           # Feast feature store
├── kubernetes/              # K8s deployment configs
├── monitoring/              # Prometheus + Grafana
├── airflow/                 # Airflow DAGs
├── .github/workflows/       # CI/CD pipeline
├── dvc.yaml                 # DVC pipeline
├── bentofile.yaml           # BentoML config
└── params.yaml              # model parameters

## 🚀 8-Week Journey

### Week 1 ✅ — Experiment Tracking
- MLflow tracking for 10+ experiments
- DVC data versioning with MD5 fingerprinting
- XGBoost vs Random Forest comparison
- Best: XGBoost AUC 0.927

### Week 2 ✅ — Feature Store
- Feast feature store with 3 feature views
- Redis online store (<1ms serving)
- Point-in-time correct retrieval
- Improved AUC to 0.930!

### Week 3 ✅ — CI/CD Pipeline
- GitHub Actions auto-runs on every push
- CML posts metrics as PR comments
- Model validation gate (AUC > 0.90)
- Data validation (25 automated checks)
- Scheduled retraining every Sunday

### Week 4 ✅ — Model Serving
- BentoML REST API
- Dockerized and deployed on Render
- Load tested with Locust (50 concurrent users)
- Blue/Green deployment strategy

### Week 5 ✅ — Kubernetes
- 3 replica pods running simultaneously
- HPA auto-scaling (2-10 pods based on CPU)
- Rolling updates with zero downtime
- Health checks and self-healing

### Week 6 ✅ — Monitoring
- Prometheus scraping API metrics every 15s
- Grafana dashboards for real-time visualization
- Custom ML metrics (prediction distribution)
- Alert rules for high error rates

### Week 7 ✅ — Drift Detection
- EvidentlyAI drift reports with HTML output
- PSI score monitoring per feature
- Model output distribution monitoring
- Automatic retraining trigger on drift

### Week 8 ✅ — Pipeline Automation
- Apache Airflow DAG orchestrating full pipeline
- Conditional retraining (only when drift detected)
- Failure notifications
- Runs every Sunday at midnight automatically

## 🔄 Automated Pipeline (Airflow DAG)

validate_data → check_drift → [retrain OR skip] → validate_model → monitor_predictions → summary

## 🚀 Quick Start
```bash
git clone https://github.com/Kanishka-Msd/mlops-platform
cd mlops-platform
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull                    # restore dataset
dvc repro                   # run full pipeline
mlflow ui                   # view experiments
```

## 📊 API Usage
```bash
curl -X POST https://income-classifier-1.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 👨‍💻 Author
**Kanishka** — Junior ML Engineer
- GitHub: https://github.com/Kanishka-Msd
- DagsHub: https://dagshub.com/Kanishka-Msd
