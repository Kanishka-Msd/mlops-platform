# MLOps Platform

A production-grade MLOps pipeline built from scratch.

## What this project does
Trains a machine learning model to predict whether a person earns
over $50K/year using US census data — with full experiment tracking,
data versioning, and reproducibility.

## Tech stack
- **MLflow** — experiment tracking, model registry
- **DVC** — data versioning and pipeline automation
- **scikit-learn** — model training (RandomForest, LogisticRegression)
- **Git** — code versioning

## Results
Best model: Random Forest (200 trees, depth 15)
- ROC-AUC: 0.920
- F1 Score: 0.691
- Accuracy: 86.7%

## How to reproduce this exactly
```bash
git clone https://github.com/Kanishka-Msd/mlops-platform
cd mlops-platform
python -m venv venv && source venv/bin/activate
pip install mlflow scikit-learn pandas numpy dvc
dvc checkout        # restores exact dataset
cd src
python train.py     # trains all 10 models
mlflow ui           # view all experiments
```

## Project structure
mlops-platform/
├── data/
│   └── adult.csv.dvc      # dataset fingerprint (DVC tracked)
├── src/
│   ├── preprocess.py      # data cleaning + train/test split
│   └── train.py           # model training + MLflow logging
├── .dvc/                  # DVC configuration
└── README.md

## Week 1 progress
- [x] Day 1: MLflow experiment tracking — 10 models compared
- [x] Day 2: DVC data versioning — fully reproducible pipeline
- [ ] Day 3: DVC pipeline automation
- [ ] Day 4: params.yaml config-driven experiments
- [ ] Day 5: Full pipeline review

