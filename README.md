<div align="center">

# DiyabetAI: Diabetes Risk Prediction with Machine Learning

**Predict diabetes risk from simple health-survey questions. Eight models trained and compared on 253,680 CDC BRFSS responses, plus a NHANES stage that looks for undiagnosed diabetes without a blood test.**

[![Live demo](https://img.shields.io/badge/Live_demo-diyabet--risk--analizi.netlify.app-00c7b7?style=for-the-badge)](https://diyabet-risk-analizi.netlify.app)

![Python](https://img.shields.io/badge/Python-3776ab?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-f7931e?style=flat-square&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189fdd?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-explainability-6f42c1?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

[Türkçe](README.tr.md) · English

</div>

> **Medical disclaimer.** DiyabetAI is a university machine-learning project for education and research.
> It is **not a medical device** and does not provide a diagnosis. The models were trained on United States
> survey data. If you are worried about diabetes, talk to a doctor.

## What it does

Diabetes often goes unnoticed because diagnosis needs a blood test. This project asks how far
**survey questions alone** can go, and compares the result with the clinical screening tools doctors use
today.

- **Stage 1: general risk model (CDC BRFSS 2015).** Eight models are trained and compared on 253,680 people
  and 21 questions.
- **Stage 2: undiagnosed diabetes (CDC NHANES 2007-2020).** A model that never sees a blood test tries to find
  people whose lab result indicates diabetes before a doctor has said so, and competes with the
  ADA Risk Test and FINDRISC.
- **A web app** serves the model with SHAP explanations, so every prediction shows which answers pushed the
  risk up or down.

## Results

All numbers come from the JSON files in [`backend/reports/`](backend/reports) and can be reproduced with the scripts below.

**Stage 1: BRFSS 2015 (ROC-AUC, from `model_sonuclari.json`)**

| Model | ROC-AUC |
|---|---|
| XGBoost (gradient boosting) | 0.8176 |
| Neural network (Keras MLP) | 0.8171 |
| Random Forest | 0.8156 |
| AdaBoost | 0.8141 |
| SVM (RBF kernel) | 0.8121 |
| Logistic Regression | 0.8097 |
| Decision Tree | 0.8062 |
| KNN | 0.8014 |

Income, education, health insurance and "could not see a doctor because of cost" were removed from the
final question set (17 questions remain). They are related to diabetes, but their effect is already carried by
weight, blood pressure and general-health answers, and the ablation in `model_sonuclari.json` shows they added only about 0.001 AUC.

**Stage 2: NHANES, trained on 2007-2016 and tested on unseen 2017-2020 data (n = 6,650)**

| Method | ROC-AUC |
|---|---|
| **DiyabetAI (logistic regression)** | **0.809** |
| FINDRISC | 0.769 |
| ADA Risk Test | 0.751 |

Six improvement candidates (XGBoost, splines, HbA1c regression, ensemble, dietary data and iterative
imputation) were tested under a strict acceptance rule (+0.003 AUC and better in 4 of 5 folds). None passed, so
the deployed model stays a calibrated logistic regression.

## Features

- 8 compared models with GridSearchCV, K-Fold cross-validation and a Keras neural network
- SHAP explanations, K-Means and hierarchical clustering, PCA and t-SNE visualisations
- FastAPI service: `/api/tahmin` (predict), `/api/sonuclar` (results), `/api/meta` (metadata)
- Build-free frontend (HTML, CSS, JavaScript) in Turkish
- **Static in-browser version** (`frontend/statik.js`): XGBoost trees, TreeSHAP, calibrated logistic regression
  and the ADA and FINDRISC scores run in JavaScript, so the site works on static hosting. `tests/test_statik.py`
  checks that it matches the Python API
- Automated tests (`tests/`) and CI

## Project structure

```
backend/
  ml/        data download, EDA, model training, SHAP and clustering, NHANES pipeline
  api/       FastAPI application
  models/    trained models (small ones are included)
  reports/   JSON results and figures
frontend/    web app (index.html, app.js, statik.js, styles.css)
tests/       API and static-version tests
```

## Quick start

```bash
git clone https://github.com/yusufgunduz-dev/diyabet-risk-ai.git
cd diyabet-risk-ai
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
```

Run the web app:

```bash
.venv/Scripts/python -m uvicorn backend.api.main:app --port 8000
# open http://localhost:8000
```

Retrain the models from scratch (datasets are downloaded, not stored in the repo):

```bash
.venv/Scripts/python -m backend.ml.veri_indir        # BRFSS from UCI
.venv/Scripts/python -m backend.ml.kesif_analizi     # exploratory analysis
.venv/Scripts/python -m backend.ml.model_egitimi     # 8 models
.venv/Scripts/python -m backend.ml.aciklama_kumeleme # SHAP and clustering
.venv/Scripts/python -m backend.ml.nhanes_indir      # 84 NHANES files, about 170 MB
.venv/Scripts/python -m backend.ml.nhanes_hazirla
.venv/Scripts/python -m backend.ml.nhanes_model
.venv/Scripts/python -m backend.ml.nhanes_uygulama
```

Run the tests:

```bash
.venv/Scripts/python -m pytest tests -q
```

## Data sources

| Dataset | Source |
|---|---|
| CDC BRFSS 2015 (Diabetes Health Indicators) | [UCI Machine Learning Repository #891](https://archive.ics.uci.edu/dataset/891) |
| CDC NHANES 2007-2020 | [CDC National Center for Health Statistics](https://www.cdc.gov/nchs/nhanes/) |

The data are not redistributed here. Please read each source's terms before using them.
See [RAPOR.md](RAPOR.md) (Turkish) for the full project report.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Yusuf Gündüz. Developed as a Machine Learning course term project.
