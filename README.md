# Delivery Duration Prediction with MLOps (ZenML)

![Python](https://img.shields.io/badge/Python-3.9-blue.svg) ![ZenML](https://img.shields.io/badge/MLOps-ZenML-purple)

## Overview

An end-to-end MLOps pipeline built using ZenML to predict delivery durations using historical food delivery data. The project modularizes data preparation, feature engineering, prep-time prediction, and final duration regression to ensure scalability and reproducibility.

---

## Problem Statement

Food delivery companies often struggle to estimate how long a delivery will take. This project aims to accurately predict **actual total delivery duration** (in seconds) using historical data.

---

## Features

* 📦 Load raw delivery data from compressed ZIP files
* 🧹 Clean and preprocess features
* 🧠 Train LightGBM to estimate **prep time**
* ➕ Combine with estimated order and driving durations
* 📊 Train final Random Forest model to predict total delivery time
* 📈 Interactive evaluation using HTML & Plotly

---

## ⚙️ Tech Stack

* **ML Models**: LightGBM, RandomForest
* **Framework**: ZenML
* **Visualization**: Plotly
* **Languages**: Python, Pandas, Sklearn

---

## 🛠 Pipeline Steps

```text
1. load_data                ➔ Load zipped CSVs
2. data_cleaning_step       ➔ Drop invalid rows, create target
3. feature_engineering_step ➔ Ratios, dummies, collinear drop, VIF
4. feature_enhancement_step ➔ Improve features (future)
5. split_data_step          ➔ Train-test split
6. train_prep_model_step    ➔ LightGBM predicts prep_time
7. merge_predictions_step   ➔ Combine with estimated_* durations
8. final_regression_model   ➔ RandomForest final regression
9. evaluate_model           ➔ RMSE, MAE, R2 + HTML Report
```

---

## 📁 Project Structure

```
.
├── steps/
│   ├── data_loader.py
│   ├── data_cleaner.py
│   ├── feature_engineering.py
│   ├── ...
├── pipelines/
│   └── training_pipeline.py
├── run_pipeline.py
├── datasets.zip
├── README.md
└── requirements.txt
```

---

## 📉 Evaluation Report

| Metric | Value     |
| ------ | --------- |
| RMSE   | 1095.40 s |
| MAE    | 717.44 s  |
| R²     | 0.1495    |

![Evaluation](assets/metrics.png)

---

## 💻 Getting Started

```bash
# Clone repo
https://github.com/Msingisi/Delivery-Duration-Prediction-Mlops.git
cd Delivery-Duration-Prediction-Mlops

# Create virtual env
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py
```

---

## 📊 Visual Reports

Interactive Plotly report showing actual vs predicted values and summary metrics is auto-generated in HTML.

---

## 🙏 Acknowledgements

* Dataset inspired by Doordash delivery analysis
* ZenML for pipeline management
* Scikit-learn, LightGBM, and Plotly for modeling and visuals

---

## ✨ Future Enhancements

* Integrate MLflow for experiment tracking
* Add feature drift monitoring with Evidently
* Deploy with FastAPI
* Add CI/CD with GitHub Actions

---