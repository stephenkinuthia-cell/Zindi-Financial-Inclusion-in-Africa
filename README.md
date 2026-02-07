# Financial Inclusion in Africa – Zindi Challenge

## Project Overview

This project is an **end-to-end machine learning pipeline** developed for the Zindi learning challenge  **“Financial Inclusion in Africa”** . The goal is to predict whether an individual is likely to have or use a bank account based on demographic and socio-economic characteristics across four East African countries:  **Kenya, Rwanda, Tanzania, and Uganda** .

The project goes beyond model training to include  **structured preprocessing, model comparison, automatic model selection, interpretability using SHAP, and preparation for deployment via a dashboard** .

---

## Problem Statement

Financial inclusion is a key driver of economic growth and human development. However, a large proportion of adults in East Africa do not have access to formal banking services.

**Objective:**
Predict whether an individual has a bank account (`Yes = 1`, `No = 0`) using survey data, and identify the key factors driving financial inclusion.

---

## Machine Learning Framing

* **Task type:** Binary classification
* **Target variable:** `bank_account`
* **Evaluation focus:** Recall (due to class imbalance)
* **Metric used by competition:** Mean Absolute Error (MAE)

Because the dataset is imbalanced (fewer people have bank accounts), **recall is prioritized** to avoid missing individuals who are banked.

---

## Project Structure

```text
financial-inclusion-zindi/
│
├── data/
│   ├── raw/                # Original Zindi datasets (ignored in Git)
│   │   ├── Train.csv
│   │   └── Test.csv
│
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb   # Model training, evaluation & SHAP
│
├── src/
│   ├── preprocess.py       # Feature engineering pipelines
│   ├── train.py            # Model training & auto-selection
│   └── predict.py          # Inference & submission generation
│
├── models/
│   ├── __init__.py
│   ├── logistic.py
│   ├── random_forest.py
│   ├── xgboost_model.py
│   └── lightgbm_model.py
│
├── artifacts/
│   ├── preprocessor.pkl
│   └── best_model.pkl
│
├── outputs/
│   ├── submission.csv
│   └── feature_importance.csv
│
├── dashboard/
│   └── app.py              # (Planned) Streamlit dashboard
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Exploratory Data Analysis (EDA)

Key EDA steps performed:

* Target variable distribution (`Yes` vs `No`)
* Bank account ownership by:
  * Country
  * Gender
  * Location type (Urban/Rural)
  * Education level
  * Job type
* Identification of class imbalance

**Key insights:**

* Higher education levels strongly correlate with bank account ownership
* Urban residents are more likely to be banked
* Employment type and cellphone access are important drivers

---

## Feature Engineering

Feature engineering is centralized in `src/preprocess.py` and includes:

### Encoding strategies

* **Binary encoding:**
  * `cellphone_access`
  * `gender_of_respondent`
  * `location_type`
* **Ordinal encoding:**
  * `education_level`
* **One-hot encoding:**
  * `job_type`
  * `marital_status`
  * `relationship_with_head`
  * `country`
* **Numerical scaling:**
  * `age_of_respondent`
  * `household_size`

### Derived features

* `is_head_or_spouse`
* `high_education`
* `urban`

A **Scikit-learn `ColumnTransformer` pipeline** ensures consistent transformations across training and testing datasets.

---

## Models Implemented

The following models were implemented and compared:

* Logistic Regression (baseline)
* Random Forest
* XGBoost
* LightGBM

Each model is defined in a separate module under `models/` and wrapped in a pipeline with the shared preprocessor.

---

## Model Evaluation & Selection

Because the dataset is imbalanced, **recall** is the primary metric used for model selection.

Metrics evaluated:

* Recall
* Precision
* F1-score
* ROC AUC

An **automatic model selection routine** chooses the model with the highest recall and saves it as the final model.

```text
Best model → Saved to artifacts/best_model.pkl
```

---

## Model Interpretability (SHAP)

SHAP (SHapley Additive exPlanations) is used to interpret tree-based models.

Outputs include:

* Global feature importance rankings
* Mean absolute SHAP values per feature

Top influential features typically include:

* Education level
* Job type
* Cellphone access
* Age
* Urban vs rural location

These insights are saved to:

```text
outputs/feature_importance.csv
```

---

## Prediction & Submission

The `src/predict.py` script:

1. Loads the saved best model
2. Applies identical preprocessing to the test data
3. Generates predictions
4. Creates a Zindi-compliant submission file

```text
outputs/submission.csv
```

---

## Future Work

* Streamlit dashboard for interactive exploration
* Threshold tuning for recall optimization
* Cross-country comparative analysis
* Fairness & bias evaluation

---

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* SHAP
* Matplotlib / Seaborn
* Streamlit (planned)

---

## Notes

* Raw data is excluded from version control in compliance with Zindi rules
* Only open-source libraries are used
* No AutoML tools were applied

---
