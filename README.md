# Customer Churn Prediction

A production-style machine learning project for predicting telecommunications customer churn.

## What this project demonstrates

- Data cleaning
- Train/test splitting with stratification
- Missing-value handling
- Numeric scaling
- One-hot encoding
- Class-imbalance handling
- Logistic Regression baseline
- Random Forest
- XGBoost
- Cross-validation
- Random Forest GridSearchCV
- ROC-AUC, precision, recall, F1 and accuracy
- Threshold analysis
- Feature importance
- SHAP explainability
- Model persistence with Joblib
- Streamlit deployment

## Standard ML project workflow

Business Problem → Data Understanding → Data Cleaning → EDA → Train/Test Split → Preprocessing → Baseline → Model Comparison → Cross Validation → Hyperparameter Tuning → Evaluation → Threshold Analysis → Explainability → Save Model → Deployment

## Setup on Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the dataset here:

```text
data/raw/Telco_Customer_Churn.csv
```

Train everything:

```powershell
python -m src.train
```

Run the app:

```powershell
streamlit run app/app.py
```

The training process creates:

```text
models/logistic_model.pkl
models/rf_model.pkl
models/xgb_model.pkl
reports/metrics.json
reports/feature_importance.csv
reports/threshold_analysis.csv
```

Do not commit the dataset or trained model binaries to GitHub unless appropriate for your project/license.
