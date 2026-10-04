#  Customer Churn Prediction

> An end-to-end Machine Learning project that predicts whether a telecom customer is likely to churn.

##  Overview

This project uses customer demographics, services, contracts, and billing information to identify customers who are at risk of leaving a telecom company.

###  Machine Learning Models

* Logistic Regression
* Random Forest
* XGBoost

###  Model Performance

| Model               |    ROC-AUC |   F1 Score |  Precision |     Recall |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.8377 |     0.6151 |     0.5023 |     0.7932 |
| **Random Forest**   | **0.8353** | **0.6265** | **0.5200** | **0.7879** |
| XGBoost             |     0.8359 |     0.6199 |     0.5147 |     0.7790 |

**Selected Model:** Random Forest

##  Project Architecture

```text
                    ┌──────────────────────┐
                    │   Telco Churn Data   │
                    │        (CSV)         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Data Cleaning     │
                    │  Missing Values      │
                    │  Feature Preparation │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Preprocessing      │
                    │                      │
                    │  Numerical Features  │
                    │  → Imputation        │
                    │  → Scaling           │
                    │                      │
                    │  Categorical         │
                    │  → Imputation        │
                    │  → One-Hot Encoding  │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌─────────────────────────────────┐
              │       Machine Learning          │
              │                                 │
              │  Logistic Regression            │
              │  Random Forest                  │
              │  XGBoost                        │
              └───────────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────────┐
                    │   Model Evaluation   │
                    │                      │
                    │ ROC-AUC • F1         │
                    │ Precision • Recall   │
                    │ Confusion Matrix     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Model Selection    │
                    │    Random Forest     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
          ┌──────────────────┐   ┌──────────────────┐
          │  SHAP & Feature  │   │    Streamlit     │
          │    Importance    │   │   Web Dashboard  │
          └──────────────────┘   └────────┬─────────┘
                                          │
                                          ▼
                               ┌──────────────────────┐
                               │ Churn Probability &  │
                               │ Business Insights    │
                               └──────────────────────┘
```

##  Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Matplotlib` `Seaborn` `SHAP` `Streamlit`

##  Features

*  Customer churn prediction
*  Churn probability estimation
*  Multiple ML model comparison
*  Feature importance analysis
*  SHAP model explanations
*  Interactive Streamlit dashboard
*  Class imbalance handling
*  Probability threshold analysis
*  Cross-validation and hyperparameter tuning

##  Project Structure

```text
customer-churn-prediction/
│
├── app/
│   └── app.py                    # Streamlit application
│
├── data/
│   └── raw/                      # Raw dataset
│
├── models/                       # Trained models
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── reports/                      # Evaluation results
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Data preprocessing
│   ├── train.py                  # Model training
│   └── predict.py                # Prediction logic
│
├── .gitignore
├── requirements.txt
└── README.md
```

##  Installation

```bash
git clone https://github.com/MuzammilAIX/customer-churn-prediction.git
cd customer-churn-prediction
python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the [IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place `Telco_Customer_Churn.csv` inside:

```text
data/raw/
```

##  Run

### Train Models

```bash
python -m src.train
```

### Launch Streamlit App

```bash
python -m streamlit run app/app.py
```

##  Business Goal

Identify customers with a high probability of churn so businesses can take proactive retention actions and reduce customer loss.

##  Author

**Muzammil AIX**

[![GitHub](https://img.shields.io/badge/GitHub-MuzammilAIX-black?logo=github)](https://github.com/MuzammilAIX)

---

⭐ If you find this project useful, consider giving it a star!
