from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "rf_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_churn(customer_data):
    model = load_model()
    customer_df = pd.DataFrame([customer_data])
    probability = float(model.predict_proba(customer_df)[0][1])
    prediction = int(model.predict(customer_df)[0])
    return {
        "prediction": prediction,
        "churn_probability": probability
    }
