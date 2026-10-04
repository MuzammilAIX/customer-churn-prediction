from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="",
    layout="wide"
)

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load(MODEL_DIR / "logistic_model.pkl"),
        "Random Forest": joblib.load(MODEL_DIR / "rf_model.pkl"),
        "XGBoost": joblib.load(MODEL_DIR / "xgb_model.pkl")
    }

@st.cache_data
def load_reports():
    with open(REPORT_DIR / "metrics.json", "r") as f:
        metrics = json.load(f)
    feature_importance = pd.read_csv(REPORT_DIR / "feature_importance.csv")
    return metrics, feature_importance

models = load_models()
metrics, feature_importance = load_reports()

tab1, tab2 = st.tabs(["Prediction", "Model Insights"])

with tab1:
    st.title(" Customer Churn Prediction App")
    st.write("Enter customer details to predict churn probability.")

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72)
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])
        multiplelines = st.selectbox(
            "Multiple Lines", ["Yes", "No", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

    with col2:
        onlinesecurity = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        onlinebackup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        deviceprotection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        techsupport = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )
        streamingtv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streamingmovies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly = st.number_input(
            "Monthly Charges", min_value=0.0,
            max_value=200.0, value=70.0
        )
        total = st.number_input(
            "Total Charges", min_value=0.0,
            max_value=10000.0, value=1000.0
        )

    data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines],
        "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity],
        "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection],
        "TechSupport": [techsupport],
        "StreamingTV": [streamingtv],
        "StreamingMovies": [streamingmovies],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly],
        "TotalCharges": [total]
    })

    if st.button("Predict Churn", type="primary", use_container_width=True):
        model = models[model_choice]
        probability = float(model.predict_proba(data)[0][1])
        prediction = int(model.predict(data)[0])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Churn Probability", f"{probability * 100:.1f}%")
        with c2:
            st.metric(
                "Prediction",
                "Likely to Churn" if prediction == 1 else "Likely to Stay"
            )

        st.progress(probability)

        if probability > 0.60:
            st.error("🔴 High Risk Customer")
        elif probability > 0.30:
            st.warning("🟠 Medium Risk Customer")
        else:
            st.success("🟢 Low Risk Customer")

        st.write(f"**Model Used:** {model_choice}")

        if model_choice == "Random Forest":
            st.subheader("Prediction Explanation (SHAP)")

            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["model"]

            X_transformed = preprocessor.transform(data)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()

            feature_names = preprocessor.get_feature_names_out()
            X_transformed_df = pd.DataFrame(
                X_transformed, columns=feature_names
            )

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer(X_transformed_df)

            if len(shap_values.values.shape) == 3:
                shap_class = shap_values[:, :, 1]
            else:
                shap_class = shap_values

            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(
                shap_class[0], max_display=15, show=False
            )
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "Individual SHAP explanation is configured for "
                "Random Forest in this version."
            )

with tab2:
    st.title("📊 Model Insights")

    st.subheader("Top Drivers of Customer Churn")
    top_features = feature_importance.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_features["feature"],
        top_features["importance"]
    )
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top Drivers of Customer Churn")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Model Performance")

    model_metrics = metrics.get("metrics", {})
    rows = []
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        if name in model_metrics:
            result = model_metrics[name]
            rows.append({
                "Model": name,
                "ROC AUC": result["roc_auc"],
                "F1 Score": result["f1"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "Accuracy": result["accuracy"]
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Business Insights")
    st.markdown("""
### Key Drivers of Customer Churn

The model helps identify customers with a higher probability of leaving.

### Customer Tenure
Customers with shorter tenure can represent a higher-risk group.

### Contract Type
Contract structure can be an important signal of customer retention.

### Monthly Charges
Monthly charges can contribute to churn risk for some customer segments.

### Business Use
Predictions can help prioritize customers for retention campaigns,
personalized offers, or proactive support.
""")
