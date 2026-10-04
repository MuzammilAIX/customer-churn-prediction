from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from src.preprocessing import (
    create_logistic_pipeline, create_rf_pipeline, create_xgb_pipeline
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Telco_Customer_Churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Put Telco_Customer_Churn.csv inside data/raw/"
        )
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
    }

def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    logistic_model = create_logistic_pipeline()
    logistic_model.fit(X_train, y_train)
    logistic_cv = cross_val_score(
        logistic_model, X_train, y_train,
        scoring="roc_auc", cv=5, n_jobs=-1
    )

    rf_model = create_rf_pipeline()
    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 5]
    }
    grid_rf = GridSearchCV(
        rf_model, param_grid, scoring="roc_auc",
        cv=3, n_jobs=-1, verbose=1
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    rf_cv = cross_val_score(
        best_rf, X_train, y_train,
        scoring="roc_auc", cv=5, n_jobs=-1
    )

    neg, pos = np.bincount(y_train)
    scale_pos_weight = float(neg / pos)
    xgb_model = create_xgb_pipeline(scale_pos_weight)
    xgb_model.fit(X_train, y_train)
    xgb_cv = cross_val_score(
        xgb_model, X_train, y_train,
        scoring="roc_auc", cv=5, n_jobs=-1
    )

    models = {
        "Logistic Regression": logistic_model,
        "Random Forest": best_rf,
        "XGBoost": xgb_model
    }
    metrics = {
        name: evaluate_model(model, X_test, y_test)
        for name, model in models.items()
    }

    metrics["Logistic Regression"]["cv_auc_mean"] = round(float(logistic_cv.mean()), 4)
    metrics["Random Forest"]["cv_auc_mean"] = round(float(rf_cv.mean()), 4)
    metrics["XGBoost"]["cv_auc_mean"] = round(float(xgb_cv.mean()), 4)

    feature_names = best_rf.named_steps["preprocessor"].get_feature_names_out()
    importances = best_rf.named_steps["model"].feature_importances_
    pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).to_csv(
        REPORT_DIR / "feature_importance.csv", index=False
    )

    rf_prob = best_rf.predict_proba(X_test)[:, 1]
    threshold_rows = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        pred = (rf_prob >= threshold).astype(int)
        threshold_rows.append({
            "threshold": round(float(threshold), 2),
            "precision": round(precision_score(y_test, pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, pred, zero_division=0), 4)
        })
    pd.DataFrame(threshold_rows).to_csv(
        REPORT_DIR / "threshold_analysis.csv", index=False
    )

    joblib.dump(logistic_model, MODEL_DIR / "logistic_model.pkl")
    joblib.dump(best_rf, MODEL_DIR / "rf_model.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgb_model.pkl")

    metadata = {
        "random_state": 42,
        "test_size": 0.30,
        "best_rf_params": grid_rf.best_params_,
        "best_rf_cv_auc": round(float(grid_rf.best_score_), 4),
        "scale_pos_weight": round(scale_pos_weight, 4),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "metrics": metrics
    }
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\nTraining completed successfully.")
    for name, result in metrics.items():
        print(
            f"{name}: ROC-AUC={result['roc_auc']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"Precision={result['precision']:.4f}, "
            f"Recall={result['recall']:.4f}"
        )

if __name__ == "__main__":
    main()
