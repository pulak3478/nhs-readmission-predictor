"""
train_model.py
Trains a baseline logistic regression and an XGBoost classifier on the
processed dataset, prints comparison metrics, and saves the best model.

Run data_prep.py first to generate data/processed.csv.
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
TARGET = "readmitted_30d"


def build_pipeline(model, numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


def main():
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Baseline: Logistic Regression ---
    log_reg = build_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        numeric_cols, categorical_cols,
    )
    log_reg.fit(X_train, y_train)
    log_reg_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])

    # --- XGBoost ---
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = build_pipeline(
        XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
        ),
        numeric_cols, categorical_cols,
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:, 1])

    print(f"Logistic Regression AUC: {log_reg_auc:.4f}")
    print(f"XGBoost AUC:             {xgb_auc:.4f}")
    print("\nXGBoost classification report:")
    print(classification_report(y_test, xgb.predict(X_test)))

    os.makedirs(MODEL_DIR, exist_ok=True)
    best_model = xgb if xgb_auc >= log_reg_auc else log_reg
    joblib.dump(best_model, os.path.join(MODEL_DIR, "readmission_model.pkl"))
    print(f"\nSaved best model to {MODEL_DIR}/readmission_model.pkl")


if __name__ == "__main__":
    main()
