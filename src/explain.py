"""
explain.py
Loads the trained model and produces a SHAP summary plot, so predictions
can be explained to a non-technical (clinical/managerial) audience —
this is the piece that makes the project credible beyond a raw accuracy score.
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "readmission_model.pkl")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
TARGET = "readmitted_30d"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET]).sample(n=300, random_state=42)  # subsample for speed

    pipeline = joblib.load(MODEL_PATH)
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()

    # Explain via the prediction function rather than the raw XGBoost model
    # object — avoids a known shap/xgboost internal-parsing version clash.
    predict_fn = lambda data: model.predict_proba(data)[:, 1]

    explainer = shap.Explainer(predict_fn, X_transformed, feature_names=feature_names)
    shap_values = explainer(X_transformed)

    os.makedirs(OUT_DIR, exist_ok=True)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "shap_summary.png"), dpi=150)
    print(f"Saved SHAP summary plot to {OUT_DIR}/shap_summary.png")


if __name__ == "__main__":
    main()