"""
data_prep.py
Loads the UCI "Diabetes 130-US Hospitals" dataset, cleans it, and saves a
model-ready CSV to data/processed.csv.

Dataset: https://archive.ics.uci.edu/dataset/296
Citation: Strack et al. (2014), BioMed Research International.
"""

import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_PATH = os.path.join(OUT_DIR, "processed.csv")

# Columns known to be near-empty or zero-variance in this dataset.
DROP_COLS = [
    "weight",            # ~97% missing
    "payer_code",        # high missingness, not clinically predictive
    "medical_specialty", # ~50% missing, optional to re-add later as a feature
    "encounter_id",
    "patient_nbr",
    "examide",            # zero variance
    "citoglipton",         # zero variance
]

# Discharge dispositions meaning the patient died or went to hospice —
# these can't meaningfully "be readmitted", so they're excluded.
EXCLUDE_DISCHARGE_IDS = [11, 13, 14, 19, 20, 21]


def load_raw() -> pd.DataFrame:
    """Fetch the dataset via the UCI ML Repository API."""
    dataset = fetch_ucirepo(id=296)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Replace UCI's "?" missing-value marker with NaN
    df = df.replace("?", pd.NA)

    # Drop sparse / non-predictive / zero-variance columns that exist
    existing_drops = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drops)

    # Drop patients discharged to hospice / deceased
    if "discharge_disposition_id" in df.columns:
        df = df[~df["discharge_disposition_id"].isin(EXCLUDE_DISCHARGE_IDS)]

    # Binary target: readmitted within 30 days vs. everything else
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Drop rows still missing key demographic fields
    df = df.dropna(subset=["race", "gender", "age"])

    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Fetching dataset from UCI ML Repository (id=296)...")
    raw = load_raw()
    print(f"Raw shape: {raw.shape}")

    processed = clean(raw)
    print(f"Processed shape: {processed.shape}")
    print(f"Positive class rate (readmitted <30d): "
          f"{processed['readmitted_30d'].mean():.3%}")

    processed.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
