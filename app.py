"""
app.py
Streamlit demo: enter a patient's details, get a readmission risk score.
This is the piece you demo live in your dissertation viva / to NHS contacts —
a working prototype is far more memorable than a notebook of metrics.

Run with: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "readmission_model.pkl")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "processed.csv")

st.set_page_config(page_title="Readmission Risk Predictor", layout="centered")
st.title("30-Day Readmission Risk Predictor")
st.caption(
    "MSc Data Science dissertation prototype — trained on the UCI Diabetes "
    "130-US Hospitals dataset as a methodological proxy for NHS Wales data."
)

if not os.path.exists(MODEL_PATH):
    st.error("No trained model found. Run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
reference = pd.read_csv(DATA_PATH).drop(columns=["readmitted_30d"])

st.subheader("Patient details")
col1, col2 = st.columns(2)

with col1:
    age = st.selectbox("Age band", sorted(reference["age"].unique()))
    gender = st.selectbox("Gender", sorted(reference["gender"].unique()))
    race = st.selectbox("Race", sorted(reference["race"].dropna().unique()))
    time_in_hospital = st.slider("Days in hospital", 1, 14, 3)

with col2:
    num_medications = st.slider("Number of medications", 0, 50, 10)
    num_lab_procedures = st.slider("Number of lab procedures", 0, 130, 40)
    number_inpatient = st.slider("Inpatient visits in prior year", 0, 20, 0)
    number_emergency = st.slider("Emergency visits in prior year", 0, 20, 0)

if st.button("Predict readmission risk"):
    # Build a single-row input using reference medians/modes for any
    # feature not exposed in this simplified demo form.
    row = reference.iloc[[0]].copy()
    row["age"] = age
    row["gender"] = gender
    row["race"] = race
    row["time_in_hospital"] = time_in_hospital
    row["num_medications"] = num_medications
    row["num_lab_procedures"] = num_lab_procedures
    row["number_inpatient"] = number_inpatient
    row["number_emergency"] = number_emergency

    risk = model.predict_proba(row)[0, 1]
    st.metric("Predicted 30-day readmission risk", f"{risk:.1%}")

    if risk > 0.5:
        st.warning("High risk — consider flagging for follow-up / discharge planning review.")
    else:
        st.success("Lower risk based on current inputs.")

st.divider()
st.caption(
    "Limitation: this demo uses a simplified subset of inputs and a US public "
    "dataset. A production NHS Wales version would require PEDW/SAIL Databank "
    "access and Information Governance approval."
)
