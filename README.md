# Predicting 30-Day Hospital Readmission Risk in Diabetic Patients

**MSc Data Science Dissertation — University of South Wales**
**Framed for NHS Wales relevance:** unplanned readmissions are a nationally tracked
NHS quality/cost metric. This project builds an explainable ML pipeline that flags
patients at high risk of readmission within 30 days of discharge, using a large,
well-documented public clinical dataset as a stand-in for NHS Wales PEDW
(Patient Episode Database for Wales) data.

## 1. Why this project

- Readmission reduction is a standing NHS Wales / Welsh Government priority
  (linked to both patient outcomes and cost).
- It's a **structured tabular** problem, not an image problem — realistic to build,
  train, and explain end-to-end in the time you have, with room to extend into
  deep learning (e.g. sequence models on visit history) if you want to go further.
- Explainability (SHAP) is built in from day one — this is what makes the project
  credible to a clinical/operational audience, not just a Kaggle leaderboard entry.

## 2. Dataset

**UCI Diabetes 130-US Hospitals (1999–2008)**
- 101,766 patient encounters, 130 US hospitals, ~50 raw features
- Target: `readmitted` → `<30` days, `>30` days, `NO`
  (this project treats it as binary: readmitted <30 days vs. not)
- Citation: Strack et al. (2014), *Impact of HbA1c Measurement on Hospital
  Readmission Rates*, BioMed Research International.
- UCI page: https://archive.ics.uci.edu/dataset/296
- Loaded via the `ucimlrepo` Python package — no manual download needed.

> Note for your dissertation write-up: be explicit that this is a **US public
> dataset used as a methodological proxy**, and that a production NHS Wales
> version would require IG (Information Governance) approval and PEDW/SAIL
> Databank access. Naming this limitation clearly is itself a strength in
> your write-up — it shows you understand real NHS data governance.

## 3. Project structure

```
nhs-readmission-predictor/
├── README.md
├── requirements.txt
├── src/
│   ├── data_prep.py      # load + clean UCI dataset
│   ├── train_model.py    # train baseline + XGBoost model
│   └── explain.py        # SHAP explainability
├── app.py                 # Streamlit demo dashboard
└── .gitignore
```

## 4. Quickstart

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/data_prep.py         # downloads + cleans data -> data/processed.csv
python src/train_model.py       # trains models -> models/xgb_model.pkl
python src/explain.py           # generates SHAP summary plot -> outputs/shap_summary.png
streamlit run app.py            # interactive risk-prediction demo
```

## 5. Suggested dissertation roadmap

| Phase | Weeks | What |
|---|---|---|
| 1 | 1–2 | Lit review: NHS readmission policy, prior ML readmission studies |
| 2 | 3–4 | EDA + data cleaning (this repo's `data_prep.py` is your starting point) |
| 3 | 5–7 | Baseline models (logistic regression) → gradient boosting (XGBoost/LightGBM) |
| 4 | 8–9 | Explainability (SHAP), fairness check across race/age/gender subgroups |
| 5 | 10–11 | Streamlit dashboard for a non-technical (clinical/managerial) audience |
| 6 | 12+ | Write-up: explicitly map findings back to NHS Wales strategic priorities |

## 6. Ideas to extend further (pick 1 if you want more depth)

- Compare against a **deep learning tabular model** (TabNet or a simple MLP) as
  a secondary contribution.
- Add a **fairness audit**: does the model perform equally well across
  demographic subgroups? (very relevant to NHS equality duties)
- Swap in **StatsWales / NHS Wales open datasets** for a second, smaller
  case study (e.g. A&E attendance trends) to explicitly localise the work to Wales.
