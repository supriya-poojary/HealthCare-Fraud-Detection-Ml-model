# Healthcare Provider Fraud Detection 🏥

> **AI-powered system to detect fraudulent healthcare insurance providers using XGBoost + SMOTE**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎯 Problem Statement

Healthcare fraud costs the US insurance industry over **$60 billion annually**. This project builds a machine learning pipeline to predict potentially fraudulent providers based on their Inpatient, Outpatient, and Beneficiary claims data.

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── raw/          ← Place your CSV datasets here
│   └── processed/    ← Auto-generated engineered features
├── notebooks/
│   ├── generate_data.py        ← Synthetic data generator
│   ├── data_loader.py          ← Step 1: Data management
│   ├── feature_engineering.py  ← Step 3: Feature engineering
│   ├── modelling.py            ← Steps 4-6: ML pipeline
│   └── run_pipeline.py         ← MAIN SCRIPT — runs everything
├── models/
│   ├── best_model.pkl          ← Trained XGBoost model
│   └── pipeline_artifacts.pkl  ← All artifacts for Streamlit
├── app/
│   └── streamlit_app.py        ← Streamlit web application
├── outputs/
│   └── Supriyo_Submission.csv  ← Final submission file
├── reports/                    ← EDA plots, evaluation charts
└── requirements.txt
```

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your datasets
Place the 8 CSV files in `data/raw/`. If you don't have them, generate synthetic data:
```bash
python notebooks/generate_data.py
```

### 3. Run the full pipeline
```bash
python notebooks/run_pipeline.py
```
This runs all 7 steps and generates:
- `models/best_model.pkl`
- `outputs/Supriyo_Submission.csv`
- `reports/*.png` (EDA and evaluation charts)

### 4. Launch Streamlit app
```bash
streamlit run app/streamlit_app.py
```

## 📊 Methodology

| Step | Task | Details |
|------|------|---------|
| 1 | Data Management | Load & clean 8 CSV files, parse dates |
| 2 | EDA | Fraud patterns, class imbalance analysis |
| 3 | Feature Engineering | 70+ provider-level aggregated features |
| 4 | Feature Selection | Random Forest importance thresholding |
| 5 | Modelling | LR, RF, XGBoost, LightGBM with SMOTE |
| 6 | Evaluation | ROC-AUC, Confusion Matrix, P-R Curve |
| 7 | Deployment | Streamlit app + CSV submission |

## 🤖 Models Compared

- Logistic Regression (baseline)
- Random Forest
- **XGBoost** ← Best performer
- LightGBM

## 📈 Key Engineered Features

- `Total_Claims`, `IP/OP_TotalClaims`
- `IP_AvgReimbursed`, `IP_MaxReimbursed`
- `IP_AvgHospitalStay`, `IP_MaxHospitalStay`
- `IP_Unique_AttendingPhysician`
- `Bene_AvgChronicCond`, `Bene_DeadCount`
- `IP_OP_ClaimRatio`, `Avg_Reimbursed_Per_Claim`

## ⚙️ Handling Class Imbalance

- `class_weight='balanced'` on all models
- **SMOTE** (Synthetic Minority Oversampling) on best model
- Threshold tuning for optimal F1

## 📜 Citations & References

- [CMS Medicare Provider Data](https://data.cms.gov/)
- [NHCAA — National Health Care Anti-Fraud Association](https://www.nhcaa.org/)
- Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR.

## 🔒 Privacy

This data is confidential and not shared or published publicly.

---
Built with ❤️by Supriya using Python, XGBoost, and Streamlit
