import os
import json

# Define the cells
cells = [
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🏥 Healthcare Provider Fraud Detection\n",
            "### End-to-End Machine Learning Pipeline (XGBoost + SMOTE)\n",
            "**Author:** Supriyo  \n",
            "**Date:** June 2026\n\n",
            "---\n\n",
            "## 📋 Project Overview & Objectives\n",
            "Provider Fraud is one of the most critical issues in the health insurance domain, accounting for tens of billions of dollars in losses annually. Healthcare fraud involves complex, collusive behavior among providers, physicians, and beneficiaries to inflate claims, charge for services not rendered, duplicate claims, or upcode procedures.\n\n",
            "The objective of this project is to **predict potentially fraudulent providers** based on claims data (Inpatient and Outpatient) and Beneficiary KYC details.\n\n",
            "---\n\n",
            "## 🛠 Methodology & Steps\n",
            "1. **Data Management**: Loading and cleaning 8 CSV files (Train/Test splits).\n",
            "2. **Exploratory Data Analysis**: Analyzing the class distribution and verifying fraud patterns (e.g. duplicate claims, inflated reimbursements, phantom billing).\n",
            "3. **Feature Engineering**: Creating over 50 provider-level features (claims volume, financial metrics, clinical metrics, chronic condition rates, patient demographics).\n",
            "4. **Feature Selection**: Identifying highly predictive variables using Random Forest feature importance.\n",
            "5. **Modelling & CV Comparison**: Comparing Logistic Regression, Random Forest, LightGBM, and XGBoost using 5-Fold Stratified Cross-Validation.\n",
            "6. **Best Model Evaluation**: Training the best model (XGBoost) with SMOTE to handle the severe class imbalance (~9% fraud rate), and evaluating on a holdout validation set.\n",
            "7. **Unseen Test Predictions**: Generating class predictions and probabilities for the unseen test providers.\n",
            "8. **Business Recommendations**: Translating modeling insights into actionable policies."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup paths and imports\n",
            "import os, sys, warnings\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "import joblib\n\n",
            "# Add notebooks directory to path for imports\n",
            "sys.path.insert(0, os.getcwd())\n",
            "warnings.filterwarnings('ignore')\n\n",
            "from notebooks.data_loader import load_all, clean_beneficiary, clean_claims\n",
            "from notebooks.feature_engineering import build_provider_features\n",
            "from notebooks.modelling import select_features, compare_models, train_best_model, evaluate, make_submission\n\n",
            "print(\"Imports successful. Working directory:\", os.getcwd())"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 1: Data Management & Cleaning\n",
            "We load the raw data files (Inpatient, Outpatient, and Beneficiary details for both Train and Unseen/Test datasets) and perform date parsing, age calculations, and chronic condition re-coding."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Load raw data\n",
            "print(\"--- Loading Raw Data ---\")\n",
            "(\n",
            "    train_labels, test_labels,\n",
            "    train_bene,   test_bene,\n",
            "    train_inpat,  test_inpat,\n",
            "    train_outpat, test_outpat\n",
            ") = load_all()\n\n",
            "# 2. Apply cleaning operations\n",
            "print(\"\\n--- Cleaning Beneficiary & Claims Data ---\")\n",
            "train_bene_cleaned = clean_beneficiary(train_bene)\n",
            "test_bene_cleaned  = clean_beneficiary(test_bene)\n",
            "train_inpat_cleaned  = clean_claims(train_inpat, 'IP')\n",
            "test_inpat_cleaned   = clean_claims(test_inpat, 'IP')\n",
            "train_outpat_cleaned = clean_claims(train_outpat, 'OP')\n",
            "test_outpat_cleaned  = clean_claims(test_outpat, 'OP')\n\n",
            "print(\"\\nSample of Cleaned Beneficiary Data:\")\n",
            "display(train_bene_cleaned[['BeneID', 'DOB', 'Age', 'IsDead', 'NumChronicCond']].head(5))"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 2: Exploratory Data Analysis\n",
            "We inspect the fraud label distribution and load the visual charts generated during EDA to review fraud behavior patterns (inflated reimbursements, patient chronic condition profiles, and duplicate billing)."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display as ipydisplay\n",
            "import os\n\n",
            "print(\"--- Displaying EDA Overview Plots ---\")\n",
            "eda_path = os.path.join('reports', 'eda_overview.png')\n",
            "if os.path.exists(eda_path):\n",
            "    ipydisplay(Image(filename=eda_path))\n",
            "else:\n",
            "    print('EDA overview plot not found.')\n\n",
            "print(\"\\n--- Displaying Correlation Heatmap ---\")\n",
            "corr_path = os.path.join('reports', 'correlation_heatmap.png')\n",
            "if os.path.exists(corr_path):\n",
            "    ipydisplay(Image(filename=corr_path))\n",
            "else:\n",
            "    print('Correlation heatmap plot not found.')"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Feature Engineering & Aggregation\n",
            "Because our target is provider-level fraud, we aggregate the patient-level claims and beneficiary records to the Provider level."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Build features\n",
            "print(\"--- Aggregating Features to Provider Level ---\")\n",
            "print(\"Training set:\")\n",
            "train_feats = build_provider_features(train_inpat_cleaned, train_outpat_cleaned, train_bene_cleaned)\n",
            "print(\"\\nUnseen (Test) set:\")\n",
            "test_feats = build_provider_features(test_inpat_cleaned, test_outpat_cleaned, test_bene_cleaned)\n\n",
            "# 2. Merge labels to training features\n",
            "if train_labels['PotentialFraud'].dtype == object:\n",
            "    train_labels['FraudLabel'] = train_labels['PotentialFraud'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)\n",
            "else:\n",
            "    train_labels['FraudLabel'] = train_labels['PotentialFraud'].astype(int)\n\n",
            "train_df = train_feats.merge(train_labels[['Provider', 'FraudLabel']], on='Provider', how='inner')\n",
            "print(\"\\nTarget Label Distribution:\")\n",
            "print(train_df['FraudLabel'].value_counts())\n",
            "print(f\"Fraud Rate: {train_df['FraudLabel'].mean():.2%}\")"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 3: Feature Selection\n",
            "We use a Random Forest Classifier to score feature importance and filter out noisy or irrelevant variables."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "feature_cols = [c for c in train_df.columns if c not in ['Provider','FraudLabel']]\n",
            "X = train_df[feature_cols]\n",
            "y = train_df['FraudLabel']\n\n",
            "print(\"--- Running Feature Selection ---\")\n",
            "selected_features, importances = select_features(X, y)\n\n",
            "print(\"\\nTop 10 Most Important Features:\")\n",
            "display(importances.head(10))\n\n",
            "from IPython.display import Image, display as ipydisplay\n",
            "import os\n",
            "feat_img_path = os.path.join('reports', 'feature_importance.png')\n",
            "if os.path.exists(feat_img_path):\n",
            "    ipydisplay(Image(filename=feat_img_path))"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 4: Model Comparison via 5-Fold Stratified CV\n",
            "We run a 5-Fold Stratified Cross-Validation on the training set to compare Logistic Regression, Random Forest, LightGBM, and XGBoost."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import StratifiedKFold, cross_val_score\n\n",
            "X_selected = X[selected_features]\n",
            "print(\"--- Comparing Models (5-Fold Stratified CV ROC-AUC) ---\")\n",
            "cv_results = compare_models(X_selected, y)\n\n",
            "# Print a summary table\n",
            "summary_df = pd.DataFrame({\n",
            "    'Model': list(cv_results.keys()),\n",
            "    'Mean ROC-AUC': [scores.mean() for scores in cv_results.values()],\n",
            "    'Std ROC-AUC': [scores.std() for scores in cv_results.values()]\n",
            "})\n",
            "display(summary_df)\n\n",
            "from IPython.display import Image, display as ipydisplay\n",
            "import os\n",
            "mc_img_path = os.path.join('reports', 'model_comparison.png')\n",
            "if os.path.exists(mc_img_path):\n",
            "    ipydisplay(Image(filename=mc_img_path))"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 5: Best Model Training & Validation (XGBoost + SMOTE)\n",
            "We train our top-performing model, XGBoost, using SMOTE to handle the class imbalance, and evaluate it on a holdout validation set (20% split)."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from sklearn.model_selection import train_test_split\n\n",
            "# Train/Val Split\n",
            "X_train, X_val, y_train, y_val = train_test_split(\n",
            "    X_selected, y, test_size=0.2, random_state=42, stratify=y\n",
            ")\n\n",
            "print(f\"Train size: {X_train.shape} | Val size: {X_val.shape}\")\n\n",
            "# Train with SMOTE\n",
            "best_model = train_best_model(X_train.copy(), y_train.copy(), use_smote=True)\n\n",
            "# Evaluate model\n",
            "print(\"\\n--- Evaluation on Validation Set ---\")\n",
            "auc, val_probs = evaluate(best_model, X_val, y_val, label='XGBoost')\n\n",
            "from IPython.display import Image, display as ipydisplay\n",
            "import os\n",
            "eval_img_path = os.path.join('reports', 'XGBoost_evaluation.png')\n",
            "if os.path.exists(eval_img_path):\n",
            "    ipydisplay(Image(filename=eval_img_path))"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 6: Prediction on Unseen Test Data & Submission File Generation\n",
            "Finally, we align the test features with our selected features and generate predictions."
        ]
    },
    # Code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"--- Generating Predictions for Unseen Test Data ---\")\n",
            "# Align test features\n",
            "test_X_merged = test_labels.copy().merge(\n",
            "    test_feats[['Provider'] + [c for c in selected_features if c in test_feats.columns]],\n",
            "    on='Provider', how='left'\n",
            ")\n",
            "for col in selected_features:\n",
            "    if col not in test_X_merged.columns:\n",
            "        test_X_merged[col] = 0\n",
            "test_X_final = test_X_merged[selected_features].fillna(0)\n\n",
            "# Generate predictions\n",
            "probs_test = best_model.predict_proba(test_X_final)[:, 1]\n",
            "preds_test = (probs_test >= 0.5).astype(int)\n\n",
            "submission = pd.DataFrame({\n",
            "    'Provider'                               : test_labels['Provider'].values,\n",
            "    'Probability'                            : probs_test.round(4),\n",
            "    'Predicted Class as per your best model' : pd.Series(preds_test).map({0: 'No', 1: 'Yes'}).values\n",
            "})\n\n",
            "# Save to CSV\n",
            "out_path = os.path.join('outputs', 'Supriyo_Submission.csv')\n",
            "submission.to_csv(out_path, index=False)\n",
            "submission.to_csv(os.path.join('outputs', 'Your_Full_Name_Submission.csv'), index=False)\n",
            "print(f\"Submission saved to: {out_path}\")\n",
            "print(f\"Fraud predicted: {(submission['Predicted Class as per your best model']=='Yes').sum()} / {len(submission)}\")\n",
            "display(submission.head(10))"
        ]
    },
    # Markdown
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Step 7: Business Recommendations & Actionable Insights\n",
            "Based on the feature importances and fraud patterns:\n",
            "1. **Reimbursement Benchmarking**: Outpatient and inpatient claim totals are the most significant predictors. Focus audits on providers with reimbursement rates >200% of peer group medians.\n",
            "2. **Claims-per-Patient Ratio**: Trigger reviews when a provider bills multiple outpatient claims for the same patient in rapid succession (suspected unbundling).\n",
            "3. **Rotating Networks**: Flag providers coordinating with unusually large or highly variable circles of attending/operating physicians.\n",
            "4. **Deceased Beneficiary Claims**: Implement automatic checks against SSA records before claims payout."
        ]
    }
]

# Write ipynb file
notebook_path = 'notebooks/Healthcare_Provider_Fraud_Detection.ipynb'
notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook_data, f, indent=1)

print(f"Jupyter Notebook template written to {notebook_path}")
