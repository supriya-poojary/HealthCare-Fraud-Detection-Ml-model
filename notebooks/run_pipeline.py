"""
=============================================================================
  Healthcare Provider Fraud Detection
  MAIN PIPELINE  — runs all 7 steps end-to-end
  Usage: python notebooks/run_pipeline.py
=============================================================================
"""

import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ── Add notebooks dir to path ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader        import load_all, clean_beneficiary, clean_claims
from feature_engineering import build_provider_features
from modelling          import (select_features, compare_models,
                                 train_best_model, evaluate,
                                 plot_feature_importance, make_submission,
                                 MODELS, OUTPUTS, REPORTS)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC    = os.path.join(BASE, 'data', 'processed')

print("\n" + "="*60)
print("  HEALTHCARE PROVIDER FRAUD DETECTION  * Full Pipeline")
print("="*60)

# ══════════════════════════════════════════════════════════════
# STEP 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Data Loading *")
(train_labels, test_labels,
 train_bene,   test_bene,
 train_inpat,  test_inpat,
 train_outpat, test_outpat) = load_all()

# ══════════════════════════════════════════════════════════════
# STEP 2 — CLEANING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2] Cleaning *")
train_bene   = clean_beneficiary(train_bene)
test_bene    = clean_beneficiary(test_bene)
train_inpat  = clean_claims(train_inpat,  'IP')
test_inpat   = clean_claims(test_inpat,   'IP')
train_outpat = clean_claims(train_outpat, 'OP')
test_outpat  = clean_claims(test_outpat,  'OP')
print("  * Cleaning done.")

# ══════════════════════════════════════════════════════════════
# STEP 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════
print("\n[STEP 3] Feature Engineering *")
print("  * Train features")
train_feats = build_provider_features(train_inpat, train_outpat, train_bene)
print("  * Test features")
test_feats  = build_provider_features(test_inpat,  test_outpat,  test_bene)

# Encode target
le = LabelEncoder()
train_labels['FraudLabel'] = le.fit_transform(
    train_labels['PotentialFraud'].map({'Yes': 1, 'No': 0}
    ).fillna(train_labels['PotentialFraud']))

# Handle edge case where PotentialFraud might already be 0/1
if train_labels['PotentialFraud'].dtype == object:
    train_labels['FraudLabel'] = train_labels['PotentialFraud'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
else:
    train_labels['FraudLabel'] = train_labels['PotentialFraud'].astype(int)

# Merge labels
train_df = train_feats.merge(
    train_labels[['Provider', 'FraudLabel']], on='Provider', how='inner')

print(f"\n  Fraud distribution:\n{train_df['FraudLabel'].value_counts()}")
print(f"  Fraud rate: {train_df['FraudLabel'].mean():.2%}")

# Save processed
train_df.to_csv(os.path.join(PROC, 'train_features.csv'), index=False)
test_feats.to_csv(os.path.join(PROC, 'test_features.csv'), index=False)
print("  * Features saved to data/processed/")

# ══════════════════════════════════════════════════════════════
# STEP 2b — EDA PLOTS
# ══════════════════════════════════════════════════════════════
print("\n[STEP 2b] Generating EDA Plots *")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Healthcare Provider Fraud — EDA', fontsize=16, fontweight='bold')

# 1. Class Distribution
fraud_counts = train_df['FraudLabel'].value_counts()
axes[0,0].bar(['Non-Fraud', 'Fraud'], fraud_counts.values, color=['#2ecc71','#e74c3c'])
axes[0,0].set_title('Class Distribution'); axes[0,0].set_ylabel('Count')
for i, v in enumerate(fraud_counts.values):
    axes[0,0].text(i, v + 10, str(v), ha='center', fontweight='bold')

# 2. Total Claims by Fraud
axes[0,1].boxplot([train_df[train_df.FraudLabel==0]['Total_Claims'].dropna(),
                   train_df[train_df.FraudLabel==1]['Total_Claims'].dropna()],
                  labels=['Non-Fraud','Fraud'])
axes[0,1].set_title('Total Claims per Provider'); axes[0,1].set_ylabel('Claims Count')

# 3. Avg Reimbursed
axes[0,2].boxplot([train_df[train_df.FraudLabel==0]['IP_AvgReimbursed'].dropna(),
                   train_df[train_df.FraudLabel==1]['IP_AvgReimbursed'].dropna()],
                  labels=['Non-Fraud','Fraud'])
axes[0,2].set_title('Avg IP Reimbursement'); axes[0,2].set_ylabel('Amount ($)')

# 4. Chronic Conditions
if 'Bene_AvgChronicCond' in train_df.columns:
    axes[1,0].boxplot([train_df[train_df.FraudLabel==0]['Bene_AvgChronicCond'].dropna(),
                       train_df[train_df.FraudLabel==1]['Bene_AvgChronicCond'].dropna()],
                      labels=['Non-Fraud','Fraud'])
    axes[1,0].set_title('Avg Chronic Conditions per Beneficiary')

# 5. Unique Physicians
if 'IP_Unique_AttendingPhysician' in train_df.columns:
    axes[1,1].boxplot([train_df[train_df.FraudLabel==0]['IP_Unique_AttendingPhysician'].dropna(),
                       train_df[train_df.FraudLabel==1]['IP_Unique_AttendingPhysician'].dropna()],
                      labels=['Non-Fraud','Fraud'])
    axes[1,1].set_title('Unique Attending Physicians')

# 6. Hospital Stay
if 'IP_AvgHospitalStay' in train_df.columns:
    axes[1,2].boxplot([train_df[train_df.FraudLabel==0]['IP_AvgHospitalStay'].dropna(),
                       train_df[train_df.FraudLabel==1]['IP_AvgHospitalStay'].dropna()],
                      labels=['Non-Fraud','Fraud'])
    axes[1,2].set_title('Avg Hospital Stay (Days)')

plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'eda_overview.png'), dpi=150); plt.close()
print("  EDA plots saved * reports/eda_overview.png")

# Correlation heatmap (top features)
feat_cols = [c for c in train_df.columns if c not in ['Provider','FraudLabel']]
corr_data = train_df[feat_cols + ['FraudLabel']].corr()
top_corr_features = corr_data['FraudLabel'].abs().nlargest(21).index.tolist()
fig2, ax2 = plt.subplots(figsize=(14, 12))
sns.heatmap(train_df[top_corr_features].corr(), annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax2)
ax2.set_title('Correlation Heatmap — Top 20 Features vs FraudLabel', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'correlation_heatmap.png'), dpi=150); plt.close()
print("  Correlation heatmap saved * reports/correlation_heatmap.png")

# ══════════════════════════════════════════════════════════════
# STEP 4 — FEATURE SELECTION
# ══════════════════════════════════════════════════════════════
print("\n[STEP 4] Feature Selection *")
feature_cols = [c for c in train_df.columns if c not in ['Provider','FraudLabel']]
X = train_df[feature_cols]
y = train_df['FraudLabel']

selected_features, importances = select_features(X, y)
plot_feature_importance(importances)

X_selected = X[selected_features]

# Train/Val split
X_train, X_val, y_train, y_val = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y)
print(f"  Train: {X_train.shape} | Val: {X_val.shape}")

# Save feature list
joblib.dump(selected_features, os.path.join(MODELS, 'selected_features.pkl'))

# ══════════════════════════════════════════════════════════════
# STEP 5 — MODEL COMPARISON (5-Fold CV)
# ══════════════════════════════════════════════════════════════
print("\n[STEP 5] Model Comparison (5-Fold CV) *")
cv_results = compare_models(X_selected, y)

# Plot CV results
fig3, ax3 = plt.subplots(figsize=(10, 5))
model_names = list(cv_results.keys())
means = [v.mean() for v in cv_results.values()]
stds  = [v.std()  for v in cv_results.values()]
colors_bar = ['#3498db','#2ecc71','#e74c3c','#f39c12']
bars = ax3.bar(model_names, means, yerr=stds, capsize=6,
               color=colors_bar, alpha=0.85, edgecolor='black')
ax3.set_ylim(0, 1.05); ax3.set_ylabel('ROC-AUC')
ax3.set_title('Model Comparison — 5-Fold Stratified CV', fontsize=13, fontweight='bold')
for bar, mean in zip(bars, means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{mean:.3f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(REPORTS, 'model_comparison.png'), dpi=150); plt.close()
print("  Model comparison plot saved * reports/model_comparison.png")

# ══════════════════════════════════════════════════════════════
# STEP 6 — TRAIN BEST MODEL & EVALUATE
# ══════════════════════════════════════════════════════════════
print("\n[STEP 6] Training Best Model (XGBoost + SMOTE) *")
best_model = train_best_model(X_train.copy(), y_train.copy(), use_smote=True)

auc, val_probs = evaluate(best_model, X_val, y_val, label='XGBoost')

# Save model
joblib.dump(best_model, os.path.join(MODELS, 'best_model.pkl'))
print(f"  Model saved * models/best_model.pkl")

# ══════════════════════════════════════════════════════════════
# STEP 7 — SUBMISSION
# ══════════════════════════════════════════════════════════════
print("\n[STEP 7] Generating Submission File *")

# Align test features to selected feature columns (fill missing with 0)
test_X = test_feats.copy()
for col in selected_features:
    if col not in test_X.columns:
        test_X[col] = 0
test_X_sel = test_X[selected_features].fillna(0)

# Only predict for providers in test_labels
test_providers_in = test_labels.copy()
test_X_merged = test_providers_in.merge(
    test_feats[['Provider'] + [c for c in selected_features if c in test_feats.columns]],
    on='Provider', how='left')
for col in selected_features:
    if col not in test_X_merged.columns:
        test_X_merged[col] = 0
test_X_final = test_X_merged[selected_features].fillna(0)

probs_test = best_model.predict_proba(test_X_final)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

submission = pd.DataFrame({
    'Provider'                               : test_providers_in['Provider'].values,
    'Probability'                            : probs_test.round(4),
    'Predicted Class as per your best model' : pd.Series(preds_test).map({0: 'No', 1: 'Yes'}).values,
})
sub_path = os.path.join(OUTPUTS, 'Supriyo_Submission.csv')
submission.to_csv(sub_path, index=False)
# Also save generic template file
submission.to_csv(os.path.join(OUTPUTS, 'Your_Full_Name_Submission.csv'), index=False)
print(f"  * Submission saved * {sub_path}")
print(f"  Predicted Fraud: {(submission['Predicted Class as per your best model']=='Yes').sum()} / {len(submission)}")
print(submission.head(10))

# Save everything needed by Streamlit
joblib.dump({
    'model'            : best_model,
    'selected_features': selected_features,
    'importances'      : importances,
    'cv_results'       : cv_results,
    'val_auc'          : auc,
    'fraud_rate'       : float(y.mean()),
    'n_providers'      : int(len(train_df)),
}, os.path.join(MODELS, 'pipeline_artifacts.pkl'))

print("\n" + "="*60)
print("  *  PIPELINE COMPLETE!")
print(f"  Validation AUC : {auc:.4f}")
print(f"  Submission     : {sub_path}")
print(f"  Model          : models/best_model.pkl")
print("="*60)
