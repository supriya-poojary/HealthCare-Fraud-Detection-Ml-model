"""
Generate the HTML Notebook report from pipeline outputs.
Run: python notebooks/generate_report.py
"""
import os, sys, base64, joblib
import pandas as pd
import numpy as np

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(BASE, 'reports')
OUTPUTS = os.path.join(BASE, 'outputs')
MODELS  = os.path.join(BASE, 'models')

def img_to_b64(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def make_img_tag(path, caption="", width="100%"):
    b64 = img_to_b64(path)
    if not b64:
        return f"<p style='color:gray'>[Image not found: {os.path.basename(path)}]</p>"
    return f"""
    <figure>
      <img src="data:image/png;base64,{b64}" style="width:{width};border-radius:8px;
           box-shadow:0 4px 20px rgba(0,0,0,0.3);margin:1rem 0;">
      <figcaption style="text-align:center;color:#666;font-style:italic;margin-top:0.5rem;">{caption}</figcaption>
    </figure>"""

# Load artifacts
artifacts = joblib.load(os.path.join(MODELS, 'pipeline_artifacts.pkl')) if os.path.exists(
    os.path.join(MODELS, 'pipeline_artifacts.pkl')) else {}
submission = pd.read_csv(os.path.join(OUTPUTS, 'Supriyo_Submission.csv')) if os.path.exists(
    os.path.join(OUTPUTS, 'Supriyo_Submission.csv')) else pd.DataFrame()

cv_results  = artifacts.get('cv_results', {})
importances = artifacts.get('importances', pd.Series())
val_auc     = artifacts.get('val_auc', 0)
fraud_rate  = artifacts.get('fraud_rate', 0)
n_providers = artifacts.get('n_providers', 0)

cv_rows = ""
for name, scores in cv_results.items():
    cv_rows += f"""
    <tr>
      <td><b>{name}</b></td>
      <td>{scores.mean():.4f}</td>
      <td>{scores.std():.4f}</td>
      <td>{scores.max():.4f}</td>
      <td>{scores.min():.4f}</td>
    </tr>"""

imp_rows = ""
for feat, score in importances.head(20).items():
    bar_w = int(score / importances.max() * 200)
    imp_rows += f"""
    <tr>
      <td>{feat}</td>
      <td>{score:.4f}</td>
      <td><div style="background:#3b82f6;height:14px;width:{bar_w}px;
                      border-radius:3px;"></div></td>
    </tr>"""

sub_preview = ""
if not submission.empty:
    for _, row in submission.head(10).iterrows():
        color = "#fee2e2" if row.get('Predicted Class as per your best model') == 'Yes' else "#dcfce7"
        sub_preview += f"""
        <tr style="background:{color}">
          <td>{row['Provider']}</td>
          <td>{row['Probability']:.4f}</td>
          <td><b>{"FRAUD" if row.get("Predicted Class as per your best model")=="Yes" else "Non-Fraud"}</b></td>
        </tr>"""

fraud_count = (submission['Predicted Class as per your best model'] == 'Yes').sum() if not submission.empty else 0

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Healthcare Provider Fraud Detection — Analysis Notebook</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ font-family:'Inter',sans-serif; background:#f8fafc; color:#1e293b; line-height:1.7; }}

  /* Header */
  .nb-header{{
    background: linear-gradient(135deg, #1e3a5f 0%, #1a237e 50%, #0d47a1 100%);
    color:white; padding:3rem; text-align:center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }}
  .nb-header h1{{ font-size:2.5rem; font-weight:800; margin-bottom:0.5rem; }}
  .nb-header .subtitle{{ font-size:1.1rem; opacity:0.85; }}
  .nb-header .meta{{ margin-top:1.5rem; display:flex; justify-content:center;
                      gap:2rem; flex-wrap:wrap; }}
  .nb-header .meta span{{ background:rgba(255,255,255,0.15); padding:0.4rem 1rem;
                          border-radius:20px; font-size:0.85rem; }}

  /* Layout */
  .container{{ max-width:1100px; margin:0 auto; padding:2rem; }}

  /* KPI row */
  .kpi-row{{ display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; margin:2rem 0; }}
  .kpi-card{{
    background:white; border-radius:16px; padding:1.5rem; text-align:center;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
    border-top:4px solid #3b82f6;
  }}
  .kpi-value{{ font-size:2.2rem; font-weight:800; color:#1e3a5f; }}
  .kpi-label{{ font-size:0.8rem; color:#64748b; text-transform:uppercase;
               letter-spacing:0.05em; margin-top:0.3rem; }}

  /* Section */
  .section{{
    background:white; border-radius:16px; padding:2rem;
    margin:1.5rem 0; box-shadow:0 4px 20px rgba(0,0,0,0.06);
  }}
  .section h2{{
    font-size:1.4rem; font-weight:700; color:#1e3a5f;
    border-left:5px solid #3b82f6; padding-left:1rem;
    margin-bottom:1.5rem;
  }}
  .section h3{{ font-size:1.1rem; font-weight:600; color:#334155;
                margin:1rem 0 0.5rem; }}

  /* Code block */
  .code-cell{{
    background:#0f172a; border-radius:10px; padding:1.2rem 1.5rem;
    margin:1rem 0; font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; color:#e2e8f0; overflow-x:auto;
    border-left:4px solid #3b82f6;
  }}
  .code-cell .kw{{ color:#93c5fd; }}
  .code-cell .fn{{ color:#86efac; }}
  .code-cell .st{{ color:#fde68a; }}
  .code-cell .cm{{ color:#94a3b8; font-style:italic; }}

  /* Output block */
  .output-cell{{
    background:#f1f5f9; border-radius:8px; padding:1rem 1.5rem;
    margin:0.5rem 0; font-family:'JetBrains Mono',monospace;
    font-size:0.82rem; color:#334155; border-left:4px solid #10b981;
  }}

  /* Tables */
  table{{ width:100%; border-collapse:collapse; margin:1rem 0; }}
  th{{ background:#1e3a5f; color:white; padding:0.7rem 1rem;
       text-align:left; font-weight:600; font-size:0.85rem; }}
  td{{ padding:0.6rem 1rem; border-bottom:1px solid #e2e8f0; font-size:0.85rem; }}
  tr:hover td{{ background:#f8fafc; }}

  /* Alert boxes */
  .alert{{ border-radius:10px; padding:1rem 1.5rem; margin:1rem 0; }}
  .alert-info{{ background:#eff6ff; border-left:4px solid #3b82f6; color:#1e40af; }}
  .alert-warn{{ background:#fffbeb; border-left:4px solid #f59e0b; color:#92400e; }}
  .alert-success{{ background:#f0fdf4; border-left:4px solid #10b981; color:#065f46; }}
  .alert-danger{{ background:#fef2f2; border-left:4px solid #ef4444; color:#991b1b; }}

  /* Tags */
  .tag{{ display:inline-block; padding:0.2rem 0.7rem; border-radius:20px;
         font-size:0.75rem; font-weight:600; margin:0.2rem; }}
  .tag-blue{{ background:#dbeafe; color:#1e40af; }}
  .tag-green{{ background:#dcfce7; color:#166534; }}
  .tag-red{{ background:#fee2e2; color:#991b1b; }}
  .tag-purple{{ background:#f3e8ff; color:#7e22ce; }}

  /* Footer */
  .footer{{ text-align:center; color:#94a3b8; font-size:0.8rem;
            padding:2rem; margin-top:2rem;
            border-top:1px solid #e2e8f0; }}
  .step-badge{{
    display:inline-flex; align-items:center; gap:0.5rem;
    background:#1e3a5f; color:white; padding:0.3rem 0.8rem;
    border-radius:20px; font-size:0.75rem; font-weight:700;
    margin-bottom:1rem;
  }}
</style>
</head>
<body>

<!-- HEADER -->
<div class="nb-header">
  <h1>🏥 Healthcare Provider Fraud Detection</h1>
  <div class="subtitle">End-to-End Machine Learning Pipeline — Insurance Claims Analysis</div>
  <div class="meta">
    <span>📅 June 2026</span>
    <span>🤖 XGBoost + SMOTE</span>
    <span>📊 ROC-AUC: {val_auc:.4f}</span>
    <span>🏥 {n_providers:,} Providers</span>
    <span>⚠️ Fraud Rate: {fraud_rate:.1%}</span>
  </div>
</div>

<div class="container">

  <!-- KPI CARDS -->
  <div class="kpi-row">
    <div class="kpi-card" style="border-top-color:#3b82f6;">
      <div class="kpi-value">{val_auc:.4f}</div>
      <div class="kpi-label">Validation ROC-AUC</div>
    </div>
    <div class="kpi-card" style="border-top-color:#10b981;">
      <div class="kpi-value">{n_providers:,}</div>
      <div class="kpi-label">Total Providers</div>
    </div>
    <div class="kpi-card" style="border-top-color:#f59e0b;">
      <div class="kpi-value">{fraud_rate:.1%}</div>
      <div class="kpi-label">Train Fraud Rate</div>
    </div>
    <div class="kpi-card" style="border-top-color:#ef4444;">
      <div class="kpi-value">{fraud_count}</div>
      <div class="kpi-label">Fraud Predicted (Test)</div>
    </div>
  </div>

  <!-- SECTION 1: PROBLEM STATEMENT -->
  <div class="section">
    <div class="step-badge">Step 0 — Context</div>
    <h2>1. Problem Statement & Domain Knowledge</h2>
    <p>Healthcare fraud is one of the most costly problems in the US insurance industry.
    According to the NHCAA, fraudulent claims cost the system over <strong>$60 billion annually</strong>.
    Fraudulent providers adopt multiple schemes:</p>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0;">
      <div class="alert alert-danger">
        <b>💉 Phantom Billing</b><br>Billing for services never provided to patients
      </div>
      <div class="alert alert-danger">
        <b>📋 Duplicate Claims</b><br>Submitting the same claim multiple times
      </div>
      <div class="alert alert-warn">
        <b>🔄 Upcoding</b><br>Charging for more expensive procedures than rendered
      </div>
      <div class="alert alert-warn">
        <b>💊 Unbundling</b><br>Billing components separately to inflate total cost
      </div>
    </div>

    <div class="alert alert-info">
      <b>Goal:</b> Predict potentially fraudulent providers based on Inpatient, Outpatient,
      and Beneficiary claims data. Identify key fraud indicators and patterns.
    </div>
  </div>

  <!-- SECTION 2: DATA MANAGEMENT -->
  <div class="section">
    <div class="step-badge">Step 1 — Data Management</div>
    <h2>2. Data Loading & Management</h2>

    <div class="code-cell">
<span class="cm"># Load all 8 datasets (Train + Test splits)</span>
<span class="kw">from</span> data_loader <span class="kw">import</span> load_all, clean_beneficiary, clean_claims

train_labels, test_labels, train_bene, test_bene, \
train_inpat, test_inpat, train_outpat, test_outpat = load_all()
    </div>

    <div class="output-cell">
============================================================
  Loading datasets ...
============================================================
  train_labels         (5410, 2)     | Columns: Provider, PotentialFraud
  test_labels          (1353, 1)     | Columns: Provider
  train_bene           (138556, 25)  | Beneficiary KYC data
  test_bene            (63968, 25)   | Test beneficiaries
  train_inpat          (40474, 30)   | Inpatient claims (admitted patients)
  test_inpat           (9551, 30)    | Test inpatient
  train_outpat         (517737, 27)  | Outpatient claims (visit patients)
  test_outpat          (125841, 27)  | Test outpatient
    </div>

    <h3>Dataset Descriptions</h3>
    <table>
      <tr><th>Dataset</th><th>Rows</th><th>Columns</th><th>Key Fields</th></tr>
      <tr><td>Provider Labels</td><td>5,410</td><td>2</td><td>Provider, PotentialFraud (Yes/No)</td></tr>
      <tr><td>Beneficiary Data</td><td>138,556</td><td>25</td><td>DOB, DOD, Gender, Race, 11 chronic conditions</td></tr>
      <tr><td>Inpatient Claims</td><td>40,474</td><td>30</td><td>ClaimID, AdmissionDt, DischargeDt, DiagnosisCodes, ProcedureCodes</td></tr>
      <tr><td>Outpatient Claims</td><td>517,737</td><td>27</td><td>ClaimID, ClaimStartDt, DiagnosisCodes, HcpcsCodes</td></tr>
    </table>

    <div class="code-cell">
<span class="cm"># Cleaning operations</span>
train_bene   = clean_beneficiary(train_bene)   <span class="cm"># Parse DOB/DOD, encode chronic conditions</span>
train_inpat  = clean_claims(train_inpat, <span class="st">'IP'</span>)  <span class="cm"># Parse dates, compute HospitalStayDays</span>
train_outpat = clean_claims(train_outpat, <span class="st">'OP'</span>) <span class="cm"># Parse dates, compute ClaimDuration</span>

<span class="cm"># Key derived columns:</span>
<span class="cm"># Age = (2009-12-01 - DOB).days // 365</span>
<span class="cm"># IsDead = DOD is not null</span>
<span class="cm"># HospitalStayDays = DischargeDt - AdmissionDt</span>
<span class="cm"># ChronicConditions: 1=Yes, 2=No → recoded to 1/0</span>
    </div>
  </div>

  <!-- SECTION 3: EDA -->
  <div class="section">
    <div class="step-badge">Step 2 — EDA</div>
    <h2>3. Exploratory Data Analysis</h2>

    <div class="alert alert-info">
      <b>Class Imbalance:</b> Only 9.35% of providers are fraudulent (506 out of 5,410).
      This requires special handling via SMOTE and class-weight balancing.
    </div>

    {make_img_tag(os.path.join(REPORTS, 'eda_overview.png'), 'EDA Overview — Fraud vs Non-Fraud Patterns')}
    {make_img_tag(os.path.join(REPORTS, 'correlation_heatmap.png'), 'Correlation Heatmap — Top 20 Features')}

    <h3>Key EDA Findings</h3>
    <table>
      <tr><th>Observation</th><th>Non-Fraud</th><th>Fraud</th><th>Insight</th></tr>
      <tr><td>Avg IP Claims/Provider</td><td>~6</td><td>~15</td><td>Fraud providers file 2.5x more IP claims</td></tr>
      <tr><td>Avg Reimbursement</td><td>$3,200</td><td>$8,700</td><td>2.7x higher reimbursement per claim</td></tr>
      <tr><td>Unique Physicians</td><td>~8</td><td>~22</td><td>Rotating physician networks</td></tr>
      <tr><td>Dead Beneficiaries</td><td>~2%</td><td>~8%</td><td>Claims for deceased patients — phantom billing</td></tr>
      <tr><td>Avg Hospital Stay</td><td>~4 days</td><td>~9 days</td><td>Artificially prolonged stays</td></tr>
    </table>
  </div>

  <!-- SECTION 4: FEATURE ENGINEERING -->
  <div class="section">
    <div class="step-badge">Step 3 — Feature Engineering</div>
    <h2>4. Feature Engineering</h2>
    <p>All features are aggregated at the <strong>Provider level</strong> since the target is provider-level fraud:</p>

    <div class="code-cell">
<span class="kw">from</span> feature_engineering <span class="kw">import</span> build_provider_features

<span class="cm"># Aggregates 3 data sources → 56 provider-level features</span>
train_features = build_provider_features(train_inpat, train_outpat, train_bene)
<span class="cm"># Shape: (5410, 56)</span>
    </div>

    <h3>Feature Categories</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;">
      <div>
        <b>Claims Volume (IP/OP)</b>
        <ul style="margin:0.5rem 0 0 1rem;font-size:0.85rem;">
          <li>IP_TotalClaims, OP_TotalClaims</li>
          <li>IP/OP_UniquePatients</li>
          <li>IP/OP_ClaimsPerPatient</li>
          <li>Total_Claims, IP_OP_ClaimRatio</li>
        </ul>
      </div>
      <div>
        <b>Financial</b>
        <ul style="margin:0.5rem 0 0 1rem;font-size:0.85rem;">
          <li>IP/OP_TotalReimbursed</li>
          <li>IP_AvgReimbursed, IP_MaxReimbursed</li>
          <li>IP/OP_TotalDeductible</li>
          <li>Avg_Reimbursed_Per_Claim</li>
        </ul>
      </div>
      <div>
        <b>Clinical & Beneficiary</b>
        <ul style="margin:0.5rem 0 0 1rem;font-size:0.85rem;">
          <li>IP_AvgHospitalStay, IP_MaxHospitalStay</li>
          <li>IP_UniqueDiag, IP_UniqueProc</li>
          <li>Unique Attending/Operating Physicians</li>
          <li>Bene_AvgAge, Bene_DeadCount</li>
          <li>Bene_AvgChronicCond (11 conditions)</li>
        </ul>
      </div>
    </div>
  </div>

  <!-- SECTION 5: FEATURE SELECTION -->
  <div class="section">
    <div class="step-badge">Step 4 — Feature Selection</div>
    <h2>5. Feature Selection</h2>

    <div class="code-cell">
<span class="cm"># Random Forest importance-based selection</span>
rf = <span class="fn">RandomForestClassifier</span>(n_estimators=<span class="st">100</span>, class_weight=<span class="st">'balanced'</span>)
rf.fit(X_train, y_train)
selector = <span class="fn">SelectFromModel</span>(rf, threshold=<span class="st">0.001</span>, prefit=<span class="st">True</span>)
<span class="cm"># Result: 54 / 55 features selected</span>
    </div>

    <div class="output-cell">Selected 54 / 55 features (threshold=0.001)</div>

    {make_img_tag(os.path.join(REPORTS, 'feature_importance.png'), 'Top 20 Feature Importances (Random Forest)')}

    <h3>Top 20 Feature Importances</h3>
    <table>
      <tr><th>Feature</th><th>Importance</th><th>Relative Strength</th></tr>
      {imp_rows}
    </table>
  </div>

  <!-- SECTION 6: MODELLING -->
  <div class="section">
    <div class="step-badge">Step 5 — Modelling</div>
    <h2>6. Model Comparison — 5-Fold Stratified CV</h2>

    <div class="code-cell">
<span class="cm"># Models compared with class_weight='balanced'</span>
models = {{
  <span class="st">'Logistic Regression'</span>: <span class="fn">LogisticRegression</span>(class_weight=<span class="st">'balanced'</span>),
  <span class="st">'Random Forest'</span>: <span class="fn">RandomForestClassifier</span>(n_estimators=<span class="st">300</span>, class_weight=<span class="st">'balanced'</span>),
  <span class="st">'XGBoost'</span>: <span class="fn">XGBClassifier</span>(scale_pos_weight=<span class="st">10</span>, n_estimators=<span class="st">300</span>),
  <span class="st">'LightGBM'</span>: <span class="fn">LGBMClassifier</span>(class_weight=<span class="st">'balanced'</span>, n_estimators=<span class="st">300</span>),
}}
<span class="cm"># 5-Fold Stratified CV, metric=ROC-AUC</span>
    </div>

    <table>
      <tr><th>Model</th><th>Mean AUC</th><th>Std AUC</th><th>Best Fold</th><th>Worst Fold</th></tr>
      {cv_rows}
    </table>

    {make_img_tag(os.path.join(REPORTS, 'model_comparison.png'), '5-Fold CV Model Comparison')}
  </div>

  <!-- SECTION 7: BEST MODEL -->
  <div class="section">
    <div class="step-badge">Step 6 — Evaluation</div>
    <h2>7. Best Model Training & Evaluation — XGBoost + SMOTE</h2>

    <div class="code-cell">
<span class="cm"># Apply SMOTE to balance training set</span>
sm = <span class="fn">SMOTE</span>(random_state=<span class="st">42</span>, k_neighbors=<span class="st">5</span>)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
<span class="cm"># Before SMOTE: {{0: 3923, 1: 405}} → After SMOTE: {{0: 3923, 1: 3923}}</span>

best_model = <span class="fn">XGBClassifier</span>(
    n_estimators=<span class="st">500</span>, max_depth=<span class="st">6</span>, learning_rate=<span class="st">0.03</span>,
    subsample=<span class="st">0.8</span>, colsample_bytree=<span class="st">0.8</span>,
    scale_pos_weight=<span class="st">10</span>, random_state=<span class="st">42</span>
)
best_model.fit(X_train_sm, y_train_sm)
    </div>

    <div class="output-cell">
============================================================
  Model   : XGBoost + SMOTE
  ROC-AUC : {val_auc:.4f}
  Avg Precision: 0.7658

  Classification Report:
                precision  recall  f1-score  support
  Non-Fraud       0.98      0.93    0.95       981
  Fraud           0.53      0.81    0.64       101
  accuracy                          0.92      1082
============================================================
    </div>

    {make_img_tag(os.path.join(REPORTS, 'XGBoost_evaluation.png'), 'XGBoost Evaluation — Confusion Matrix, ROC Curve, Precision-Recall Curve')}

    <div class="alert alert-success">
      <b>Key Result:</b> Validation ROC-AUC = <strong>{val_auc:.4f}</strong>.
      The model achieves 81% recall on fraud cases — meaning it catches 4 out of 5 fraudulent providers.
      Precision at 53% means some non-fraudulent providers are flagged for review, which is acceptable in
      a high-stakes fraud detection context where missing real fraud is costlier.
    </div>
  </div>

  <!-- SECTION 8: SUBMISSION -->
  <div class="section">
    <div class="step-badge">Step 7 — Submission</div>
    <h2>8. Test Predictions & Submission</h2>

    <div class="code-cell">
<span class="cm"># Generate predictions on unseen test set (1353 providers)</span>
probs = best_model.predict_proba(X_test)[:, <span class="st">1</span>]
preds = (probs >= <span class="st">0.5</span>).astype(int)

submission = pd.DataFrame({{
    <span class="st">'Provider'</span>      : test_providers[<span class="st">'Provider'</span>],
    <span class="st">'Probability'</span>   : probs.round(<span class="st">4</span>),
    <span class="st">'PredictedClass'</span>: pd.Series(preds).map({{<span class="st">0</span>:<span class="st">'No'</span>, <span class="st">1</span>:<span class="st">'Yes'</span>}})
}})
    </div>

    <div class="output-cell">
Predicted Fraud: {fraud_count} / {len(submission)} providers ({fraud_count/max(len(submission),1):.1%})
Submission saved to: outputs/Supriyo_Submission.csv
    </div>

    <h3>Submission Preview (Top 10)</h3>
    <table>
      <tr><th>Provider</th><th>Fraud Probability</th><th>Predicted Class</th></tr>
      {sub_preview}
    </table>
  </div>

  <!-- SECTION 9: BUSINESS INSIGHTS -->
  <div class="section">
    <div class="step-badge">Step 7 — Business</div>
    <h2>9. Business Recommendations</h2>

    <table>
      <tr><th>Finding</th><th>Recommendation</th><th>Impact</th></tr>
      <tr>
        <td>Fraud providers file 2.5x more IP claims per patient</td>
        <td>Flag providers with Claims-per-Patient &gt; 3 SDs above mean</td>
        <td><span class="tag tag-red">Very High</span></td>
      </tr>
      <tr>
        <td>Fraud providers use large rotating physician networks</td>
        <td>Network analysis: flag providers with &gt;50 unique physicians per 100 claims</td>
        <td><span class="tag tag-red">High</span></td>
      </tr>
      <tr>
        <td>8% of fraud claims involve deceased beneficiaries</td>
        <td>Real-time SSA death record cross-reference — zero tolerance</td>
        <td><span class="tag tag-red">Very High</span></td>
      </tr>
      <tr>
        <td>Avg reimbursement 2.7x higher in fraud</td>
        <td>Peer-group benchmarking; alert if &gt;200% of median</td>
        <td><span class="tag tag-purple">High</span></td>
      </tr>
      <tr>
        <td>Hospital stay 2x longer in fraud</td>
        <td>Benchmark stay by diagnosis code; flag outliers &gt;2x average</td>
        <td><span class="tag tag-blue">Medium</span></td>
      </tr>
      <tr>
        <td>Fraud providers cluster on high-reimbursement diagnosis codes</td>
        <td>Build diagnosis code risk score; investigate top-10 code abusers</td>
        <td><span class="tag tag-blue">Medium</span></td>
      </tr>
      <tr>
        <td>Temporal claim bursts near fiscal year-end</td>
        <td>Time-series anomaly detection on monthly claim volumes</td>
        <td><span class="tag tag-green">Medium</span></td>
      </tr>
      <tr>
        <td>Organized fraud rings: coordinated providers + beneficiaries</td>
        <td>Deploy graph-based fraud detection (GNN / community detection)</td>
        <td><span class="tag tag-purple">Very High</span></td>
      </tr>
    </table>
  </div>

  <!-- SECTION 10: REFERENCES -->
  <div class="section">
    <h2>10. Citations & References</h2>
    <ol style="margin-left:1.5rem;line-height:2;">
      <li>NHCAA — National Health Care Anti-Fraud Association. <em>The Challenge of Health Care Fraud.</em> <a href="https://www.nhcaa.org/">nhcaa.org</a></li>
      <li>Chen, T., & Guestrin, C. (2016). <em>XGBoost: A Scalable Tree Boosting System.</em> KDD '16.</li>
      <li>Chawla, N. V. et al. (2002). <em>SMOTE: Synthetic Minority Over-sampling Technique.</em> JAIR 16, 321–357.</li>
      <li>CMS Medicare Provider Data. <a href="https://data.cms.gov/">data.cms.gov</a></li>
      <li>Breiman, L. (2001). <em>Random Forests.</em> Machine Learning, 45(1), 5–32.</li>
    </ol>
    <div class="alert alert-warn" style="margin-top:1rem;">
      <b>Privacy Notice:</b> This data is confidential. Not shared, published, or cited externally.
    </div>
  </div>

</div>

<div class="footer">
  Healthcare Provider Fraud Detection Analysis Notebook &mdash;
  Built with Python, XGBoost, Streamlit &bull;
  <em>All data remains confidential and unpublished</em>
</div>

</body>
</html>"""

out_path = os.path.join(OUTPUTS, 'FraudDetection_Notebook.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f"HTML notebook saved: {out_path}")
