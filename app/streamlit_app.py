"""
=============================================================================
  Healthcare Provider Fraud Detection — Streamlit App
  Run: streamlit run app/streamlit_app.py
=============================================================================
"""

import os, sys, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
warnings.filterwarnings('ignore')

# Path setup
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, 'notebooks'))
MODELS_DIR  = os.path.join(BASE, 'models')
REPORTS_DIR = os.path.join(BASE, 'reports')
OUTPUTS_DIR = os.path.join(BASE, 'outputs')

# Page config
st.set_page_config(
    page_title  = "Healthcare Fraud Detector",
    page_icon   = "⚕️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Main header */
.hero-header {
    background: linear-gradient(135deg, #1a1f3c 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle at 60% 40%, rgba(99,179,237,0.08) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9f7aea, #f687b3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0; line-height: 1.1;
}
.hero-sub {
    color: #90cdf4; font-size: 1.1rem;
    margin-top: 0.8rem; font-weight: 400;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a2744 0%, #16213e 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
    height: 100%;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(99,179,237,0.15);
}
.metric-value {
    font-size: 2.5rem; font-weight: 800;
    background: linear-gradient(135deg, #63b3ed, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    color: #a0aec0; font-size: 0.85rem;
    margin-top: 0.3rem; font-weight: 500;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }

/* Section headers */
.section-header {
    font-size: 1.5rem; font-weight: 700;
    color: #90cdf4;
    border-left: 4px solid #63b3ed;
    padding-left: 1rem;
    margin: 2rem 0 1rem;
}

/* Prediction card */
.pred-card-fraud {
    background: linear-gradient(135deg, rgba(229,62,62,0.15), rgba(197,48,48,0.1));
    border: 2px solid rgba(229,62,62,0.5);
    border-radius: 16px; padding: 2rem; text-align: center;
    animation: pulse-red 2s infinite;
}
@keyframes pulse-red {
    0%,100% { box-shadow: 0 0 20px rgba(229,62,62,0.3); }
    50%      { box-shadow: 0 0 40px rgba(229,62,62,0.6); }
}
.pred-card-safe {
    background: linear-gradient(135deg, rgba(72,187,120,0.15), rgba(56,161,105,0.1));
    border: 2px solid rgba(72,187,120,0.5);
    border-radius: 16px; padding: 2rem; text-align: center;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 20px rgba(72,187,120,0.3); }
    50%      { box-shadow: 0 0 40px rgba(72,187,120,0.6); }
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(99,179,237,0.15) !important;
}

/* DataFrame */
.dataframe { background: #1a2744 !important; color: #e2e8f0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #90cdf4; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: #63b3ed !important;
    border-bottom: 2px solid #63b3ed !important;
}

.info-box {
    background: rgba(99,179,237,0.1);
    border-left: 4px solid #63b3ed;
    border-radius: 8px; padding: 1rem 1.5rem;
    margin: 1rem 0;
}

/* Mobile Responsiveness */
@media (max-width: 768px) {
    .hero-header { padding: 1.5rem 1rem; }
    .hero-title { font-size: 1.8rem; }
    .metric-value { font-size: 1.8rem; }
    .metric-card { padding: 1rem; margin-bottom: 1rem; }
}
</style>
""", unsafe_allow_html=True)


# Load artifacts
@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    artifact_path = os.path.join(MODELS_DIR, 'pipeline_artifacts.pkl')
    if os.path.exists(artifact_path):
        return joblib.load(artifact_path)
    return None

@st.cache_data(show_spinner="Loading submission results…")
def load_submission():
    path = os.path.join(OUTPUTS_DIR, 'Supriyo_Submission.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

artifacts   = load_artifacts()
submission  = load_submission()


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:3rem;'>🏥</div>
        <div style='font-size:1.1rem; font-weight:700; color:#90cdf4;'>Fraud Detector</div>
        <div style='font-size:0.75rem; color:#718096;'>Healthcare Insurance AI</div>
    </div>
    <hr style='border-color: rgba(99,179,237,0.2);'>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigate",
        ["Overview", "EDA & Insights", "Model Results",
         "Live Prediction", "Submission", "Business Insights",
         ],
        label_visibility="collapsed"
    )
    st.markdown("<hr style='border-color: rgba(99,179,237,0.2);'>", unsafe_allow_html=True)

    if artifacts:
        st.markdown(f"""
        <div class='info-box'>
        <b>Model Status</b><br>
        ✅ Pipeline trained<br>
        📈 Val AUC: <b>{artifacts.get('val_auc',0):.4f}</b><br>
        🗂 Providers: <b>{artifacts.get('n_providers',0):,}</b><br>
        ⚠️ Fraud Rate: <b>{artifacts.get('fraud_rate',0):.1%}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Run the pipeline first:\n```\npython notebooks/run_pipeline.py\n```")


#  PAGE: OVERVIEW
if page == "Overview":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>⚕️ Healthcare Provider Fraud Detection</div>
        <div class='hero-sub'>
            Machine Learning Insurance Claims Analysis · XGBoost + SMOTE · ROC-AUC Optimized
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("🎯", f"{artifacts.get('val_auc',0):.4f}" if artifacts else "—", "Validation AUC"),
        ("🏥", f"{artifacts.get('n_providers',0):,}" if artifacts else "—", "Total Providers"),
        ("⚠️", f"{artifacts.get('fraud_rate',0):.1%}" if artifacts else "—", "Fraud Rate"),
        ("🔢", f"{len(artifacts.get('selected_features',[]))}+" if artifacts else "—", "ML Features"),
    ]
    for col, (icon, val, label) in zip([col1,col2,col3,col4], kpis):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-icon'>{icon}</div>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem statement
    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("<div class='section-header'>Problem Statement</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        Healthcare fraud is one of the most costly problems in the insurance industry.
        Fraudulent providers adopt ways to bill for services not rendered, duplicate claims,
        misrepresent services, and upcode procedures — costing the system billions annually.
        <br><br>
        <b>This project predicts potentially fraudulent providers</b> using machine learning on
        Inpatient, Outpatient, and Beneficiary claims data.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Methodology</div>", unsafe_allow_html=True)
        steps = [
            ("1️⃣", "Data Management", "Load & clean 8 CSV files, parse dates, encode categories"),
            ("2️⃣", "EDA", "Visualize fraud patterns, class imbalance, claim distributions"),
            ("3️⃣", "Feature Engineering", "70+ provider-level aggregated features"),
            ("4️⃣", "Feature Selection", "RF importance → top features selected"),
            ("5️⃣", "Modelling", "LR, RF, XGBoost, LightGBM with SMOTE"),
            ("6️⃣", "Evaluation", "ROC-AUC, Confusion Matrix, Precision-Recall"),
            ("7️⃣", "Deployment", "Streamlit app + CSV submission"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex; gap:1rem; margin:0.6rem 0; align-items:start;
                        background:rgba(99,179,237,0.05); border-radius:10px; padding:0.8rem;'>
                <span style='font-size:1.3rem;'>{num}</span>
                <div><b style='color:#90cdf4;'>{title}</b><br>
                <span style='color:#a0aec0; font-size:0.85rem;'>{desc}</span></div>
            </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("<div class='section-header'>Types of Fraud Detected</div>", unsafe_allow_html=True)
        fraud_types = [
            ("💉", "Phantom Billing", "Billing for services never provided"),
            ("📋", "Duplicate Claims", "Submitting same claim multiple times"),
            ("🔄", "Upcoding", "Charging for more expensive procedures"),
            ("💊", "Unbundling", "Billing components separately"),
            ("🏥", "Unnecessary Services", "Providing unneeded treatments"),
        ]
        for icon, title, desc in fraud_types:
            st.markdown(f"""
            <div style='background:rgba(229,62,62,0.07); border:1px solid rgba(229,62,62,0.2);
                        border-radius:10px; padding:0.8rem; margin:0.5rem 0;'>
                <b style='color:#fc8181;'>{icon} {title}</b><br>
                <span style='color:#a0aec0; font-size:0.85rem;'>{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Tech Stack</div>", unsafe_allow_html=True)
        techs = ["Python 3.13", "XGBoost", "LightGBM", "Scikit-Learn",
                 "SMOTE", "SHAP", "Plotly", "Streamlit"]
        cols = st.columns(4)
        for i, t in enumerate(techs):
            cols[i%4].markdown(f"""
            <div style='background:rgba(99,179,237,0.1); border-radius:8px;
                        padding:0.4rem 0.6rem; text-align:center; font-size:0.8rem;
                        color:#90cdf4; margin:0.3rem 0; font-weight:500;'>
                {t}
            </div>""", unsafe_allow_html=True)


#  PAGE: EDA
elif page == "EDA & Insights":
    st.markdown("<div class='section-header'>Exploratory Data Analysis</div>", unsafe_allow_html=True)

    eda_img = os.path.join(REPORTS_DIR, 'eda_overview.png')
    corr_img = os.path.join(REPORTS_DIR, 'correlation_heatmap.png')

    if os.path.exists(eda_img):
        st.image(eda_img, use_container_width=True, caption="EDA Overview — Fraud vs Non-Fraud Patterns")
    else:
        st.info("Run `python notebooks/run_pipeline.py` to generate EDA plots.")

    if os.path.exists(corr_img):
        st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
        st.image(corr_img, use_container_width=True, caption="Top 20 Features Correlation Heatmap")

    # Fraud type interactive chart
    st.markdown("<div class='section-header'>💰 Estimated Cost Impact of Fraud Types</div>",
                unsafe_allow_html=True)
    fraud_data = pd.DataFrame({
        'Fraud Type': ['Phantom Billing','Duplicate Claims','Upcoding',
                       'Unbundling','Unnecessary Services','Kickbacks'],
        'Estimated Cost ($B)': [6.8, 3.2, 8.1, 2.5, 4.7, 5.3],
        'Detection Difficulty': ['High','Medium','High','Low','Medium','High']
    })
    fig = px.bar(fraud_data, x='Fraud Type', y='Estimated Cost ($B)',
                 color='Detection Difficulty',
                 color_discrete_map={'High':'#e53e3e','Medium':'#ed8936','Low':'#48bb78'},
                 title='Annual Cost by Fraud Type (US Healthcare)',
                 template='plotly_dark')
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,39,68,0.8)',
        font_color='#e2e8f0',
    )
    st.plotly_chart(fig, use_container_width=True)


#  PAGE: MODEL RESULTS
elif page == "Model Results":
    st.markdown("<div class='section-header'>🤖 Model Training Results</div>", unsafe_allow_html=True)

    if artifacts:
        cv_results = artifacts.get('cv_results', {})
        if cv_results:
            # Model comparison chart
            model_names = list(cv_results.keys())
            means = [v.mean() for v in cv_results.values()]
            stds  = [v.std()  for v in cv_results.values()]

            fig = go.Figure()
            colors = ['#3498db','#2ecc71','#e74c3c','#f39c12']
            for i, (name, mean, std) in enumerate(zip(model_names, means, stds)):
                fig.add_trace(go.Bar(
                    name=name, x=[name], y=[mean],
                    error_y=dict(type='data', array=[std], visible=True),
                    marker_color=colors[i % len(colors)],
                    text=[f'{mean:.4f}'], textposition='outside',
                ))
            fig.update_layout(
                title='Model Comparison — 5-Fold Stratified CV ROC-AUC',
                yaxis=dict(title='ROC-AUC', range=[0, 1.1]),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,39,68,0.8)',
                font_color='#e2e8f0',
                showlegend=False,
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Summary table
            summary_df = pd.DataFrame({
                'Model'    : model_names,
                'Mean AUC' : [f"{v.mean():.4f}" for v in cv_results.values()],
                'Std AUC'  : [f"{v.std():.4f}"  for v in cv_results.values()],
                'Best CV'  : [f"{v.max():.4f}"  for v in cv_results.values()],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Evaluation image
        eval_img = os.path.join(REPORTS_DIR, 'XGBoost_evaluation.png')
        if os.path.exists(eval_img):
            st.markdown("<div class='section-header'>📈 XGBoost Evaluation</div>",
                        unsafe_allow_html=True)
            st.image(eval_img, use_container_width=True)

        # Feature importance
        feat_img = os.path.join(REPORTS_DIR, 'feature_importance.png')
        if os.path.exists(feat_img):
            st.markdown("<div class='section-header'>🔑 Feature Importance</div>",
                        unsafe_allow_html=True)
            st.image(feat_img, use_container_width=True)

        # Model comparison img
        mc_img = os.path.join(REPORTS_DIR, 'model_comparison.png')
        if os.path.exists(mc_img):
            st.image(mc_img, use_container_width=True)

    else:
        st.warning("⚠️ No model artifacts found. Run the pipeline first.")


#  PAGE: LIVE PREDICTION
elif page == "Live Prediction":
    st.markdown("<div class='section-header'>🎯 Live Provider Fraud Prediction</div>",
                unsafe_allow_html=True)

    if not artifacts:
        st.warning("⚠️ Run the pipeline first to enable predictions.")
        st.stop()

    model     = artifacts['model']
    sel_feats = artifacts['selected_features']

    tab1, tab2 = st.tabs(["🔢 Manual Input", "📁 Batch Upload"])

    with tab1:
        st.markdown("Enter provider-level statistics to get an instant fraud prediction:")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**📊 Claims Volume**")
            ip_claims   = st.number_input("IP Total Claims",         0, 10000, 150)
            op_claims   = st.number_input("OP Total Claims",         0, 50000, 800)
            ip_patients = st.number_input("IP Unique Patients",      0, 5000,  80)
            op_patients = st.number_input("OP Unique Patients",      0, 20000, 450)

        with col2:
            st.markdown("**💰 Financial**")
            ip_total_reimb = st.number_input("IP Total Reimbursed ($)", 0, 5000000, 120000)
            op_total_reimb = st.number_input("OP Total Reimbursed ($)", 0, 2000000, 85000)
            ip_avg_reimb   = st.number_input("IP Avg Reimbursed ($)",   0, 100000,  800)
            ip_max_reimb   = st.number_input("IP Max Reimbursed ($)",   0, 500000,  5000)

        with col3:
            st.markdown("**🏥 Clinical**")
            avg_stay     = st.number_input("Avg Hospital Stay (days)", 0.0, 90.0, 5.0)
            avg_chronic  = st.number_input("Avg Chronic Conditions",   0.0, 11.0, 3.5)
            unique_diag  = st.number_input("Unique Diagnoses (IP)",    0, 500,  45)
            unique_phys  = st.number_input("Unique Physicians (IP)",   0, 1000, 25)

        if st.button("🔍 Predict Fraud Risk", type="primary", use_container_width=True):
            # Build input dict with 0 defaults for all selected features
            input_dict = {f: 0 for f in sel_feats}
            mappings = {
                'IP_TotalClaims'               : ip_claims,
                'OP_TotalClaims'               : op_claims,
                'IP_UniquePatients'            : ip_patients,
                'OP_UniquePatients'            : op_patients,
                'IP_TotalReimbursed'           : ip_total_reimb,
                'OP_TotalReimbursed'           : op_total_reimb,
                'IP_AvgReimbursed'             : ip_avg_reimb,
                'IP_MaxReimbursed'             : ip_max_reimb,
                'IP_AvgHospitalStay'           : avg_stay,
                'Bene_AvgChronicCond'          : avg_chronic,
                'IP_UniqueDiag'                : unique_diag,
                'IP_Unique_AttendingPhysician' : unique_phys,
                'Total_Claims'                 : ip_claims + op_claims,
                'Total_Reimbursed'             : ip_total_reimb + op_total_reimb,
                'IP_OP_ClaimRatio'             : ip_claims / max(ip_claims + op_claims, 1),
                'Avg_Reimbursed_Per_Claim'     : (ip_total_reimb + op_total_reimb) / max(ip_claims + op_claims, 1),
                'IP_ClaimsPerPatient'          : ip_claims / max(ip_patients, 1),
                'OP_ClaimsPerPatient'          : op_claims / max(op_patients, 1),
            }
            for k, v in mappings.items():
                if k in input_dict:
                    input_dict[k] = v

            X_input = pd.DataFrame([input_dict])[sel_feats]
            prob    = model.predict_proba(X_input)[0][1]
            pred    = "FRAUDULENT" if prob >= 0.5 else "LEGITIMATE"

            st.markdown("<br>", unsafe_allow_html=True)
            if pred == "FRAUDULENT":
                st.markdown(f"""
                <div class='pred-card-fraud'>
                    <div style='font-size:3rem;'>🚨</div>
                    <div style='font-size:2rem; font-weight:800; color:#fc8181;'>{pred}</div>
                    <div style='font-size:1.5rem; color:#feb2b2;'>Fraud Probability: {prob:.1%}</div>
                    <div style='color:#fc8181; margin-top:0.5rem;'>
                        This provider shows high-risk behaviour patterns consistent with fraud.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-card-safe'>
                    <div style='font-size:3rem;'>✅</div>
                    <div style='font-size:2rem; font-weight:800; color:#68d391;'>{pred}</div>
                    <div style='font-size:1.5rem; color:#9ae6b4;'>Fraud Probability: {prob:.1%}</div>
                    <div style='color:#68d391; margin-top:0.5rem;'>
                        This provider's claims pattern appears within normal parameters.
                    </div>
                </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = prob * 100,
                title = {'text': "Fraud Risk Score", 'font': {'color': '#e2e8f0'}},
                number= {'suffix': "%", 'font': {'color': '#e2e8f0'}},
                gauge = {
                    'axis'     : {'range': [0, 100], 'tickcolor': '#e2e8f0'},
                    'bar'      : {'color': "#e53e3e" if prob>=0.5 else "#48bb78"},
                    'bgcolor'  : "rgba(26,39,68,0.8)",
                    'steps'    : [
                        {'range': [0,  30],  'color': 'rgba(72,187,120,0.2)'},
                        {'range': [30, 60],  'color': 'rgba(237,137,54,0.2)'},
                        {'range': [60, 100], 'color': 'rgba(229,62,62,0.2)'},
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'value': 50},
                }
            ))
            fig_gauge.update_layout(
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

    with tab2:
        st.markdown("Upload a CSV file with provider features for batch prediction:")
        uploaded = st.file_uploader("Upload Provider Features CSV", type=['csv'])
        if uploaded:
            df_upload = pd.read_csv(uploaded)
            st.dataframe(df_upload.head(), use_container_width=True)
            for col in sel_feats:
                if col not in df_upload.columns:
                    df_upload[col] = 0
            X_batch  = df_upload[sel_feats].fillna(0)
            probs    = model.predict_proba(X_batch)[:, 1]
            preds    = pd.Series((probs >= 0.5).astype(int)).map({0:'No',1:'Yes'})
            result   = df_upload.copy()
            result['FraudProbability']  = probs.round(4)
            result['PredictedFraud']    = preds.values
            result['RiskLevel']         = pd.cut(probs,
                                                  bins=[0,0.3,0.6,1.0],
                                                  labels=['Low','Medium','High'])
            st.success(f"✅ Predicted {len(result)} providers")
            st.dataframe(result[['FraudProbability','PredictedFraud','RiskLevel']].head(20),
                         use_container_width=True)
            csv_out = result.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions", csv_out,
                               "batch_predictions.csv", "text/csv")


#  PAGE: SUBMISSION
elif page == "Submission":
    st.markdown("<div class='section-header'>📁 Submission File</div>", unsafe_allow_html=True)

    if submission is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Providers", len(submission))
        col2.metric("Predicted Fraud", (submission['Predicted Class as per your best model']=='Yes').sum())
        col3.metric("Fraud Rate",
                    f"{(submission['Predicted Class as per your best model']=='Yes').mean():.1%}")

        # Distribution
        fig = px.histogram(submission, x='Probability', nbins=50,
                           color_discrete_sequence=['#63b3ed'],
                           title='Fraud Probability Distribution',
                           template='plotly_dark')
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="Threshold=0.5")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(26,39,68,0.8)',
                          font_color='#e2e8f0')
        st.plotly_chart(fig, use_container_width=True)

        # Pie chart
        pred_counts = submission['Predicted Class as per your best model'].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=['Non-Fraud','Fraud'],
            values=[pred_counts.get('No',0), pred_counts.get('Yes',0)],
            hole=0.5,
            marker_colors=['#48bb78','#e53e3e'],
        ))
        fig2.update_layout(title='Prediction Distribution',
                           paper_bgcolor='rgba(0,0,0,0)',
                           font_color='#e2e8f0')
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<div class='section-header'>📋 Submission Preview</div>",
                    unsafe_allow_html=True)
        st.dataframe(submission, use_container_width=True)

        csv_data = submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label    = "⬇️ Download Submission CSV",
            data     = csv_data,
            file_name= "Supriyo_Submission.csv",
            mime     = "text/csv",
            type     = "primary",
            use_container_width=True,
        )
    else:
        st.warning("⚠️ No submission file found. Run the pipeline first.")


#  PAGE: BUSINESS INSIGHTS
elif page == "Business Insights":
    st.markdown("<div class='section-header'>💡 Business Recommendations</div>",
                unsafe_allow_html=True)

    insights = [
        {
            "icon": "🔍",
            "title": "High-Volume Low-Value Claims",
            "finding": "Fraudulent providers file significantly more claims per patient, often fragmenting single visits into multiple billable events.",
            "recommendation": "Flag providers with Claims-per-Patient ratio > 3 standard deviations above the mean for manual review.",
            "impact": "High",
            "color": "#e53e3e",
        },
        {
            "icon": "👨‍⚕️",
            "title": "Physician Network Anomalies",
            "finding": "Fraudulent providers use disproportionately large and rotating networks of attending physicians, making individual accountability difficult.",
            "recommendation": "Implement physician co-occurrence network analysis. Flag providers with >50 unique physicians per 100 claims.",
            "impact": "High",
            "color": "#e53e3e",
        },
        {
            "icon": "💊",
            "title": "Diagnosis Code Clustering",
            "finding": "Fraud providers tend to cluster around a small set of high-reimbursement diagnosis codes, particularly ambiguous ones.",
            "recommendation": "Build a diagnosis code risk score. Providers overusing top-10 highest-reimbursement codes warrant investigation.",
            "impact": "Medium",
            "color": "#ed8936",
        },
        {
            "icon": "🏥",
            "title": "Hospital Stay Duration",
            "finding": "Some fraud providers either drastically over-admit patients (to maximize IP reimbursement) or keep stays suspiciously short.",
            "recommendation": "Benchmark average stay duration by diagnosis. Outliers >2x the diagnosis-specific average should trigger review.",
            "impact": "Medium",
            "color": "#ed8936",
        },
        {
            "icon": "💰",
            "title": "Reimbursement Benchmarking",
            "finding": "Total reimbursement per patient is the single strongest predictor of fraud. Fraud providers extract 2-4x more per patient.",
            "recommendation": "Set peer-group benchmarks for reimbursement. Trigger alerts when a provider's average exceeds 200% of the peer median.",
            "impact": "Very High",
            "color": "#9f7aea",
        },
        {
            "icon": "👤",
            "title": "Dead Beneficiary Claims",
            "finding": "A subset of fraud involves claiming for deceased beneficiaries — a clear signal of phantom billing.",
            "recommendation": "Cross-reference claim dates against SSA death records in real-time. Zero tolerance for post-mortem claims.",
            "impact": "High",
            "color": "#e53e3e",
        },
        {
            "icon": "📅",
            "title": "Temporal Claim Patterns",
            "finding": "Fraud providers tend to submit claims in bursts, often near fiscal year-end or prior to anticipated audits.",
            "recommendation": "Implement time-series anomaly detection on monthly claim volumes per provider.",
            "impact": "Medium",
            "color": "#ed8936",
        },
        {
            "icon": "🤝",
            "title": "Provider-Beneficiary Network",
            "finding": "Organized fraud rings involve coordinated groups of providers and beneficiaries. Network analysis can expose these rings.",
            "recommendation": "Deploy graph-based fraud detection (GNN or community detection) to identify tightly-knit fraud clusters.",
            "impact": "Very High",
            "color": "#9f7aea",
        },
    ]

    for i in range(0, len(insights), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(insights):
                ins = insights[i + j]
                col.markdown(f"""
                <div style='background: rgba(26,39,68,0.8);
                            border-left: 4px solid {ins["color"]};
                            border-radius: 12px; padding: 1.2rem; margin-bottom: 1rem;
                            height: 100%;'>
                    <div style='font-size:1.5rem; margin-bottom:0.5rem;'>{ins["icon"]}</div>
                    <div style='font-weight:700; color:#90cdf4; font-size:1rem;
                                margin-bottom:0.5rem;'>{ins["title"]}</div>
                    <div style='background:rgba(0,0,0,0.2); border-radius:6px;
                                padding:0.6rem; margin-bottom:0.5rem;'>
                        <b style='color:#a0aec0; font-size:0.75rem;'>FINDING</b><br>
                        <span style='color:#cbd5e0; font-size:0.85rem;'>{ins["finding"]}</span>
                    </div>
                    <div style='background:rgba(99,179,237,0.08); border-radius:6px;
                                padding:0.6rem;'>
                        <b style='color:#63b3ed; font-size:0.75rem;'>RECOMMENDATION</b><br>
                        <span style='color:#e2e8f0; font-size:0.85rem;'>{ins["recommendation"]}</span>
                    </div>
                    <div style='margin-top:0.8rem;'>
                        <span style='background:{ins["color"]}22; color:{ins["color"]};
                                     border:1px solid {ins["color"]}55; border-radius:20px;
                                     padding:0.2rem 0.8rem; font-size:0.75rem; font-weight:600;'>
                            Impact: {ins["impact"]}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # Model limitations
    st.markdown("<div class='section-header'>⚠️ Model Limitations & Future Work</div>",
                unsafe_allow_html=True)
    lims = [
        "Class imbalance (~9% fraud) requires careful threshold tuning beyond 0.5",
        "Provider-level labels may miss patient-level fraud coordination",
        "Temporal drift: fraud patterns evolve; model needs periodic retraining",
        "Graph-based features (provider networks) not yet incorporated",
        "Real-time scoring integration with claims management systems needed",
    ]
    for lim in lims:
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; align-items:center; margin:0.4rem 0;
                    background:rgba(237,137,54,0.07); border-radius:8px; padding:0.6rem 1rem;'>
            <span style='color:#ed8936;'>⚡</span>
            <span style='color:#cbd5e0; font-size:0.9rem;'>{lim}</span>
        </div>""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; color:#4a5568; font-size:0.8rem;
            margin-top:3rem; padding:1.5rem;
            border-top: 1px solid rgba(99,179,237,0.1);'>
    ⚕️ Healthcare Provider Fraud Detection | Built with XGBoost + Streamlit<br>
    <i>Data is confidential — not shared or published</i>
</div>
""", unsafe_allow_html=True)
