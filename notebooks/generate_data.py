"""
=============================================================================
  Healthcare Provider Fraud Detection - Data Generation Script
  Generates realistic synthetic data matching the Kaggle dataset schema
  Use this ONLY if you don't have the real datasets
=============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

np.random.seed(42)
random.seed(42)

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
N_PROVIDERS     = 5410    # realistic count
N_BENEFICIARIES = 138556
N_INPATIENT     = 40474
N_OUTPATIENT    = 517737
FRAUD_RATE      = 0.089   # ~9% fraud

CHRONIC_COLS = [
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
]

DIAG_CODES   = [f'D{i:04d}' for i in range(1, 1000)]
PROC_CODES   = [f'P{i:05d}' for i in range(1, 5000)]
HCPCS_CODES  = [f'H{i:05d}' for i in range(1, 2000)]
STATES       = list(range(1, 53))
COUNTIES     = list(range(1, 3000))

print("Generating synthetic healthcare fraud dataset...")

# ─── Providers ────────────────────────────────────────────────────────────────
provider_ids = [f'PRV{str(i).zfill(5)}' for i in range(1, N_PROVIDERS + 1)]
fraud_labels = np.random.choice(['Yes', 'No'], size=N_PROVIDERS,
                                p=[FRAUD_RATE, 1 - FRAUD_RATE])

train_providers = pd.DataFrame({'Provider': provider_ids, 'PotentialFraud': fraud_labels})
test_providers  = pd.DataFrame({'Provider': provider_ids[:1000]})

# ─── Beneficiaries ────────────────────────────────────────────────────────────
bene_ids = [f'B{str(i).zfill(10)}' for i in range(1, N_BENEFICIARIES + 1)]
dob_list = [datetime(1940, 1, 1) + timedelta(days=np.random.randint(0, 20000))
            for _ in range(N_BENEFICIARIES)]
dod_list = [None if np.random.rand() > 0.15 else
            d + timedelta(days=np.random.randint(100, 5000)) for d in dob_list]

bene_df = pd.DataFrame({
    'BeneID'        : bene_ids,
    'DOB'           : [d.strftime('%Y-%m-%d') for d in dob_list],
    'DOD'           : [d.strftime('%Y-%m-%d') if d else np.nan for d in dod_list],
    'Gender'        : np.random.choice([1, 2], N_BENEFICIARIES),
    'Race'          : np.random.choice([1, 2, 3, 5], N_BENEFICIARIES),
    'State'         : np.random.choice(STATES, N_BENEFICIARIES),
    'County'        : np.random.choice(COUNTIES, N_BENEFICIARIES),
    'NoOfMonths_PartACov': np.random.randint(0, 12, N_BENEFICIARIES),
    'NoOfMonths_PartBCov': np.random.randint(0, 12, N_BENEFICIARIES),
    'IPAnnualReimbursementAmt': np.random.exponential(2000, N_BENEFICIARIES).astype(int),
    'IPAnnualDeductibleAmt'   : np.random.exponential(800, N_BENEFICIARIES).astype(int),
    'OPAnnualReimbursementAmt': np.random.exponential(1500, N_BENEFICIARIES).astype(int),
    'OPAnnualDeductibleAmt'   : np.random.exponential(400, N_BENEFICIARIES).astype(int),
})
for col in CHRONIC_COLS:
    bene_df[col] = np.random.choice([1, 2], N_BENEFICIARIES, p=[0.3, 0.7])

# ─── Inpatient Claims ─────────────────────────────────────────────────────────
claim_ids_ip = [f'CLM{str(i).zfill(9)}' for i in range(1, N_INPATIENT + 1)]
admit_dates  = [datetime(2009, 1, 1) + timedelta(days=np.random.randint(0, 730))
                for _ in range(N_INPATIENT)]
stay_days    = np.random.randint(1, 30, N_INPATIENT)
discharge_dates = [a + timedelta(days=int(s)) for a, s in zip(admit_dates, stay_days)]

ip_df = pd.DataFrame({
    'ClaimID'               : claim_ids_ip,
    'BeneID'                : np.random.choice(bene_ids, N_INPATIENT),
    'ClaimStartDt'          : [d.strftime('%Y-%m-%d') for d in admit_dates],
    'ClaimEndDt'            : [d.strftime('%Y-%m-%d') for d in discharge_dates],
    'Provider'              : np.random.choice(provider_ids, N_INPATIENT),
    'InscClaimAmtReimbursed': np.random.exponential(5000, N_INPATIENT).astype(int),
    'AttendingPhysician'    : [f'PHY{np.random.randint(1000,9999)}' for _ in range(N_INPATIENT)],
    'OperatingPhysician'    : [f'PHY{np.random.randint(1000,9999)}' if np.random.rand()>0.3 else np.nan
                               for _ in range(N_INPATIENT)],
    'OtherPhysician'        : [f'PHY{np.random.randint(1000,9999)}' if np.random.rand()>0.5 else np.nan
                               for _ in range(N_INPATIENT)],
    'AdmissionDt'           : [d.strftime('%Y-%m-%d') for d in admit_dates],
    'ClmAdmitDiagnosisCode' : np.random.choice(DIAG_CODES, N_INPATIENT),
    'DischargeDt'           : [d.strftime('%Y-%m-%d') for d in discharge_dates],
    'DiagnosisGroupCode'    : np.random.choice([f'G{i:03d}' for i in range(1, 300)], N_INPATIENT),
    'ClmDiagnosisCode_1'    : np.random.choice(DIAG_CODES, N_INPATIENT),
    'ClmDiagnosisCode_2'    : np.random.choice(DIAG_CODES + [np.nan]*200, N_INPATIENT),
    'ClmDiagnosisCode_3'    : np.random.choice(DIAG_CODES + [np.nan]*400, N_INPATIENT),
    'ClmDiagnosisCode_4'    : np.random.choice(DIAG_CODES + [np.nan]*600, N_INPATIENT),
    'ClmDiagnosisCode_5'    : np.random.choice(DIAG_CODES + [np.nan]*800, N_INPATIENT),
    'ClmProcedureCode_1'    : np.random.choice(PROC_CODES + [np.nan]*300, N_INPATIENT),
    'ClmProcedureCode_2'    : np.random.choice(PROC_CODES + [np.nan]*500, N_INPATIENT),
    'ClmProcedureCode_3'    : np.random.choice(PROC_CODES + [np.nan]*700, N_INPATIENT),
    'DeductibleAmtPaid'     : np.random.exponential(300, N_INPATIENT).astype(int),
    'ClaimSettlementDelay'  : np.random.randint(1, 60, N_INPATIENT),
})

# ─── Outpatient Claims ────────────────────────────────────────────────────────
claim_ids_op = [f'CLMOP{str(i).zfill(8)}' for i in range(1, N_OUTPATIENT + 1)]
claim_dates  = [datetime(2009, 1, 1) + timedelta(days=np.random.randint(0, 730))
                for _ in range(N_OUTPATIENT)]

op_df = pd.DataFrame({
    'ClaimID'               : claim_ids_op,
    'BeneID'                : np.random.choice(bene_ids, N_OUTPATIENT),
    'ClaimStartDt'          : [d.strftime('%Y-%m-%d') for d in claim_dates],
    'ClaimEndDt'            : [(d + timedelta(days=np.random.randint(0, 5))).strftime('%Y-%m-%d')
                               for d in claim_dates],
    'Provider'              : np.random.choice(provider_ids, N_OUTPATIENT),
    'InscClaimAmtReimbursed': np.random.exponential(800, N_OUTPATIENT).astype(int),
    'AttendingPhysician'    : [f'PHY{np.random.randint(1000,9999)}' for _ in range(N_OUTPATIENT)],
    'OperatingPhysician'    : [f'PHY{np.random.randint(1000,9999)}' if np.random.rand()>0.5 else np.nan
                               for _ in range(N_OUTPATIENT)],
    'OtherPhysician'        : [f'PHY{np.random.randint(1000,9999)}' if np.random.rand()>0.7 else np.nan
                               for _ in range(N_OUTPATIENT)],
    'ClmDiagnosisCode_1'    : np.random.choice(DIAG_CODES, N_OUTPATIENT),
    'ClmDiagnosisCode_2'    : np.random.choice(DIAG_CODES + [np.nan]*200, N_OUTPATIENT),
    'ClmDiagnosisCode_3'    : np.random.choice(DIAG_CODES + [np.nan]*400, N_OUTPATIENT),
    'ClmDiagnosisCode_4'    : np.random.choice(DIAG_CODES + [np.nan]*600, N_OUTPATIENT),
    'ClmProcedureCode_1'    : np.random.choice(PROC_CODES + [np.nan]*300, N_OUTPATIENT),
    'ClmProcedureCode_2'    : np.random.choice(PROC_CODES + [np.nan]*500, N_OUTPATIENT),
    'ClmHcpcsCode'          : np.random.choice(HCPCS_CODES + [np.nan]*400, N_OUTPATIENT),
    'DeductibleAmtPaid'     : np.random.exponential(100, N_OUTPATIENT).astype(int),
    'ClaimSettlementDelay'  : np.random.randint(1, 30, N_OUTPATIENT),
})

# ─── Save files ───────────────────────────────────────────────────────────────
print("Saving CSV files...")
train_providers.to_csv(os.path.join(RAW_DIR, 'Train-1542865627584.csv'), index=False)
test_providers.to_csv(os.path.join(RAW_DIR, 'Test-1542969243754.csv'), index=False)
bene_df.to_csv(os.path.join(RAW_DIR, 'Train_Beneficiarydata-1542865627584.csv'), index=False)
bene_df.sample(N_BENEFICIARIES//5, random_state=42).to_csv(
    os.path.join(RAW_DIR, 'Test_Beneficiarydata-1542969243754.csv'), index=False)
ip_df.to_csv(os.path.join(RAW_DIR, 'Train_Inpatientdata-1542865627584.csv'), index=False)
ip_df.sample(N_INPATIENT//5, random_state=42).to_csv(
    os.path.join(RAW_DIR, 'Test_Inpatientdata-1542969243754.csv'), index=False)
op_df.to_csv(os.path.join(RAW_DIR, 'Train_Outpatientdata-1542865627584.csv'), index=False)
op_df.sample(N_OUTPATIENT//5, random_state=42).to_csv(
    os.path.join(RAW_DIR, 'Test_Outpatientdata-1542969243754.csv'), index=False)

print("✅ All synthetic datasets saved to data/raw/")
print(f"   Providers  : {N_PROVIDERS} (Train) | 1000 (Test)")
print(f"   Benes      : {N_BENEFICIARIES}")
print(f"   Inpatient  : {N_INPATIENT}")
print(f"   Outpatient : {N_OUTPATIENT}")
