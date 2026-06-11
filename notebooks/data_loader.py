"""
=============================================================================
  Healthcare Provider Fraud Detection
  Step 1 — Data Loading & Management
=============================================================================
"""

import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings('ignore')

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW     = os.path.join(BASE, 'data', 'raw')
PROC    = os.path.join(BASE, 'data', 'processed')
TRAIN_DIR = os.path.join(RAW, 'Case Study', 'Training Data')
TEST_DIR  = os.path.join(RAW, 'Case Study', 'Unseen Data')
os.makedirs(PROC, exist_ok=True)

# ── File map (auto-detect actual filenames) ─────────────────────────────────
def _find(pattern, search_dir=None):
    """Return first file in search_dir (or RAW) matching pattern."""
    dirs_to_check = [search_dir] if search_dir else [TRAIN_DIR, TEST_DIR, RAW]
    for d in dirs_to_check:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if pattern.lower() in f.lower():
                return os.path.join(d, f)
    raise FileNotFoundError(f"No file matching '{pattern}' in searched directories")

def load_all():
    print("=" * 60)
    print("  Loading datasets ...")
    print("=" * 60)

    train_labels   = pd.read_csv(_find('Train-',             TRAIN_DIR))
    test_labels    = pd.read_csv(_find('Unseen-',            TEST_DIR))
    train_bene     = pd.read_csv(_find('Train_Beneficiary',  TRAIN_DIR))
    test_bene      = pd.read_csv(_find('Unseen_Beneficiary', TEST_DIR))
    train_inpat    = pd.read_csv(_find('Train_Inpatient',    TRAIN_DIR))
    test_inpat     = pd.read_csv(_find('Unseen_Inpatient',   TEST_DIR))
    train_outpat   = pd.read_csv(_find('Train_Outpatient',   TRAIN_DIR))
    test_outpat    = pd.read_csv(_find('Unseen_Outpatient',  TEST_DIR))

    for name, df in [
        ('train_labels', train_labels), ('test_labels', test_labels),
        ('train_bene',   train_bene),   ('test_bene',   test_bene),
        ('train_inpat',  train_inpat),  ('test_inpat',  test_inpat),
        ('train_outpat', train_outpat), ('test_outpat', test_outpat),
    ]:
        print(f"  {name:<20} {df.shape}")

    return (train_labels, test_labels,
            train_bene,   test_bene,
            train_inpat,  test_inpat,
            train_outpat, test_outpat)


# ── Date columns ────────────────────────────────────────────────────────────
DATE_COLS = ['DOB', 'DOD', 'ClaimStartDt', 'ClaimEndDt',
             'AdmissionDt', 'DischargeDt']

def parse_dates(df):
    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df


# ── Basic cleaning ───────────────────────────────────────────────────────────
def clean_beneficiary(df):
    df = parse_dates(df.copy())
    df['Age']     = (pd.Timestamp('2009-12-01') - df['DOB']).dt.days // 365
    df['IsDead']  = df['DOD'].notna().astype(int)
    CHRONIC = [c for c in df.columns if 'ChronicCond' in c]
    # recode: 2 → 0 (no condition), 1 → 1 (has condition)
    for c in CHRONIC:
        df[c] = df[c].map({1: 1, 2: 0}).fillna(0).astype(int)
    df['NumChronicCond'] = df[CHRONIC].sum(axis=1)
    return df

def clean_claims(df, claim_type='IP'):
    df = parse_dates(df.copy())
    df['ClaimDuration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days.fillna(0)
    if claim_type == 'IP' and 'AdmissionDt' in df.columns and 'DischargeDt' in df.columns:
        df['HospitalStayDays'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days.fillna(0)
    df['ClaimType'] = claim_type
    return df


if __name__ == '__main__':
    data = load_all()
    print("\n* Data loading complete.")
