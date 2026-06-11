"""
=============================================================================
  Healthcare Provider Fraud Detection
  Step 3 — Feature Engineering  (Provider-level aggregation)
=============================================================================
"""

import pandas as pd
import numpy as np

CHRONIC_COLS = [
    'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
    'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
    'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
    'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
]


def _agg_claims(claims_df, claim_type):
    """Aggregate one claims table to Provider level."""
    diag_cols = [c for c in claims_df.columns if 'DiagnosisCode' in c]
    proc_cols  = [c for c in claims_df.columns if 'ProcedureCode' in c]

    grp = claims_df.groupby('Provider')

    feats = pd.DataFrame(index=grp.groups.keys())
    feats.index.name = 'Provider'

    prefix = claim_type  # 'IP' or 'OP'

    feats[f'{prefix}_TotalClaims']        = grp['ClaimID'].count()
    feats[f'{prefix}_TotalReimbursed']    = grp['InscClaimAmtReimbursed'].sum()
    feats[f'{prefix}_AvgReimbursed']      = grp['InscClaimAmtReimbursed'].mean()
    feats[f'{prefix}_MaxReimbursed']      = grp['InscClaimAmtReimbursed'].max()
    feats[f'{prefix}_StdReimbursed']      = grp['InscClaimAmtReimbursed'].std().fillna(0)
    feats[f'{prefix}_TotalDeductible']    = grp['DeductibleAmtPaid'].sum()
    feats[f'{prefix}_AvgDeductible']      = grp['DeductibleAmtPaid'].mean()
    feats[f'{prefix}_UniquePatients']     = grp['BeneID'].nunique()

    # Physician features
    for phy in ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']:
        if phy in claims_df.columns:
            feats[f'{prefix}_Unique_{phy}'] = grp[phy].nunique()

    feats[f'{prefix}_ClaimsPerPatient']   = (feats[f'{prefix}_TotalClaims'] /
                                              feats[f'{prefix}_UniquePatients'].replace(0, 1))

    # Diagnosis diversity
    all_diag = claims_df.melt(id_vars='Provider', value_vars=diag_cols,
                               value_name='diag').dropna(subset=['diag'])
    diag_grp = all_diag.groupby('Provider')['diag'].nunique().rename(f'{prefix}_UniqueDiag')
    feats = feats.join(diag_grp, how='left')
    feats[f'{prefix}_UniqueDiag'].fillna(0, inplace=True)

    # Procedure diversity
    if proc_cols:
        all_proc = claims_df.melt(id_vars='Provider', value_vars=proc_cols,
                                   value_name='proc').dropna(subset=['proc'])
        proc_grp = all_proc.groupby('Provider')['proc'].nunique().rename(f'{prefix}_UniqueProc')
        feats = feats.join(proc_grp, how='left')
        feats[f'{prefix}_UniqueProc'].fillna(0, inplace=True)

    # Duration features
    if 'ClaimDuration' in claims_df.columns:
        feats[f'{prefix}_AvgClaimDuration'] = grp['ClaimDuration'].mean()
        feats[f'{prefix}_MaxClaimDuration'] = grp['ClaimDuration'].max()

    if 'HospitalStayDays' in claims_df.columns:
        feats[f'{prefix}_AvgHospitalStay'] = grp['HospitalStayDays'].mean()
        feats[f'{prefix}_MaxHospitalStay'] = grp['HospitalStayDays'].max()

    return feats.reset_index()


def _agg_beneficiary(bene_df, claims_df):
    """Aggregate beneficiary features at Provider level via claims."""
    bene_claims = claims_df[['Provider', 'BeneID']].merge(
        bene_df, on='BeneID', how='left')

    grp = bene_claims.groupby('Provider')

    feats = pd.DataFrame(index=grp.groups.keys())
    feats.index.name = 'Provider'

    feats['Bene_AvgAge']          = grp['Age'].mean() if 'Age' in bene_claims.columns else 0
    feats['Bene_DeadCount']       = grp['IsDead'].sum() if 'IsDead' in bene_claims.columns else 0
    feats['Bene_AvgChronicCond']  = grp['NumChronicCond'].mean() if 'NumChronicCond' in bene_claims.columns else 0
    feats['Bene_MaxChronicCond']  = grp['NumChronicCond'].max() if 'NumChronicCond' in bene_claims.columns else 0

    for col in CHRONIC_COLS:
        if col in bene_claims.columns:
            feats[f'Bene_{col}_Rate'] = grp[col].mean()

    if 'IPAnnualReimbursementAmt' in bene_claims.columns:
        feats['Bene_AvgIPReimbursement'] = grp['IPAnnualReimbursementAmt'].mean()
        feats['Bene_AvgOPReimbursement'] = grp['OPAnnualReimbursementAmt'].mean()

    return feats.reset_index()


def build_provider_features(train_ip, train_op, train_bene):
    """
    Full provider-level feature table.
    Returns a DataFrame indexed by Provider.
    """
    print("  Building IP features *")
    ip_feats   = _agg_claims(train_ip, 'IP')

    print("  Building OP features *")
    op_feats   = _agg_claims(train_op, 'OP')

    # Combined claims for beneficiary lookup
    combined   = pd.concat([
        train_ip[['Provider', 'BeneID']],
        train_op[['Provider', 'BeneID']]
    ], ignore_index=True).drop_duplicates()

    print("  Building Beneficiary features *")
    bene_feats = _agg_beneficiary(train_bene, combined)

    print("  Merging all feature tables *")
    features = ip_feats.merge(op_feats,   on='Provider', how='outer')
    features = features.merge(bene_feats, on='Provider', how='outer')

    # Derived cross-table features
    features['Total_Claims']       = features['IP_TotalClaims'].fillna(0) + features['OP_TotalClaims'].fillna(0)
    features['Total_Reimbursed']   = features['IP_TotalReimbursed'].fillna(0) + features['OP_TotalReimbursed'].fillna(0)
    features['IP_OP_ClaimRatio']   = (features['IP_TotalClaims'].fillna(0) /
                                       features['Total_Claims'].replace(0, 1))
    features['Avg_Reimbursed_Per_Claim'] = (features['Total_Reimbursed'] /
                                             features['Total_Claims'].replace(0, 1))

    features.fillna(0, inplace=True)
    print(f"  * Feature matrix shape: {features.shape}")
    return features


if __name__ == '__main__':
    print("Feature engineering module loaded.")
