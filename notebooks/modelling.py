"""
=============================================================================
  Healthcare Provider Fraud Detection
  Steps 4-6 — Modelling, Evaluation, Submission
=============================================================================
"""

import os, warnings, joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing   import LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (roc_auc_score, classification_report,
                                     confusion_matrix, roc_curve, precision_recall_curve,
                                     average_precision_score)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling    import SMOTE
import xgboost  as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS  = os.path.join(BASE, 'models')
OUTPUTS = os.path.join(BASE, 'outputs')
REPORTS = os.path.join(BASE, 'reports')
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


# ─── Feature Selection ──────────────────────────────────────────────────────
def select_features(X_train, y_train, threshold=0.001):
    """Use Random Forest importance to select top features."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                 class_weight='balanced')
    rf.fit(X_train, y_train)
    selector = SelectFromModel(rf, threshold=threshold, prefit=True)
    mask     = selector.get_support()
    selected = X_train.columns[mask].tolist()
    importances = pd.Series(rf.feature_importances_, index=X_train.columns)
    print(f"  Selected {len(selected)} / {len(X_train.columns)} features")
    return selected, importances.sort_values(ascending=False)


# ─── Models to Compare ──────────────────────────────────────────────────────
def get_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=10, use_label_encoder=False,
            eval_metric='logloss', random_state=42),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            class_weight='balanced', random_state=42, verbose=-1),
    }


# ─── Cross-Validation Comparison ────────────────────────────────────────────
def compare_models(X, y, cv=5):
    results = {}
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    for name, model in get_models().items():
        print(f"  CV * {name} *", end=' ', flush=True)
        scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
        results[name] = scores
        print(f"AUC = {scores.mean():.4f} * {scores.std():.4f}")
    return results


# ─── Train Best Model ───────────────────────────────────────────────────────
def train_best_model(X_train, y_train, use_smote=True):
    if use_smote:
        print("  Applying SMOTE *")
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

    best = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )
    print("  Training XGBoost *")
    best.fit(X_train, y_train)
    return best


# ─── Evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, label='XGBoost', save_dir=REPORTS):
    prob  = model.predict_proba(X_test)[:, 1]
    pred  = (prob >= 0.5).astype(int)
    auc   = roc_auc_score(y_test, prob)
    ap    = average_precision_score(y_test, prob)

    print(f"\n{'='*50}")
    print(f"  Model  : {label}")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, pred, target_names=['Non-Fraud','Fraud'])}")

    # ── Confusion matrix plot ────────────────────────────────────────────────
    cm  = confusion_matrix(y_test, pred)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{label} — Evaluation', fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud','Fraud'],
                yticklabels=['Non-Fraud','Fraud'], ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

    # ── ROC curve ───────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, prob)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={auc:.3f}')
    axes[1].plot([0,1],[0,1],'k--')
    axes[1].set_xlabel('False Positive Rate'); axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve'); axes[1].legend()

    # ── Precision-Recall curve ──────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, prob)
    axes[2].plot(rec, prec, color='steelblue', lw=2, label=f'AP={ap:.3f}')
    axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
    axes[2].set_title('Precision-Recall Curve'); axes[2].legend()

    plt.tight_layout()
    out = os.path.join(save_dir, f'{label.replace(" ","_")}_evaluation.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Evaluation plot saved * {out}")
    return auc, prob


# ─── Feature Importance Plot ─────────────────────────────────────────────────
def plot_feature_importance(importances, top_n=20, save_dir=REPORTS):
    top = importances.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(top))]
    top.iloc[::-1].plot(kind='barh', ax=ax, color=colors[::-1])
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    out = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Feature importance plot saved * {out}")


# ─── Submission ──────────────────────────────────────────────────────────────
def make_submission(model, X_test_feats, test_providers, name='Supriyo_Submission'):
    prob = model.predict_proba(X_test_feats)[:, 1]
    pred = pd.Series((prob >= 0.5).astype(int)).map({0: 'No', 1: 'Yes'})
    submission = pd.DataFrame({
        'Provider'                               : test_providers['Provider'].values,
        'Probability'                            : prob,
        'Predicted Class as per your best model' : pred.values,
    })
    out = os.path.join(OUTPUTS, f'{name}.csv')
    submission.to_csv(out, index=False)
    print(f"\n  * Submission saved * {out}")
    print(submission.head())
    return submission


if __name__ == '__main__':
    print("Modelling module loaded.")
