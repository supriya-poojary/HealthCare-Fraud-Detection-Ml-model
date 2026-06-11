import os
import pandas as pd

print("=== OUTPUT FILES ===")
for f in sorted(os.listdir("outputs")):
    size = os.path.getsize(os.path.join("outputs", f))
    print(f"  {f}  ({size:,} bytes)")

print()
print("=== MODEL FILES ===")
for f in sorted(os.listdir("models")):
    size = os.path.getsize(os.path.join("models", f))
    print(f"  {f}  ({size:,} bytes)")

print()
print("=== REPORT PLOTS ===")
for f in sorted(os.listdir("reports")):
    size = os.path.getsize(os.path.join("reports", f))
    print(f"  {f}  ({size:,} bytes)")

print()
sub = pd.read_csv("outputs/Supriyo_Submission.csv")
print("=== SUBMISSION FILE ===")
print("Columns:", list(sub.columns))
print("Shape:  ", sub.shape)
print(sub.head())
pred_col = sub.columns[2]
fraud_count = (sub[pred_col] == "Yes").sum()
print(f"Fraud predicted: {fraud_count} / {len(sub)}")

# Verify notebook files exist
nb_path  = "notebooks/Healthcare_Provider_Fraud_Detection.ipynb"
html1    = "outputs/Healthcare_Provider_Fraud_Detection.html"
html2    = "outputs/FraudDetection_Notebook.html"
sub2     = "outputs/Your_Full_Name_Submission.csv"
print()
print("=== KEY FILE CHECKS ===")
for p in [nb_path, html1, html2, sub2]:
    exists = "OK" if os.path.exists(p) else "MISSING"
    print(f"  [{exists}]  {p}")
