import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print("--- Step 1: Creating Notebook Template ---")
# Run create_notebook.py to generate the template
os.system(f"python {os.path.join(BASE_DIR, 'notebooks', 'create_notebook.py')}")

notebook_path = os.path.join(BASE_DIR, 'notebooks', 'Healthcare_Provider_Fraud_Detection.ipynb')
html_out_path = os.path.join(BASE_DIR, 'outputs', 'Healthcare_Provider_Fraud_Detection.html')

print("--- Step 2: Loading Notebook ---")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

print("--- Step 3: Executing Notebook (this may take a minute) ---")
# Configure execution path to be the project root directory
ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
ep.preprocess(nb, {'metadata': {'path': BASE_DIR}})

print("--- Step 4: Saving Executed Notebook (.ipynb) ---")
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Executed notebook saved: {notebook_path}")

print("--- Step 5: Exporting Notebook to HTML ---")
html_exporter = HTMLExporter()
# Exclude input code prompt if needed, or leave it standard
(body, resources) = html_exporter.from_notebook_node(nb)

os.makedirs(os.path.dirname(html_out_path), exist_ok=True)
with open(html_out_path, 'w', encoding='utf-8') as f:
    f.write(body)
print(f"HTML notebook report saved: {html_out_path}")
print("[OK] Notebook compilation successful!")
