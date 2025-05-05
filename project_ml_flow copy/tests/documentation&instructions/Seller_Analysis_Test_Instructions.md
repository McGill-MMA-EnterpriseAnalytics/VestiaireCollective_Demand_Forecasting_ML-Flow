
# 🧪 How to Run Each Test Notebook

Each test is implemented in a separate Jupyter notebook under the `testing_notebooks/` folder.  
You can run the tests manually using Jupyter Notebook or VSCode.

---

## ✅ Step-by-Step Instructions

1. **Install the required packages**:
```bash
pip install -r requirements.txt
```

2. **Open Jupyter Notebook or JupyterLab**:
```bash
jupyter notebook
```

3. **Navigate to** the `testing_notebooks/` directory.

4. **Open one of the notebooks below** and **run all cells**.

---

## 📂 Notebook Guide

| Notebook | What It Tests | How to Run |
|:--|:--|:--|
| `1_Unit_Tests.ipynb` | Checks missing value handling and column dropping logic | Run all cells to validate basic data cleaning |
| `2_Data_Tests.ipynb` | Checks for duplicates, type mismatches, and missing values | Run to confirm data integrity |
| `3_Model_Validation_Tests.ipynb` | Placeholder: Will test model quality like AUC (if model is added) | Customize with model evaluation later |
| `4_Model_Performance_Tests.ipynb` | Placeholder: Measures training or prediction time | Use with large datasets or deployed models |
| `5_Integration_Tests.ipynb` | Validates the full pipeline (cleaning + type detection) | Run all to ensure end-to-end logic works |
| `6_Data_Skew_Tests.ipynb` | Compares feature distributions between current and new data | Use when evaluating new datasets |
| `7_Load_Tests.ipynb` | Simulates 10x dataset and tests scalability | Run to verify performance under load |

---

## 🛠 Tips

- If import paths don’t work, ensure this line is near the top of each notebook:
```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src')))
```

- You can edit test values or inputs inside each notebook to match new datasets or use cases.
